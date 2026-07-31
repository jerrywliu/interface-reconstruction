"""Isolated adversarial probe for the script-only publication transaction."""

from __future__ import annotations

import ast
import json
import os
import shutil
import stat
import sys
from pathlib import Path


def _load_script_boundary(path: Path) -> dict:
    descriptor = os.open(path.with_name("run_final_figure_orchestrator"), os.O_RDONLY)
    if descriptor != 9:
        os.dup2(descriptor, 9)
        os.close(descriptor)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
        )
    ]
    ast.fix_missing_locations(tree)
    namespace = {"__file__": str(path), "__name__": "__main__"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def _write_candidate_pdf(path: Path, label: str) -> None:
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_name = "TransactionProbeVera"
    font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    drawing = canvas.Canvas(
        str(path),
        pagesize=(72, 72),
        initialFontName=font_name,
        initialFontSize=7,
    )
    drawing.setFont(font_name, 7)
    drawing.drawString(4, 36, label[:18])
    drawing.line(4, 30, 68, 30)
    drawing.save()


def _make_staging(namespace: dict, root: Path):
    staging = root / "staging"
    staging.mkdir(parents=True)
    specs = namespace["load_candidate_allowlist"]()
    candidates = []
    for spec in specs:
        relative = Path("candidates") / spec.root / spec.pdf
        candidate = staging / relative
        _write_candidate_pdf(candidate, spec.candidate_id)
        candidates.append(
            {
                "candidate_id": spec.candidate_id,
                "path": relative.as_posix(),
                "sha256": namespace["file_sha256"](candidate),
                "generator": "transaction_probe",
            }
        )

    allowlist = staging / namespace["PRIVATE_ALLOWLIST"]
    allowlist.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(namespace["DEFAULT_ALLOWLIST"], allowlist)
    allowlist.chmod(0o400)
    manifest = staging / namespace["ORCHESTRATION_MANIFEST"]
    namespace["atomic_write_json"](
        manifest,
        {
            "schema_version": namespace["ORCHESTRATION_SCHEMA_VERSION"],
            "allowlist": {
                "path": namespace["PRIVATE_ALLOWLIST"],
                "sha256": namespace["file_sha256"](allowlist),
            },
            "candidates": candidates,
            "snapshot_artifacts": [],
        },
    )
    release_anchor = {
        "name": "transaction_probe",
        "source_commit": "5" * 40,
        "reconstruction_profile": dict(namespace["PROFILE"]),
        "artifacts": {},
    }
    state = namespace["_create_acceptance_state"](
        figure_root=staging / "candidates" / "figure_root",
        c0_root=staging / "candidates" / "c0_root",
        snapshot_root=staging,
        release_anchor=release_anchor,
        generator_source_commit="6" * 40,
        orchestration_record=manifest,
        allowlist_path=allowlist,
        candidate_records=candidates,
    )
    return staging, manifest, specs, state


def _run_success(namespace: dict, root: Path, runtime) -> dict:
    root.mkdir()
    staging, manifest, specs, state = _make_staging(namespace, root)
    output = root / "publication"
    reservation = namespace["_reserve_publication"](output)
    reservation_path = reservation.path
    namespace["_complete_publication_transaction"](
        staging=staging,
        output_root=output,
        reservation=reservation,
        manifest_path=manifest,
        acceptance_state=state,
        candidate_specs=specs,
        runtime=runtime,
    )

    qa = namespace["load_json_object"](
        output / "review" / "figure_candidate_vector_qa.json"
    )
    source_map = namespace["load_json_object"](
        output / "review" / "figure_candidate_source_map.json"
    )
    result = {
        "candidate_pdfs": len(list((output / "candidates").rglob("*.pdf"))),
        "previews": len(list((output / "review" / "previews").glob("*.png"))),
        "candidate_rasters": sum(
            report["image_objects"] for report in qa["candidate_reports"]
        ),
        "candidate_fonts_embedded": all(
            report["fonts"] and not report["issues"]
            for report in qa["candidate_reports"]
        ),
        "review_rasters": qa["review_report"]["image_objects"],
        "review_fonts_embedded": bool(qa["review_report"]["fonts"])
        and not qa["review_report"]["issues"],
        "review_pages": qa["measured_review_page_count"],
        "preview_dpi": sorted({row["png_dpi_x"] for row in source_map["candidates"]}),
        "root_mode": stat.S_IMODE(output.stat().st_mode),
        "reservation_cleaned": not reservation_path.exists(),
        "staging_cleaned": not staging.exists(),
        "publish_temporaries": len(list(root.glob(".publication.publish-*"))),
    }
    namespace["_remove_tree"](output)
    return result


def _run_destination_attack(namespace: dict, root: Path, runtime) -> dict:
    root.mkdir()
    staging, manifest, specs, state = _make_staging(namespace, root)
    output = root / "publication"
    reservation = namespace["_reserve_publication"](output)
    reservation_path = reservation.path

    output.mkdir()
    sentinel = output / "winner.txt"
    sentinel.write_text("competing winner\n", encoding="utf-8")
    error = None
    try:
        namespace["_complete_publication_transaction"](
            staging=staging,
            output_root=output,
            reservation=reservation,
            manifest_path=manifest,
            acceptance_state=state,
            candidate_specs=specs,
            runtime=runtime,
        )
    except namespace["FinalFigureOrchestrationError"] as exc:
        error = str(exc)
    result = {
        "error": error,
        "winner": sentinel.read_text(encoding="utf-8"),
        "reservation_cleaned": not reservation_path.exists(),
        "staging_cleaned": not staging.exists(),
        "acceptance_temporaries": len(list(root.glob(".review.staging-*"))),
        "publish_temporaries": len(list(root.glob(".publication.publish-*"))),
    }
    shutil.rmtree(output)
    return result


def main() -> int:
    boundary = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    namespace = _load_script_boundary(boundary)
    runtime = namespace["prepare_trusted_figure_runtime"](work / "runtime")
    result = {
        "success": _run_success(namespace, work / "success", runtime),
        "destination_attack": _run_destination_attack(
            namespace, work / "destination-attack", runtime
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
