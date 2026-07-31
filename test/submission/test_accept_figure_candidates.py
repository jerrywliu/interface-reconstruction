import csv
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from submission.accept_figure_candidates import (
    DEFAULT_ALLOWLIST,
    EXPECTED_COUNTS,
    PROVENANCE_MANIFEST_SPECS,
    AcceptedCandidate,
    CandidateSpec,
    FigureAcceptanceError,
    PdfPageInfo,
    PreviewRecord,
    ProvenanceEvidence,
    accept_figure_candidates,
    build_vector_review_pdf,
    load_candidate_allowlist,
    release_figure_anchor,
    render_pdf_preview,
    verify_review_page_map,
)
from submission.audit_final_release import AuditReport
from submission.pdf_vector_qa import FontRecord, PdfQaReport, inspect_pdf


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "5" * 40
PROFILE = {
    "plic_fallback": "LVIRA",
    "corner_behavior_profile": "pre_f8_corner",
    "rescue_profile": "exact_linear_support_only",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _passing_report(path, *, require_fonts=True):
    assert require_fonts
    return PdfQaReport(
        path=str(path),
        image_objects=0,
        fonts=(FontRecord("Embedded", True, True, "TrueType"),),
        issues=(),
    )


def _passing_release_audit(path):
    return AuditReport(Path(path))


def _generation():
    return {
        "generated_at_utc": "2026-07-31T00:00:00+00:00",
        "source_commit": SOURCE_COMMIT,
        "source_branch": "submission-source",
        "source_dirty": False,
        "source_status": [],
        "reconstruction_profile": dict(PROFILE),
        "profile_application": "test fixture",
    }


def _make_release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    _write_json(
        root / "submission_config.resolved.json",
        {
            "source": {"target_commit": SOURCE_COMMIT},
            "production_method": {
                "unresolved_orientation_fallback": PROFILE["plic_fallback"],
                "corner_behavior_profile": PROFILE["corner_behavior_profile"],
                "rescue_profile": PROFILE["rescue_profile"],
            },
        },
    )
    _write_json(root / "sweep_manifest.json", {"status": "completed"})
    (root / "perturbed_sweep.csv").write_text("metric,value\nhausdorff,0\n")
    (root / "representative_geometry.dat").write_bytes(b"final release geometry")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )
    return root


def _provenance_input(path, role, release_root):
    path = Path(path).resolve()
    try:
        release_relative = path.relative_to(release_root.resolve()).as_posix()
    except ValueError:
        release_relative = None
    return {
        "role": role,
        "path": str(path),
        "sha256": _sha256(path),
        "release_relative_path": release_relative,
    }


def _populate_candidate_roots(tmp_path):
    roots = {
        "figure_root": tmp_path / "figures",
        "c0_root": tmp_path / "c0",
    }
    for root in roots.values():
        root.mkdir()
    for spec in load_candidate_allowlist():
        pdf = roots[spec.root] / spec.pdf
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"%PDF-1.4\n{spec.candidate_id}\n%%EOF\n".encode())
    return roots


def _populate_authoritative_provenance(roots, release_root):
    candidates = {spec.candidate_id: spec for spec in load_candidate_allowlist()}
    anchor = release_figure_anchor(release_root)
    for index, contract in enumerate(PROVENANCE_MANIFEST_SPECS):
        manifest_path = roots[contract.root] / contract.path
        producer_path = manifest_path.parent / f"producer_{index}.json"
        _write_json(
            producer_path,
            {
                "schema_version": 1,
                "status": "completed",
                "generation_provenance": _generation(),
            },
        )
        inputs = [_provenance_input(producer_path, "producer_manifest", release_root)]
        for role in contract.required_input_roles:
            if role == "producer_manifest":
                continue
            if role == "final_release_metrics":
                input_path = release_root / "perturbed_sweep.csv"
            elif role == "final_release_plot_artifact":
                input_path = release_root / "representative_geometry.dat"
            else:
                input_path = manifest_path.parent / f"{role}.dat"
                input_path.parent.mkdir(parents=True, exist_ok=True)
                input_path.write_bytes(f"{contract.generator}:{role}".encode())
            inputs.append(_provenance_input(input_path, role, release_root))
        outputs = []
        for candidate_id in contract.candidate_ids:
            spec = candidates[candidate_id]
            output = (roots[spec.root] / spec.pdf).resolve()
            outputs.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(output),
                    "sha256": _sha256(output),
                }
            )
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "manifest_type": "final_figure_generation",
                "status": "completed",
                "generator": contract.generator,
                "generation_provenance": _generation(),
                "release": anchor,
                "inputs": inputs,
                "outputs": outputs,
            },
        )


def _acceptance_fixture(tmp_path):
    release_root = _make_release(tmp_path)
    roots = _populate_candidate_roots(tmp_path)
    _populate_authoritative_provenance(roots, release_root)
    return roots, release_root


def _fake_page_inspector(path):
    path = Path(path)
    if path.name == "figure_candidate_review.pdf":
        return PdfPageInfo(41, 792.0, 612.0)
    return PdfPageInfo(1, 72.0, 72.0)


def _fake_preview_renderer(pdf_path, output_path, *, dpi, page):
    assert dpi == 300
    assert page >= 1
    Image.new("RGB", (300, 300), "white").save(output_path)


def _fake_review_builder(records, output, *, source_commit):
    assert len(records) == 38
    assert source_commit == SOURCE_COMMIT
    output.write_bytes(b"%PDF-1.4\nsynthetic vector review\n%%EOF\n")


def _accept(tmp_path, roots, release_root, **overrides):
    options = {
        "figure_root": roots["figure_root"],
        "c0_root": roots["c0_root"],
        "release_root": release_root,
        "output_dir": tmp_path / "review",
        "pdf_inspector": _passing_report,
        "page_inspector": _fake_page_inspector,
        "preview_renderer": _fake_preview_renderer,
        "review_builder": _fake_review_builder,
        "review_map_verifier": lambda *args, **kwargs: None,
        "release_auditor": _passing_release_audit,
    }
    options.update(overrides)
    return accept_figure_candidates(**options)


def test_explicit_allowlist_has_exact_38_candidate_contract():
    specs = load_candidate_allowlist()
    assert len(specs) == EXPECTED_COUNTS["candidate_pdfs"] == 38
    assert sum(spec.variant == "unpaired" for spec in specs) == 14
    assert len({spec.slot_id for spec in specs if spec.variant == "unpaired"}) == 14
    paired = [spec for spec in specs if spec.variant != "unpaired"]
    assert len(paired) == 24
    slots = {}
    for spec in paired:
        slots.setdefault(spec.slot_id, set()).add(spec.variant)
    assert len(slots) == 12
    assert all(variants == {"with_endpoints", "clean"} for variants in slots.values())
    assert len({(spec.root, spec.pdf) for spec in specs}) == 38
    assert {
        candidate
        for spec in PROVENANCE_MANIFEST_SPECS
        for candidate in spec.candidate_ids
    } == {spec.candidate_id for spec in specs}


def test_acceptance_fails_on_missing_and_unexpected_candidate_pdfs(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    missing = roots["figure_root"] / load_candidate_allowlist()[0].pdf
    missing.unlink()
    unexpected = roots["c0_root"] / "summary_plots/unexpected.pdf"
    unexpected.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(FigureAcceptanceError) as caught:
        _accept(tmp_path, roots, release_root)
    message = str(caught.value)
    assert "Missing candidate PDFs" in message
    assert "Unexpected candidate PDFs" in message
    assert "perfect_reconstruction_plic_stencil.pdf" in message
    assert "unexpected.pdf" in message


def test_historical_file_masquerade_fails_provenance_checksum(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    spec = load_candidate_allowlist()[0]
    candidate = roots[spec.root] / spec.pdf
    candidate.write_bytes(b"%PDF-1.4\nhistorical candidate with same name\n%%EOF\n")

    with pytest.raises(FigureAcceptanceError, match="candidate checksum mismatch"):
        _accept(tmp_path, roots, release_root)


def test_unproven_source_commit_and_profile_fail_closed(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    contract = PROVENANCE_MANIFEST_SPECS[0]
    manifest_path = roots[contract.root] / contract.path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_provenance"]["source_commit"] = "6" * 40
    _write_json(manifest_path, manifest)

    with pytest.raises(
        FigureAcceptanceError, match="source commit is not authoritative"
    ):
        _accept(tmp_path, roots, release_root)

    _populate_authoritative_provenance(roots, release_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_provenance"]["reconstruction_profile"][
        "plic_fallback"
    ] = "Youngs"
    _write_json(manifest_path, manifest)
    with pytest.raises(FigureAcceptanceError, match="profile does not match release"):
        _accept(tmp_path, roots, release_root)


def test_clean_generator_commit_may_be_a_tooling_descendant_of_release(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    generator_commit = "6" * 40
    for contract in PROVENANCE_MANIFEST_SPECS:
        manifest_path = roots[contract.root] / contract.path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generation_provenance"]["source_commit"] = generator_commit
        producer_input = next(
            item for item in manifest["inputs"] if item["role"] == "producer_manifest"
        )
        producer_path = Path(producer_input["path"])
        producer = json.loads(producer_path.read_text(encoding="utf-8"))
        producer["generation_provenance"]["source_commit"] = generator_commit
        _write_json(producer_path, producer)
        producer_input["sha256"] = _sha256(producer_path)
        _write_json(manifest_path, manifest)

    outputs = _accept(tmp_path, roots, release_root)
    source_map = json.loads(outputs.source_map_json.read_text(encoding="utf-8"))
    assert source_map["release_source_commit"] == SOURCE_COMMIT
    assert {
        candidate["generator_source_commit"] for candidate in source_map["candidates"]
    } == {generator_commit}


def test_release_input_checksum_must_be_proven(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    (release_root / "representative_geometry.dat").write_bytes(b"stale geometry")

    with pytest.raises(FigureAcceptanceError, match="checksum verification failed"):
        _accept(tmp_path, roots, release_root)


@pytest.mark.parametrize("size", [(1, 1), (300, 150)])
def test_tiny_or_stale_rendered_preview_fails(tmp_path, size):
    roots, release_root = _acceptance_fixture(tmp_path)

    def bad_renderer(pdf_path, output_path, *, dpi, page):
        Image.new("RGB", size, "white").save(output_path)

    with pytest.raises(FigureAcceptanceError, match="tiny|dimensions do not match"):
        _accept(
            tmp_path,
            roots,
            release_root,
            preview_renderer=bad_renderer,
        )
    assert not (tmp_path / "review").exists()


def test_multi_page_candidate_fails(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    failed_name = load_candidate_allowlist()[3].candidate_id

    def page_inspector(path):
        if failed_name in Path(path).read_text(encoding="utf-8"):
            return PdfPageInfo(2, 72.0, 72.0)
        return _fake_page_inspector(path)

    with pytest.raises(FigureAcceptanceError, match="exactly one page"):
        _accept(tmp_path, roots, release_root, page_inspector=page_inspector)


def test_wrong_merged_review_page_count_fails(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)

    def page_inspector(path):
        if Path(path).name == "figure_candidate_review.pdf":
            return PdfPageInfo(40, 792.0, 612.0)
        return PdfPageInfo(1, 72.0, 72.0)

    with pytest.raises(FigureAcceptanceError, match="Merged review page count"):
        _accept(tmp_path, roots, release_root, page_inspector=page_inspector)
    assert not (tmp_path / "review").exists()


def test_acceptance_reuses_pdf_qa_and_fails_closed(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    failed_name = load_candidate_allowlist()[7].candidate_id

    def inspector(path, *, require_fonts=True):
        if failed_name in Path(path).read_text(encoding="utf-8"):
            return PdfQaReport(str(path), 1, (), ("contains 1 raster image object(s)",))
        return _passing_report(path, require_fonts=require_fonts)

    with pytest.raises(FigureAcceptanceError, match="contains 1 raster image"):
        _accept(tmp_path, roots, release_root, pdf_inspector=inspector)


def test_success_writes_hash_source_maps_and_generated_previews(tmp_path):
    roots, release_root = _acceptance_fixture(tmp_path)
    inspected = []

    def inspector(path, *, require_fonts=True):
        inspected.append(Path(path))
        return _passing_report(path, require_fonts=require_fonts)

    outputs = _accept(tmp_path, roots, release_root, pdf_inspector=inspector)

    assert len(inspected) == 39
    assert len(list((tmp_path / "review" / "previews").glob("*.png"))) == 38
    payload = json.loads(outputs.source_map_json.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["release"]["source_commit"] == SOURCE_COMMIT
    assert payload["allowlist"]["sha256"] == _sha256(DEFAULT_ALLOWLIST)
    assert payload["review"]["index_pages"] == 3
    assert payload["review"]["page_count"] == 41
    assert payload["review"]["sha256"] == _sha256(outputs.review_pdf)
    assert len(payload["candidates"]) == 38
    assert payload["candidates"][0]["review_page_start"] == 4
    assert payload["candidates"][-1]["review_page_end"] == 41
    assert all(len(row["pdf_sha256"]) == 64 for row in payload["candidates"])
    assert all(len(row["png_sha256"]) == 64 for row in payload["candidates"])
    assert all(row["png_width_px"] == 300 for row in payload["candidates"])
    assert all(
        len(row["provenance_manifest_sha256"]) == 64 for row in payload["candidates"]
    )

    with outputs.source_map_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 38
    assert rows[0]["review_page_start"] == "4"
    assert rows[-1]["review_page_end"] == "41"
    qa = json.loads(outputs.vector_qa_json.read_text(encoding="utf-8"))
    assert qa["passed"]
    assert qa["candidate_pdf_count"] == 38
    assert qa["measured_review_page_count"] == 41
    assert qa["review_page_map_verified"]


def test_late_write_failure_removes_staging_and_publishes_nothing(
    tmp_path, monkeypatch
):
    roots, release_root = _acceptance_fixture(tmp_path)
    module = importlib.import_module("submission.accept_figure_candidates")

    def fail_late(*args, **kwargs):
        raise OSError("simulated late write failure")

    monkeypatch.setattr(module, "_write_csv", fail_late)
    with pytest.raises(FigureAcceptanceError, match="simulated late write failure"):
        _accept(tmp_path, roots, release_root)
    assert not (tmp_path / "review").exists()
    assert not list(tmp_path.glob(".review.staging-*"))


@pytest.mark.skipif(
    shutil.which("pdfunite") is None
    or shutil.which("pdfinfo") is None
    or shutil.which("pdftocairo") is None,
    reason="Poppler PDF tools are unavailable",
)
def test_wrong_review_page_map_fails(tmp_path):
    from reportlab.pdfgen import canvas

    records = []
    for index, color in enumerate(((1, 0, 0), (0, 0, 1))):
        pdf = tmp_path / f"candidate_{index}.pdf"
        drawing = canvas.Canvas(str(pdf), pagesize=(72, 72))
        drawing.setFillColorRGB(*color)
        drawing.rect(0, 0, 72, 72, stroke=0, fill=1)
        drawing.save()
        preview = tmp_path / f"candidate_{index}.png"
        render_pdf_preview(pdf, preview, dpi=300, page=1)
        records.append(_review_record(index, pdf, preview))

    review = tmp_path / "wrong_order.pdf"
    build_vector_review_pdf(
        list(reversed(records)), review, source_commit=SOURCE_COMMIT
    )
    with pytest.raises(FigureAcceptanceError, match="Review page map mismatch"):
        verify_review_page_map(
            records,
            review,
            {record.spec.candidate_id: record.preview.path for record in records},
        )


def _review_record(index, pdf, preview):
    return AcceptedCandidate(
        order=index + 1,
        spec=CandidateSpec(
            candidate_id=f"candidate_{index}",
            slot_id=f"slot_{index}",
            section="Test",
            title=f"Candidate {index}",
            variant="unpaired",
            root="figure_root",
            pdf=pdf.name,
        ),
        pdf_path=pdf,
        pdf_sha256=_sha256(pdf),
        pdf_page_count=1,
        pdf_width_points=72.0,
        pdf_height_points=72.0,
        preview=PreviewRecord(
            path=preview,
            sha256=_sha256(preview),
            width_px=300,
            height_px=300,
            dpi_x=300.0,
            dpi_y=300.0,
        ),
        pdf_qa=_passing_report(pdf),
        provenance=ProvenanceEvidence(
            manifest_path=pdf.with_suffix(".json"),
            manifest_sha256="a" * 64,
            generator="test",
            generator_source_commit=SOURCE_COMMIT,
        ),
        review_page_start=index + 2,
        review_page_end=index + 2,
    )


@pytest.mark.skipif(
    shutil.which("pdfunite") is None or shutil.which("pdfinfo") is None,
    reason="Poppler PDF composition tools are unavailable",
)
def test_vector_review_builder_concatenates_pdf_pages_without_rasterizing(tmp_path):
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
    font_name = "AcceptanceTestVera"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))

    records = []
    for index in range(2):
        pdf = tmp_path / f"candidate_{index}.pdf"
        drawing = canvas.Canvas(
            str(pdf),
            pagesize=(180, 120),
            initialFontName=font_name,
            initialFontSize=10,
        )
        drawing.setFont(font_name, 10)
        drawing.drawString(20, 60, f"Vector candidate {index}")
        drawing.line(20, 50, 160, 50)
        drawing.save()
        preview = tmp_path / f"candidate_{index}.png"
        Image.new("RGB", (750, 500), "white").save(preview)
        qa = inspect_pdf(pdf)
        assert qa.passed
        record = _review_record(index, pdf, preview)
        records.append(
            AcceptedCandidate(
                **{
                    **record.__dict__,
                    "pdf_qa": qa,
                    "pdf_width_points": 180.0,
                    "pdf_height_points": 120.0,
                }
            )
        )

    output = tmp_path / "review.pdf"
    build_vector_review_pdf(records, output, source_commit=SOURCE_COMMIT)
    info = subprocess.run(
        ["pdfinfo", str(output)], check=True, capture_output=True, text=True
    ).stdout
    assert "Pages:           3" in info
    assert inspect_pdf(output).passed


def test_documented_direct_cli_entry_point_runs():
    completed = subprocess.run(
        [sys.executable, "submission/accept_figure_candidates.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Fail-closed acceptance gate" in completed.stdout
    assert "--release-root" in completed.stdout
    assert "--source-commit" not in completed.stdout
