from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import multiprocessing
import os
import subprocess
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path

import pytest

import submission.package_submission as package_submission
from submission.audit_final_release import AuditReport
from submission.package_submission import (
    DEFAULT_PAPER_ENTRYPOINT,
    GENERATOR_EXPERIMENT_MAP,
    RELEASE_PAYLOADS,
    SubmissionPackagingError,
    _extract_archive_safely,
    _safe_relative_path,
    _validate_final_figure_candidate_contract,
    build_submission_package,
    discover_paper_source_files,
    load_approved_figures,
    plan_submission_package,
    verify_package_checksums,
)
from submission.pdf_vector_qa import PdfQaReport


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_identifier(release: Path) -> str:
    return f"sha256:{_sha256(release / 'SHA256SUMS')}"


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def _tree_state(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (_sha256(path), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_existing_package(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "INVENTORY.json").write_text("{}\n", encoding="utf-8")
    (root / "SHA256SUMS").write_text(f"{'0' * 64}  INVENTORY.json\n", encoding="utf-8")
    (root / "published-sentinel.txt").write_text(
        "existing package must remain unchanged\n", encoding="utf-8"
    )
    return root


def _passing_audit(root: Path) -> AuditReport:
    report = AuditReport(root.resolve())
    report.summaries.update(
        {
            "expected_runs": 970,
            "expected_cases": 24_250,
            "raw_bundles": 970,
        }
    )
    return report


def _failed_audit(root: Path) -> AuditReport:
    report = AuditReport(root.resolve())
    report.add_error("sweep status is running")
    return report


def _checksums_pass(root: Path, manifest: str) -> list[str]:
    assert manifest == "SHA256SUMS"
    return []


def _vector_pdf(path: Path, *, require_fonts: bool) -> PdfQaReport:
    assert require_fonts is False
    return PdfQaReport(str(path), 0, (), ())


def _write_source_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        b"# Paper Experiment Map\n\nSynthetic map.\n"
        b"Historical root: /Users/test/project on private-host.local\n"
    )
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("docs/PAPER_EXPERIMENT_MAP.md")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def _candidate_specs() -> list[dict]:
    specs = []
    for index in range(14):
        candidate_id = f"unpaired_{index:02d}"
        specs.append(
            {
                "candidate_id": candidate_id,
                "slot_id": candidate_id,
                "section": "Synthetic",
                "title": f"Unpaired {index}",
                "variant": "unpaired",
                "root": "figure_root",
                "pdf": f"synthetic/{candidate_id}.pdf",
            }
        )
    selectable_slots = [
        "lines_resolution",
        "ellipses_resolution",
        "zalesak_resolution",
        *(f"paired_{index:02d}" for index in range(9)),
    ]
    for index, slot_id in enumerate(selectable_slots):
        for variant in ("with_endpoints", "clean"):
            candidate_id = f"{slot_id}_{variant}"
            specs.append(
                {
                    "candidate_id": candidate_id,
                    "slot_id": slot_id,
                    "section": "Synthetic",
                    "title": f"Paired {index}",
                    "variant": variant,
                    "root": "figure_root",
                    "pdf": f"synthetic/{candidate_id}.pdf",
                }
            )
        if slot_id in {
            "lines_resolution",
            "ellipses_resolution",
            "zalesak_resolution",
        }:
            variant = "hybrid_endpoints_n16_n32"
            candidate_id = f"{slot_id}_{variant}"
            specs.append(
                {
                    "candidate_id": candidate_id,
                    "slot_id": slot_id,
                    "section": "Synthetic",
                    "title": f"Paired {index}",
                    "variant": variant,
                    "root": "figure_root",
                    "pdf": f"synthetic/{candidate_id}.pdf",
                }
            )
    return specs


def _allowlist_payload() -> dict:
    return {
        "schema_version": 1,
        "expected_counts": {
            "candidate_pdfs": 41,
            "unpaired_candidates": 14,
            "paired_slots": 12,
            "paired_candidates": 24,
            "hybrid_slots": 3,
            "hybrid_candidates": 3,
        },
        "candidates": _candidate_specs(),
    }


def _selected_candidate_specs() -> list[dict]:
    return [
        spec
        for spec in _candidate_specs()
        if spec["variant"] == "unpaired"
        or (
            spec["slot_id"] == "lines_resolution"
            and spec["variant"] == "hybrid_endpoints_n16_n32"
        )
        or (spec["slot_id"] != "lines_resolution" and spec["variant"] == "clean")
    ]


def _write_final_figure_publication(
    root: Path,
    *,
    generator_commit: str,
    generator_tree: str,
    scientific_commit: str,
    release_ledger_sha256: str,
    paper_figure_bytes: dict[str, bytes],
) -> dict[str, dict]:
    allowlist = _allowlist_payload()
    allowlist_path = root / "provenance" / "approved_candidate_allowlist.json"
    allowlist_path.parent.mkdir(parents=True)
    allowlist_path.write_text(json.dumps(allowlist, indent=2) + "\n", encoding="utf-8")
    allowlist_sha256 = _sha256(allowlist_path)
    approval_payload = {
        "schema_version": 2,
        "record_type": "final_figure_orchestration_approval",
        "approval_status": "approved",
        "revoked": False,
        "approved_generator_commit": generator_commit,
        "approved_generator_tree": generator_tree,
        "scientific_release_commit": scientific_commit,
        "release_sha256sums_sha256": release_ledger_sha256,
        "allowlist_sha256": allowlist_sha256,
        "candidate_contract": allowlist["expected_counts"],
        "orchestrator_schema_version": 4,
        "approved_by": "Independent Reviewer",
        "approved_at_utc": "2026-07-31T12:00:00Z",
    }
    approval_path = root / "provenance" / "external_approval_record.json"
    approval_path.write_text(
        json.dumps(approval_payload, indent=2) + "\n", encoding="utf-8"
    )
    approval_sha256 = _sha256(approval_path)

    candidate_rows = []
    orchestration_candidates = []
    approvals: dict[str, dict] = {}
    selected = {spec["candidate_id"] for spec in _selected_candidate_specs()}
    selected_specs = [
        spec for spec in allowlist["candidates"] if spec["candidate_id"] in selected
    ]
    selected_paper_paths = [
        "interface-reconstruction-paper/figs/approved.pdf",
        *[
            f"interface-reconstruction-paper/figs/approved_{index:02d}.pdf"
            for index in range(1, len(selected_specs))
        ],
    ]
    selected_paths_by_id = {
        spec["candidate_id"]: paper_path
        for spec, paper_path in zip(selected_specs, selected_paper_paths)
    }
    for order, spec in enumerate(allowlist["candidates"], start=1):
        candidate_path = root / "candidates" / spec["root"] / Path(spec["pdf"])
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        paper_path = selected_paths_by_id.get(spec["candidate_id"])
        if paper_path is not None:
            data = paper_figure_bytes[paper_path]
            approvals[spec["candidate_id"]] = {
                "candidate_id": spec["candidate_id"],
                "slot_id": spec["slot_id"],
                "variant": spec["variant"],
                "paper_path": paper_path,
            }
        else:
            data = (
                b"%PDF-1.4\nsynthetic unselected candidate "
                + spec["candidate_id"].encode("ascii")
                + b"\n%%EOF\n"
            )
        candidate_path.write_bytes(data)
        relative = candidate_path.relative_to(root).as_posix()
        digest = _sha256(candidate_path)
        orchestration_candidates.append(
            {
                "candidate_id": spec["candidate_id"],
                "path": relative,
                "sha256": digest,
                "generator": "synthetic-generator",
            }
        )
        candidate_rows.append(
            {
                "order": order,
                **spec,
                "pdf_path": relative,
                "pdf_sha256": digest,
                "generator_source_commit": generator_commit,
            }
        )

    release = {
        "root": "provenance/release_input_snapshot",
        "name": "synthetic-release",
        "source_commit": scientific_commit,
        "artifacts": {
            "SHA256SUMS": {
                "path": "provenance/release_input_snapshot/SHA256SUMS",
                "sha256": release_ledger_sha256,
            }
        },
    }
    orchestration = {
        "schema_version": 4,
        "manifest_type": "final_figure_orchestration",
        "status": "ready_for_internal_acceptance",
        "created_at_utc": "2026-07-31T12:30:00+00:00",
        "generator_checkout": {
            "repository": "/Users/test/project",
            "approved_commit": generator_commit,
            "commit_tree": generator_tree,
            "scientific_release_commit": scientific_commit,
            "tracked_file_count": 421,
            "checkout_manifest_sha256": "1" * 64,
            "materialized_manifest_sha256": "1" * 64,
        },
        "trusted_figure_runtime": {
            "python": {
                "executable": "/Users/test/.pyenv/versions/3.9.13/bin/python3.9",
                "version": "3.9.13",
            },
            "runtime_root": "/Users/test/private-runtime",
            "fontconfig_file": "/Users/test/private-runtime/fonts.conf",
        },
        "execution_config_authority": {
            "path": "provenance/execution_config_authority.json",
            "sha256": "2" * 64,
            "file_count": 7,
            "source": "approved_materialized_generator_commit",
            "verification": "per_yaml_read_and_before_after_generator",
        },
        "external_approval": {
            "sha256": approval_sha256,
            "approved_generator_commit": generator_commit,
            "approved_generator_tree": generator_tree,
            "scientific_release_commit": scientific_commit,
            "release_sha256sums_sha256": release_ledger_sha256,
            "allowlist_sha256": allowlist_sha256,
            "candidate_contract": allowlist["expected_counts"],
            "orchestrator_schema_version": 4,
            "approval_status": "approved",
            "revoked": False,
            "approved_by": "Independent Reviewer",
            "approved_at_utc": "2026-07-31T12:00:00Z",
            "snapshot_path": "provenance/external_approval_record.json",
        },
        "scientific_release": release,
        "audited_release_authority": {
            "source_commit": scientific_commit,
            "sha256sums_sha256": release_ledger_sha256,
            "resolved_config_sha256": "3" * 64,
            "audit_root": "private_complete_release_snapshot",
            "checksum_verification": "complete_inventory_passed",
            "snapshotted_file_count": 123_857,
            "snapshotted_size_bytes": 1_000_000,
        },
        "release_input_snapshot": {
            "root": "provenance/release_input_snapshot",
            "representative_alias_sources": {},
            "artifact_count": 12,
        },
        "scientific_contracts": {"final_release": {"passed": True}},
        "allowlist": {
            "path": "provenance/approved_candidate_allowlist.json",
            "sha256": allowlist_sha256,
            "expected_counts": allowlist["expected_counts"],
        },
        "candidates": orchestration_candidates,
        "snapshot_artifacts": [
            {
                "role": "external_approval_record",
                "path": "provenance/external_approval_record.json",
                "sha256": approval_sha256,
                "size_bytes": approval_path.stat().st_size,
            }
        ],
    }
    orchestration_path = root / "provenance" / "final_figure_orchestration.json"
    orchestration_path.write_text(
        json.dumps(orchestration, indent=2) + "\n", encoding="utf-8"
    )
    orchestration_sha256 = _sha256(orchestration_path)
    for row in candidate_rows:
        row.update(
            {
                "provenance_manifest": "provenance/final_figure_orchestration.json",
                "provenance_manifest_sha256": orchestration_sha256,
            }
        )

    review = root / "review"
    review.mkdir()
    source_map = {
        "schema_version": 2,
        "passed": True,
        "source_commit": scientific_commit,
        "release_source_commit": scientific_commit,
        "release": release,
        "allowlist": {
            "path": "provenance/approved_candidate_allowlist.json",
            "sha256": allowlist_sha256,
            "expected_counts": allowlist["expected_counts"],
        },
        "candidates": candidate_rows,
    }
    (review / "figure_candidate_source_map.json").write_text(
        json.dumps(source_map, indent=2) + "\n", encoding="utf-8"
    )
    csv_fields = (
        "candidate_id",
        "slot_id",
        "variant",
        "source_commit",
        "pdf_relative_path",
        "pdf_sha256",
        "generator_source_commit",
        "provenance_manifest",
        "provenance_manifest_sha256",
    )
    with (review / "figure_candidate_source_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "slot_id": row["slot_id"],
                    "variant": row["variant"],
                    "source_commit": scientific_commit,
                    "pdf_relative_path": row["pdf_path"],
                    "pdf_sha256": row["pdf_sha256"],
                    "generator_source_commit": row["generator_source_commit"],
                    "provenance_manifest": row["provenance_manifest"],
                    "provenance_manifest_sha256": row["provenance_manifest_sha256"],
                }
            )

    records = []
    ledger_path = root / "provenance" / "published_tree_sha256.json"
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != ledger_path:
            records.append(
                {
                    "role": "published_artifact",
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    ledger_path.write_text(
        json.dumps({"schema_version": 1, "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in root.rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    return approvals


def _refresh_final_figure_publication(
    tmp_path: Path, paper: Path, release: Path, generator_commit: str
) -> None:
    root = tmp_path / "final-figures"
    if root.exists():
        for path in root.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        root.chmod(0o700)
        for path in sorted(
            root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        root.rmdir()
    figure_paths = sorted(
        path.relative_to(paper).as_posix()
        for path in (paper / "interface-reconstruction-paper" / "figs").glob(
            "approved*.pdf"
        )
    )
    paper_figure_bytes = {
        relative: (paper / relative).read_bytes() for relative in figure_paths
    }
    config = json.loads(
        (release / "submission_config.resolved.json").read_text(encoding="utf-8")
    )
    _write_final_figure_publication(
        root,
        generator_commit=generator_commit,
        generator_tree=_git(paper, "rev-parse", f"{generator_commit}^{{tree}}"),
        scientific_commit=config["source"]["target_commit"],
        release_ledger_sha256=_sha256(release / "SHA256SUMS"),
        paper_figure_bytes=paper_figure_bytes,
    )


def _open_final_figure_publication(root: Path) -> None:
    root.chmod(0o700)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)


def _reseal_final_figure_publication(root: Path) -> None:
    ledger_path = root / "provenance" / "published_tree_sha256.json"
    ledger_path.unlink(missing_ok=True)
    records = [
        {
            "role": "published_artifact",
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != ledger_path
    ]
    ledger_path.write_text(
        json.dumps({"schema_version": 1, "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in root.rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_fake_latexmk(
    path: Path,
    *,
    fail_at: int | None = None,
    block_at: int | None = None,
    block_marker: Path | None = None,
    block_release: Path | None = None,
) -> Path:
    if block_at is not None and (block_marker is None or block_release is None):
        raise ValueError("blocking compiler requires marker and release paths")
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys, time\n"
        "if '-norc' not in sys.argv:\n"
        "    print('missing -norc', file=sys.stderr)\n"
        "    raise SystemExit(10)\n"
        "for variable in ('TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS', 'TEXMFHOME', "
        "'TEXMFCONFIG', 'TEXMFVAR', 'TEXMFCACHE', 'TEXMFOUTPUT', 'HOME', "
        "'XDG_CONFIG_HOME', 'LATEXMKRC', 'LATEXMKRCSYS', 'PERL5LIB', "
        "'PERL5OPT'):\n"
        "    if 'hostile' in os.environ.get(variable, ''):\n"
        "        print(f'hostile environment leaked through {variable}', file=sys.stderr)\n"
        "        raise SystemExit(11)\n"
        "for variable in ('TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS'):\n"
        "    if os.environ.get(variable) != 'interface-reconstruction-paper//:':\n"
        "        print(f'unsanitized search path {variable}', file=sys.stderr)\n"
        "        raise SystemExit(12)\n"
        "script = pathlib.Path(__file__)\n"
        "counter = script.with_suffix('.count')\n"
        "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        f"fail_at = {fail_at!r}\n"
        f"block_at = {block_at!r}\n"
        f"block_marker = {str(block_marker) if block_marker else None!r}\n"
        f"block_release = {str(block_release) if block_release else None!r}\n"
        "if fail_at is not None and count >= fail_at:\n"
        "    print('synthetic compile failure', file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
        "if block_at is not None and count == block_at:\n"
        "    pathlib.Path(block_marker).write_text('blocked\\n')\n"
        "    deadline = time.monotonic() + 30\n"
        "    while not pathlib.Path(block_release).exists():\n"
        "        if time.monotonic() >= deadline:\n"
        "            print('synthetic compile block timed out', file=sys.stderr)\n"
        "            raise SystemExit(13)\n"
        "        time.sleep(0.01)\n"
        "outdir = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('-outdir='))\n"
        "entrypoint = pathlib.Path(sys.argv[-1])\n"
        "if not entrypoint.is_file():\n"
        "    raise SystemExit(3)\n"
        "figure = pathlib.Path('interface-reconstruction-paper/figs/approved.pdf')\n"
        "if not figure.is_file():\n"
        "    raise SystemExit(4)\n"
        "output = pathlib.Path(outdir) / (entrypoint.stem + '.pdf')\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_bytes(b'%PDF-1.4\\ncompiled fixture\\n%%EOF\\n')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _build_package_process(plan, create_archive, label, result_queue) -> None:
    try:
        package, archive = build_submission_package(plan, create_archive=create_archive)
    except Exception as exc:
        result_queue.put((label, "error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(
            (
                label,
                "ok",
                str(package),
                str(archive) if archive is not None else None,
            )
        )


def _wait_for_path(path: Path, process, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if not process.is_alive():
            raise AssertionError(f"process exited before creating {path}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paper = tmp_path / "paper-worktree"
    source = paper / "interface-reconstruction-paper"
    figure_dir = source / "figs"
    figure_dir.mkdir(parents=True)
    selected_specs = _selected_candidate_specs()
    paper_paths = [
        "interface-reconstruction-paper/figs/approved.pdf",
        *[
            f"interface-reconstruction-paper/figs/approved_{index:02d}.pdf"
            for index in range(1, len(selected_specs))
        ],
    ]
    paper_figure_bytes = {}
    for index, paper_path in enumerate(paper_paths):
        data = (
            b"%PDF-1.4\nsynthetic vector fixture "
            + str(index).encode("ascii")
            + b"\n%%EOF\n"
        )
        (paper / paper_path).write_bytes(data)
        paper_figure_bytes[paper_path] = data
    includes = "".join(f"\\includegraphics{{{path}}}\n" for path in paper_paths)
    (source / "interface-reconstruction.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n" + includes + "\\end{document}\n",
        encoding="utf-8",
    )
    (source / "references.bib").write_text("% bibliography\n", encoding="utf-8")
    generator_map = paper / "docs" / "PAPER_EXPERIMENT_MAP.md"
    generator_map.parent.mkdir(parents=True)
    generator_map.write_text(
        "# Paper Experiment Map\n\nPinned final-generator map.\n"
        "Historical root: /Users/test/project on private-host.local\n",
        encoding="utf-8",
    )
    generator_submission = paper / "submission"
    generator_submission.mkdir()
    (generator_submission / "final_figure_orchestrator.py").write_text(
        "print('synthetic orchestrator')\n", encoding="utf-8"
    )
    (generator_submission / "run_final_figure_orchestrator").write_text(
        '#!/bin/sh\nexec python3 submission/final_figure_orchestrator.py "$@"\n',
        encoding="utf-8",
    )
    (generator_submission / "run_final_figure_orchestrator").chmod(0o755)
    (generator_submission / "final_figure_candidates.json").write_text(
        json.dumps(_allowlist_payload(), indent=2) + "\n", encoding="utf-8"
    )
    (paper / ".gitignore").write_text("*.aux\n", encoding="utf-8")
    (source / "interface-reconstruction.aux").write_text(
        "generated\n", encoding="utf-8"
    )

    subprocess.run(["git", "init", "-q", str(paper)], check=True)
    _git(paper, "config", "user.email", "tests@example.com")
    _git(paper, "config", "user.name", "Submission Tests")
    _git(paper, "add", ".")
    _git(paper, "commit", "-q", "-m", "Synthetic paper")
    paper_commit = _git(paper, "rev-parse", "HEAD")
    paper_tree = _git(paper, "rev-parse", "HEAD^{tree}")

    release = tmp_path / "release_final"
    for source_relative, _, _ in RELEASE_PAYLOADS:
        if source_relative == "SHA256SUMS":
            continue
        path = release / source_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if source_relative == "diagnostics/source_snapshot.tar.gz":
            _write_source_snapshot(path)
        elif source_relative == "submission_config.resolved.json":
            path.write_text(
                json.dumps({"source": {"target_commit": paper_commit}}, indent=2)
                + "\n",
                encoding="utf-8",
            )
        elif source_relative == "diagnostics/source_state.json":
            path.write_text(
                json.dumps(
                    {
                        "source_commit": paper_commit,
                        "source_dirty": False,
                        "source_status": [],
                        "snapshot_path": "/Users/test/release/source_snapshot.tar.gz",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        elif source_relative == "environment.json":
            path.write_text(
                json.dumps(
                    {
                        "repository": {"root": "/Users/test/project"},
                        "system": {"uname": {"node": "private-host.local"}},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        elif source_relative == "sweep_manifest.json":
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "artifacts": {"root": "/Users/test/release"},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        elif source_relative == "diagnostics/run_manifests.jsonl":
            path.write_text(
                json.dumps(
                    {
                        "command": "/Users/test/project/run.py",
                        "hostname": "private-host.local",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"synthetic {source_relative}\n", encoding="utf-8")
    release_manifest = release / "SHA256SUMS"
    release_files = sorted(path for path in release.rglob("*") if path.is_file())
    release_manifest.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(release).as_posix()}\n"
            for path in release_files
        ),
        encoding="utf-8",
    )

    approvals = _write_final_figure_publication(
        tmp_path / "final-figures",
        generator_commit=paper_commit,
        generator_tree=paper_tree,
        scientific_commit=paper_commit,
        release_ledger_sha256=_sha256(release_manifest),
        paper_figure_bytes=paper_figure_bytes,
    )

    manifest = tmp_path / "approved_figures.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "candidate_id",
                "slot_id",
                "variant",
                "paper_path",
                "source_path",
                "sha256",
                "approval_status",
                "approval_reference",
            ),
        )
        writer.writeheader()
        for candidate_id, approval in approvals.items():
            paper_path = approval["paper_path"]
            writer.writerow(
                {
                    **approval,
                    "source_path": paper_path,
                    "sha256": _sha256(paper / paper_path),
                    "approval_status": "approved",
                    "approval_reference": f"synthetic-review-{candidate_id}",
                }
            )
    latexmk = _write_fake_latexmk(tmp_path / "fake-latexmk")
    return release, paper, manifest, paper_commit, latexmk


def _plan(tmp_path: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    return plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )


def test_safe_relative_path_rejects_escape_and_absolute_paths():
    assert _safe_relative_path("figs/approved.pdf", "test") == "figs/approved.pdf"
    with pytest.raises(SubmissionPackagingError, match="safe relative"):
        _safe_relative_path("../secret", "test")
    with pytest.raises(SubmissionPackagingError, match="safe relative"):
        _safe_relative_path("/tmp/secret", "test")


def test_documented_direct_cli_entry_point_runs():
    completed = subprocess.run(
        [sys.executable, "submission/package_submission.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Build a deterministic, fail-closed submission package" in completed.stdout


def test_paper_source_allowlist_excludes_generated_and_binary_files(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text("content\n", encoding="utf-8")
    (paper / "refs.bib").write_text("content\n", encoding="utf-8")
    (paper / "main.aux").write_text("generated\n", encoding="utf-8")
    (paper / "preview.png").write_bytes(b"png")

    included, excluded = discover_paper_source_files(paper)

    assert [path.name for path in included] == ["main.tex", "refs.bib"]
    assert excluded == ("main.aux", "preview.png")


def test_committed_excluded_build_directory_is_not_packaged(tmp_path):
    release, paper, manifest, _, latexmk = _make_inputs(tmp_path)
    stale = paper / "interface-reconstruction-paper" / "build" / "stale.tex"
    stale.parent.mkdir()
    stale.write_text("hostile committed build product\n", encoding="utf-8")
    _git(paper, "add", stale.relative_to(paper).as_posix())
    _git(paper, "commit", "-q", "-m", "Commit stale build product")
    paper_commit = _git(paper, "rev-parse", "HEAD")
    _refresh_final_figure_publication(tmp_path, paper, release, paper_commit)

    plan = plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=tmp_path / "package",
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )

    destinations = {item.destination for item in plan.files}
    assert (
        "manuscript/source/interface-reconstruction-paper/build/stale.tex"
        not in destinations
    )


@pytest.mark.parametrize(
    "relationship",
    ("inside_package", "contains_package", "archive_contains_package"),
)
def test_plan_rejects_existing_package_namespace_overlap(tmp_path, relationship):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path / "inputs")
    destination_parent = tmp_path / "destinations"
    destination_parent.mkdir()
    if relationship == "inside_package":
        existing = _write_existing_package(destination_parent / "bundle")
        output = existing / "nested"
    elif relationship == "contains_package":
        output = destination_parent / "wrapper"
        existing = _write_existing_package(output / "bundle")
    else:
        output = destination_parent / "bundle"
        archive_path = output.with_suffix(".tar.gz")
        existing = _write_existing_package(archive_path / "nested")
    before = _tree_state(existing)

    with pytest.raises(
        SubmissionPackagingError,
        match="inside existing package|contains existing package",
    ):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=output,
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )

    assert _tree_state(existing) == before
    assert not list(destination_parent.rglob(".*.package_submission.lock"))
    assert not list(destination_parent.rglob(".*.staging-*"))


@pytest.mark.parametrize(
    "relationship",
    ("inside_package", "contains_package", "archive_contains_package"),
)
def test_stale_plan_rejects_package_overlap_before_writing(tmp_path, relationship):
    destination_parent = tmp_path / "destinations"
    destination_parent.mkdir()
    if relationship == "inside_package":
        plain_parent = destination_parent / "bundle"
        plain_parent.mkdir()
        output = plain_parent / "nested"
    else:
        output = destination_parent / "bundle"
    plan = _plan(tmp_path / "inputs", output)

    if relationship == "inside_package":
        existing = _write_existing_package(output.parent)
    elif relationship == "contains_package":
        existing = _write_existing_package(output / "nested")
    else:
        existing = _write_existing_package(output.with_suffix(".tar.gz") / "nested")
    before = _tree_state(existing)

    with pytest.raises(
        SubmissionPackagingError,
        match="inside existing package|contains existing package",
    ):
        build_submission_package(plan)

    assert _tree_state(existing) == before
    assert not list(destination_parent.rglob(".*.package_submission.lock"))
    assert not list(destination_parent.rglob(".*.staging-*"))
    assert not list(destination_parent.rglob(".*.tmp-*"))


def test_approved_figure_manifest_requires_approval_and_exact_checksum(tmp_path):
    _, paper, manifest, _, _ = _make_inputs(tmp_path)
    source_map = json.loads(
        (
            tmp_path / "final-figures" / "review" / "figure_candidate_source_map.json"
        ).read_text(encoding="utf-8")
    )
    candidates = {
        row["candidate_id"]: {
            **row,
            "candidate_path": row["pdf_path"],
        }
        for row in source_map["candidates"]
    }
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[0]["approval_status"] = "pending"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(SubmissionPackagingError, match="not explicitly approved"):
        load_approved_figures(manifest, paper, candidates)

    rows[0]["approval_status"] = "approved"
    rows[0]["sha256"] = "0" * 64
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SubmissionPackagingError, match="checksum mismatch"):
        load_approved_figures(manifest, paper, candidates)


def test_approved_figures_require_exactly_one_candidate_per_slot(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[1]["slot_id"] = rows[0]["slot_id"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        SubmissionPackagingError, match="mismatched or duplicate slot/variant"
    ):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_candidate_contract_requires_exact_pairs_and_named_hybrid_slots():
    specs = {spec["candidate_id"]: dict(spec) for spec in _candidate_specs()}
    _validate_final_figure_candidate_contract(specs)

    specs["paired_00_clean"]["variant"] = "with_endpoints"
    specs["paired_01_with_endpoints"]["variant"] = "clean"
    with pytest.raises(SubmissionPackagingError, match="14 unique unpaired slots"):
        _validate_final_figure_candidate_contract(specs)

    specs = {spec["candidate_id"]: dict(spec) for spec in _candidate_specs()}
    specs["lines_resolution_hybrid_endpoints_n16_n32"]["slot_id"] = "paired_00"
    with pytest.raises(SubmissionPackagingError, match="lines_resolution"):
        _validate_final_figure_candidate_contract(specs)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("generator", "generator commit/tree"),
        ("scientific", "scientific commit"),
        ("release", "sealed release ledger digest"),
    ),
)
def test_plan_rejects_final_figure_authority_mismatch(tmp_path, mismatch, message):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    figure_root = tmp_path / "final-figures"
    _open_final_figure_publication(figure_root)
    orchestration_path = figure_root / "provenance" / "final_figure_orchestration.json"
    orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
    if mismatch == "generator":
        orchestration["external_approval"]["approved_generator_commit"] = "0" * 40
    elif mismatch == "scientific":
        orchestration["audited_release_authority"]["source_commit"] = "0" * 40
    else:
        orchestration["audited_release_authority"]["sha256sums_sha256"] = "0" * 64
    orchestration_path.write_text(
        json.dumps(orchestration, indent=2) + "\n", encoding="utf-8"
    )
    _reseal_final_figure_publication(figure_root)

    with pytest.raises(SubmissionPackagingError, match=message):
        plan_submission_package(
            release_root=release,
            final_figure_root=figure_root,
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("revoked", "not approved and nonrevoked"),
        ("embedded_mismatch", "do not exactly match the verified snapshot"),
    ),
)
def test_plan_validates_complete_external_approval_snapshot(
    tmp_path, mutation, message
):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    figure_root = tmp_path / "final-figures"
    _open_final_figure_publication(figure_root)
    approval_path = figure_root / "provenance" / "external_approval_record.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    orchestration_path = figure_root / "provenance" / "final_figure_orchestration.json"
    orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
    if mutation == "revoked":
        approval["revoked"] = True
        orchestration["external_approval"]["revoked"] = True
    else:
        approval["approved_by"] = "Second Independent Reviewer"
    approval_path.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
    orchestration["external_approval"]["sha256"] = _sha256(approval_path)
    orchestration_path.write_text(
        json.dumps(orchestration, indent=2) + "\n", encoding="utf-8"
    )
    _reseal_final_figure_publication(figure_root)

    with pytest.raises(SubmissionPackagingError, match=message):
        plan_submission_package(
            release_root=release,
            final_figure_root=figure_root,
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_build_rejects_final_figure_publication_mutated_after_planning(tmp_path):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    figure_root = plan.final_figure_publication.root
    _open_final_figure_publication(figure_root)
    orchestration_path = figure_root / "provenance" / "final_figure_orchestration.json"
    orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
    orchestration["created_at_utc"] = "changed-after-planning"
    orchestration_path.write_text(
        json.dumps(orchestration, indent=2) + "\n", encoding="utf-8"
    )
    _reseal_final_figure_publication(figure_root)

    with pytest.raises(SubmissionPackagingError, match="candidate metadata mismatch"):
        build_submission_package(plan)
    assert not output.exists()


def test_plan_rejects_candidate_hash_changed_in_published_source_map(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    figure_root = tmp_path / "final-figures"
    _open_final_figure_publication(figure_root)
    source_map_path = figure_root / "review" / "figure_candidate_source_map.json"
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    source_map["candidates"][0]["pdf_sha256"] = "0" * 64
    source_map_path.write_text(
        json.dumps(source_map, indent=2) + "\n", encoding="utf-8"
    )
    _reseal_final_figure_publication(figure_root)

    with pytest.raises(SubmissionPackagingError, match="candidate checksum mismatch"):
        plan_submission_package(
            release_root=release,
            final_figure_root=figure_root,
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


@pytest.mark.parametrize(
    "location",
    (
        "https://doi.org/10.xxxx/record",
        "doi:10.0000/interface-release",
        "https://doi.org/10.1234/record",
        "https://doi.org/10.1234/xxxx",
        "https://doi.org/10.1234/xxxx_release",
        "https://doi.org/10.1234/xxxxtest",
        "https://doi.org/10.1234/XXXXTEST",
    ),
)
def test_plan_rejects_placeholder_doi_patterns(tmp_path, location):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    with pytest.raises(SubmissionPackagingError, match="contains a placeholder"):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition=location,
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


@pytest.mark.parametrize(
    "location",
    (
        "https://doi.org/10.1234/exact-interface",
        "doi:10.1234/complex.dataset",
    ),
)
def test_deposition_location_allows_ordinary_single_x(location):
    assert package_submission._validate_deposition_location(location) == location


def test_plan_fails_closed_when_release_audit_fails(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    output = tmp_path / "package"

    with pytest.raises(SubmissionPackagingError, match="final release audit failed"):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=output,
            audit_runner=_failed_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )

    assert not output.exists()


def test_plan_requires_existing_private_output_parent(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    common = {
        "release_root": release,
        "final_figure_root": release.parent / "final-figures",
        "generator_worktree_root": paper,
        "generator_commit": paper_commit,
        "documentation_commit": paper_commit,
        "paper_worktree_root": paper,
        "paper_commit": paper_commit,
        "approved_figures_manifest": manifest,
        "raw_data_deposition": "https://doi.org/10.1234/interface.release",
        "raw_data_manifest_identifier": _manifest_identifier(release),
        "acknowledge_unverified_remote_deposit": True,
        "latexmk_executable": str(latexmk),
        "audit_runner": _passing_audit,
        "checksum_verifier": _checksums_pass,
        "pdf_inspector": _vector_pdf,
    }

    with pytest.raises(SubmissionPackagingError, match="parent must already exist"):
        plan_submission_package(
            output_dir=tmp_path / "missing-parent" / "bundle", **common
        )

    writable_parent = tmp_path / "writable-parent"
    writable_parent.mkdir(mode=0o777)
    writable_parent.chmod(0o777)
    with pytest.raises(SubmissionPackagingError, match="group- or other-writable"):
        plan_submission_package(output_dir=writable_parent / "bundle", **common)


def test_build_rejects_replaced_output_parent_from_stale_plan(tmp_path):
    output = tmp_path / "destinations" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    original_parent = output.parent.with_name("destinations-original")
    output.parent.rename(original_parent)
    output.parent.mkdir()

    with pytest.raises(SubmissionPackagingError, match="parent changed after"):
        build_submission_package(plan)

    assert not output.exists()
    assert not list(output.parent.glob(".*.package_submission.lock"))
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_plan_rejects_unapproved_manuscript_graphic(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    entrypoint = paper / DEFAULT_PAPER_ENTRYPOINT
    entrypoint.write_text(
        "\\includegraphics{interface-reconstruction-paper/figs/not-approved.pdf}\n",
        encoding="utf-8",
    )
    _git(paper, "add", DEFAULT_PAPER_ENTRYPOINT)
    _git(paper, "commit", "-q", "-m", "Reference unapproved figure")
    paper_commit = _git(paper, "rev-parse", "HEAD")
    _refresh_final_figure_publication(tmp_path, paper, release, paper_commit)

    with pytest.raises(SubmissionPackagingError, match="absent from"):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_build_stages_compact_payload_and_valid_checksums(tmp_path):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)

    package, archive = build_submission_package(plan)

    assert package == output
    assert archive == output.with_suffix(".tar.gz")
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as packaged_archive:
        archive_names = packaged_archive.getnames()
    assert archive_names[0] == output.name
    assert all(
        name == output.name or name.startswith(f"{output.name}/")
        for name in archive_names
    )
    scientific_snapshot = package / "code" / "scientific_source_snapshot.tar.gz"
    generator_snapshot = package / "code" / "figure_generator_snapshot.tar.gz"
    assert scientific_snapshot.is_file()
    assert generator_snapshot.is_file()
    assert scientific_snapshot.read_bytes() != generator_snapshot.read_bytes()
    assert (package / "docs" / "PAPER_EXPERIMENT_MAP.md").is_file()
    assert "Pinned final-generator map" in (
        package / "docs" / "PAPER_EXPERIMENT_MAP.md"
    ).read_text(encoding="utf-8")
    with tarfile.open(generator_snapshot, "r:gz") as generator_archive:
        names = generator_archive.getnames()
    assert (
        "interface-reconstruction-generator/submission/final_figure_orchestrator.py"
        in names
    )
    assert "interface-reconstruction-generator/docs/PAPER_EXPERIMENT_MAP.md" in names
    assert (
        package
        / "manuscript"
        / "source"
        / "interface-reconstruction-paper"
        / "figs"
        / "approved.pdf"
    ).is_file()
    assert not (package / "raw_runs").exists()
    assert not (package / "diagnostics").exists()
    assert not list((package / "manuscript" / "source").rglob("*.aux"))
    assert not list((package / "manuscript" / "source").rglob("*.log"))
    assert verify_package_checksums(package) == []

    inventory = json.loads((package / "INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["raw_data"]["included"] is False
    assert inventory["raw_data"]["deposition"]["manifest_identifier"].startswith(
        "sha256:"
    )
    assert (
        inventory["raw_data"]["deposition"]["verification_status"]
        == "manual_acknowledgment_remote_contents_unverified"
    )
    assert inventory["raw_data"]["deposition"]["network_assertion_made"] is False
    assert not (package / "provenance" / "deposit").exists()
    assert inventory["release"]["audit_passed"] is True
    assert inventory["release"]["source_payloads_verified_against_release_manifest"]
    assert inventory["release"]["packaged_presentation_metadata_privacy_sanitized"]
    assert inventory["code"]["scientific_source_snapshot"]["sha256"] == _sha256(
        scientific_snapshot
    )
    generator_inventory = inventory["code"]["figure_generator_snapshot"]
    assert generator_inventory["git_commit"] == plan.generator_commit
    assert generator_inventory["git_tree"] == plan.generator_tree_object_id
    assert generator_inventory["sha256"] == _sha256(generator_snapshot)
    map_inventory = inventory["code"]["paper_experiment_map"]
    assert map_inventory["git_commit"] == plan.documentation_commit
    assert map_inventory["git_tree"] == plan.documentation_tree_object_id
    assert map_inventory["sha256"] == _sha256(
        package / "docs" / "PAPER_EXPERIMENT_MAP.md"
    )
    assert inventory["paper"]["git_commit"] == plan.paper_commit
    assert inventory["paper"]["clean_pinned_worktree_verified"] is True
    assert inventory["paper"]["bytes_materialized_from_pinned_git_objects"] is True
    code_record = json.loads(
        (package / "provenance" / "code_snapshots.json").read_text(encoding="utf-8")
    )
    assert code_record["scientific_source_snapshot"]["path"] == (
        "code/scientific_source_snapshot.tar.gz"
    )
    assert code_record["scientific_source_snapshot"]["git_commit"] == (
        plan.scientific_commit
    )
    assert code_record["scientific_source_snapshot"]["git_tree"] == (
        plan.scientific_tree_object_id
    )
    assert code_record["figure_generator_snapshot"]["git_commit"] == (
        plan.generator_commit
    )
    assert code_record["figure_generator_snapshot"]["git_tree"] == (
        plan.generator_tree_object_id
    )
    assert code_record["paper_experiment_map"]["git_object_id"] == (
        plan.documentation_experiment_map_object_id
    )
    assert len(inventory["approved_figures"]) == 26
    assert inventory["approved_figures"][0]["candidate_id"]
    assert any(
        figure["variant"] == "hybrid_endpoints_n16_n32"
        for figure in inventory["approved_figures"]
    )
    binding = json.loads(
        (
            package / "provenance" / "figures" / "approved_figure_bindings.json"
        ).read_text(encoding="utf-8")
    )
    assert binding["passed"] is True
    assert binding["candidate_count"] == 41
    assert binding["approved_slot_count"] == 26
    assert binding["generator_commit"] == plan.generator_commit
    assert binding["generator_tree"] == plan.generator_tree_object_id
    assert binding["scientific_commit"] == plan.scientific_commit
    assert (
        _sha256(package / "provenance" / "figures" / "final_figure_orchestration.json")
        == binding["orchestration_manifest_public_sha256"]
    )
    assert (
        plan.final_figure_publication.orchestration_sha256
        == binding["orchestration_manifest_authority_sha256"]
    )
    assert (
        _sha256(package / "provenance" / "figures" / "external_approval_record.json")
        == binding["external_approval_record_public_sha256"]
    )
    assert (
        plan.final_figure_publication.external_approval_sha256
        == binding["external_approval_record_authority_sha256"]
    )
    assert (
        _sha256(package / "provenance" / "figures" / "figure_candidate_source_map.json")
        == binding["candidate_source_map_sha256"]
    )
    privacy = json.loads(
        (package / "provenance" / "privacy_redactions.json").read_text(encoding="utf-8")
    )
    assert privacy["passed"] is True
    assert sum(row["replacement_count"] for row in privacy["records"]) >= 5
    privacy_by_path = {row["package_path"]: row for row in privacy["records"]}
    orchestration_privacy = privacy_by_path[
        "provenance/figures/final_figure_orchestration.json"
    ]
    assert orchestration_privacy["authority_sha256"] == (
        plan.final_figure_publication.orchestration_sha256
    )
    assert orchestration_privacy["public_sha256"] == _sha256(
        package / "provenance" / "figures" / "final_figure_orchestration.json"
    )
    approval_privacy = privacy_by_path[
        "provenance/figures/external_approval_record.json"
    ]
    assert approval_privacy["authority_sha256"] == (
        plan.final_figure_publication.external_approval_sha256
    )
    packaged_bytes = b"".join(
        path.read_bytes() for path in package.rglob("*") if path.is_file()
    )
    assert b"/Users/test" not in packaged_bytes
    assert b"private-host.local" not in packaged_bytes
    for snapshot in (scientific_snapshot, generator_snapshot):
        with tarfile.open(snapshot, "r:gz") as archive_reader:
            for member in archive_reader.getmembers():
                if not member.isfile():
                    continue
                stream = archive_reader.extractfile(member)
                member_bytes = stream.read() if stream is not None else b""
                assert b"/Users/test" not in member_bytes
                assert b"private-host.local" not in member_bytes
    build_record = json.loads(
        (package / "provenance" / "manuscript_build.json").read_text(encoding="utf-8")
    )
    assert build_record["staged_compile_passed"] is True
    assert build_record["compile_outputs_in_package"] is False
    assert build_record["external_tex_search_environment_discarded"] is True
    assert build_record["user_texmf_and_latexmkrc_disabled"] is True
    assert int(Path(plan.latexmk_executable).with_suffix(".count").read_text()) == 3


def test_concurrent_build_reservation_prevents_winner_deletion(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("deterministic collision regression requires fork")
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    marker = tmp_path / "slow-build.blocked"
    release = tmp_path / "slow-build.release"
    _write_fake_latexmk(
        Path(plan.latexmk_executable),
        block_at=2,
        block_marker=marker,
        block_release=release,
    )
    fast_latexmk = _write_fake_latexmk(tmp_path / "fast-latexmk")
    collision_plan = replace(plan, latexmk_executable=str(fast_latexmk))

    context = multiprocessing.get_context("fork")
    results = context.Queue()
    slow = context.Process(
        target=_build_package_process,
        args=(plan, True, "owner", results),
    )
    collision = context.Process(
        target=_build_package_process,
        args=(collision_plan, True, "collision", results),
    )
    slow.start()
    try:
        _wait_for_path(marker, slow)
        output_lock = output.with_name(f".{output.name}.package_submission.lock")
        archive = output.with_suffix(".tar.gz")
        archive_lock = archive.with_name(f".{archive.name}.package_submission.lock")
        assert output_lock.is_file()
        assert archive_lock.is_file()

        collision.start()
        collision.join(20)
        assert not collision.is_alive()
        release.write_text("continue\n", encoding="utf-8")
        slow.join(30)
        assert not slow.is_alive()
    finally:
        release.write_text("continue\n", encoding="utf-8")
        for process in (collision, slow):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)

    records = {}
    for _ in range(2):
        record = results.get(timeout=5)
        records[record[0]] = record
    results.close()
    assert records.keys() == {"owner", "collision"}
    assert records["owner"][1] == "ok"
    assert records["collision"][1] == "error"
    assert "reserved by another packaging invocation" in records["collision"][3]
    assert (output / "README.md").is_file()
    assert archive.is_file()
    assert not output_lock.exists()
    assert not archive_lock.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert not list(output.parent.glob(f".{archive.name}.tmp-*"))


def test_publish_collision_cleanup_preserves_unowned_sentinel(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("deterministic collision regression requires fork")
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    marker = tmp_path / "sentinel-build.blocked"
    release = tmp_path / "sentinel-build.release"
    _write_fake_latexmk(
        Path(plan.latexmk_executable),
        block_at=2,
        block_marker=marker,
        block_release=release,
    )

    context = multiprocessing.get_context("fork")
    results = context.Queue()
    builder = context.Process(
        target=_build_package_process,
        args=(plan, False, "builder", results),
    )
    builder.start()
    try:
        _wait_for_path(marker, builder)
        output.mkdir(parents=True)
        sentinel = output / "published-by-other-invocation.txt"
        sentinel.write_text("must survive losing cleanup\n", encoding="utf-8")
        release.write_text("continue\n", encoding="utf-8")
        builder.join(30)
        assert not builder.is_alive()
    finally:
        release.write_text("continue\n", encoding="utf-8")
        if builder.is_alive():
            builder.terminate()
            builder.join(5)

    record = results.get(timeout=5)
    results.close()
    assert record[0:2] == ("builder", "error")
    assert "appeared during packaging" in record[3]
    assert sentinel.read_text(encoding="utf-8") == "must survive losing cleanup\n"
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert not list(output.parent.glob(".*.package_submission.lock"))


def test_late_hostile_final_path_replacements_are_left_untouched(tmp_path, monkeypatch):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    archive = output.with_suffix(".tar.gz")
    displaced_output = output.with_name("displaced-owned-output")
    displaced_archive = archive.with_name("displaced-owned-archive.tar.gz")
    original_publish = package_submission._publish_archive

    def publish_then_replace(temporary_owner, archive_path):
        published = original_publish(temporary_owner, archive_path)
        output.rename(displaced_output)
        output.mkdir()
        (output / "replacement-sentinel.txt").write_text(
            "replacement output must survive\n", encoding="utf-8"
        )
        archive_path.rename(displaced_archive)
        archive_path.write_text("replacement archive must survive\n", encoding="utf-8")
        raise RuntimeError(f"hostile replacement after {published.path}")

    monkeypatch.setattr(package_submission, "_publish_archive", publish_then_replace)

    with pytest.raises(SubmissionPackagingError, match="left untouched"):
        build_submission_package(plan)

    assert (output / "replacement-sentinel.txt").read_text(encoding="utf-8") == (
        "replacement output must survive\n"
    )
    assert archive.read_text(encoding="utf-8") == "replacement archive must survive\n"
    assert (displaced_output / "INVENTORY.json").is_file()
    assert displaced_archive.is_file()
    assert not list(output.parent.glob(".*.package_submission.lock"))
    assert not list(output.parent.glob(f".{archive.name}.tmp-*"))


def test_plan_requires_exact_clean_paper_commit(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    common = {
        "release_root": release,
        "final_figure_root": release.parent / "final-figures",
        "generator_worktree_root": paper,
        "generator_commit": paper_commit,
        "documentation_commit": paper_commit,
        "paper_worktree_root": paper,
        "approved_figures_manifest": manifest,
        "raw_data_deposition": "https://doi.org/10.1234/interface.release",
        "raw_data_manifest_identifier": _manifest_identifier(release),
        "acknowledge_unverified_remote_deposit": True,
        "latexmk_executable": str(latexmk),
        "output_dir": tmp_path / "package",
        "audit_runner": _passing_audit,
        "checksum_verifier": _checksums_pass,
        "pdf_inspector": _vector_pdf,
    }

    with pytest.raises(SubmissionPackagingError, match="paper commit mismatch"):
        plan_submission_package(paper_commit="0" * 40, **common)

    (paper / "uncommitted-note.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(SubmissionPackagingError, match="worktree is not clean"):
        plan_submission_package(paper_commit=paper_commit, **common)


def test_plan_requires_exact_clean_generator_commit(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    common = {
        "release_root": release,
        "final_figure_root": release.parent / "final-figures",
        "generator_worktree_root": paper,
        "documentation_commit": paper_commit,
        "paper_worktree_root": paper,
        "paper_commit": paper_commit,
        "approved_figures_manifest": manifest,
        "raw_data_deposition": "https://doi.org/10.1234/interface.release",
        "raw_data_manifest_identifier": _manifest_identifier(release),
        "acknowledge_unverified_remote_deposit": True,
        "latexmk_executable": str(latexmk),
        "output_dir": tmp_path / "package",
        "audit_runner": _passing_audit,
        "checksum_verifier": _checksums_pass,
        "pdf_inspector": _vector_pdf,
    }

    with pytest.raises(SubmissionPackagingError, match="generator commit mismatch"):
        plan_submission_package(generator_commit="0" * 40, **common)

    (paper / "docs" / "PAPER_EXPERIMENT_MAP.md").write_text(
        "dirty generator map\n", encoding="utf-8"
    )
    with pytest.raises(
        SubmissionPackagingError, match="generator worktree is not clean"
    ):
        plan_submission_package(generator_commit=paper_commit, **common)


def test_plan_requires_explicit_non_none_documentation_commit(tmp_path):
    parameter = inspect.signature(plan_submission_package).parameters[
        "documentation_commit"
    ]
    assert parameter.default is inspect.Parameter.empty

    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    with pytest.raises(
        SubmissionPackagingError, match="documentation commit is required"
    ):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=None,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_build_rechecks_generator_worktree_after_planning(tmp_path):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    generator = (
        plan.generator_worktree_root / "submission" / "final_figure_orchestrator.py"
    )
    generator.write_text("changed after planning\n", encoding="utf-8")

    with pytest.raises(
        SubmissionPackagingError, match="generator worktree is not clean"
    ):
        build_submission_package(plan)

    assert not output.exists()


def test_generator_bytes_come_from_commit_despite_hidden_worktree_edit(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    map_relative = "docs/PAPER_EXPERIMENT_MAP.md"
    committed_map = subprocess.run(
        ["git", "-C", str(paper), "show", f"{paper_commit}:{map_relative}"],
        check=True,
        capture_output=True,
    ).stdout
    worktree_map = paper / map_relative
    worktree_map.write_text("hostile hidden map\n", encoding="utf-8")
    _git(paper, "update-index", "--assume-unchanged", map_relative)
    assert _git(paper, "status", "--porcelain", "--untracked-files=all") == ""

    output = tmp_path / "bundle"
    plan = plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )
    package, _ = build_submission_package(plan, create_archive=False)

    packaged_map = package / map_relative
    assert packaged_map.read_bytes() != committed_map
    assert packaged_map.read_bytes() != worktree_map.read_bytes()
    assert b"Pinned final-generator map" in packaged_map.read_bytes()
    assert b"<HOME>" in packaged_map.read_bytes()
    assert b"<HOSTNAME>" in packaged_map.read_bytes()
    with tarfile.open(
        package / "code" / "figure_generator_snapshot.tar.gz", "r:gz"
    ) as archive:
        stream = archive.extractfile(
            f"interface-reconstruction-generator/{map_relative}"
        )
        assert stream is not None
        archived_map = stream.read()
        assert b"Pinned final-generator map" in archived_map
        assert b"<HOME>" in archived_map
        assert b"<HOSTNAME>" in archived_map


def test_experiment_map_uses_independent_pinned_documentation_commit(tmp_path):
    release, paper, manifest, generator_commit, latexmk = _make_inputs(tmp_path)
    generator_map = paper / GENERATOR_EXPERIMENT_MAP
    generator_map.write_text(
        "# Corrected Paper Experiment Map\n\nDocumentation-only repair.\n",
        encoding="utf-8",
    )
    _git(paper, "add", GENERATOR_EXPERIMENT_MAP)
    _git(paper, "commit", "-q", "-m", "Repair experiment map")
    documentation_commit = _git(paper, "rev-parse", "HEAD")
    documentation_tree = _git(paper, "rev-parse", "HEAD^{tree}")
    generator_worktree = tmp_path / "approved-generator-worktree"
    _git(
        paper,
        "worktree",
        "add",
        "-q",
        "--detach",
        str(generator_worktree),
        generator_commit,
    )

    output = tmp_path / "deliverable" / "bundle"
    output.parent.mkdir(parents=True)
    plan = plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=generator_worktree,
        generator_commit=generator_commit,
        documentation_commit=documentation_commit,
        paper_worktree_root=paper,
        paper_commit=documentation_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )
    package, _ = build_submission_package(plan, create_archive=False)

    assert plan.generator_commit == generator_commit
    assert plan.documentation_commit == documentation_commit
    assert plan.documentation_tree_object_id == documentation_tree
    assert (package / GENERATOR_EXPERIMENT_MAP).read_text(encoding="utf-8") == (
        "# Corrected Paper Experiment Map\n\nDocumentation-only repair.\n"
    )
    code_record = json.loads(
        (package / "provenance" / "code_snapshots.json").read_text(encoding="utf-8")
    )
    assert code_record["figure_generator_snapshot"]["git_commit"] == generator_commit
    assert code_record["paper_experiment_map"]["git_commit"] == documentation_commit


def test_generator_object_substitution_fails_closed(tmp_path, monkeypatch):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    original_read = package_submission._read_git_blob

    def substituted_blob(repository, object_id):
        if object_id == plan.documentation_experiment_map_object_id:
            return b"substituted object bytes\n"
        return original_read(repository, object_id)

    monkeypatch.setattr(package_submission, "_read_git_blob", substituted_blob)
    with pytest.raises(SubmissionPackagingError, match="pinned object ID"):
        build_submission_package(plan)

    assert not output.exists()
    assert not output.with_suffix(".tar.gz").exists()


def test_plan_rejects_inner_paper_directory_as_worktree_root(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    with pytest.raises(SubmissionPackagingError, match="Git top level"):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper / "interface-reconstruction-paper",
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_plan_binds_deposit_to_exact_release_manifest(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    with pytest.raises(SubmissionPackagingError, match="does not match"):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=f"sha256:{'0' * 64}",
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_plan_fails_when_disposable_manuscript_compile_fails(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    _write_fake_latexmk(latexmk, fail_at=1)
    with pytest.raises(SubmissionPackagingError, match="manuscript compile failed"):
        plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=tmp_path / "package",
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )


def test_build_fails_closed_when_extracted_manuscript_compile_fails(tmp_path):
    inputs = tmp_path / "inputs"
    release, paper, manifest, paper_commit, latexmk = _make_inputs(inputs)
    _write_fake_latexmk(latexmk, fail_at=3)
    output = tmp_path / "deliverable" / "bundle"
    output.parent.mkdir()
    plan = plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )

    with pytest.raises(SubmissionPackagingError, match="manuscript compile failed"):
        build_submission_package(plan)

    assert not output.exists()
    assert not output.with_suffix(".tar.gz").exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert not list(output.parent.glob(".*.package_submission.lock"))
    assert not list(output.parent.glob(".*.tmp-*"))


def test_build_rechecks_paper_worktree_after_planning(tmp_path):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    entrypoint = plan.paper_worktree_root / plan.paper_entrypoint
    entrypoint.write_text(entrypoint.read_text(encoding="utf-8") + "% changed\n")

    with pytest.raises(SubmissionPackagingError, match="worktree is not clean"):
        build_submission_package(plan)

    assert not output.exists()


@pytest.mark.parametrize("relative", ("perturbed_sweep.csv", "SHA256SUMS"))
def test_build_rejects_release_payload_mutated_after_planning(tmp_path, relative):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    payload = plan.release_root / relative
    payload.write_text("mutated after planning\n", encoding="utf-8")

    with pytest.raises(
        SubmissionPackagingError,
        match="checksum mismatch|sealed release ledger digest",
    ):
        build_submission_package(plan)

    assert not output.exists()
    assert not output.with_suffix(".tar.gz").exists()


@pytest.mark.parametrize("index_flag", ("--assume-unchanged", "--skip-worktree"))
def test_paper_bytes_come_from_commit_despite_hidden_worktree_edit(
    tmp_path, index_flag
):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    entrypoint = paper / DEFAULT_PAPER_ENTRYPOINT
    committed_bytes = subprocess.run(
        ["git", "-C", str(paper), "show", f"{paper_commit}:{DEFAULT_PAPER_ENTRYPOINT}"],
        check=True,
        capture_output=True,
    ).stdout
    figure_relative = "interface-reconstruction-paper/figs/approved.pdf"
    committed_figure = subprocess.run(
        ["git", "-C", str(paper), "show", f"{paper_commit}:{figure_relative}"],
        check=True,
        capture_output=True,
    ).stdout
    entrypoint.write_text("hostile worktree replacement\n", encoding="utf-8")
    worktree_figure = paper / figure_relative
    worktree_figure.write_bytes(b"hostile worktree figure\n")
    _git(
        paper,
        "update-index",
        index_flag,
        DEFAULT_PAPER_ENTRYPOINT,
        figure_relative,
    )
    assert _git(paper, "status", "--porcelain", "--untracked-files=all") == ""

    output = tmp_path / "bundle"
    plan = plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )
    package, _ = build_submission_package(plan, create_archive=False)

    packaged_entrypoint = package / "manuscript" / "source" / DEFAULT_PAPER_ENTRYPOINT
    assert packaged_entrypoint.read_bytes() == committed_bytes
    assert packaged_entrypoint.read_bytes() != entrypoint.read_bytes()
    packaged_figure = package / "manuscript" / "source" / figure_relative
    assert packaged_figure.read_bytes() == committed_figure
    assert packaged_figure.read_bytes() != worktree_figure.read_bytes()


def test_compile_environment_discards_hostile_tex_configuration(tmp_path, monkeypatch):
    hostile_home = tmp_path / "hostile-home"
    hostile_home.mkdir()
    (hostile_home / ".latexmkrc").write_text("die 'hostile rc loaded';\n")
    for variable in (
        "TEXINPUTS",
        "BIBINPUTS",
        "BSTINPUTS",
        "TEXMFHOME",
        "TEXMFCONFIG",
        "TEXMFVAR",
        "TEXMFCACHE",
        "TEXMFOUTPUT",
        "LATEXMKRC",
        "LATEXMKRCSYS",
        "PERL5LIB",
        "PERL5OPT",
    ):
        monkeypatch.setenv(variable, f"/hostile/{variable.lower()}")
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(hostile_home / ".config"))

    output = tmp_path / "bundle"
    plan = _plan(tmp_path / "inputs", output)

    assert plan.paper_commit
    assert not output.exists()


def test_remote_deposit_requires_evidence_or_explicit_manual_acknowledgment(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    common = {
        "release_root": release,
        "final_figure_root": release.parent / "final-figures",
        "generator_worktree_root": paper,
        "generator_commit": paper_commit,
        "documentation_commit": paper_commit,
        "paper_worktree_root": paper,
        "paper_commit": paper_commit,
        "approved_figures_manifest": manifest,
        "raw_data_deposition": "https://doi.org/10.1234/interface.release",
        "raw_data_manifest_identifier": _manifest_identifier(release),
        "latexmk_executable": str(latexmk),
        "output_dir": tmp_path / "bundle",
        "audit_runner": _passing_audit,
        "checksum_verifier": _checksums_pass,
        "pdf_inspector": _vector_pdf,
    }
    with pytest.raises(SubmissionPackagingError, match="remote deposit contents"):
        plan_submission_package(**common)

    bad_manifest = tmp_path / "bad-downloaded-SHA256SUMS"
    bad_manifest.write_text("different bytes\n", encoding="utf-8")
    with pytest.raises(SubmissionPackagingError, match="bytes do not match"):
        plan_submission_package(
            deposited_release_manifest=bad_manifest,
            **common,
        )


def test_supplied_deposit_manifest_is_checked_and_packaged(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    downloaded = tmp_path / "downloaded-SHA256SUMS"
    downloaded.write_bytes((release / "SHA256SUMS").read_bytes())
    output = tmp_path / "bundle"
    plan = plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        deposited_release_manifest=downloaded,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )
    package, _ = build_submission_package(plan, create_archive=False)

    evidence = package / "provenance" / "deposit" / "SHA256SUMS.downloaded"
    assert evidence.read_bytes() == (release / "SHA256SUMS").read_bytes()
    inventory = json.loads((package / "INVENTORY.json").read_text(encoding="utf-8"))
    deposition = inventory["raw_data"]["deposition"]
    assert deposition["verification_status"] == "supplied_manifest_bytes_verified"
    assert deposition["supplied_manifest_bytes_verified"] is True
    assert deposition["network_assertion_made"] is False


def test_dry_run_plan_does_not_mutate_release_paper_or_output(tmp_path):
    inputs = tmp_path / "inputs"
    release, paper, manifest, paper_commit, latexmk = _make_inputs(inputs)
    entrypoint = paper / DEFAULT_PAPER_ENTRYPOINT
    entrypoint_metadata = entrypoint.stat()
    os.utime(
        entrypoint,
        ns=(
            entrypoint_metadata.st_atime_ns,
            entrypoint_metadata.st_mtime_ns + 5_000_000_000,
        ),
    )
    git_index = paper / ".git" / "index"
    index_before = (git_index.read_bytes(), git_index.stat().st_mtime_ns)
    release_before = _tree_snapshot(release)
    paper_before = _tree_snapshot(paper)
    output = tmp_path / "deliverable" / "bundle"
    output.parent.mkdir()
    output_parent_before = (
        output.parent.stat().st_mtime_ns,
        tuple(sorted(path.name for path in output.parent.iterdir())),
    )

    plan_submission_package(
        release_root=release,
        final_figure_root=release.parent / "final-figures",
        generator_worktree_root=paper,
        generator_commit=paper_commit,
        documentation_commit=paper_commit,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
        acknowledge_unverified_remote_deposit=True,
        latexmk_executable=str(latexmk),
        output_dir=output,
        audit_runner=_passing_audit,
        checksum_verifier=_checksums_pass,
        pdf_inspector=_vector_pdf,
    )

    assert _tree_snapshot(release) == release_before
    assert _tree_snapshot(paper) == paper_before
    assert (git_index.read_bytes(), git_index.stat().st_mtime_ns) == index_before
    assert not (paper / ".git" / "index.lock").exists()
    assert not output.exists()
    assert not output.with_suffix(".tar.gz").exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert not list(output.parent.glob(".*.package_submission.lock"))
    assert (
        output.parent.stat().st_mtime_ns,
        tuple(sorted(path.name for path in output.parent.iterdir())),
    ) == output_parent_before


@pytest.mark.parametrize("member_kind", ("traversal", "symlink", "special"))
def test_archive_extraction_rejects_hostile_members(tmp_path, member_kind):
    archive_path = tmp_path / f"{member_kind}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        if member_kind == "traversal":
            info = tarfile.TarInfo("../outside")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        elif member_kind == "symlink":
            info = tarfile.TarInfo("bundle/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            archive.addfile(info)
        else:
            info = tarfile.TarInfo("bundle/device")
            info.type = tarfile.CHRTYPE
            archive.addfile(info)

    with pytest.raises(SubmissionPackagingError, match="unsafe|link|special"):
        _extract_archive_safely(archive_path, tmp_path / "extract")

    assert not (tmp_path / "outside").exists()


def test_deterministic_archives_match_for_identical_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    release, paper, manifest, paper_commit, latexmk = _make_inputs(inputs)

    archives = []
    generator_snapshots = []
    for parent in (tmp_path / "first", tmp_path / "second"):
        parent.mkdir()
        output = parent / "bundle"
        plan = plan_submission_package(
            release_root=release,
            final_figure_root=release.parent / "final-figures",
            generator_worktree_root=paper,
            generator_commit=paper_commit,
            documentation_commit=paper_commit,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            acknowledge_unverified_remote_deposit=True,
            latexmk_executable=str(latexmk),
            output_dir=output,
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )
        package, archive = build_submission_package(plan)
        archives.append(archive)
        generator_snapshots.append(
            _sha256(package / "code" / "figure_generator_snapshot.tar.gz")
        )

    assert archives[0] is not None and archives[1] is not None
    assert _sha256(archives[0]) == _sha256(archives[1])
    assert generator_snapshots[0] == generator_snapshots[1]


def test_paper_source_symlink_is_rejected(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    target = tmp_path / "target.tex"
    target.write_text("content\n", encoding="utf-8")
    (paper / "main.tex").symlink_to(target)

    with pytest.raises(SubmissionPackagingError, match="symbolic link"):
        discover_paper_source_files(paper)
