import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from submission.audit_final_release import AuditReport
from submission.package_submission import (
    RELEASE_PAYLOADS,
    SubmissionPackagingError,
    _safe_relative_path,
    build_submission_package,
    discover_paper_source_files,
    load_approved_figures,
    plan_submission_package,
    verify_package_checksums,
)
from submission.pdf_vector_qa import PdfQaReport


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    data = b"# Paper Experiment Map\n\nSynthetic map.\n"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("docs/PAPER_EXPERIMENT_MAP.md")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    release = tmp_path / "release_final"
    for source_relative, _, _ in RELEASE_PAYLOADS:
        path = release / source_relative
        if source_relative == "diagnostics/source_snapshot.tar.gz":
            _write_source_snapshot(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {source_relative}\n", encoding="utf-8")

    paper = tmp_path / "paper"
    figure = paper / "figs" / "approved.pdf"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"%PDF-1.4\nsynthetic vector fixture\n%%EOF\n")
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        "\\includegraphics{figs/approved.pdf}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "references.bib").write_text("% bibliography\n", encoding="utf-8")
    (paper / "main.aux").write_text("generated\n", encoding="utf-8")

    manifest = tmp_path / "approved_figures.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "paper_path",
                "source_path",
                "sha256",
                "approval_status",
                "approval_reference",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "paper_path": "figs/approved.pdf",
                "source_path": "figs/approved.pdf",
                "sha256": _sha256(figure),
                "approval_status": "approved",
                "approval_reference": "synthetic-review-1",
            }
        )
    return release, paper, manifest


def _plan(tmp_path: Path, output: Path):
    release, paper, manifest = _make_inputs(tmp_path)
    return plan_submission_package(
        release_root=release,
        paper_source_root=paper,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
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


def test_approved_figure_manifest_requires_approval_and_exact_checksum(tmp_path):
    _, paper, manifest = _make_inputs(tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[0]["approval_status"] = "pending"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(SubmissionPackagingError, match="not explicitly approved"):
        load_approved_figures(manifest, paper)

    rows[0]["approval_status"] = "approved"
    rows[0]["sha256"] = "0" * 64
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SubmissionPackagingError, match="checksum mismatch"):
        load_approved_figures(manifest, paper)


def test_plan_fails_closed_when_release_audit_fails(tmp_path):
    release, paper, manifest = _make_inputs(tmp_path)
    output = tmp_path / "package"

    with pytest.raises(SubmissionPackagingError, match="final release audit failed"):
        plan_submission_package(
            release_root=release,
            paper_source_root=paper,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            output_dir=output,
            audit_runner=_failed_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )

    assert not output.exists()


def test_plan_rejects_unapproved_manuscript_graphic(tmp_path):
    release, paper, manifest = _make_inputs(tmp_path)
    (paper / "main.tex").write_text(
        "\\includegraphics{figs/not-approved.pdf}\n", encoding="utf-8"
    )

    with pytest.raises(SubmissionPackagingError, match="absent from"):
        plan_submission_package(
            release_root=release,
            paper_source_root=paper,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
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
    assert (package / "code" / "source_snapshot.tar.gz").is_file()
    assert (package / "docs" / "PAPER_EXPERIMENT_MAP.md").is_file()
    assert (package / "manuscript" / "source" / "figs" / "approved.pdf").is_file()
    assert not (package / "raw_runs").exists()
    assert not (package / "diagnostics").exists()
    assert verify_package_checksums(package) == []

    inventory = json.loads((package / "INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["raw_data"]["included"] is False
    assert inventory["release"]["audit_passed"] is True
    assert inventory["approved_figures"][0]["approval_reference"] == "synthetic-review-1"


def test_deterministic_archives_match_for_identical_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    release, paper, manifest = _make_inputs(inputs)

    archives = []
    for parent in (tmp_path / "first", tmp_path / "second"):
        output = parent / "bundle"
        plan = plan_submission_package(
            release_root=release,
            paper_source_root=paper,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            output_dir=output,
            audit_runner=_passing_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )
        _, archive = build_submission_package(plan)
        archives.append(archive)

    assert archives[0] is not None and archives[1] is not None
    assert _sha256(archives[0]) == _sha256(archives[1])


def test_paper_source_symlink_is_rejected(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    target = tmp_path / "target.tex"
    target.write_text("content\n", encoding="utf-8")
    (paper / "main.tex").symlink_to(target)

    with pytest.raises(SubmissionPackagingError, match="symbolic link"):
        discover_paper_source_files(paper)
