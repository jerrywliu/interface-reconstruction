from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from submission.audit_final_release import AuditReport
from submission.package_submission import (
    DEFAULT_PAPER_ENTRYPOINT,
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


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_identifier(release: Path) -> str:
    return f"sha256:{_sha256(release / 'SHA256SUMS')}"


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


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_fake_latexmk(path: Path, *, fail_at: int | None = None) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "script = pathlib.Path(__file__)\n"
        "counter = script.with_suffix('.count')\n"
        "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        f"fail_at = {fail_at!r}\n"
        "if fail_at is not None and count >= fail_at:\n"
        "    print('synthetic compile failure', file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
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


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    release = tmp_path / "release_final"
    for source_relative, _, _ in RELEASE_PAYLOADS:
        path = release / source_relative
        if source_relative == "diagnostics/source_snapshot.tar.gz":
            _write_source_snapshot(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {source_relative}\n", encoding="utf-8")

    paper = tmp_path / "paper-worktree"
    source = paper / "interface-reconstruction-paper"
    figure = source / "figs" / "approved.pdf"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"%PDF-1.4\nsynthetic vector fixture\n%%EOF\n")
    (source / "interface-reconstruction.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        "\\includegraphics{interface-reconstruction-paper/figs/approved.pdf}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (source / "references.bib").write_text("% bibliography\n", encoding="utf-8")
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
                "paper_path": "interface-reconstruction-paper/figs/approved.pdf",
                "source_path": "interface-reconstruction-paper/figs/approved.pdf",
                "sha256": _sha256(figure),
                "approval_status": "approved",
                "approval_reference": "synthetic-review-1",
            }
        )
    latexmk = _write_fake_latexmk(tmp_path / "fake-latexmk")
    return release, paper, manifest, paper_commit, latexmk


def _plan(tmp_path: Path, output: Path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    return plan_submission_package(
        release_root=release,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
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


def test_approved_figure_manifest_requires_approval_and_exact_checksum(tmp_path):
    _, paper, manifest, _, _ = _make_inputs(tmp_path)
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
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    output = tmp_path / "package"

    with pytest.raises(SubmissionPackagingError, match="final release audit failed"):
        plan_submission_package(
            release_root=release,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            latexmk_executable=str(latexmk),
            output_dir=output,
            audit_runner=_failed_audit,
            checksum_verifier=_checksums_pass,
            pdf_inspector=_vector_pdf,
        )

    assert not output.exists()


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

    with pytest.raises(SubmissionPackagingError, match="absent from"):
        plan_submission_package(
            release_root=release,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
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
    assert (package / "code" / "source_snapshot.tar.gz").is_file()
    assert (package / "docs" / "PAPER_EXPERIMENT_MAP.md").is_file()
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
    assert inventory["release"]["audit_passed"] is True
    assert inventory["paper"]["git_commit"] == plan.paper_commit
    assert inventory["paper"]["clean_pinned_worktree_verified"] is True
    assert (
        inventory["approved_figures"][0]["approval_reference"] == "synthetic-review-1"
    )
    build_record = json.loads(
        (package / "provenance" / "manuscript_build.json").read_text(encoding="utf-8")
    )
    assert build_record["staged_compile_passed"] is True
    assert build_record["compile_outputs_in_package"] is False
    assert int(Path(plan.latexmk_executable).with_suffix(".count").read_text()) == 3


def test_plan_requires_exact_clean_paper_commit(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    common = {
        "release_root": release,
        "paper_worktree_root": paper,
        "approved_figures_manifest": manifest,
        "raw_data_deposition": "https://doi.org/10.1234/interface.release",
        "raw_data_manifest_identifier": _manifest_identifier(release),
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


def test_plan_rejects_inner_paper_directory_as_worktree_root(tmp_path):
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    with pytest.raises(SubmissionPackagingError, match="Git top level"):
        plan_submission_package(
            release_root=release,
            paper_worktree_root=paper / "interface-reconstruction-paper",
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
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
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
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
    plan = plan_submission_package(
        release_root=release,
        paper_worktree_root=paper,
        paper_commit=paper_commit,
        approved_figures_manifest=manifest,
        raw_data_deposition="https://doi.org/10.1234/interface.release",
        raw_data_manifest_identifier=_manifest_identifier(release),
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


def test_build_rechecks_paper_worktree_after_planning(tmp_path):
    output = tmp_path / "deliverable" / "bundle"
    plan = _plan(tmp_path / "inputs", output)
    entrypoint = plan.paper_worktree_root / plan.paper_entrypoint
    entrypoint.write_text(entrypoint.read_text(encoding="utf-8") + "% changed\n")

    with pytest.raises(SubmissionPackagingError, match="worktree is not clean"):
        build_submission_package(plan)

    assert not output.exists()


def test_deterministic_archives_match_for_identical_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    release, paper, manifest, paper_commit, latexmk = _make_inputs(inputs)

    archives = []
    for parent in (tmp_path / "first", tmp_path / "second"):
        output = parent / "bundle"
        plan = plan_submission_package(
            release_root=release,
            paper_worktree_root=paper,
            paper_commit=paper_commit,
            approved_figures_manifest=manifest,
            raw_data_deposition="https://doi.org/10.1234/interface.release",
            raw_data_manifest_identifier=_manifest_identifier(release),
            latexmk_executable=str(latexmk),
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
