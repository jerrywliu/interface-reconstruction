from __future__ import annotations

import csv
import hashlib
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
    RELEASE_PAYLOADS,
    SubmissionPackagingError,
    _extract_archive_safely,
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
    release = tmp_path / "release_final"
    for source_relative, _, _ in RELEASE_PAYLOADS:
        if source_relative == "SHA256SUMS":
            continue
        path = release / source_relative
        if source_relative == "diagnostics/source_snapshot.tar.gz":
            _write_source_snapshot(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    release, paper, manifest, paper_commit, latexmk = _make_inputs(tmp_path)
    return plan_submission_package(
        release_root=release,
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

    plan = plan_submission_package(
        release_root=release,
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

    with pytest.raises(SubmissionPackagingError, match="absent from"):
        plan_submission_package(
            release_root=release,
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
    assert (
        inventory["raw_data"]["deposition"]["verification_status"]
        == "manual_acknowledgment_remote_contents_unverified"
    )
    assert inventory["raw_data"]["deposition"]["network_assertion_made"] is False
    assert not (package / "provenance" / "deposit").exists()
    assert inventory["release"]["audit_passed"] is True
    assert inventory["release"]["staged_payloads_verified_against_release_manifest"]
    assert inventory["paper"]["git_commit"] == plan.paper_commit
    assert inventory["paper"]["clean_pinned_worktree_verified"] is True
    assert inventory["paper"]["bytes_materialized_from_pinned_git_objects"] is True
    assert (
        inventory["approved_figures"][0]["approval_reference"] == "synthetic-review-1"
    )
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
        SubmissionPackagingError, match="staged source checksum mismatch"
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
    for parent in (tmp_path / "first", tmp_path / "second"):
        parent.mkdir()
        output = parent / "bundle"
        plan = plan_submission_package(
            release_root=release,
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
