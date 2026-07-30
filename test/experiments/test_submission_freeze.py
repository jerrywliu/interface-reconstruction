import csv
import json
import subprocess
import sys

from submission.check_submission_freeze import (
    CONFIG_PATH,
    PROVENANCE_PATH,
    REPO,
    active_paper_figures,
    expected_run_count,
    uncommitted_source_paths,
)


def test_submission_scope_matches_machine_readable_totals():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runs = expected_run_count(config)
    assert runs == 970
    assert runs == config["planned_totals"]["runs"]
    assert runs * config["benchmark_grid"]["trials_per_setting"] == 24250


def test_all_active_paper_figures_have_provenance_rows():
    with PROVENANCE_PATH.open(newline="", encoding="utf-8") as stream:
        recorded = {row["paper_file"] for row in csv.DictReader(stream)}
    assert active_paper_figures() == recorded


def test_source_audit_allows_generated_roots_but_rejects_source_and_tests(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "module.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )

    tracked.write_text("value = 2\n", encoding="utf-8")
    for path in (
        tmp_path / "submission" / "new_config.json",
        tmp_path / "test" / "experiments" / "new_test.py",
        tmp_path / "results" / "run" / "metrics.csv",
        tmp_path / "plots" / "run" / "mesh.vtk",
        tmp_path / "output" / "figure.pdf",
        tmp_path / "tmp" / "scratch.txt",
        tmp_path / "custom_release" / "bundle.csv",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated or source\n", encoding="utf-8")

    assert uncommitted_source_paths(tmp_path) == [
        "custom_release/bundle.csv",
        "module.py",
        "submission/new_config.json",
        "test/experiments/new_test.py",
    ]
    assert uncommitted_source_paths(
        tmp_path, allowed_generated_paths=(tmp_path / "custom_release",)
    ) == [
        "module.py",
        "submission/new_config.json",
        "test/experiments/new_test.py",
    ]


def test_final_sweep_cli_dry_plan_matches_submission_config(tmp_path):
    release_root = tmp_path / "submission-dry-run"
    command = [
        sys.executable,
        "-m",
        "experiments.static.run_perturbed_sweeps",
        "--only",
        "lines,circles,ellipses,squares,zalesak",
        "--wiggles",
        "0.0,0.05,0.1,0.2,0.3",
        "--seeds",
        "0",
        "--lines",
        "25",
        "--circles",
        "25",
        "--ellipses",
        "25",
        "--squares",
        "25",
        "--zalesak",
        "25",
        "--plic_fallback",
        "LVIRA",
        "--rescue_profile",
        "exact_linear_support_only",
        "--corner_behavior_profile",
        "pre_f8_corner",
        "--max_workers",
        "5",
        "--run_namespace",
        "submission_test",
        "--raw_bundle_dir",
        str(release_root / "raw_runs"),
        "--out_csv",
        str(release_root / "perturbed_sweep.csv"),
        "--diagnostics_dir",
        str(release_root / "diagnostics"),
        "--summary_dir",
        str(release_root / "summary_plots"),
        "--log_dir",
        str(release_root / "logs"),
        "--dry_run",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Planned runs: 970" in completed.stdout
    assert "Planned cases: 24250" in completed.stdout
    assert "- lines: 150" in completed.stdout
    assert "- circles: 210" in completed.stdout
    assert "- ellipses: 210" in completed.stdout
    assert "- squares: 200" in completed.stdout
    assert "- zalesak: 200" in completed.stdout
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert 970 * config["benchmark_grid"]["trials_per_setting"] == 24250
    assert not release_root.exists()
