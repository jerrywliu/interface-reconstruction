import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.static import run_perturbed_sweeps
from submission.final_figure_orchestrator import (
    ALL_METHOD_FILES,
    C0_RESOLUTIONS,
    C0_VARIANTS,
    C0_WIGGLES,
    MAINTEXT_CASES,
    PROFILE,
    RESOLUTION_CASES,
    RESOLUTION_VALUES,
    RESOLUTION_WIGGLES,
    FinalFigureOrchestrationError,
    _numbers,
    finalize_publication,
    stage_all_method_candidates,
    validate_c0_manifests,
    validate_maintext_manifest,
    validate_plic_metadata,
    validate_resolution_manifest,
    validate_staged_metadata,
)
from submission.final_figure_provenance import atomic_write_json, file_sha256
from submission.generator_checkout import (
    GeneratorCheckoutError,
    verify_generator_checkout,
)
from submission.accept_figure_candidates import load_candidate_allowlist


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _generation():
    return {
        "source_commit": COMMIT,
        "source_dirty": False,
        "source_status": [],
        "reconstruction_profile": dict(PROFILE),
    }


def _run_manifest(
    path, experiment, method, resolution, wiggle, case_index, do_c0=False
):
    count_key = {
        "lines": "num_lines",
        "squares": "num_squares",
        "circles": "num_circles",
        "ellipses": "num_ellipses",
        "zalesak": "num_cases",
    }[experiment]
    parameters = {
        "facet_algo": method,
        "resolution": resolution,
        "perturb_wiggle": wiggle,
        "perturb_seed": 0,
        "plic_fallback": PROFILE["plic_fallback"],
        "corner_behavior_profile": PROFILE["corner_behavior_profile"],
        "case_indices": None if case_index is None else [case_index],
        "do_c0": do_c0,
        count_key: 25,
    }
    if experiment == "zalesak":
        parameters["rescue_profile"] = PROFILE["rescue_profile"]
    _write_json(
        path,
        {
            "schema_version": 1,
            "experiment": experiment,
            "source_commit": COMMIT,
            "parameters": parameters,
        },
    )


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    tracked = repo / "generator.py"
    tracked.write_text("old = True\n", encoding="utf-8")
    _git(repo, "add", "generator.py")
    _git(repo, "commit", "-m", "historical")
    historical = _git(repo, "rev-parse", "HEAD")
    tracked.write_text("release = True\n", encoding="utf-8")
    _git(repo, "commit", "-am", "scientific release")
    release = _git(repo, "rev-parse", "HEAD")
    tracked.write_text("approved = True\n", encoding="utf-8")
    _git(repo, "commit", "-am", "approved generator")
    approved = _git(repo, "rev-parse", "HEAD")
    return repo, tracked, historical, release, approved


def test_checkout_rejects_nonexistent_and_historical_generator_commits(tmp_path):
    repo, _tracked, historical, release, approved = _git_repo(tmp_path)
    with pytest.raises(GeneratorCheckoutError, match="does not exist"):
        verify_generator_checkout(repo, "0" * 40, release)
    with pytest.raises(GeneratorCheckoutError, match="does not descend"):
        verify_generator_checkout(repo, historical, release)
    attestation = verify_generator_checkout(repo, approved, release)
    assert attestation.approved_commit == approved
    assert attestation.tracked_file_count == 1


def test_checkout_rejects_coherent_old_bytes_hidden_by_assume_unchanged(tmp_path):
    repo, tracked, _historical, release, approved = _git_repo(tmp_path)
    old_bytes = _git(repo, "show", f"{release}:generator.py") + "\n"
    tracked.write_text(old_bytes, encoding="utf-8")
    _git(repo, "update-index", "--assume-unchanged", "generator.py")
    assert _git(repo, "status", "--porcelain") == ""
    with pytest.raises(GeneratorCheckoutError, match="assume-unchanged/skip-worktree"):
        verify_generator_checkout(repo, approved, release)


def test_parameterless_or_wrong_maintext_manifest_fails(tmp_path):
    manifest = tmp_path / "maintext.json"
    _write_json(manifest, {"quantitative": {}, "representative": {}})
    with pytest.raises(FinalFigureOrchestrationError, match="scientific specs"):
        validate_maintext_manifest(manifest)

    representatives = {}
    for experiment, values in MAINTEXT_CASES.items():
        representatives[experiment] = {
            **values,
            "methods": [["wrong", "Wrong"]],
        }
    _write_json(
        manifest,
        {
            "specs": {"representative": representatives},
            "quantitative": {name: "x" for name in MAINTEXT_CASES},
            "representative": {
                name: {"with_endpoints": "x", "clean": "y"} for name in MAINTEXT_CASES
            },
        },
    )
    with pytest.raises(FinalFigureOrchestrationError, match="methods differ"):
        validate_maintext_manifest(manifest)


def _resolution_fixture(tmp_path, experiment="lines", status="completed"):
    plots = tmp_path / "plots"
    case_index, method = RESOLUTION_CASES[experiment]
    runs = []
    for index, (resolution, wiggle) in enumerate(
        (
            pair
            for pair in ((r, w) for r in RESOLUTION_VALUES for w in RESOLUTION_WIGGLES)
        )
    ):
        save_name = f"resolution_{index}"
        _run_manifest(
            plots / save_name / "run_manifest.json",
            experiment,
            method,
            resolution,
            wiggle,
            case_index,
        )
        runs.append(
            {
                "experiment": experiment,
                "algo": method,
                "resolution": resolution,
                "wiggle": wiggle,
                "seed": 0,
                "case_index": case_index,
                "save_name": save_name,
                "status": status,
            }
        )
    manifest = tmp_path / "resolution.json"
    _write_json(
        manifest,
        {
            "status": "completed",
            "generation_provenance": _generation(),
            "endpoint_variants": "paired",
            "runs": runs,
            "summary_plots": {experiment: {"with_endpoints": {}, "clean": {}}},
        },
    )
    return manifest, plots


def test_resolution_rejects_planned_plot_only_runs(tmp_path):
    manifest, plots = _resolution_fixture(tmp_path, status="planned")
    with pytest.raises(FinalFigureOrchestrationError, match="planned/plot-only"):
        validate_resolution_manifest(manifest, plots, "lines", COMMIT)


def _c0_fixture(tmp_path):
    plots = tmp_path / "plots"
    paths = {}
    index = 0
    for experiment in ("ellipses", "zalesak"):
        runs = []
        for resolution in C0_RESOLUTIONS[experiment]:
            for wiggle in C0_WIGGLES:
                for label, (method, do_c0) in C0_VARIANTS[experiment].items():
                    save_name = f"c0_{index:03d}"
                    index += 1
                    _run_manifest(
                        plots / save_name / "run_manifest.json",
                        experiment,
                        method,
                        resolution,
                        wiggle,
                        None,
                        do_c0,
                    )
                    runs.append(
                        {
                            "experiment": experiment,
                            "variant": label,
                            "resolution": resolution,
                            "wiggle": wiggle,
                            "seed": 0,
                            "save_name": save_name,
                            "status": "completed",
                        }
                    )
        path = tmp_path / f"{experiment}.json"
        _write_json(
            path,
            {
                "status": "completed",
                "generation_provenance": _generation(),
                "parameters": {
                    "only": experiment,
                    "algos": ",".join(C0_VARIANTS[experiment]),
                    "resolutions": _numbers(C0_RESOLUTIONS[experiment]),
                    "wiggles": _numbers(C0_WIGGLES),
                    "seeds": "0",
                    "case_indices": None,
                    "endpoint_variants": "paired",
                },
                "runs": runs,
                "outputs": {
                    "summary": {experiment: {}},
                    "representative": {experiment: {"with_endpoints": {}, "clean": {}}},
                },
            },
        )
        paths[experiment] = path
    return paths, plots


def test_guarded_c0_requires_exactly_165_completed_settings(tmp_path):
    paths, plots = _c0_fixture(tmp_path)
    assert len(validate_c0_manifests(paths, plots, COMMIT)) == 165
    payload = json.loads(paths["zalesak"].read_text(encoding="utf-8"))
    payload["runs"].pop()
    _write_json(paths["zalesak"], payload)
    with pytest.raises(FinalFigureOrchestrationError, match="setting count"):
        validate_c0_manifests(paths, plots, COMMIT)


def test_wrong_deterministic_parameters_fail(tmp_path):
    plic = tmp_path / "plic.json"
    _write_json(
        plic,
        {
            "generation_provenance": _generation(),
            "case_index": 4,
            "center_cell": [14, 13],
            "resolution": 0.32,
            "perturbation_magnitude": 0.3,
            "mesh_seed": 9,
        },
    )
    with pytest.raises(FinalFigureOrchestrationError, match="mesh_seed"):
        validate_plic_metadata(plic, COMMIT)

    staged = tmp_path / "staged.json"
    _write_json(
        staged,
        {
            "metadata": {
                "generation_provenance": _generation(),
                "case_index": 22,
                "resolution": 1.0,
                "wiggle": 0.1,
                "seed": 0,
                "radius": 15.0,
                "slot_width": 6.0,
                "slot_top_rel": 10.0,
            }
        },
    )
    with pytest.raises(FinalFigureOrchestrationError, match="slot_width"):
        validate_staged_metadata(staged, COMMIT)


def test_48_pdf_all_method_output_stages_only_five(tmp_path):
    source = tmp_path / "generated"
    source.mkdir()
    for index in range(43):
        (source / f"auxiliary_{index:02d}.pdf").write_bytes(b"aux")
    for filename in ALL_METHOD_FILES.values():
        (source / filename).write_bytes(filename.encode())
    destination = tmp_path / "accepted"
    copied = stage_all_method_candidates(source, destination)
    assert len(copied) == 5
    assert {path.name for path in destination.glob("*.pdf")} == set(
        ALL_METHOD_FILES.values()
    )


def test_existing_plot_from_csv_callers_remain_compatible(tmp_path, monkeypatch):
    source = (
        REPO_ROOT / "experiments/static/merge_section6_with_lvira.py"
    ).read_text() + (
        REPO_ROOT / "experiments/static/finalize_sharded_zalesak.py"
    ).read_text()
    assert source.count('"--plot_from_csv"') == 2
    assert "--release-root" not in source and "--release_root" not in source

    csv_path = tmp_path / "ordinary.csv"
    csv_path.write_text(
        "experiment,algo,resolution,wiggle,seed,metric_key,metric_value,save_name\n"
    )
    calls = []
    monkeypatch.setattr(
        run_perturbed_sweeps,
        "_generate_summary_plots",
        lambda csv_value, output: calls.append((csv_value, Path(output))) or {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_perturbed_sweeps.py",
            "--plot_from_csv",
            str(csv_path),
            "--summary_dir",
            str(tmp_path / "plots"),
            "--no-notify",
        ],
    )
    run_perturbed_sweeps.main()
    assert calls == [(str(csv_path), (tmp_path / "plots").resolve())]


@pytest.mark.parametrize(
    "module",
    [
        "experiments.static.generate_section6_maintext_figures",
        "experiments.static.run_appendix_resolution_visuals",
        "experiments.static.run_appendix_c0_study",
        "experiments.static.run_perturbed_sweeps",
        "experiments.static.generate_plic_baseline_stencil_figure",
        "experiments.static.generate_staged_reconstruction_figure",
    ],
)
def test_general_generator_clis_do_not_require_submission_provenance(module):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--release-root" not in completed.stdout
    assert "--release_root" not in completed.stdout


def test_candidate_mutation_after_acceptance_cleans_up_and_publishes_nothing(tmp_path):
    staging = tmp_path / ".publication.staging-test"
    staging.mkdir()
    candidates = []
    for spec in load_candidate_allowlist():
        path = staging / "candidates" / spec.root / spec.pdf
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spec.candidate_id.encode())
        candidates.append(
            {
                "candidate_id": spec.candidate_id,
                "root": spec.root,
                "pdf": spec.pdf,
                "sha256": file_sha256(path),
                "generator": "test",
            }
        )
    manifest = staging / "provenance" / "final_figure_orchestration.json"
    atomic_write_json(manifest, {"snapshot_artifacts": [], "candidates": candidates})
    output = tmp_path / "publication"

    def mutate(root):
        first = load_candidate_allowlist()[0]
        (root / "candidates" / first.root / first.pdf).write_bytes(b"mutated")

    with pytest.raises(FinalFigureOrchestrationError, match="mutated before publish"):
        finalize_publication(
            staging=staging,
            output_root=output,
            manifest_path=manifest,
            acceptance_runner=lambda **_kwargs: None,
            acceptance_kwargs={},
            after_acceptance_hook=mutate,
        )
    assert not output.exists()
    assert not staging.exists()
