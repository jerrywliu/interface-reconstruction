import json
import os
import shutil
import hashlib
import csv
import subprocess
import sys
import stat
from types import SimpleNamespace
from pathlib import Path

import pytest
from PIL import Image

from experiments.static import run_perturbed_sweeps
import submission.final_figure_orchestration as orchestration
from submission.final_figure_orchestration import (
    ALL_METHOD_FILES,
    C0_RESOLUTIONS,
    C0_VARIANTS,
    C0_WIGGLES,
    MAINTEXT_CASES,
    MAINTEXT_METHODS,
    PROFILE,
    ORCHESTRATION_SCHEMA_VERSION,
    RESOLUTION_CASES,
    RESOLUTION_VALUES,
    RESOLUTION_WIGGLES,
    FinalFigureOrchestrationError,
    _capture_release_audit_pin,
    _generator_environment,
    _numbers,
    _rehash_before_publish,
    _remove_tree,
    _snapshot_release_inputs,
    _snapshot_complete_release,
    _seal_execution_config,
    _verify_frozen_publication_tree,
    _write_command_record,
    resolution_input_paths,
    stage_all_method_candidates,
    validate_c0_metrics,
    validate_c0_manifests,
    validate_maintext_manifest,
    validate_plic_metadata,
    validate_resolution_manifest,
    validate_staged_metadata,
)
from submission.final_figure_provenance import atomic_write_json, file_sha256
from submission.generator_checkout import (
    GeneratorCheckoutError,
    materialize_approved_source,
    verify_external_approval_record,
    verify_generator_checkout,
    verify_materialized_source,
)
from submission.accept_figure_candidates import (
    EXPECTED_COUNTS,
    load_candidate_allowlist,
    pdf_page_info,
)
from submission.trusted_figure_runtime import prepare_trusted_figure_runtime
from util.config import ConfigAuthorityError, read_yaml
from test.submission.final_figure_test_support import freeze_staging_for_test


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
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    tracked.write_text("old = True\n", encoding="utf-8")
    _git(repo, "add", "generator.py", ".gitignore")
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
    assert attestation.tracked_file_count == 2


def test_checkout_rejects_coherent_old_bytes_hidden_by_assume_unchanged(tmp_path):
    repo, tracked, _historical, release, approved = _git_repo(tmp_path)
    old_bytes = _git(repo, "show", f"{release}:generator.py") + "\n"
    tracked.write_text(old_bytes, encoding="utf-8")
    _git(repo, "update-index", "--assume-unchanged", "generator.py")
    assert _git(repo, "status", "--porcelain") == ""
    with pytest.raises(GeneratorCheckoutError, match="assume-unchanged/skip-worktree"):
        verify_generator_checkout(repo, approved, release)


def test_git_replace_refs_and_caller_git_environment_cannot_substitute_source(
    tmp_path, monkeypatch
):
    repo, _tracked, historical, release, approved = _git_repo(tmp_path)
    _git(repo, "replace", approved, historical)
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "objects"))
    monkeypatch.setenv("PATH", str(tmp_path / "attacker-bin"))

    attestation = verify_generator_checkout(repo, approved, release)
    source = tmp_path / "materialized"
    materialized = materialize_approved_source(repo, approved, source, attestation)
    try:
        assert (source / "generator.py").read_text(
            encoding="utf-8"
        ) == "approved = True\n"
        assert materialized.materialized_manifest_sha256 == (
            attestation.checkout_manifest_sha256
        )
    finally:
        _remove_tree(source)


def test_materialized_source_excludes_ignored_pyc_and_survives_live_edit(tmp_path):
    repo, tracked, _historical, release, approved = _git_repo(tmp_path)
    pycache = repo / "__pycache__"
    pycache.mkdir()
    (pycache / "generator.cpython-39.pyc").write_bytes(b"malicious ignored bytecode")
    attestation = verify_generator_checkout(repo, approved, release)
    source = tmp_path / "materialized"
    materialize_approved_source(repo, approved, source, attestation)
    try:
        assert not (source / "__pycache__").exists()
        tracked.write_text("concurrent = 'live edit'\n", encoding="utf-8")
        assert (source / "generator.py").read_text(
            encoding="utf-8"
        ) == "approved = True\n"
        runtime = prepare_trusted_figure_runtime(tmp_path / "trusted-runtime")
        env = _generator_environment(repo, source, runtime)
        assert env["PYTHONPATH"] == str(source.resolve())
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert env["HOME"].startswith(str(tmp_path / "trusted-runtime"))
    finally:
        _remove_tree(source)


def test_materialized_source_reattestation_rejects_late_pyc(tmp_path):
    repo, _tracked, _historical, release, approved = _git_repo(tmp_path)
    attestation = verify_generator_checkout(repo, approved, release)
    source = tmp_path / "materialized"
    attestation = materialize_approved_source(repo, approved, source, attestation)
    source.chmod(0o755)
    pycache = source / "__pycache__"
    pycache.mkdir()
    (pycache / "generator.cpython-39.pyc").write_bytes(b"late malicious bytecode")
    with pytest.raises(GeneratorCheckoutError, match="missing or unexpected"):
        verify_materialized_source(repo, approved, source, attestation)
    _remove_tree(source)


def test_ordinary_import_cannot_reach_sensitive_publication_stages(tmp_path):
    probe = r"""
import importlib
import inspect
import json
import sys

sys.path.insert(0, sys.argv[1])
try:
    importlib.import_module("submission.final_figure_orchestrator")
except ImportError as exc:
    import_error = str(exc)
else:
    raise SystemExit("script boundary was importable")

import submission.accept_figure_candidates as acceptance
import submission.final_figure_orchestration as library

sensitive = {
    "reservation": {
        "PublicationReservation",
        "_reserve_publication",
        "_verify_reservation",
        "_release_reservation",
    },
    "generation": {"run_command", "orchestrate_final_figures"},
    "acceptance": {
        "_ORCHESTRATION_AUTHORITY",
        "_OrchestratedAcceptanceState",
        "_AcceptanceState",
        "_create_orchestrated_acceptance_state",
        "_create_acceptance_state",
        "_accept_orchestrated_candidates",
        "_accept_candidates",
    },
    "publication": {
        "finalize_publication",
        "_complete_publication_transaction",
        "_rename_directory_noreplace",
    },
}
reachable = {
    stage: sorted(
        name
        for name in names
        if name in vars(acceptance) or name in vars(library)
    )
    for stage, names in sensitive.items()
}
injectable_parameters = {
    "pdf_inspector",
    "page_inspector",
    "preview_renderer",
    "review_builder",
    "review_map_verifier",
    "renderer",
}
injectable = {}
for name, value in vars(acceptance).items():
    if inspect.isfunction(value) and value.__module__ == acceptance.__name__:
        found = sorted(
            set(inspect.signature(value).parameters) & injectable_parameters
        )
        if found:
            injectable[name] = found
result = {
    "import_refused": "script-only publication boundary" in import_error,
    "module_cached": "submission.final_figure_orchestrator" in sys.modules,
    "reachable": reachable,
    "injectable": injectable,
}
print(json.dumps(result, sort_keys=True))
if not result["import_refused"] or result["module_cached"]:
    raise SystemExit(8)
if any(reachable.values()) or injectable:
    raise SystemExit(9)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(REPO_ROOT)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "import_refused": True,
        "module_cached": False,
        "reachable": {
            "acceptance": [],
            "generation": [],
            "publication": [],
            "reservation": [],
        },
        "injectable": {},
    }


@pytest.mark.skipif(
    any(
        shutil.which(tool) is None
        for tool in ("pdfimages", "pdffonts", "pdfinfo", "pdftocairo", "pdfunite")
    ),
    reason="Poppler PDF tools are unavailable",
)
def test_real_transaction_accepts_freezes_publishes_and_cleans_attack(tmp_path):
    worker = REPO_ROOT / "test/submission/final_figure_transaction_probe.py"
    boundary = REPO_ROOT / "submission/final_figure_orchestrator.py"
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            str(worker),
            str(boundary),
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(completed.stdout)

    assert result["success"] == {
        "candidate_fonts_embedded": True,
        "candidate_pdfs": 38,
        "candidate_rasters": 0,
        "preview_dpi": [300.0],
        "previews": 38,
        "publish_temporaries": 0,
        "reservation_cleaned": True,
        "review_fonts_embedded": True,
        "review_pages": 41,
        "review_rasters": 0,
        "root_mode": 0o500,
        "staging_cleaned": True,
    }
    attack = result["destination_attack"]
    assert "destination appeared before publish" in attack["error"]
    assert attack["winner"] == "competing winner\n"
    assert attack["reservation_cleaned"]
    assert attack["staging_cleaned"]
    assert attack["acceptance_temporaries"] == 0
    assert attack["publish_temporaries"] == 0


def test_trusted_launcher_ignores_pythonpath_and_user_sitecustomize(tmp_path):
    attack_root = tmp_path / "python-attack"
    attack_root.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (attack_root / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('owned')\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        f"#!/bin/sh\necho fake > {str(marker)!r}\nexit 99\n", encoding="utf-8"
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(attack_root)
    env["PYTHONUSERBASE"] = str(attack_root)
    launcher = REPO_ROOT / "submission" / "run_final_figure_orchestrator"
    trusted_python = Path(sys.executable).resolve()

    completed = subprocess.run(
        [str(launcher), "--python", str(trusted_python), "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "approved-generator-commit" in completed.stdout
    assert not marker.exists()


def test_orchestrator_cli_rejects_unisolated_direct_python_startup():
    completed = subprocess.run(
        [sys.executable, "submission/final_figure_orchestrator.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert (
        "use the trusted submission/run_final_figure_orchestrator" in completed.stderr
    )


def test_orchestrator_cli_rejects_isolated_direct_python_startup():
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "submission/final_figure_orchestrator.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "inherited descriptor is missing or invalid" in completed.stderr


def test_orchestrator_cli_rejects_preloaded_repository_modules():
    script = REPO_ROOT / "submission" / "final_figure_orchestrator.py"
    code = (
        "import runpy,sys,types; "
        "sys.modules['submission.generator_checkout'] = types.ModuleType('submission.generator_checkout'); "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    completed = subprocess.run(
        [str(Path(sys.executable).resolve()), "-I", "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "preloaded=submission.generator_checkout" in completed.stderr


def test_poppler_ignores_fake_caller_path_and_records_exact_tools(
    tmp_path, monkeypatch
):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-tool-ran"
    for name in ("pdfinfo", "pdftocairo", "pdfunite", "pdfimages", "pdffonts"):
        tool = fake_bin / name
        tool.write_text(
            f"#!/bin/sh\necho forged > {marker}\nexit 99\n", encoding="utf-8"
        )
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    runtime = prepare_trusted_figure_runtime(tmp_path / "trusted-runtime")
    assert set(runtime.tools) == {
        "pdfinfo",
        "pdftocairo",
        "pdfunite",
        "pdfimages",
        "pdffonts",
    }
    assert all(
        not record.path.startswith(str(fake_bin)) for record in runtime.tools.values()
    )
    assert all(
        len(record.sha256) == 64 and record.version for record in runtime.tools.values()
    )
    assert all(font["version"] != "unknown" for font in runtime.attestation["fonts"])

    from reportlab.pdfgen import canvas

    pdf = tmp_path / "one-page.pdf"
    drawing = canvas.Canvas(str(pdf), pagesize=(72, 72))
    drawing.drawString(5, 36, "trusted")
    drawing.save()
    assert pdf_page_info(pdf, runtime=runtime).page_count == 1
    assert not marker.exists()


def test_generator_environment_ignores_hostile_matplotlib_and_home_config(
    tmp_path, monkeypatch
):
    repo, _tracked, _historical, release, approved = _git_repo(tmp_path)
    attestation = verify_generator_checkout(repo, approved, release)
    source = tmp_path / "materialized"
    materialize_approved_source(repo, approved, source, attestation)
    attacker = tmp_path / "attacker-home"
    attacker_mpl = attacker / ".config" / "matplotlib"
    attacker_mpl.mkdir(parents=True)
    (attacker_mpl / "matplotlibrc").write_text(
        "figure.facecolor: red\nsavefig.facecolor: red\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(attacker))
    monkeypatch.setenv("MPLCONFIGDIR", str(attacker_mpl))
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    monkeypatch.setenv("FONTCONFIG_PATH", str(attacker))
    monkeypatch.setenv("TEXINPUTS", str(attacker))

    runtime = prepare_trusted_figure_runtime(tmp_path / "trusted-runtime")
    env = _generator_environment(repo, source, runtime)
    output = tmp_path / "facecolor.png"
    script = (
        "import matplotlib, matplotlib.pyplot as plt; "
        "assert matplotlib.rcParams['figure.facecolor'] == 'white'; "
        "fig=plt.figure(figsize=(1,1)); fig.savefig(r'" + str(output) + "', dpi=20)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    with Image.open(output) as image:
        image.load()
        assert image.convert("RGB").getpixel((0, 0)) == (255, 255, 255)
    assert env["HOME"] != str(attacker)
    assert env["MPLCONFIGDIR"] != str(attacker_mpl)
    assert env["MPLBACKEND"] == "Agg"
    assert env["LC_ALL"] == "C" and env["TZ"] == "UTC"
    assert env["PYTHONHASHSEED"] == "0"
    _remove_tree(source)


def test_external_approval_record_pins_final_commit_tree_and_digest(tmp_path):
    repo, _tracked, _historical, release, approved = _git_repo(tmp_path)
    attestation = verify_generator_checkout(repo, approved, release)
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text("{}\n", encoding="utf-8")
    approval = tmp_path / "approval.json"
    release_ledger_sha256 = "b" * 64
    payload = {
        "schema_version": 2,
        "record_type": "final_figure_orchestration_approval",
        "approval_status": "approved",
        "revoked": False,
        "approved_generator_commit": approved,
        "approved_generator_tree": attestation.commit_tree,
        "scientific_release_commit": release,
        "release_sha256sums_sha256": release_ledger_sha256,
        "allowlist_sha256": hashlib.sha256(allowlist.read_bytes()).hexdigest(),
        "candidate_contract": EXPECTED_COUNTS,
        "orchestrator_schema_version": ORCHESTRATION_SCHEMA_VERSION,
        "approved_by": "independent reviewer",
        "approved_at_utc": "2026-07-31T12:00:00Z",
    }
    _write_json(approval, payload)
    digest = hashlib.sha256(approval.read_bytes()).hexdigest()
    record = verify_external_approval_record(
        approval,
        digest,
        repository=repo,
        approved_commit=approved,
        approved_tree=attestation.commit_tree,
        scientific_release_commit=release,
        release_sha256sums_sha256=release_ledger_sha256,
        allowlist_sha256=payload["allowlist_sha256"],
        candidate_contract=EXPECTED_COUNTS,
        orchestrator_schema_version=ORCHESTRATION_SCHEMA_VERSION,
    )
    assert record.approved_generator_commit == approved
    assert record.release_sha256sums_sha256 == release_ledger_sha256
    assert record.candidate_contract == EXPECTED_COUNTS
    payload["approved_generator_commit"] = release
    _write_json(approval, payload)
    with pytest.raises(GeneratorCheckoutError, match="SHA-256 does not match"):
        verify_external_approval_record(
            approval,
            digest,
            repository=repo,
            approved_commit=approved,
            approved_tree=attestation.commit_tree,
            scientific_release_commit=release,
            release_sha256sums_sha256=release_ledger_sha256,
            allowlist_sha256=record.allowlist_sha256,
            candidate_contract=EXPECTED_COUNTS,
            orchestrator_schema_version=ORCHESTRATION_SCHEMA_VERSION,
        )
    approval_link = tmp_path / "approval-link.json"
    approval_link.symlink_to(approval)
    with pytest.raises(GeneratorCheckoutError, match="symbolic link"):
        verify_external_approval_record(
            approval_link,
            hashlib.sha256(approval.read_bytes()).hexdigest(),
            repository=repo,
            approved_commit=approved,
            approved_tree=attestation.commit_tree,
            scientific_release_commit=release,
            release_sha256sums_sha256=release_ledger_sha256,
            allowlist_sha256=record.allowlist_sha256,
            candidate_contract=EXPECTED_COUNTS,
            orchestrator_schema_version=ORCHESTRATION_SCHEMA_VERSION,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"approval_status": "pending"}, "approval_status"),
        ({"revoked": True}, "revoked"),
        ({"release_sha256sums_sha256": "c" * 64}, "release_sha256sums"),
        ({"approved_by": "x"}, "approved_by"),
        ({"approved_at_utc": "not-a-time"}, "approved_at_utc"),
        ({"unexpected_authority": "forged"}, "unknown fields"),
    ],
)
def test_external_approval_rejects_wrong_status_revocation_or_schema_fields(
    tmp_path, mutation, match
):
    repo, _tracked, _historical, release, approved = _git_repo(tmp_path)
    attestation = verify_generator_checkout(repo, approved, release)
    allowlist_sha256 = "a" * 64
    release_ledger_sha256 = "b" * 64
    payload = {
        "schema_version": 2,
        "record_type": "final_figure_orchestration_approval",
        "approval_status": "approved",
        "revoked": False,
        "approved_generator_commit": approved,
        "approved_generator_tree": attestation.commit_tree,
        "scientific_release_commit": release,
        "release_sha256sums_sha256": release_ledger_sha256,
        "allowlist_sha256": allowlist_sha256,
        "candidate_contract": EXPECTED_COUNTS,
        "orchestrator_schema_version": ORCHESTRATION_SCHEMA_VERSION,
        "approved_by": "independent reviewer",
        "approved_at_utc": "2026-07-31T12:00:00Z",
    }
    payload.update(mutation)
    approval = tmp_path / "approval.json"
    _write_json(approval, payload)
    with pytest.raises(GeneratorCheckoutError, match=match):
        verify_external_approval_record(
            approval,
            file_sha256(approval),
            repository=repo,
            approved_commit=approved,
            approved_tree=attestation.commit_tree,
            scientific_release_commit=release,
            release_sha256sums_sha256=release_ledger_sha256,
            allowlist_sha256=allowlist_sha256,
            candidate_contract=EXPECTED_COUNTS,
            orchestrator_schema_version=ORCHESTRATION_SCHEMA_VERSION,
        )


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


def _write_complete_c0_metrics(path, experiment):
    metric_bases = {
        "ellipses": (
            "curvature_error",
            "facet_gap",
            "hausdorff",
            "tangent_error",
            "curvature_proxy_error",
        ),
        "zalesak": ("area_error", "facet_gap", "hausdorff"),
    }[experiment]
    fieldnames = (
        "experiment",
        "algo",
        "facet_algo",
        "do_c0",
        "resolution",
        "wiggle",
        "seed",
        "metric_key",
        "metric_value",
        "save_name",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for resolution in C0_RESOLUTIONS[experiment]:
            for wiggle in C0_WIGGLES:
                for label, (method, do_c0) in C0_VARIANTS[experiment].items():
                    for metric in metric_bases:
                        for statistic in ("mean", "median", "p25", "p75"):
                            writer.writerow(
                                {
                                    "experiment": experiment,
                                    "algo": label,
                                    "facet_algo": method,
                                    "do_c0": int(do_c0),
                                    "resolution": resolution,
                                    "wiggle": wiggle,
                                    "seed": 0,
                                    "metric_key": f"{metric}_{statistic}",
                                    "metric_value": 0.0,
                                    "save_name": (
                                        f"c0_{experiment}_{label}_{resolution}_{wiggle}"
                                    ),
                                }
                            )


@pytest.mark.parametrize(
    "experiment,expected_rows", [("ellipses", 1800), ("zalesak", 900)]
)
def test_guarded_c0_metrics_require_exact_setting_metric_coverage(
    tmp_path, experiment, expected_rows
):
    path = tmp_path / experiment / "metrics.csv"
    _write_complete_c0_metrics(path, experiment)
    contract = validate_c0_metrics(path, experiment)
    assert contract["row_count"] == expected_rows

    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(
        FinalFigureOrchestrationError, match="metric coverage|row count"
    ):
        validate_c0_metrics(path, experiment)


def test_resolution_inputs_require_quantitative_and_geometry_evidence(tmp_path):
    run = tmp_path / "plots" / "resolution"
    metrics = run / "metrics"
    facets = run / "vtk" / "reconstructed" / "facets"
    facets.mkdir(parents=True)
    metrics.mkdir(parents=True)
    (run / "vtk" / "mesh.vtk").write_bytes(b"mesh")
    (facets / "0.vtp").write_bytes(b"facet")
    _write_json(
        facets / "0.facet_metadata.json",
        {"schema_version": 2, "primitives": []},
    )
    with (metrics / "case_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_index",
                "hausdorff",
                "facet_gap",
                "area_error",
                "curvature_error",
                "tangent_error",
                "curvature_proxy_error",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_index": 0,
                "hausdorff": 0.0,
                "facet_gap": 0.0,
                "area_error": 0.0,
                "curvature_error": 0.0,
                "tangent_error": 0.0,
                "curvature_proxy_error": 0.0,
            }
        )
    (metrics / "case_geometry.jsonl").write_text(
        json.dumps({"case_index": 0, "geometry_type": "line"}) + "\n",
        encoding="utf-8",
    )
    truth = run / "vtk" / "true" / "true_line0.vtp"
    truth.parent.mkdir(parents=True)
    # Line truth is analytic in the plotter; a saved VTP must remain unclaimed.
    truth.write_bytes(b"truth")
    paths = resolution_input_paths(
        tmp_path / "plots",
        experiment="lines",
        save_name="resolution",
        case_index=0,
    )
    assert len(paths) == 5
    assert all(role != "resolution_truth_geometry" for _path, role in paths)

    square_truth = run / "vtk" / "true" / "true_square0.vtp"
    square_truth.write_bytes(b"square truth consumed by the plot")
    square_paths = resolution_input_paths(
        tmp_path / "plots",
        experiment="squares",
        save_name="resolution",
        case_index=0,
        include_consumed_truth=True,
    )
    assert square_paths[-1] == (square_truth, "resolution_truth_geometry")

    circle_truth = run / "vtk" / "true" / "true_circle0.vtp"
    circle_truth.write_bytes(b"circle truth consumed by the plot")
    circle_paths = resolution_input_paths(
        tmp_path / "plots",
        experiment="circles",
        save_name="resolution",
        case_index=0,
        include_consumed_truth=True,
    )
    assert circle_paths[-1] == (circle_truth, "resolution_truth_geometry")
    (metrics / "case_geometry.jsonl").unlink()
    with pytest.raises(FinalFigureOrchestrationError, match="JSONL is missing"):
        resolution_input_paths(
            tmp_path / "plots",
            experiment="lines",
            save_name="resolution",
            case_index=0,
        )


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


def _release_snapshot_fixture(tmp_path):
    release = tmp_path / "release"
    raw = release / "raw_runs"
    raw.mkdir(parents=True)
    _write_json(
        release / "submission_config.resolved.json",
        {
            "source": {"target_commit": COMMIT},
            "benchmark_grid": {
                "seed": 0,
                "trials_per_setting": 25,
                "wiggles": [0.0, 0.05, 0.1, 0.2, 0.3],
                "full_resolutions": [0.32, 0.5, 0.64, 1.0, 1.28, 1.5],
                "short_resolutions": [0.5, 0.64, 1.0, 1.28, 1.5],
            },
            "benchmarks": {
                experiment: {
                    "methods": list(methods),
                    "planned_runs": {
                        "lines": 150,
                        "circles": 210,
                        "ellipses": 210,
                        "squares": 200,
                        "zalesak": 200,
                    }[experiment],
                }
                for experiment, methods in orchestration.RELEASE_METHODS.items()
            },
            "planned_totals": {"runs": 970, "cases": 24250},
            "production_method": {
                "unresolved_orientation_fallback": PROFILE["plic_fallback"],
                "corner_behavior_profile": PROFILE["corner_behavior_profile"],
                "rescue_profile": PROFILE["rescue_profile"],
            },
        },
    )
    _write_json(
        release / "sweep_manifest.json",
        {
            "status": "completed",
            "planned_run_count": 970,
            "successful_run_count": 970,
            "failure_count": 0,
            "failures": [],
        },
    )
    fields = (
        "experiment",
        "algo",
        "resolution",
        "wiggle",
        "seed",
        "metric_key",
        "metric_value",
        "save_name",
    )
    with (release / "perturbed_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment, expected in MAINTEXT_CASES.items():
            for method in MAINTEXT_METHODS[experiment]:
                save_name = (
                    f"release_{experiment}_{method.lower().replace('+', 'plus')}"
                )
                bundle = raw / save_name
                bundle.mkdir()
                _write_json(bundle / "run_manifest.json", {"save_name": save_name})
                (bundle / "geometry.bin").write_bytes(save_name.encode("utf-8"))
                writer.writerow(
                    {
                        "experiment": experiment,
                        "algo": method,
                        "resolution": expected["resolution"],
                        "wiggle": expected["wiggle"],
                        "seed": expected["seed"],
                        "metric_key": "hausdorff_median",
                        "metric_value": 0.0,
                        "save_name": save_name,
                    }
                )
    files = sorted(path for path in release.rglob("*") if path.is_file())
    (release / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(release).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return release


def test_release_snapshot_rejects_transient_input_mutation(tmp_path):
    release = _release_snapshot_fixture(tmp_path)
    audit_pin = _capture_release_audit_pin(release)
    staging = tmp_path / "staging"
    staging.mkdir()
    mutated = False

    def mutate_after_open(path):
        nonlocal mutated
        if not mutated and path.name == "perturbed_sweep.csv":
            mutated = True
            path.write_bytes(path.read_bytes() + b"# concurrent mutation\n")

    with pytest.raises(ValueError, match="changed while being read|checksum mismatch"):
        _snapshot_release_inputs(
            release,
            staging / "provenance" / "release_input_snapshot",
            audit_pin=audit_pin,
            staging_root=staging,
            after_open_hook=mutate_after_open,
        )
    assert not (staging / "provenance" / "release_input_snapshot").exists()


def test_complete_snapshot_is_the_only_release_audit_root_during_substitution(
    tmp_path, monkeypatch
):
    release = _release_snapshot_fixture(tmp_path)
    live_pin = _capture_release_audit_pin(release)
    complete = _snapshot_complete_release(
        release, tmp_path / "complete-release", live_pin=live_pin
    )
    audited_roots = []
    verified_roots = []
    real_verify = orchestration.verify_sha256_manifest

    def substitute_live_release(_snapshot_root):
        audited_roots.append(Path(_snapshot_root))
        old_release = tmp_path / "substituted-live-release"
        release.rename(old_release)
        shutil.copytree(old_release, release)
        csv_path = release / "perturbed_sweep.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"# malicious coherent release\n")
        files = sorted(
            path
            for path in release.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (release / "SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(release).as_posix()}\n"
                for path in files
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(ok=True, errors=[])

    def verify_snapshot(root):
        verified_roots.append(Path(root))
        return real_verify(root)

    monkeypatch.setattr(orchestration, "audit_final_release", substitute_live_release)
    monkeypatch.setattr(orchestration, "verify_sha256_manifest", verify_snapshot)
    audited = orchestration.validate_final_release_contract(complete.root)

    assert audited_roots == [complete.root]
    assert verified_roots == [complete.root]
    assert audited.root == complete.root
    assert (
        b"malicious coherent release"
        not in (complete.root / "perturbed_sweep.csv").read_bytes()
    )
    assert (
        b"malicious coherent release" in (release / "perturbed_sweep.csv").read_bytes()
    )


def test_complete_release_snapshot_has_exact_full_checksum_inventory(tmp_path):
    release = _release_snapshot_fixture(tmp_path)
    live_pin = _capture_release_audit_pin(release)
    complete = _snapshot_complete_release(
        release, tmp_path / "complete-release", live_pin=live_pin
    )

    assert complete.file_count == len(
        orchestration.parse_sha256_manifest(release / "SHA256SUMS")
    )
    assert orchestration.verify_sha256_manifest(complete.root) == []
    assert (complete.root / "SHA256SUMS").read_bytes() == live_pin.sha256sums_bytes


def test_release_snapshot_preserves_audited_ledger_and_source_commit(tmp_path):
    release = _release_snapshot_fixture(tmp_path)
    audit_pin = _capture_release_audit_pin(release)
    staging = tmp_path / "staging"
    staging.mkdir()
    snapshot = _snapshot_release_inputs(
        release,
        staging / "provenance" / "release_input_snapshot",
        audit_pin=audit_pin,
        staging_root=staging,
    )

    assert audit_pin.source_commit == COMMIT
    assert (
        audit_pin.sha256sums_sha256
        == hashlib.sha256((release / "SHA256SUMS").read_bytes()).hexdigest()
    )
    assert (snapshot.root / "SHA256SUMS").read_bytes() == audit_pin.sha256sums_bytes
    assert snapshot.anchor["source_commit"] == audit_pin.source_commit


def test_release_snapshot_rejects_release_root_replacement_after_audit(tmp_path):
    release = _release_snapshot_fixture(tmp_path)
    audit_pin = _capture_release_audit_pin(release)
    original = tmp_path / "release-before-replacement"
    release.rename(original)
    shutil.copytree(original, release)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(
        FinalFigureOrchestrationError, match="release root was replaced"
    ):
        _snapshot_release_inputs(
            release,
            staging / "provenance" / "release_input_snapshot",
            audit_pin=audit_pin,
            staging_root=staging,
        )
    assert not (staging / "provenance" / "release_input_snapshot").exists()


def test_release_snapshot_rejects_release_ledger_change_after_audit(tmp_path):
    release = _release_snapshot_fixture(tmp_path)
    audit_pin = _capture_release_audit_pin(release)
    ledger = release / "SHA256SUMS"
    ledger.write_bytes(ledger.read_bytes() + b"# coherent-looking replacement\n")
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(FinalFigureOrchestrationError, match="SHA256SUMS differs"):
        _snapshot_release_inputs(
            release,
            staging / "provenance" / "release_input_snapshot",
            audit_pin=audit_pin,
            staging_root=staging,
        )
    assert not (staging / "provenance" / "release_input_snapshot").exists()


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


def test_command_provenance_does_not_record_deleted_private_roots(tmp_path):
    staging = tmp_path / "private-staging"
    execution = tmp_path / "private-execution"
    staging.mkdir()
    execution.mkdir()
    record = _write_command_record(
        execution / "command.json",
        [
            "python",
            str(staging / "provenance" / "release_input_snapshot" / "data.csv"),
            f"--out={execution / 'generated' / 'figure.pdf'}",
        ],
        COMMIT,
        staging_root=staging,
        execution_root=execution,
    )

    text = record.read_text(encoding="utf-8")
    assert str(staging) not in text
    assert str(execution) not in text
    assert "<publication-root>/provenance/release_input_snapshot/data.csv" in text
    assert "--out=<private-execution>/generated/figure.pdf" in text


def _config_authority_fixture(tmp_path, monkeypatch):
    config_root = tmp_path / "approved-source" / "config"
    override = config_root / "static" / "test.yaml"
    override.parent.mkdir(parents=True)
    (config_root / "base.yaml").write_text(
        "value: 1\nnested:\n  base: true\n", encoding="utf-8"
    )
    override.write_text("value: 2\nnested:\n  override: true\n", encoding="utf-8")
    authority = _seal_execution_config(
        config_root, tmp_path / "private-execution" / "config_authority.json"
    )
    monkeypatch.setenv("INTERFACE_CONFIG_ROOT", str(authority.config_root))
    monkeypatch.setenv("INTERFACE_CONFIG_AUTHORITY", str(authority.manifest_path))
    monkeypatch.setenv("INTERFACE_CONFIG_AUTHORITY_SHA256", authority.manifest_sha256)
    return authority, override


def test_nested_config_read_consumes_digest_attested_source_bytes(
    tmp_path, monkeypatch
):
    authority, _override = _config_authority_fixture(tmp_path, monkeypatch)

    assert read_yaml("config/static/test.yaml") == {
        "value": 2,
        "nested": {"base": True, "override": True},
    }
    orchestration._verify_execution_config(authority)


def test_nested_config_read_rejects_source_and_authority_substitution(
    tmp_path, monkeypatch
):
    authority, override = _config_authority_fixture(tmp_path, monkeypatch)
    override.write_text("value: 999\n", encoding="utf-8")

    with pytest.raises(ConfigAuthorityError, match="size differs|digest differs"):
        read_yaml("config/static/test.yaml")
    with pytest.raises(
        FinalFigureOrchestrationError, match="execution config bytes mutated"
    ):
        orchestration._verify_execution_config(authority)

    override.write_text("value: 2\nnested:\n  override: true\n", encoding="utf-8")
    authority.manifest_path.chmod(0o600)
    payload = json.loads(authority.manifest_path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    _write_json(authority.manifest_path, payload)
    with pytest.raises(ConfigAuthorityError, match="manifest digest differs"):
        read_yaml("config/static/test.yaml")


def test_nested_config_read_rejects_symlink_swap(tmp_path, monkeypatch):
    _authority, override = _config_authority_fixture(tmp_path, monkeypatch)
    original = override.with_name("test-original.yaml")
    override.rename(original)
    malicious = override.with_name("malicious.yaml")
    malicious.write_text("value: 999\n", encoding="utf-8")
    override.symlink_to(malicious)

    with pytest.raises(ConfigAuthorityError, match="open attested config|symlink"):
        read_yaml("config/static/test.yaml")


def test_config_authority_is_enforced_in_nested_experiment_process(
    tmp_path, monkeypatch
):
    authority, override = _config_authority_fixture(tmp_path, monkeypatch)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    command = [
        sys.executable,
        "-c",
        (
            "from util.config import read_yaml; "
            "print(read_yaml('config/static/test.yaml')['value'])"
        ),
    ]
    valid = subprocess.run(
        command, cwd=tmp_path, env=env, check=False, capture_output=True, text=True
    )
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.strip() == "2"

    override.write_text("value: 999\n", encoding="utf-8")
    attacked = subprocess.run(
        command, cwd=tmp_path, env=env, check=False, capture_output=True, text=True
    )
    assert attacked.returncode != 0
    assert "Config size differs from authority" in attacked.stderr


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


def test_candidate_mutation_is_rejected_before_frozen_tree_build(tmp_path):
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
                "path": (Path("candidates") / spec.root / spec.pdf).as_posix(),
                "sha256": file_sha256(path),
                "generator": "test",
            }
        )
    manifest = staging / "provenance" / "final_figure_orchestration.json"
    atomic_write_json(manifest, {"snapshot_artifacts": [], "candidates": candidates})
    manifest_digest = file_sha256(manifest)
    first = load_candidate_allowlist()[0]
    (staging / "candidates" / first.root / first.pdf).write_bytes(b"mutated")

    with pytest.raises(FinalFigureOrchestrationError, match="mutated before publish"):
        _rehash_before_publish(
            staging,
            manifest,
            manifest_digest,
            load_candidate_allowlist(),
        )
    assert not list(tmp_path.glob(".publication.publish-*"))


def test_mutation_of_frozen_publish_tree_fails_final_locked_rehash(tmp_path):
    staging = tmp_path / ".publication.staging-test"
    staging.mkdir()
    manifest = staging / "provenance" / "final_figure_orchestration.json"
    atomic_write_json(manifest, {"snapshot_artifacts": [], "candidates": []})
    output = tmp_path / "publication"
    frozen = tmp_path / "frozen-publication"
    ledger_sha256 = freeze_staging_for_test(
        staging=staging,
        destination=frozen,
        manifest_path=manifest,
    )
    target = frozen / "provenance" / "final_figure_orchestration.json"
    target.chmod(0o600)
    target.write_text("late mutation\n", encoding="utf-8")
    target.chmod(0o400)

    with pytest.raises(
        FinalFigureOrchestrationError, match="Frozen publication artifact mutated"
    ):
        _verify_frozen_publication_tree(frozen, ledger_sha256=ledger_sha256)
    assert not output.exists()
    _remove_tree(frozen)


def test_frozen_publication_builder_seals_complete_tree(tmp_path):
    staging = tmp_path / ".publication.staging-test"
    staging.mkdir()
    manifest = staging / "provenance" / "final_figure_orchestration.json"
    atomic_write_json(manifest, {"snapshot_artifacts": [], "candidates": []})
    output = tmp_path / "publication"
    frozen = tmp_path / "frozen-publication"
    ledger_sha256 = freeze_staging_for_test(
        staging=staging,
        destination=frozen,
        manifest_path=manifest,
    )
    _verify_frozen_publication_tree(frozen, ledger_sha256=ledger_sha256)
    assert (frozen / "provenance" / "published_tree_sha256.json").is_file()
    assert stat.S_IMODE(frozen.stat().st_mode) == 0o500
    assert all(
        stat.S_IMODE(path.stat().st_mode) == (0o500 if path.is_dir() else 0o400)
        for path in frozen.rglob("*")
    )
    assert staging.exists()
    assert not output.exists()
    _remove_tree(frozen)
