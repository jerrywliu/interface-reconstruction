import csv
import json
from pathlib import Path

import pytest

from experiments.static import run_appendix_c0_study as study
from experiments.static.figure_generation_provenance import (
    frozen_reconstruction_profile,
)


def _ellipse_spec():
    return next(
        spec for spec in study.APPENDIX_EXPERIMENTS if spec["name"] == "ellipses"
    )


def _variant(spec, label):
    return next(item for item in spec["variants"] if item["label"] == label)


def test_joint_is_explicit_in_generated_command():
    spec = _ellipse_spec()
    command = study._build_command(
        spec,
        _variant(spec, "linear+C0"),
        resolution=0.32,
        wiggle=0.1,
        seed=0,
        num_cases=25,
        save_name="joint-test",
        c0_mode="joint",
        case_indices=None,
        reconstruction_profile=frozen_reconstruction_profile(),
    )

    assert command[command.index("--c0_mode") + 1] == "joint"
    assert command[command.index("--do_c0") + 1] == "1"
    assert command[command.index("--plic_fallback") + 1] == "LVIRA"


def test_guarded_remains_an_explicit_command_mode():
    spec = _ellipse_spec()
    command = study._build_command(
        spec,
        _variant(spec, "linear+C0"),
        resolution=0.32,
        wiggle=0.1,
        seed=0,
        num_cases=25,
        save_name="guarded-test",
        c0_mode="guarded",
        case_indices="9",
        reconstruction_profile=frozen_reconstruction_profile(),
    )

    assert command[command.index("--c0_mode") + 1] == "guarded"
    assert command[command.index("--case_indices") + 1] == "9"


def test_authoritative_rows_record_mode_and_hash(tmp_path):
    spec = _ellipse_spec()
    root = tmp_path / "guarded"
    source = root / "ellipses"
    source.mkdir(parents=True)
    fields = [
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
    ]
    rows = [
        {
            "experiment": "ellipses",
            "algo": "linear",
            "facet_algo": "linear",
            "do_c0": 0,
            "resolution": 0.32,
            "wiggle": 0.1,
            "seed": 0,
            "metric_key": "hausdorff_median",
            "metric_value": 0.2,
            "save_name": "linear-run",
        },
        {
            "experiment": "ellipses",
            "algo": "linear+C0",
            "facet_algo": "linear",
            "do_c0": 1,
            "resolution": 0.32,
            "wiggle": 0.1,
            "seed": 0,
            "metric_key": "hausdorff_median",
            "metric_value": 0.1,
            "save_name": "guarded-run",
        },
    ]
    with (source / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (source / "manifest.json").write_text(
        json.dumps({"generation_provenance": {"source_commit": "guarded-commit"}}),
        encoding="utf-8",
    )

    loaded, provenance = study._load_authoritative_rows(
        spec,
        root,
        variants=[_variant(spec, "linear"), _variant(spec, "linear+C0")],
        grid={(0.32, 0.1, 0)},
        c0_mode="guarded",
        include_c0=True,
    )

    assert [row["c0_mode"] for row in loaded] == ["none", "guarded"]
    assert all(row["data_source"] == "authoritative_guarded_reuse" for row in loaded)
    assert all(len(row["source_sha256"]) == 64 for row in loaded)
    assert provenance["source_commit"] == "guarded-commit"


def test_guarded_joint_comparison_matches_exact_setting():
    shared = {
        "experiment": "ellipses",
        "algo": "linear+C0",
        "do_c0": 1,
        "resolution": 0.32,
        "wiggle": 0.1,
        "seed": 0,
        "metric_key": "hausdorff_median",
    }
    comparison, summary = study._compare_guarded_joint(
        [{**shared, "metric_value": 0.05}],
        [{**shared, "metric_value": 0.1}],
    )

    assert comparison[0]["joint_over_guarded"] == 0.5
    assert summary[0]["joint_better_settings"] == 1
    assert summary[0]["joint_worse_settings"] == 0


def test_joint_qa_summary_accounts_for_components():
    row = {
        "experiment": "ellipses",
        "c0_mode": "joint",
        "num_c0_joint_components": 3,
        "num_c0_joint_components_solved": 2,
        "num_c0_joint_components_failed": 1,
        "num_c0_exact_c1_components": 1,
        "num_c0_conservative_fallback_components": 1,
        "num_c0_bad_joins_before_joint": 5,
        "num_c0_bad_joins_after_joint": 1,
        "num_final_missing_cells": 0,
        "max_c0_relative_area_residual": 1.0e-10,
        "independent_max_relative_conservation_residual": 2.0e-10,
        "independent_global_relative_conservation_residual": 3.0e-12,
        "hausdorff": 0.01,
        "facet_gap": 1.0e-6,
    }

    summary = study._qa_summary([row])

    assert summary["joint_components"] == 3
    assert summary["joint_components_solved"] == 2
    assert summary["cases_with_failed_components"] == 1
    assert summary["cases_with_remaining_bad_joins"] == 1
    assert summary["max_independent_relative_conservation_residual"] == 2.0e-10


def test_analytic_disk_polygon_area_handles_negative_radius_merged_arc():
    polygon = [
        [44.08518170371095, 56.04333794258135],
        [45.99170397728862, 56.040844193280655],
        [46.01973237648558, 57.932845849594955],
        [43.91535661343454, 58.08808752969412],
    ]
    geometry = {
        "class": "circular",
        "center": [44.89393341985056, 57.12722428038411],
        "radius": -1.3523642008106553,
        "p_left": [44.08518170371095, 56.04333794258135],
        "p_right": [43.94042358284147, 58.086238316966025],
    }

    area = study._robust_facet_area(polygon, geometry)

    assert 0.0 <= area <= abs(study.getArea(polygon))
    assert area == pytest.approx(0.040926819094817546, abs=1.0e-12)


def test_high_precision_disk_area_avoids_large_radius_cancellation():
    polygon = [
        [44.04803647146923, 50.09330437526912],
        [44.573747471240296, 49.80233968935035],
        [44.81769592860848, 50.65473130291138],
        [44.029482605682304, 50.67968452912513],
    ]
    center = [-10821948.139556998, -39022924.76177308]
    radius = 40495778.56223259

    ordinary = study._disk_polygon_intersection_area(polygon, center, radius)
    high_precision = study._disk_polygon_intersection_area_high_precision(
        polygon, center, radius
    )

    assert high_precision == pytest.approx(0.04032716834772022, abs=1.0e-14)
    assert abs(ordinary - high_precision) > 1.0e-2


def test_publication_labels_distinguish_joint_from_guarded():
    assert study._publication_label("linear+C0", "joint").endswith("joint C0")
    assert study._publication_label("linear+C0", "guarded").endswith("guarded C0")
    assert study._publication_label("linear", "joint") == "Graph-coordinated linear"


def test_manifest_validation_rejects_partial_case_support(tmp_path):
    run_dir = tmp_path / "partial"
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "facet_algo": "linear",
                    "do_c0": True,
                    "mesh_type": "perturbed_quads",
                    "perturb_wiggle": 0.1,
                    "perturb_seed": 0,
                    "perturb_fix_boundary": 1,
                    "c0_mode": "joint",
                    "plic_fallback": "LVIRA",
                    "corner_behavior_profile": "pre_f8_corner",
                    "resolution": 0.32,
                    "case_indices": None,
                }
            }
        ),
        encoding="utf-8",
    )
    (metrics / "case_metrics.csv").write_text(
        "case_index,hausdorff\n0,0.1\n",
        encoding="utf-8",
    )
    job = {
        "experiment": "ellipses",
        "facet_algo": "linear",
        "do_c0": True,
        "wiggle": 0.1,
        "seed": 0,
        "resolution": 0.32,
        "num_cases": 2,
    }

    with pytest.raises(ValueError, match="incomplete case support"):
        study._validate_run_manifest(
            run_dir,
            job,
            case_indices=None,
            c0_mode="joint",
        )


def test_reuse_replaces_invalid_partial_run(tmp_path, monkeypatch):
    plots = tmp_path / "plots"
    run_dir = plots / "partial-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    validations = 0

    def validate(path, *_args, **_kwargs):
        nonlocal validations
        validations += 1
        if validations == 1:
            raise ValueError("partial")
        assert (path / "rerun.marker").is_file()
        return {}

    def rerun(_command, _log_path):
        assert not run_dir.exists()
        run_dir.mkdir(parents=True)
        (run_dir / "rerun.marker").write_text("complete", encoding="utf-8")
        return 0

    monkeypatch.setattr(study, "_validate_run_manifest", validate)
    monkeypatch.setattr(study, "_run_subprocess", rerun)
    record = study._run_job(
        {"save_name": "partial-run", "command": ["python"], "experiment": "ellipses"},
        log_dir=tmp_path / "logs",
        reuse_existing=True,
        case_indices=None,
        c0_mode="joint",
    )

    assert validations == 2
    assert record["status"] == "completed"
