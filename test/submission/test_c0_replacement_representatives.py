import json

import pytest

from experiments.submission import generate_c0_replacement_representatives as c0_reps


def test_endpoint_continuity_accepts_a_closed_primitive_loop(tmp_path):
    metadata = {
        "primitives": [
            {"p_left": [0.0, 0.0], "p_right": [1.0, 0.0]},
            {"p_left": [1.0, 0.0], "p_right": [1.0, 1.0]},
            {"p_left": [1.0, 1.0], "p_right": [0.0, 1.0]},
            {"p_left": [0.0, 1.0], "p_right": [0.0, 0.0]},
        ]
    }
    path = tmp_path / "facets.json"
    path.write_text(json.dumps(metadata))

    audit = c0_reps._endpoint_continuity(path)

    assert audit["globally_continuous"] is True
    assert audit["max_endpoint_partner_gap"] == 0.0
    assert audit["endpoints_above_tolerance"] == 0


def test_endpoint_continuity_rejects_an_open_join(tmp_path):
    metadata = {
        "primitives": [
            {"p_left": [0.0, 0.0], "p_right": [1.0, 0.0]},
            {"p_left": [1.1, 0.0], "p_right": [0.0, 0.0]},
        ]
    }
    path = tmp_path / "facets.json"
    path.write_text(json.dumps(metadata))

    audit = c0_reps._endpoint_continuity(path)

    assert audit["globally_continuous"] is False
    assert audit["endpoints_above_tolerance"] == 2


def test_replacement_sources_distinguish_joint_and_guarded_c0():
    assert "joint_c0" in c0_reps.ELLIPSE_RUNS["linear+C0"]
    assert "continuous_case" in c0_reps.ZALESAK_RUN
    assert c0_reps.ELLIPSE_CASE == 9
    assert c0_reps.ZALESAK_CASE == 22


def test_run_provenance_accepts_comma_separated_case_indices(tmp_path):
    run_name = "selected_cases"
    run_root = tmp_path / "plots" / run_name
    facet_root = run_root / "vtk" / "reconstructed" / "facets"
    facet_root.mkdir(parents=True)
    metrics_root = run_root / "metrics"
    metrics_root.mkdir()
    (facet_root / "9.vtp").write_text("vtp")
    (facet_root / "9.facet_metadata.json").write_text("{}")
    (run_root / "vtk" / "mesh.vtk").write_text("mesh")
    (metrics_root / "case_metrics.csv").write_text("case_index\n9\n")
    (metrics_root / "case_geometry.jsonl").write_text('{"case_index": 9}\n')
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "parameters": {"case_indices": "9,24", "facet_algo": "linear"},
            }
        )
    )
    provenance = c0_reps._run_provenance(
        tmp_path / "plots",
        run_name,
        9,
        expected={"facet_algo": "linear"},
    )

    assert provenance["source_commit"] == "a" * 40
    assert set(provenance["files"]) == {
        "run_manifest",
        "case_metrics",
        "case_geometry",
        "mesh",
        "facet_vtp",
        "facet_metadata",
    }


def test_run_map_is_closed_and_rejects_missing_labels():
    parsed = c0_reps._run_map(
        ["linear=a", "linear+C0=b", "circular=c"], c0_reps.ELLIPSE_RUNS
    )
    assert parsed == {"linear": "a", "linear+C0": "b", "circular": "c"}
    with pytest.raises(ValueError, match="Run labels"):
        c0_reps._run_map(["linear=a", "circular=c"], c0_reps.ELLIPSE_RUNS)
