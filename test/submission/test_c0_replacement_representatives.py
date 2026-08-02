import json

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


def test_run_provenance_accepts_comma_separated_case_indices(tmp_path, monkeypatch):
    run_name = "selected_cases"
    run_root = tmp_path / "plots" / run_name
    facet_root = run_root / "vtk" / "reconstructed" / "facets"
    facet_root.mkdir(parents=True)
    (facet_root / "9.vtp").write_text("vtp")
    (facet_root / "9.facet_metadata.json").write_text("{}")
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "parameters": {"case_indices": "9,24", "facet_algo": "linear"},
            }
        )
    )
    monkeypatch.setattr(c0_reps, "REPO_ROOT", tmp_path)

    provenance = c0_reps._run_provenance(run_name, 9, expected={"facet_algo": "linear"})

    assert provenance["source_commit"] == "a" * 40
