import inspect
from types import SimpleNamespace

from experiments.static.run_perturbed_sweeps import (
    DISPLAY_LABELS,
    EXPERIMENTS,
    _build_run_spec,
)
from main.structs.meshes.merge_mesh import MergeMesh


def test_production_reconstruction_defaults_are_frozen():
    fit_defaults = inspect.signature(MergeMesh.fitFacets).parameters

    assert MergeMesh.default_corner_behavior_profile == "pre_f8_corner"
    assert MergeMesh.default_rescue_profile == "exact_linear_support_only"
    assert fit_defaults["plic_fallback"].default == "LVIRA"
    assert fit_defaults["rescue_profile"].default == "exact_linear_support_only"


def test_orientation_profile_ablation_uses_distinct_outputs_for_every_driver():
    args = SimpleNamespace(
        plic_fallback="LVIRA",
        rescue_profile=MergeMesh.default_rescue_profile,
    )

    for experiment_name in ("lines", "circles", "ellipses", "squares", "zalesak"):
        experiment = next(
            item for item in EXPERIMENTS if item["name"] == experiment_name
        )
        algo = "linear" if experiment_name == "lines" else "circular"
        specs = [
            _build_run_spec(
                experiment,
                algo,
                1.5,
                0.3,
                0,
                1,
                args,
                profile,
            )
            for profile in ("current", MergeMesh.default_corner_behavior_profile)
        ]

        assert specs[0]["save_name"] != specs[1]["save_name"]
        for spec, profile in zip(
            specs, ("current", MergeMesh.default_corner_behavior_profile)
        ):
            profile_index = spec["cmd"].index("--corner_behavior_profile") + 1
            assert spec["cmd"][profile_index] == profile


def test_final_namespace_and_approved_circular_labels():
    args = SimpleNamespace(
        plic_fallback="LVIRA",
        rescue_profile=MergeMesh.default_rescue_profile,
        run_namespace="submission_20260730",
    )
    experiment = next(item for item in EXPERIMENTS if item["name"] == "circles")
    spec = _build_run_spec(
        experiment,
        "safe_circle",
        1.0,
        0.1,
        0,
        25,
        args,
        MergeMesh.default_corner_behavior_profile,
    )

    assert spec["save_name"].startswith("submission_20260730_perturb_sweep_")
    assert DISPLAY_LABELS["safe_circle"] == "Ours (circular, independent cells)"
    assert DISPLAY_LABELS["circular"] == "Ours (circular, topology + merging)"
