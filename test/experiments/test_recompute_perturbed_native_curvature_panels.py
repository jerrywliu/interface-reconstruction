import math

from main.algos.baselines.external_geometry import (
    ExternalArcPrimitive,
    ExternalLinePrimitive,
)
from main.algos.baselines.external_metrics import geometric_curvature_error_external
from experiments.static.recompute_perturbed_native_curvature_panels import (
    patch_sweep_rows,
    truth_primitive,
)


def test_circle_truth_supports_common_native_curvature_observable():
    truth = truth_primitive(
        {"geometry_type": "circle", "center": [2.0, 3.0], "radius": 4.0}
    )
    exact = ExternalArcPrimitive((2.0, 3.0), 4.0, 0.0, 2.0 * math.pi)
    line = ExternalLinePrimitive((6.0, 3.0), (-2.0, 3.0))

    assert geometric_curvature_error_external((exact,), (truth,))[
        "mean_absolute_error"
    ] < 1.0e-14
    assert math.isclose(
        geometric_curvature_error_external((line,), (truth,))[
            "mean_absolute_error"
        ],
        0.25,
        rel_tol=1.0e-13,
    )


def test_patch_sweep_rows_replaces_all_four_curvature_statistics_only():
    key_fields = {
        "experiment": "circles",
        "algo": "circular",
        "resolution": "0.64",
        "wiggle": "0.1",
        "seed": "0",
    }
    rows = [
        {**key_fields, "metric_key": f"curvature_error_{stat}", "metric_value": "9"}
        for stat in ("mean", "median", "p25", "p75")
    ]
    rows.append({**key_fields, "metric_key": "hausdorff_median", "metric_value": "3"})
    aggregates = {
        ("circles", "circular", 0.64, 0.1, 0): {
            "mean": 1.0,
            "median": 2.0,
            "p25": 3.0,
            "p75": 4.0,
        }
    }

    patched = patch_sweep_rows(rows, aggregates)

    assert [float(row["metric_value"]) for row in patched[:4]] == [1.0, 2.0, 3.0, 4.0]
    assert patched[4]["metric_value"] == "3"
