import copy

import pytest

from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.baselines.run_graph_circular_ellipse_common_metrics import (
    SUMMARY_METRICS,
    summarize_case_results,
    verify_canonical_ellipse_geometry,
)


def _saved_geometry(case):
    return {
        "case_index": case.case_index,
        "geometry_type": "ellipse",
        **dict(case.parameters),
    }


def test_saved_ellipse_geometry_matches_canonical_full_rng_sequence():
    cases = canonical_benchmark_cases("ellipses", (0, 4, 24))
    for case in cases:
        verify_canonical_ellipse_geometry(_saved_geometry(case), case)


def test_canonical_match_rejects_geometry_drift():
    case = canonical_benchmark_cases("ellipses", (4,))[0]
    saved = copy.deepcopy(_saved_geometry(case))
    saved["theta"] += 1.0e-6

    with pytest.raises(ValueError, match="theta differs"):
        verify_canonical_ellipse_geometry(saved, case)


def test_summary_recovers_median_geometry_and_curvature_orders():
    rows = []
    for resolution in (32, 64, 128):
        h = 100.0 / resolution
        for case_index, factor in enumerate((0.8, 0.9, 1.0, 1.1, 1.2)):
            row = {
                "case_index": case_index,
                "cells_per_side": resolution,
                "cell_size": h,
                "primitive_count": 10,
                "arc_count": 10,
                "line_count": 0,
            }
            for metric in SUMMARY_METRICS:
                power = 2.0 if "curvature" not in metric else 1.25
                row[metric] = factor * h**power
            rows.append(row)

    summary, case_orders = summarize_case_results(rows)

    assert len(summary) == 3
    assert len(case_orders) == 5
    assert summary[-1]["native_symmetric_hausdorff_fit_order"] == pytest.approx(2.0)
    assert summary[-1][
        "geometric_curvature_mean_absolute_error_fit_order"
    ] == pytest.approx(1.25)
    assert case_orders[2]["native_symmetric_hausdorff_fit_order"] == pytest.approx(2.0)
