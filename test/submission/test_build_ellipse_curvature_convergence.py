import math

import pytest

from experiments.static.build_ellipse_curvature_convergence import (
    ellipse_curvature,
    ellipse_parameter,
    primitive_curvature,
    summarize,
    CaseMetric,
)


def test_exact_arc_curvature_is_unsigned():
    assert primitive_curvature({"kind": "arc", "radius": -4.0}) == 0.25
    assert primitive_curvature({"kind": "line"}) == 0.0


def test_rotated_ellipse_parameter_and_curvature():
    geometry = {
        "center": [2.0, 3.0],
        "major_axis": 4.0,
        "minor_axis": 2.0,
        "theta": math.pi / 2,
    }
    # The positive local major-axis endpoint rotates onto the positive y-axis.
    assert ellipse_parameter([2.0, 7.0], geometry) == pytest.approx(0.0)
    assert ellipse_curvature(geometry, 0.0) == pytest.approx(1.0)


def test_summary_recovers_known_order():
    rows = []
    for resolution in (16, 32, 64):
        for case_index, factor in enumerate((0.9, 1.0, 1.1)):
            rows.append(
                CaseMetric(
                    source_run=f"n{resolution}",
                    source_commit="test",
                    resolution=resolution,
                    cell_size=100.0 / resolution,
                    perturbation=0.0,
                    case_index=case_index,
                    primitive_count=1,
                    arc_count=1,
                    line_count=0,
                    mean_absolute_curvature_error=factor * (100.0 / resolution) ** 2,
                )
            )
    _, order = summarize(rows)
    assert order == pytest.approx(2.0)
