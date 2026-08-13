import math

import pytest

from experiments.static.diagnose_ellipse_curvature_metric_sensitivity import (
    CaseMetric,
    ellipse_curvature,
    ellipse_parameter_projection,
    fit_order,
    native_primitive_midpoint,
    primitive_diagnostic,
    summarize,
    true_interval_statistics,
)


def circle_geometry(radius=2.0):
    return {
        "center": [0.0, 0.0],
        "major_axis": radius,
        "minor_axis": radius,
        "theta": 0.0,
    }


def arc_primitive(radius=2.0, start=0.0, delta=0.5):
    return {
        "kind": "arc",
        "source_name": "arc",
        "center": [0.0, 0.0],
        "radius": radius,
        "p_left": [radius * math.cos(start), radius * math.sin(start)],
        "p_right": [
            radius * math.cos(start + delta),
            radius * math.sin(start + delta),
        ],
        "signed_delta": delta,
    }


def test_native_arc_midpoint_uses_serialized_center_radius_and_delta():
    midpoint = native_primitive_midpoint(arc_primitive(start=0.2, delta=0.6))
    assert midpoint == pytest.approx([2.0 * math.cos(0.5), 2.0 * math.sin(0.5)])


def test_circle_interval_average_and_point_targets_are_exact():
    primitive = arc_primitive(radius=2.0, start=0.3, delta=0.8)
    length, average = true_interval_statistics(primitive, circle_geometry())
    errors, diagnostic_length = primitive_diagnostic(primitive, circle_geometry())
    assert length == pytest.approx(1.6, rel=1.0e-12)
    assert diagnostic_length == pytest.approx(length)
    assert average == pytest.approx(0.5, rel=1.0e-12)
    assert errors == pytest.approx(
        {
            "chord_midpoint": 0.0,
            "native_arc_midpoint": 0.0,
            "interval_average": 0.0,
        },
        abs=1.0e-13,
    )


def test_rotated_parameter_projection_and_ellipse_curvature():
    geometry = {
        "center": [2.0, 3.0],
        "major_axis": 4.0,
        "minor_axis": 2.0,
        "theta": math.pi / 2.0,
    }
    assert ellipse_parameter_projection([2.0, 7.0], geometry) == pytest.approx(0.0)
    assert ellipse_curvature(geometry, 0.0) == pytest.approx(1.0)


def _case(resolution, case_index, factor):
    h = 100.0 / resolution
    values = {
        target: factor * h**2
        for target in (
            "chord_midpoint_equal_facet",
            "chord_midpoint_arc_length_weighted",
            "native_arc_midpoint_equal_facet",
            "native_arc_midpoint_arc_length_weighted",
            "interval_average_equal_facet",
            "interval_average_arc_length_weighted",
        )
    }
    return CaseMetric(
        source_run=f"n{resolution}",
        source_commit="test",
        resolution=resolution,
        cell_size=h,
        case_index=case_index,
        primitive_count=2,
        arc_count=1,
        non_arc_count=1,
        fallback_non_arc_count=1,
        total_true_interval_length=1.0,
        **values,
    )


def test_summary_recovers_order_and_tracks_non_arc_population():
    rows = [
        _case(resolution, case_index, factor)
        for resolution in (16, 32, 64, 128)
        for case_index, factor in enumerate((0.9, 1.0, 1.1))
    ]
    summaries = summarize(rows)
    selected = [
        row
        for row in summaries
        if row["target"] == "interval_average"
        and row["aggregation"] == "arc_length_weighted"
    ]
    assert fit_order(selected) == pytest.approx(2.0)
    assert selected[0]["order_all_resolutions"] == pytest.approx(2.0)
    assert selected[0]["order_finest_four"] == pytest.approx(2.0)
    assert selected[0]["primitive_count"] == 6
    assert selected[0]["non_arc_count"] == 3
    assert selected[0]["fallback_non_arc_count"] == 3
    assert selected[0]["non_arc_fraction"] == pytest.approx(0.5)
