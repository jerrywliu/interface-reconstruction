import math

import pytest

from experiments.submission.conservation_analyzer import (
    GridRecord,
    analyze_case_records,
    compare_c0_stages,
    facet_area_in_polygon,
)


UNIT_CELL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def _grid(nx=1):
    return GridRecord(
        points=[[[float(x), 0.0], [float(x), 1.0]] for x in range(nx + 1)],
        source="test",
    )


def _row(cell_id, fraction, merge_id, facet):
    x, y = map(int, cell_id.split(","))
    return {
        "experiment": "synthetic",
        "algo": "test",
        "case_index": 0,
        "cell_id": cell_id,
        "cell_x": x,
        "cell_y": y,
        "cell_fraction": fraction,
        "merge_id": merge_id,
        "facet_geometry_json": facet,
    }


def test_linear_facet_area_and_case_conservation():
    facet = {
        "class": "linear",
        "p_left": [0.3, 0.0],
        "p_right": [0.3, 1.0],
    }
    assert facet_area_in_polygon(UNIT_CELL, facet) == pytest.approx(0.3)
    analysis = analyze_case_records(
        _grid(),
        [_row("0,0", 0.3, 0, facet)],
        total_prescribed_phase_area=0.3,
    )
    assert analysis.summary["global_relative_phase_area_error"] < 1.0e-14


def test_arc_facet_uses_signed_circle_phase():
    facet = {
        "class": "circular",
        "center": [0.5, 0.5],
        "radius": 0.25,
        "p_left": [0.75, 0.5],
        "p_right": [0.5, 0.75],
    }
    assert facet_area_in_polygon(UNIT_CELL, facet) == pytest.approx(math.pi / 16)
    facet["radius"] = -0.25
    assert facet_area_in_polygon(UNIT_CELL, facet) == pytest.approx(1 - math.pi / 16)


def test_linear_corner_facet_area():
    facet = {
        "class": "linear_corner",
        "p_left": [0.5, 1.0],
        "corner": [0.5, 0.5],
        "p_right": [1.0, 0.5],
        "left_branch": {"class": "linear"},
        "right_branch": {"class": "linear"},
    }
    assert facet_area_in_polygon(UNIT_CELL, facet) == pytest.approx(0.25)


def test_merged_zone_total_can_hide_constituent_cell_residuals():
    facet = {
        "class": "linear",
        "p_left": [0.75, 0.0],
        "p_right": [0.75, 1.0],
    }
    analysis = analyze_case_records(
        _grid(nx=2),
        [
            _row("0,0", 0.5, 7, facet),
            _row("1,0", 0.25, 7, facet),
        ],
        total_prescribed_phase_area=0.75,
    )

    assert analysis.summary["num_merged_zones"] == 1
    assert analysis.zone_rows[0]["absolute_residual"] == pytest.approx(0.0)
    assert analysis.summary["global_relative_phase_area_error"] == pytest.approx(0.0)
    assert analysis.summary["max_cell_absolute_residual"] == pytest.approx(0.25)
    assert sorted(row["signed_residual"] for row in analysis.cell_rows) == pytest.approx(
        [-0.25, 0.25]
    )


def test_c0_comparison_requires_both_saved_stages():
    base = {
        "experiment": "synthetic",
        "save_name": "paired",
        "case_index": 0,
        "global_relative_phase_area_error": 1.0e-12,
        "max_merged_zone_absolute_residual": 2.0e-10,
        "max_merged_cell_absolute_residual": 3.0e-4,
    }
    before = {**base, "algo": "circular", "stage": "before_c0"}
    after = {
        **base,
        "algo": "circular+C0",
        "stage": "after_c0",
        "global_relative_phase_area_error": 4.0e-12,
    }

    assert compare_c0_stages([before]) == []
    comparison = compare_c0_stages([before, after])
    assert len(comparison) == 1
    assert comparison[0][
        "global_relative_phase_area_error_delta_after_minus_before"
    ] == pytest.approx(3.0e-12)
