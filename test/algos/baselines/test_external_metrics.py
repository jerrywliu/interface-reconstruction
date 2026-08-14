import math

import pytest

from main.algos.baselines.external_geometry import (
    ExternalBaselineResult,
    ExternalArcPrimitive,
    ExternalCellReconstruction,
    ExternalInterfaceComponent,
    ExternalLinePrimitive,
    ExternalParabolicPrimitive,
    ExternalReconstructionStatus,
)
from main.algos.baselines.external_metrics import (
    conservation_metrics,
    geometric_curvature_error_external,
    shared_edge_gap_metrics,
    symmetric_hausdorff_external,
    tangent_error_external,
)


def _record(index, polygon, primitives, area=0.5):
    return ExternalCellReconstruction(
        cell_index=index,
        polygon=polygon,
        components=tuple(
            ExternalInterfaceComponent((primitive,)) for primitive in primitives
        ),
        source_method="test",
        source_variant="native",
        status=ExternalReconstructionStatus.RECONSTRUCTED,
        target_phase_area=area,
        exact_area_callback=lambda points: area,
    )


def test_shared_edge_metric_matches_every_disconnected_crossing():
    left_polygon = ((0, 0), (1, 0), (1, 1), (0, 1))
    right_polygon = ((1, 0), (2, 0), (2, 1), (1, 1))
    left = _record(
        (0, 0),
        left_polygon,
        (
            ExternalLinePrimitive((0.0, 0.2), (1.0, 0.2)),
            ExternalLinePrimitive((1.0, 0.8), (0.0, 0.8)),
        ),
    )
    right = _record(
        (1, 0),
        right_polygon,
        (
            ExternalLinePrimitive((1.0, 0.21), (2.0, 0.21)),
            ExternalLinePrimitive((2.0, 0.77), (1.0, 0.77)),
        ),
    )
    result = ExternalBaselineResult("test", "native", {(0, 0): left, (1, 0): right})

    metrics = shared_edge_gap_metrics(result)

    assert metrics["matched_count"] == 2
    assert metrics["unmatched_count"] == 0
    assert metrics["max"] == pytest.approx(0.03)
    assert metrics["mean"] == pytest.approx(0.02)


def test_crossing_count_mismatch_is_explicit():
    left_polygon = ((0, 0), (1, 0), (1, 1), (0, 1))
    right_polygon = ((1, 0), (2, 0), (2, 1), (1, 1))
    left = _record(
        (0, 0),
        left_polygon,
        (
            ExternalLinePrimitive((0.0, 0.2), (1.0, 0.2)),
            ExternalLinePrimitive((1.0, 0.8), (0.0, 0.8)),
        ),
    )
    right = _record(
        (1, 0), right_polygon, (ExternalLinePrimitive((1.0, 0.2), (2.0, 0.2)),)
    )
    metrics = shared_edge_gap_metrics(
        ExternalBaselineResult("test", "native", {(0, 0): left, (1, 0): right})
    )
    assert metrics["matched_count"] == 1
    assert metrics["unmatched_count"] == 1


def test_native_parabola_hausdorff_and_tangent_metrics_do_not_require_facets():
    primitive = ExternalParabolicPrimitive((0, 0), 0.2, 0.4, 0.1, -0.5, 0.5)
    duplicate = ExternalParabolicPrimitive((0, 0), 0.2, 0.4, 0.1, -0.5, 0.5)
    assert (
        symmetric_hausdorff_external((primitive,), (duplicate,), minimum_spacing=0.01)
        < 1.0e-7
    )
    tangent = tangent_error_external(
        (primitive,), (duplicate,), samples_per_primitive=20
    )
    assert tangent["max"] < 1.0e-7


def test_conservation_uses_per_method_exact_area_callback():
    record = _record(
        (0, 0),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
        (ExternalLinePrimitive((0.0, 0.5), (1.0, 0.5)),),
        area=0.5,
    )
    metrics = conservation_metrics(
        ExternalBaselineResult("test", "native", {(0, 0): record})
    )
    assert metrics["evaluated_cells"] == 1
    assert metrics["max_absolute_residual"] == pytest.approx(0.0)


def test_native_hausdorff_is_invariant_to_equivalent_line_partition():
    whole = (ExternalLinePrimitive((0.0, 0.0), (2.0, 0.0)),)
    split = (
        ExternalLinePrimitive((0.0, 0.0), (0.7, 0.0)),
        ExternalLinePrimitive((0.7, 0.0), (2.0, 0.0)),
    )

    assert symmetric_hausdorff_external(whole, split) < 1.0e-12

    offset = (ExternalLinePrimitive((0.0, 0.25), (2.0, 0.25)),)
    assert symmetric_hausdorff_external(whole, offset) == pytest.approx(0.25)
    assert symmetric_hausdorff_external(split, offset) == pytest.approx(0.25)


def test_native_circle_metrics_are_invariant_to_arc_partition():
    halves = (
        ExternalArcPrimitive((0.0, 0.0), 2.0, 0.0, math.pi),
        ExternalArcPrimitive((0.0, 0.0), 2.0, math.pi, math.pi),
    )
    quarters = tuple(
        ExternalArcPrimitive((0.0, 0.0), 2.0, index * math.pi / 2.0, math.pi / 2.0)
        for index in range(4)
    )

    assert symmetric_hausdorff_external(halves, quarters) < 1.0e-11
    curvature = geometric_curvature_error_external(halves, quarters)
    assert curvature["mean_absolute_error"] < 1.0e-14
    assert curvature["relative_l1_error"] < 1.0e-14


def test_nonzero_native_distance_is_stable_under_parabola_partition():
    whole = (ExternalParabolicPrimitive((0, 0), 0.0, 0.5, 0.0, -1.0, 1.0),)
    split = (
        ExternalParabolicPrimitive((0, 0), 0.0, 0.5, 0.0, -1.0, 0.0),
        ExternalParabolicPrimitive((0, 0), 0.0, 0.5, 0.0, 0.0, 1.0),
    )
    chord = (ExternalLinePrimitive(whole[0].p_left, whole[0].p_right),)

    whole_distance = symmetric_hausdorff_external(whole, chord)
    split_distance = symmetric_hausdorff_external(split, chord)
    assert whole_distance > 0.0
    assert split_distance == pytest.approx(whole_distance, abs=1.0e-10)
