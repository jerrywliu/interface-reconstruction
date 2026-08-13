import json
import math

import numpy as np
import pytest

from main.algos.baselines.external_geometry import (
    ExternalArcPrimitive,
    ExternalBaselineResult,
    ExternalCellReconstruction,
    ExternalInterfaceComponent,
    ExternalLinePrimitive,
    ExternalParabolicPrimitive,
    ExternalQuadraticPrimitive,
    ExternalReconstructionStatus,
    adapt_linear_facet,
    adapt_parabolic_interval,
    adapt_quadratic_facet,
)
from main.algos.baselines.plvira import ParabolicInterface
from main.algos.baselines.quasi import QuadraticFacet
from main.structs.facets.linear_facet import LinearFacet


SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_all_native_primitive_adapters_preserve_geometry_and_distance():
    line = ExternalLinePrimitive((0.0, 0.0), (1.0, 0.0))
    arc = ExternalArcPrimitive((0.5, 0.5), 0.5, math.pi, -math.pi)
    parabola = ExternalParabolicPrimitive((0.5, 0.5), 0.0, 0.4, 0.1, -0.5, 0.5)
    quadratic = ExternalQuadraticPrimitive((0.0, 0.2), (1.0, 0.2), 0.1)

    assert line.distance_to_point((0.5, 0.25)) == pytest.approx(0.25)
    assert arc.distance_to_point((0.5, 1.0)) == pytest.approx(0.0, abs=1.0e-14)
    assert parabola.distance_to_point(parabola.point(0.37)) < 1.0e-7
    assert quadratic.distance_to_point(quadratic.point(0.63)) < 1.0e-7
    assert all(
        primitive.length() > 0.0 for primitive in (line, arc, parabola, quadratic)
    )


def test_disconnected_four_crossing_cell_round_trips_losslessly():
    lower = ExternalArcPrimitive(
        (0.5, 0.5),
        0.45,
        math.pi,
        math.pi,
        {"crossing_pair": [0, 3]},
    )
    upper = ExternalArcPrimitive(
        (0.5, 0.5),
        0.45,
        0.0,
        math.pi,
        {"crossing_pair": [2, 1]},
    )
    record = ExternalCellReconstruction(
        cell_index=(2, 4),
        polygon=SQUARE,
        components=(
            ExternalInterfaceComponent((lower,), metadata={"component_id": 0}),
            ExternalInterfaceComponent((upper,), metadata={"component_id": 1}),
        ),
        source_method="PCIC",
        source_variant="center_translation",
        status=ExternalReconstructionStatus.RECONSTRUCTED,
        diagnostics={"intersections": 4, "policy": {"phase": "disk"}},
        target_phase_area=0.2,
        exact_area_callback=lambda polygon: 0.2,
    )
    result = ExternalBaselineResult(
        "PCIC", "center_translation", {(2, 4): record}, {"seed": 7}
    )

    payload = result.to_json()
    restored = ExternalBaselineResult.from_json(payload)

    assert json.loads(restored.to_json()) == json.loads(payload)
    assert len(restored.cells[(2, 4)].components) == 2
    assert len(restored.cells[(2, 4)].primitives()) == 2
    assert restored.cells[(2, 4)].exact_phase_area() == pytest.approx(0.2)


def test_status_contract_does_not_hide_missing_geometry():
    with pytest.raises(ValueError, match="requires geometry"):
        ExternalCellReconstruction(
            (0, 0),
            SQUARE,
            (),
            "QUASI",
            "published",
            ExternalReconstructionStatus.RECONSTRUCTED,
        )
    with pytest.raises(ValueError, match="cannot carry geometry"):
        ExternalCellReconstruction(
            (0, 0),
            SQUARE,
            (ExternalInterfaceComponent((ExternalLinePrimitive((0, 0), (1, 1)),)),),
            "QUASI",
            "published",
            ExternalReconstructionStatus.UNRESOLVED,
        )


def test_non_json_diagnostics_are_rejected_instead_of_stringified():
    with pytest.raises(TypeError, match="non-JSON"):
        ExternalLinePrimitive((0, 0), (1, 1), {"bad": object()})


def test_duck_typed_adapters_preserve_published_kernel_parameters():
    line = adapt_linear_facet(LinearFacet([0.0, 0.2], [1.0, 0.3]))
    quadratic = adapt_quadratic_facet(
        QuadraticFacet([0.0, 0.2], [1.0, 0.3], bulge=0.04)
    )
    interface = ParabolicInterface(
        center=(0.5, 0.5),
        angle=0.3,
        curvature=0.2,
        shift=0.1,
        objective=0.0,
        optimizer_success=True,
        optimizer_message="test",
        curvature_source="exact-curvature-oracle",
        ghf_diagnostics=None,
    )
    parabola = adapt_parabolic_interval(interface, -0.4, 0.6)

    assert line.p_left == pytest.approx((0.0, 0.2))
    assert quadratic.bulge == pytest.approx(0.04)
    assert parabola.curvature == pytest.approx(0.2)
    assert parabola.s_start == pytest.approx(-0.4)


@pytest.mark.parametrize(
    "primitive",
    [
        ExternalParabolicPrimitive((0.2, -0.1), 0.7, 3.5, 0.2, -0.8, 0.9),
        ExternalQuadraticPrimitive((-0.4, 0.2), (1.1, 0.7), 0.9),
    ],
)
def test_high_order_point_distance_enumerates_global_stationary_points(primitive):
    dense_points = np.asarray(primitive.sample(20001))
    for target in ((-0.3, 0.8), (0.4, -0.7), (1.2, 1.1), (0.1, 0.2)):
        dense_distance = float(
            np.min(np.linalg.norm(dense_points - np.asarray(target), axis=1))
        )
        native_distance = primitive.distance_to_point(target)
        assert native_distance <= dense_distance + 1.0e-10
        assert dense_distance - native_distance < 1.0e-6
