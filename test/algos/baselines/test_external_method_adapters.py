import math

import pytest

from main.algos.baselines.external_method_adapters import (
    adapt_pcic_cell,
    adapt_plvira_cell,
    adapt_quasi_cell,
)
from main.algos.baselines.external_geometry import ExternalReconstructionStatus
from main.algos.baselines.pcic import PCICArcComponent, PCICCircle
from main.algos.baselines.plvira import ParabolicInterface
from main.algos.baselines.quasi import QuadraticFacet


SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_plvira_adapter_preserves_clipped_parabola_and_exact_area():
    interface = ParabolicInterface(
        center=(0.5, 0.5),
        angle=0.0,
        curvature=0.0,
        shift=0.1,
        objective=0.0,
        optimizer_success=True,
        optimizer_message="test",
        curvature_source="exact-curvature-oracle",
        ghf_diagnostics=None,
    )
    record = adapt_plvira_cell(
        interface, cell_index=(0, 0), polygon=SQUARE, target_phase_area=0.6
    )

    assert record.status is ExternalReconstructionStatus.RECONSTRUCTED
    assert record.source_variant == "PLVIRA (exact-curvature oracle)"
    assert len(record.components) == 1
    assert record.exact_phase_area() == pytest.approx(0.6)


def test_pcic_adapter_preserves_two_disconnected_components():
    components = (
        PCICArcComponent((0.5, 0.5), 0.6, (0.0, 0.5), (0.5, 1.0), math.pi, -math.pi / 2),
        PCICArcComponent((0.5, 0.5), 0.6, (1.0, 0.5), (0.5, 0.0), 0.0, -math.pi / 2),
    )
    circle = PCICCircle(
        center=(0.5, 0.5),
        radius=-0.6,
        intersections=tuple(point for component in components for point in (component.p_start, component.p_end)),
        source_center=(0.5, 0.5),
        source_radius=0.6,
        correction="adjust_radius",
        phase="complement",
        components=components,
        component_pairing_status="paired",
    )
    target = circle.fraction_in(SQUARE)
    record = adapt_pcic_cell(
        circle, cell_index=(0, 0), polygon=SQUARE, target_phase_area=target
    )

    assert record.status is ExternalReconstructionStatus.RECONSTRUCTED
    assert len(record.components) == 2
    assert record.exact_phase_area() == pytest.approx(target)


def test_quasi_adapter_retains_native_quadratic_and_area():
    facet = QuadraticFacet([0.0, 0.4], [1.0, 0.4], bulge=0.05)
    target = facet.represented_area(SQUARE)
    record = adapt_quasi_cell(
        facet, cell_index=(0, 0), polygon=SQUARE, target_phase_area=target
    )

    assert record.status is ExternalReconstructionStatus.RECONSTRUCTED
    assert record.primitives()[0].kind == "quadratic"
    assert record.exact_phase_area() == pytest.approx(target)
