"""Lossless adapters from source-method kernels to external baseline records."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from main.algos.baselines.external_geometry import (
    ExternalArcPrimitive,
    ExternalCellReconstruction,
    ExternalInterfaceComponent,
    ExternalReconstructionStatus,
    adapt_linear_facet,
    adapt_parabolic_interval,
    adapt_quadratic_facet,
)
from main.algos.baselines.pcic import PCICCircle
from main.algos.baselines.plvira import ParabolicInterface
from main.algos.baselines.quasi import QuadraticFacet
from main.structs.facets.linear_facet import LinearFacet


CellIndex = Tuple[int, int]


def _points(polygon) -> Tuple[Tuple[float, float], ...]:
    values = getattr(polygon, "points", polygon)
    return tuple((float(point[0]), float(point[1])) for point in values)


def _area(points: Sequence[Sequence[float]]) -> float:
    return abs(
        0.5
        * sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[index][1] * points[(index + 1) % len(points)][0]
            for index in range(len(points))
        )
    )


def adapt_plvira_cell(
    interface: ParabolicInterface,
    *,
    cell_index: CellIndex,
    polygon,
    target_phase_area: float,
) -> ExternalCellReconstruction:
    polygon_points = _points(polygon)
    intervals = interface.intervals_in_polygon(polygon_points)
    variant = (
        "PLVIRA"
        if interface.curvature_source == "cartesian-ghf"
        else "PLVIRA (exact-curvature oracle)"
    )
    if not intervals:
        return ExternalCellReconstruction(
            cell_index=cell_index,
            polygon=polygon_points,
            components=(),
            source_method="PLVIRA",
            source_variant=variant,
            status=ExternalReconstructionStatus.UNRESOLVED,
            diagnostics={"reason": "no clipped parabolic interval"},
            target_phase_area=target_phase_area,
        )
    ghf = interface.ghf_diagnostics
    components = tuple(
        ExternalInterfaceComponent(
            (
                adapt_parabolic_interval(
                    interface,
                    start,
                    end,
                    {"curvature_source": interface.curvature_source},
                ),
            )
        )
        for start, end in intervals
    )
    return ExternalCellReconstruction(
        cell_index=cell_index,
        polygon=polygon_points,
        components=components,
        source_method="PLVIRA",
        source_variant=variant,
        status=ExternalReconstructionStatus.RECONSTRUCTED,
        diagnostics={
            "curvature_source": interface.curvature_source,
            "curvature": interface.curvature,
            "objective": interface.objective,
            "optimizer_success": interface.optimizer_success,
            "optimizer_message": interface.optimizer_message,
            "interval_count": len(intervals),
            "ghf_method": None if ghf is None else ghf.method,
            "ghf_direction": None if ghf is None else ghf.direction,
            "ghf_height_points": None if ghf is None else ghf.height_points,
            "ghf_independent_points": (
                None if ghf is None else ghf.independent_points
            ),
            "ghf_fit_points": None if ghf is None else ghf.fit_points,
            "ghf_diagnostic": None if ghf is None else ghf.diagnostic,
        },
        target_phase_area=target_phase_area,
        exact_area_callback=interface.intersect_area,
    )


def adapt_pcic_cell(
    result,
    *,
    cell_index: CellIndex,
    polygon,
    target_phase_area: float,
    source_variant: str | None = None,
) -> ExternalCellReconstruction:
    polygon_points = _points(polygon)
    if isinstance(result, LinearFacet):
        if not source_variant:
            raise ValueError("a PCIC paper-fallback record requires its variant")
        return ExternalCellReconstruction(
            cell_index=cell_index,
            polygon=polygon_points,
            components=(ExternalInterfaceComponent((adapt_linear_facet(result),)),),
            source_method="PCIC",
            source_variant=source_variant,
            status=ExternalReconstructionStatus.PAPER_FALLBACK,
            diagnostics={"reason": "published straight-line limit"},
            target_phase_area=target_phase_area,
            stored_exact_phase_area=target_phase_area,
        )
    if not isinstance(result, PCICCircle):
        raise TypeError("PCIC adapter requires PCICCircle or LinearFacet")
    variant = result.source_variant
    if result.component_pairing_status != "paired" or not result.components:
        return ExternalCellReconstruction(
            cell_index=cell_index,
            polygon=polygon_points,
            components=(),
            source_method="PCIC",
            source_variant=variant,
            status=ExternalReconstructionStatus.UNRESOLVED,
            diagnostics={
                "reason": "unpaired circle/cell components",
                "boundary_crossings": len(result.intersections),
            },
            target_phase_area=target_phase_area,
        )
    components = tuple(
        ExternalInterfaceComponent(
            (
                ExternalArcPrimitive(
                    component.center,
                    abs(component.radius),
                    component.start_angle,
                    component.sweep_angle,
                    {"phase": result.phase},
                ),
            ),
            closed=component.closed,
        )
        for component in result.components
    )
    return ExternalCellReconstruction(
        cell_index=cell_index,
        polygon=polygon_points,
        components=components,
        source_method="PCIC",
        source_variant=variant,
        status=ExternalReconstructionStatus.RECONSTRUCTED,
        diagnostics={
            "phase": result.phase,
            "boundary_crossings": len(result.intersections),
            "component_count": len(result.components),
            "correction": result.correction,
        },
        target_phase_area=target_phase_area,
        exact_area_callback=lambda points: result.fraction_in(points) * _area(points),
    )


def adapt_quasi_cell(
    facet: QuadraticFacet | None,
    *,
    cell_index: CellIndex,
    polygon,
    target_phase_area: float,
    unresolved_reason: str | None = None,
) -> ExternalCellReconstruction:
    polygon_points = _points(polygon)
    variant = "QUASI (frozen Cartesian port)"
    if facet is None:
        return ExternalCellReconstruction(
            cell_index=cell_index,
            polygon=polygon_points,
            components=(),
            source_method="QUASI",
            source_variant=variant,
            status=ExternalReconstructionStatus.UNRESOLVED,
            diagnostics={"reason": unresolved_reason or "missing quadratic facet"},
            target_phase_area=target_phase_area,
        )
    return ExternalCellReconstruction(
        cell_index=cell_index,
        polygon=polygon_points,
        components=(ExternalInterfaceComponent((adapt_quadratic_facet(facet),)),),
        source_method="QUASI",
        source_variant=variant,
        status=ExternalReconstructionStatus.RECONSTRUCTED,
        diagnostics={},
        target_phase_area=target_phase_area,
        exact_area_callback=facet.represented_area,
    )


__all__ = ["adapt_pcic_cell", "adapt_plvira_cell", "adapt_quasi_cell"]
