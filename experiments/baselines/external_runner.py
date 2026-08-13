"""Cartesian-only runner for component-aware external baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from main.algos.baselines.external_geometry import (
    CellIndex,
    ExternalBaselineResult,
    ExternalCellReconstruction,
    ExternalReconstructionStatus,
    Point,
)


class ExternalBaselineCellError(RuntimeError):
    """Base class for explicit non-reconstructed cell outcomes."""

    status: ExternalReconstructionStatus

    def __init__(self, message: str, diagnostics: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class UnsupportedExternalCell(ExternalBaselineCellError):
    status = ExternalReconstructionStatus.UNSUPPORTED


class UnresolvedExternalCell(ExternalBaselineCellError):
    status = ExternalReconstructionStatus.UNRESOLVED


@dataclass(frozen=True)
class ExternalCellContext:
    cell_index: CellIndex
    polygon: Any
    polygon_points: Tuple[Point, ...]
    fraction: float
    target_phase_area: float
    stencil_3x3: Tuple[Tuple[Optional[Any], ...], ...]
    complete_3x3: bool


class ExternalCellMethod(Protocol):
    def __call__(self, context: ExternalCellContext) -> ExternalCellReconstruction: ...


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def _polygon_area(points: Sequence[Point]) -> float:
    return abs(
        0.5
        * sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[index][1] * points[(index + 1) % len(points)][0]
            for index in range(len(points))
        )
    )


def _is_axis_aligned_rectangle(points: Sequence[Point], tolerance: float) -> bool:
    if len(points) != 4 or _polygon_area(points) <= tolerance:
        return False
    for start, end in zip(points, points[1:] + points[:1]):
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if not ((dx <= tolerance) ^ (dy <= tolerance)):
            return False
    x_values = sorted({round(point[0] / tolerance) for point in points})
    y_values = sorted({round(point[1] / tolerance) for point in points})
    return len(x_values) == 2 and len(y_values) == 2


def validate_cartesian_mesh(mesh: Any, geometry_tolerance: float = 1.0e-10) -> None:
    if not geometry_tolerance > 0.0:
        raise ValueError("geometry_tolerance must be positive")
    if not getattr(mesh, "polys", None) or not mesh.polys[0]:
        raise ValueError("external baseline mesh has no cells")
    height = len(mesh.polys[0])
    if any(len(column) != height for column in mesh.polys):
        raise ValueError("external baseline mesh must be a rectangular cell grid")
    for x, column in enumerate(mesh.polys):
        for y, polygon in enumerate(column):
            points = tuple(_point(point) for point in polygon.points)
            if not _is_axis_aligned_rectangle(points, geometry_tolerance):
                raise ValueError(
                    "unqualified external baselines require axis-aligned Cartesian "
                    f"cells; cell {(x, y)} is unsupported"
                )


def _stencil(mesh: Any, x: int, y: int) -> Tuple[Tuple[Optional[Any], ...], ...]:
    if hasattr(mesh, "get3x3Stencil"):
        raw = mesh.get3x3Stencil(x, y)
    else:
        raw = []
        for dx in (-1, 0, 1):
            row = []
            for dy in (-1, 0, 1):
                candidate_x, candidate_y = x + dx, y + dy
                if 0 <= candidate_x < len(mesh.polys) and 0 <= candidate_y < len(
                    mesh.polys[0]
                ):
                    row.append(mesh.polys[candidate_x][candidate_y])
                else:
                    row.append(None)
            raw.append(row)
    return tuple(tuple(row) for row in raw)


def _fraction(polygon: Any) -> float:
    if hasattr(polygon, "getFraction"):
        value = polygon.getFraction()
    else:
        value = getattr(polygon, "fraction", None)
    if value is None:
        raise ValueError("every external-baseline cell must carry a volume fraction")
    return float(value)


def _target_area(polygon: Any, fraction: float, points: Sequence[Point]) -> float:
    if hasattr(polygon, "getArea"):
        return float(polygon.getArea())
    return fraction * _polygon_area(points)


def run_external_static_baseline(
    mesh: Any,
    method: ExternalCellMethod,
    *,
    source_method: str,
    source_variant: str,
    config: Optional[Mapping[str, Any]] = None,
    mixed_tolerance: float = 1.0e-10,
    geometry_tolerance: float = 1.0e-10,
) -> ExternalBaselineResult:
    """Run one source method on every original Cartesian mixed cell.

    The callback must either return a complete record or raise one of the
    explicit outcome exceptions. No project-method fallback is inserted.
    """

    validate_cartesian_mesh(mesh, geometry_tolerance)
    cells = {}
    for x, column in enumerate(mesh.polys):
        for y, polygon in enumerate(column):
            fraction = _fraction(polygon)
            if not mixed_tolerance < fraction < 1.0 - mixed_tolerance:
                continue
            points = tuple(_point(point) for point in polygon.points)
            stencil = _stencil(mesh, x, y)
            context = ExternalCellContext(
                cell_index=(x, y),
                polygon=polygon,
                polygon_points=points,
                fraction=fraction,
                target_phase_area=_target_area(polygon, fraction, points),
                stencil_3x3=stencil,
                complete_3x3=all(item is not None for row in stencil for item in row),
            )
            try:
                record = method(context)
            except ExternalBaselineCellError as error:
                diagnostics = dict(error.diagnostics)
                diagnostics.setdefault("message", str(error))
                record = ExternalCellReconstruction(
                    cell_index=context.cell_index,
                    polygon=context.polygon_points,
                    components=(),
                    source_method=source_method,
                    source_variant=source_variant,
                    status=error.status,
                    diagnostics=diagnostics,
                    target_phase_area=context.target_phase_area,
                )
            if record.cell_index != context.cell_index:
                raise ValueError(
                    "external method returned geometry for a different cell"
                )
            if record.polygon != context.polygon_points:
                raise ValueError("external method changed the owning cell polygon")
            if (
                record.source_method != source_method
                or record.source_variant != source_variant
            ):
                raise ValueError(
                    "external method returned inconsistent source provenance"
                )
            if record.target_phase_area is None:
                raise ValueError("external method must retain target_phase_area")
            cells[context.cell_index] = record

    status_counts = {status.value: 0 for status in ExternalReconstructionStatus}
    for record in cells.values():
        status_counts[record.status.value] += 1
    return ExternalBaselineResult(
        source_method=source_method,
        source_variant=source_variant,
        cells=cells,
        metadata={
            "mesh_class": "axis_aligned_cartesian",
            "mesh_shape": [len(mesh.polys), len(mesh.polys[0])],
            "mixed_tolerance": mixed_tolerance,
            "geometry_tolerance": geometry_tolerance,
            "config": dict(config or {}),
            "status_counts": status_counts,
        },
    )


__all__ = [
    "ExternalBaselineCellError",
    "ExternalCellContext",
    "ExternalCellMethod",
    "UnsupportedExternalCell",
    "UnresolvedExternalCell",
    "run_external_static_baseline",
    "validate_cartesian_mesh",
]
