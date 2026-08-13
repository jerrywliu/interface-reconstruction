"""Static reconstruction kernel for the published PCIC method.

This module deliberately does not call this repository's circular fitter.  It
implements the point selection, Riemann-sphere fit, and the two conservative
corrections described by Maity, Sundararajan, and Velusamy (2021).  The caller
must supply the paper's LLS/Parker--Young PLIC prediction; see the accompanying
implementation report for the boundary of this prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np

from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, getDistance
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


VolumeCorrection = Literal["translate_center", "adjust_radius"]


class PCICError(RuntimeError):
    """Base class for a PCIC reconstruction failure."""


class PCICDegenerateFit(PCICError):
    """Raised when the Riemann fit represents a line rather than a circle."""


class PCICConvergenceError(PCICError):
    """Raised when the published bisection correction cannot be bracketed."""


@dataclass(frozen=True)
class PCICConfig:
    """Numerical choices stated in Sections 2.2.1--2.2.3 of the paper."""

    sample_alphas: tuple[float, ...] = (0.2, 0.25, 0.3, 0.7, 0.75, 0.8)
    fit_fraction_min: float = 0.01
    fit_fraction_max: float = 0.99
    reconstruction_fraction_min: float = 1.0e-6
    volume_fraction_tolerance: float = 1.0e-6
    normalized_bisection_tolerance: float = 1.0e-10
    straight_radius_scale: float = 1.0e6
    max_bisection_iterations: int = 200
    max_bracket_expansions: int = 60


@dataclass(frozen=True)
class PCICCircle:
    """A conservative PCIC circle and all of its target-cell intersections.

    A negative radius denotes the complement of the disk, matching the signed
    radius convention used by ``getCircleIntersectArea`` in this repository.
    ``intersections`` is intentionally not truncated: the PCIC paper includes
    cells with four crossings, while ``ArcFacet`` can encode only one arc.
    """

    center: tuple[float, float]
    radius: float
    intersections: tuple[tuple[float, float], ...]
    source_center: tuple[float, float]
    source_radius: float
    correction: VolumeCorrection

    def fraction_in(self, polygon: object) -> float:
        points = _polygon_points(polygon)
        area, _ = getCircleIntersectArea(list(self.center), self.radius, points)
        return area / abs(getArea(points))

    def to_arc_facet(self) -> ArcFacet:
        """Convert a two-crossing result to the repository's facet type."""

        if len(self.intersections) != 2:
            raise PCICError(
                "ArcFacet cannot represent a PCIC cell with "
                f"{len(self.intersections)} circle-boundary intersections"
            )
        return ArcFacet(
            list(self.center),
            self.radius,
            list(self.intersections[0]),
            list(self.intersections[1]),
        )


PCICCellFacet = Union[LinearFacet, PCICCircle]


def sample_plic_segment(
    facet: LinearFacet, alphas: Sequence[float] = PCICConfig().sample_alphas
) -> list[list[float]]:
    """Return the six PLIC-derived samples from paper Equations (8)--(10)."""

    p = facet.pLeft
    q = facet.pRight
    return [
        [alpha * p[0] + (1.0 - alpha) * q[0], alpha * p[1] + (1.0 - alpha) * q[1]]
        for alpha in alphas
    ]


def collect_stencil_samples(
    polygon_stencil: Sequence[Sequence[Optional[object]]],
    plic_stencil: Sequence[Sequence[Optional[LinearFacet]]],
    config: PCICConfig = PCICConfig(),
) -> list[list[float]]:
    """Collect PCIC samples from eligible mixed cells in a 3-by-3 stencil."""

    if len(polygon_stencil) != 3 or any(len(row) != 3 for row in polygon_stencil):
        raise ValueError("PCIC requires a 3-by-3 polygon stencil")
    if len(plic_stencil) != 3 or any(len(row) != 3 for row in plic_stencil):
        raise ValueError("PCIC requires a 3-by-3 PLIC stencil")

    samples: list[list[float]] = []
    for i in range(3):
        for j in range(3):
            polygon = polygon_stencil[i][j]
            facet = plic_stencil[i][j]
            if polygon is None or facet is None:
                continue
            fraction = _polygon_fraction(polygon)
            if config.fit_fraction_min <= fraction <= config.fit_fraction_max:
                samples.extend(sample_plic_segment(facet, config.sample_alphas))
    return samples


def fit_riemann_sphere(points: Iterable[Sequence[float]]) -> tuple[list[float], float]:
    """Fit a circle with the paper's weighted Riemann-sphere method.

    This is Equations (12)--(20): stereographic projection, the weighted
    scatter matrix, its smallest-eigenvalue eigenvector, and inverse mapping.
    """

    xy = np.asarray(list(points), dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 3:
        raise ValueError("At least three two-dimensional points are required")
    if not np.all(np.isfinite(xy)):
        raise ValueError("PCIC fit points must be finite")

    squared_radius = np.einsum("ij,ij->i", xy, xy)
    denominator = 1.0 + squared_radius
    mapped = np.column_stack(
        (xy[:, 0] / denominator, xy[:, 1] / denominator, squared_radius / denominator)
    )
    weights = denominator**2
    weight_sum = float(np.sum(weights))
    weighted_mean = np.sum(weights[:, None] * mapped, axis=0) / weight_sum
    centered = mapped - weighted_mean
    scatter = (centered * weights[:, None]).T @ centered / weight_sum

    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    alpha, beta, gamma = (float(value) for value in normal)
    c = -float(np.dot(normal, weighted_mean))
    inverse_denominator = c + gamma
    scale = max(1.0, abs(c), abs(gamma))
    if abs(inverse_denominator) <= 1.0e-13 * scale:
        raise PCICDegenerateFit("The Riemann plane maps to a straight line")

    center = [
        -alpha / (2.0 * inverse_denominator),
        -beta / (2.0 * inverse_denominator),
    ]
    radius_squared = (alpha * alpha + beta * beta - 4.0 * c * inverse_denominator) / (
        4.0 * inverse_denominator * inverse_denominator
    )
    if radius_squared <= 0.0 or not math.isfinite(radius_squared):
        raise PCICDegenerateFit("The inverse Riemann fit has no finite circle")
    return center, math.sqrt(radius_squared)


def reconstruct_bare_pcic_cell(
    target_polygon: object,
    polygon_stencil: Sequence[Sequence[Optional[object]]],
    plic_stencil: Sequence[Sequence[Optional[LinearFacet]]],
    *,
    correction: VolumeCorrection,
    config: PCICConfig = PCICConfig(),
) -> PCICCellFacet:
    """Reconstruct one static cell through the published bare-PCIC sequence.

    ``correction`` is mandatory because the paper documents two alternatives
    but does not identify which one produced every reported bare-PCIC result.
    """

    if correction not in ("translate_center", "adjust_radius"):
        raise ValueError(f"Unknown PCIC volume correction: {correction!r}")

    target_fraction = _polygon_fraction(target_polygon)
    if not (
        config.reconstruction_fraction_min
        < target_fraction
        < 1.0 - config.reconstruction_fraction_min
    ):
        raise ValueError("PCIC reconstructs only mixed cells inside its 1e-6 cutoff")

    central_plic = plic_stencil[1][1]
    if central_plic is None:
        raise ValueError("The target cell requires its LLS/Parker--Young PLIC facet")

    samples = collect_stencil_samples(polygon_stencil, plic_stencil, config)
    try:
        fit_center, fit_radius = fit_riemann_sphere(samples)
    except PCICDegenerateFit:
        return central_plic

    points = _polygon_points(target_polygon)
    cell_width = _cell_width(points)
    cell_diagonal = _cell_diagonal(points)
    if fit_radius >= config.straight_radius_scale * cell_width:
        return central_plic

    sign = _select_phase_sign(points, target_fraction, fit_center, fit_radius)
    if fit_radius < cell_diagonal:
        fit_radius = cell_diagonal
        fit_center = _center_from_plic_chord(
            central_plic, fit_center, fit_radius, sign, target_polygon
        )

    if correction == "adjust_radius":
        corrected_center = fit_center
        corrected_radius = _correct_radius(
            target_polygon, fit_center, sign, fit_radius, cell_diagonal, config
        )
    else:
        direction = _plic_normal(central_plic)
        corrected_center = _correct_center(
            target_polygon, fit_center, sign * fit_radius, direction, cell_width, config
        )
        corrected_radius = sign * fit_radius

    _, intersections = getCircleIntersectArea(
        corrected_center, corrected_radius, points
    )
    return PCICCircle(
        center=(float(corrected_center[0]), float(corrected_center[1])),
        radius=float(corrected_radius),
        intersections=tuple((float(p[0]), float(p[1])) for p in intersections),
        source_center=(float(fit_center[0]), float(fit_center[1])),
        source_radius=float(sign * fit_radius),
        correction=correction,
    )


def _correct_radius(
    polygon: object,
    center: Sequence[float],
    sign: float,
    initial_radius: float,
    minimum_radius: float,
    config: PCICConfig,
) -> float:
    target = _polygon_fraction(polygon)
    points = _polygon_points(polygon)

    def residual(radius_magnitude: float) -> float:
        area, _ = getCircleIntersectArea(list(center), sign * radius_magnitude, points)
        return area / abs(getArea(points)) - target

    low = max(minimum_radius, np.finfo(float).eps)
    high = max(initial_radius, low) * 2.0
    low_value = residual(low)
    high_value = residual(high)
    expansions = 0
    while low_value * high_value > 0.0 and expansions < config.max_bracket_expansions:
        high *= 2.0
        high_value = residual(high)
        expansions += 1
    if low_value * high_value > 0.0:
        raise PCICConvergenceError(
            "Could not bracket the fixed-center radius correction"
        )
    return sign * _bisect(residual, low, high, target, config)


def _correct_center(
    polygon: object,
    center: Sequence[float],
    signed_radius: float,
    direction: Sequence[float],
    cell_width: float,
    config: PCICConfig,
) -> list[float]:
    target = _polygon_fraction(polygon)
    points = _polygon_points(polygon)

    def candidate(offset: float) -> list[float]:
        return [center[0] + offset * direction[0], center[1] + offset * direction[1]]

    def residual(offset: float) -> float:
        area, _ = getCircleIntersectArea(candidate(offset), signed_radius, points)
        return area / abs(getArea(points)) - target

    samples = [(0.0, residual(0.0))]
    step = cell_width
    for _ in range(config.max_bracket_expansions):
        samples.extend([(-step, residual(-step)), (step, residual(step))])
        samples.sort(key=lambda pair: pair[0])
        bracket = _closest_bracket(samples)
        if bracket is not None:
            offset = _bisect(residual, bracket[0], bracket[1], target, config)
            return candidate(offset)
        step *= 2.0
    raise PCICConvergenceError("Could not bracket the fixed-radius center correction")


def _bisect(
    function, low: float, high: float, target: float, config: PCICConfig
) -> float:
    low_value = function(low)
    high_value = function(high)
    if low_value == 0.0:
        return low
    if high_value == 0.0:
        return high
    if low_value * high_value > 0.0:
        raise PCICConvergenceError("Bisection endpoints do not bracket the target")

    for _ in range(config.max_bisection_iterations):
        middle = 0.5 * (low + high)
        middle_value = function(middle)
        if (
            abs(middle_value) <= config.volume_fraction_tolerance
            and abs(middle_value) / target <= config.normalized_bisection_tolerance
        ):
            return middle
        if low_value * middle_value <= 0.0:
            high = middle
            high_value = middle_value
        else:
            low = middle
            low_value = middle_value
    raise PCICConvergenceError("PCIC volume correction exceeded its iteration limit")


def _closest_bracket(
    samples: Sequence[tuple[float, float]]
) -> Optional[tuple[float, float]]:
    brackets = []
    for left, right in zip(samples, samples[1:]):
        if left[1] == 0.0:
            return (left[0], left[0])
        if left[1] * right[1] <= 0.0:
            brackets.append((left[0], right[0]))
    if not brackets:
        return None
    return min(brackets, key=lambda pair: abs(pair[0]) + abs(pair[1]))


def _center_from_plic_chord(
    facet: LinearFacet,
    fitted_center: Sequence[float],
    radius: float,
    sign: float,
    polygon: object,
) -> list[float]:
    """Apply the paper's minimum-radius reset using the PLIC chord."""

    chord = getDistance(facet.pLeft, facet.pRight)
    if chord > 2.0 * radius:
        return list(fitted_center)
    midpoint = [
        0.5 * (facet.pLeft[0] + facet.pRight[0]),
        0.5 * (facet.pLeft[1] + facet.pRight[1]),
    ]
    normal = _plic_normal(facet)
    distance = math.sqrt(max(0.0, radius * radius - 0.25 * chord * chord))
    candidates = [
        [midpoint[0] + distance * normal[0], midpoint[1] + distance * normal[1]],
        [midpoint[0] - distance * normal[0], midpoint[1] - distance * normal[1]],
    ]
    target = _polygon_fraction(polygon)
    points = _polygon_points(polygon)
    return min(
        candidates,
        key=lambda candidate: abs(
            getCircleIntersectArea(candidate, sign * radius, points)[0]
            / abs(getArea(points))
            - target
        ),
    )


def _select_phase_sign(
    points: Sequence[Sequence[float]],
    target_fraction: float,
    center: Sequence[float],
    radius: float,
) -> float:
    polygon_area = abs(getArea(points))
    inside = (
        getCircleIntersectArea(list(center), radius, list(points))[0] / polygon_area
    )
    outside = (
        getCircleIntersectArea(list(center), -radius, list(points))[0] / polygon_area
    )
    return (
        1.0 if abs(inside - target_fraction) <= abs(outside - target_fraction) else -1.0
    )


def _plic_normal(facet: LinearFacet) -> list[float]:
    dx = facet.pRight[0] - facet.pLeft[0]
    dy = facet.pRight[1] - facet.pLeft[1]
    magnitude = math.hypot(dx, dy)
    if magnitude == 0.0:
        raise PCICError("Degenerate PLIC facet")
    return [dy / magnitude, -dx / magnitude]


def _polygon_points(polygon: object) -> list[list[float]]:
    points = getattr(polygon, "points", polygon)
    return [[float(point[0]), float(point[1])] for point in points]


def _polygon_fraction(polygon: object) -> float:
    if hasattr(polygon, "getFraction"):
        fraction = polygon.getFraction()
    else:
        fraction = getattr(polygon, "fraction", None)
    if fraction is None:
        raise ValueError("Every PCIC polygon must carry a volume fraction")
    return float(fraction)


def _cell_diagonal(points: Sequence[Sequence[float]]) -> float:
    return max(getDistance(a, b) for a in points for b in points)


def _cell_width(points: Sequence[Sequence[float]]) -> float:
    return max(
        getDistance(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


__all__ = [
    "PCICCellFacet",
    "PCICCircle",
    "PCICConfig",
    "PCICConvergenceError",
    "PCICDegenerateFit",
    "PCICError",
    "collect_stencil_samples",
    "fit_riemann_sphere",
    "reconstruct_bare_pcic_cell",
    "sample_plic_segment",
]
