"""Static reconstruction kernel for the published PCIC method.

This module deliberately does not call this repository's circular fitter.  It
implements the cited linear least-squares (LLS)/Parker--Young predictor, the
point selection, Riemann-sphere fit, and the two conservative corrections
described by Maity, Sundararajan, and Velusamy (2021).  Only uniform Cartesian
blocks are accepted by the end-to-end predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np

from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, getDistance, pointInPoly
from main.geoms.linear_facet import getLinearFacetFromNormal
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


VolumeCorrection = Literal["translate_center", "adjust_radius"]
PCICPhase = Literal["disk", "complement"]
PCICPhasePolicy = Literal["infer_from_plic"]
CenterTranslationRootPolicy = Literal["nearest_bracket"]

CENTER_TRANSLATION_VARIANT = "bare PCIC (center translation)"
RADIUS_ADJUSTMENT_VARIANT = "bare PCIC (radius adjustment)"


class PCICError(RuntimeError):
    """Base class for a PCIC reconstruction failure."""


class PCICDegenerateFit(PCICError):
    """Raised when the Riemann fit represents a line rather than a circle."""


class PCICConvergenceError(PCICError):
    """Raised when the published bisection correction cannot be bracketed."""


class PCICAmbiguousSourceChoice(PCICError):
    """Raised when the cited source requires an unspecified material choice."""


class PCICUnsupportedGeometry(PCICError):
    """Raised for geometry outside the published Cartesian construction."""


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
    lls_max_cut_cells: int = 5
    lls_overcrowded_radius_scale: float = 0.5
    center_translation_samples_per_cell: int = 32
    cartesian_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class PCICArcComponent:
    """One phase-oriented connected circular arc inside a target cell."""

    center: tuple[float, float]
    radius: float
    p_start: tuple[float, float]
    p_end: tuple[float, float]
    start_angle: float
    sweep_angle: float
    closed: bool = False

    def sample(self, count: int) -> tuple[tuple[float, float], ...]:
        if count < 2:
            raise ValueError("A PCIC arc sample requires at least two points")
        magnitude = abs(self.radius)
        return tuple(
            (
                self.center[0]
                + magnitude
                * math.cos(self.start_angle + self.sweep_angle * index / (count - 1)),
                self.center[1]
                + magnitude
                * math.sin(self.start_angle + self.sweep_angle * index / (count - 1)),
            )
            for index in range(count)
        )


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
    phase: PCICPhase
    components: tuple[PCICArcComponent, ...] = ()
    component_pairing_status: Literal["paired", "unresolved"] = "unresolved"

    @property
    def source_variant(self) -> str:
        return source_variant_for_correction(self.correction)

    def fraction_in(self, polygon: object) -> float:
        points = _polygon_points(polygon)
        area, _ = getCircleIntersectArea(list(self.center), self.radius, points)
        return area / abs(getArea(points))

    def to_arc_facet(self) -> ArcFacet:
        """Convert a two-crossing result to the repository's facet type."""

        if (
            self.component_pairing_status != "paired"
            or len(self.components) != 1
            or self.components[0].closed
        ):
            raise PCICError(
                "ArcFacet cannot represent a PCIC cell with "
                f"{len(self.components)} connected component(s) and "
                f"{len(self.intersections)} circle-boundary intersections"
            )
        component = self.components[0]
        return ArcFacet(
            list(self.center),
            self.radius,
            list(component.p_start),
            list(component.p_end),
        )

    def to_arc_facets(self) -> tuple[ArcFacet, ...]:
        """Convert every paired open component without dropping extra arcs."""

        if self.component_pairing_status != "paired" or any(
            component.closed for component in self.components
        ):
            raise PCICError("PCIC components are not representable as open ArcFacets")
        return tuple(
            ArcFacet(
                list(self.center),
                self.radius,
                list(component.p_start),
                list(component.p_end),
            )
            for component in self.components
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


def source_variant_for_correction(correction: VolumeCorrection) -> str:
    """Return the paper-facing name of one published bare-PCIC variant."""

    if correction == "translate_center":
        return CENTER_TRANSLATION_VARIANT
    if correction == "adjust_radius":
        return RADIUS_ADJUSTMENT_VARIANT
    raise ValueError(f"Unknown PCIC volume correction: {correction!r}")


def parker_young_normal(
    polygon_stencil: Sequence[Sequence[object]],
    config: PCICConfig = PCICConfig(),
) -> list[float]:
    """Return the Parker--Young normal for a Cartesian 3-by-3 block.

    This is Scardovelli and Zaleski (2003), Section 2.1: compute the
    volume-fraction gradient at each corner of the central cell and average
    the four corner values.  Algebraically it is the Parker--Young ``a=2``
    stencil reported by Pilliod and Puckett (2004), Section 2.4.
    """

    _validate_cartesian_block(polygon_stencil, 3, config)
    fractions = [
        [_polygon_fraction(polygon_stencil[i][j]) for j in range(3)] for i in range(3)
    ]
    x_gradient = (
        fractions[2][0]
        + 2.0 * fractions[2][1]
        + fractions[2][2]
        - fractions[0][0]
        - 2.0 * fractions[0][1]
        - fractions[0][2]
    ) / 8.0
    y_gradient = (
        fractions[0][2]
        + 2.0 * fractions[1][2]
        + fractions[2][2]
        - fractions[0][0]
        - 2.0 * fractions[1][0]
        - fractions[2][0]
    ) / 8.0
    magnitude = math.hypot(x_gradient, y_gradient)
    if magnitude <= np.finfo(float).eps:
        raise PCICDegenerateFit("The Parker--Young volume-fraction gradient is zero")
    return [x_gradient / magnitude, y_gradient / magnitude]


def reconstruct_parker_young_plic(
    polygon_stencil: Sequence[Sequence[object]],
    config: PCICConfig = PCICConfig(),
) -> LinearFacet:
    """Place the conservative Parker--Young PLIC in the central cell."""

    normal = parker_young_normal(polygon_stencil, config)
    target = polygon_stencil[1][1]
    fraction = _polygon_fraction(target)
    if not _is_reconstructed_fraction(fraction, config):
        raise ValueError("Parker--Young PLIC requires a PCIC mixed target cell")
    points = _polygon_points(target)
    area_tolerance = config.normalized_bisection_tolerance * abs(getArea(points))
    p_left, p_right = getLinearFacetFromNormal(points, fraction, normal, area_tolerance)
    return LinearFacet(p_left, p_right, name="PCIC Parker-Young")


def reconstruct_lls_plic(
    polygon_stencil: Sequence[Sequence[object]],
    parker_young_stencil: Sequence[Sequence[Optional[LinearFacet]]],
    config: PCICConfig = PCICConfig(),
) -> LinearFacet:
    """Apply the cited one-pass linear least-squares fit to one mixed cell.

    The fitted points are the two endpoints and midpoint of each preliminary
    Parker--Young segment inside the source's radius of influence.  The fit is
    ordinary least squares in ``y(x)`` or ``x(y)``, selected from the central
    preliminary segment to avoid the vertical-line singularity.  Only the
    fitted direction is retained; the final line is repositioned to conserve
    the central-cell volume fraction.
    """

    bounds = _validate_cartesian_block(polygon_stencil, 3, config)
    _validate_square_stencil(parker_young_stencil, 3, "Parker--Young")
    central_facet = parker_young_stencil[1][1]
    if central_facet is None:
        raise ValueError("LLS requires a central Parker--Young PLIC facet")

    cut_facets = [
        facet for row in parker_young_stencil for facet in row if facet is not None
    ]
    radius_scale = 1.0
    if len(cut_facets) > config.lls_max_cut_cells:
        radius_scale = config.lls_overcrowded_radius_scale
        if not 0.0 < radius_scale < 1.0:
            raise ValueError(
                "The explicit overcrowded LLS radius scale must be in (0, 1)"
            )

    midpoint = _facet_midpoint(central_facet)
    influence_radius = radius_scale * min(
        midpoint[0] - bounds[0],
        bounds[1] - midpoint[0],
        midpoint[1] - bounds[2],
        bounds[3] - midpoint[1],
    )
    if influence_radius <= 0.0:
        raise PCICUnsupportedGeometry("The LLS radius of influence is not positive")

    fit_points: list[list[float]] = []
    inclusion_tolerance = config.cartesian_tolerance * max(1.0, influence_radius)
    for facet in cut_facets:
        for point in (facet.pLeft, _facet_midpoint(facet), facet.pRight):
            if getDistance(point, midpoint) <= influence_radius + inclusion_tolerance:
                fit_points.append([float(point[0]), float(point[1])])
    if len(fit_points) < 3:
        raise PCICDegenerateFit("Fewer than three PLIC points enter the LLS fit")

    dx = central_facet.pRight[0] - central_facet.pLeft[0]
    dy = central_facet.pRight[1] - central_facet.pLeft[1]
    if abs(dx) >= abs(dy):
        independent = np.asarray([point[0] for point in fit_points], dtype=float)
        dependent = np.asarray([point[1] for point in fit_points], dtype=float)
        slope, _ = _ordinary_line_fit(independent, dependent)
        normal = [slope, 1.0]
    else:
        independent = np.asarray([point[1] for point in fit_points], dtype=float)
        dependent = np.asarray([point[0] for point in fit_points], dtype=float)
        slope, _ = _ordinary_line_fit(independent, dependent)
        normal = [1.0, slope]

    initial_normal = _plic_normal(central_facet)
    if np.dot(normal, initial_normal) < 0.0:
        normal = [-normal[0], -normal[1]]
    magnitude = math.hypot(normal[0], normal[1])
    normal = [normal[0] / magnitude, normal[1] / magnitude]

    target = polygon_stencil[1][1]
    points = _polygon_points(target)
    area_tolerance = config.normalized_bisection_tolerance * abs(getArea(points))
    p_left, p_right = getLinearFacetFromNormal(
        points, _polygon_fraction(target), normal, area_tolerance
    )
    return LinearFacet(p_left, p_right, name="PCIC LLS/Parker-Young")


def build_lls_parker_young_plic_stencil(
    polygon_block: Sequence[Sequence[object]],
    config: PCICConfig = PCICConfig(),
) -> list[list[Optional[LinearFacet]]]:
    """Build the target 3-by-3 LLS PLIC stencil from a Cartesian 7-by-7 halo.

    Each of the nine final LLS lines needs preliminary Parker--Young lines in
    its own 3-by-3 block, and each of those preliminary lines needs a 3-by-3
    volume-fraction stencil.  A complete 7-by-7 volume-fraction block is thus
    the smallest boundary-free input for all nine final lines.
    """

    _validate_cartesian_block(polygon_block, 7, config)
    parker_young: list[list[Optional[LinearFacet]]] = [
        [None for _ in range(7)] for _ in range(7)
    ]
    for i in range(1, 6):
        for j in range(1, 6):
            fraction = _polygon_fraction(polygon_block[i][j])
            if not _is_reconstructed_fraction(fraction, config):
                continue
            parker_young[i][j] = reconstruct_parker_young_plic(
                _subblock(polygon_block, i, j, 1), config
            )

    lls_stencil: list[list[Optional[LinearFacet]]] = [
        [None for _ in range(3)] for _ in range(3)
    ]
    for output_i, block_i in enumerate(range(2, 5)):
        for output_j, block_j in enumerate(range(2, 5)):
            fraction = _polygon_fraction(polygon_block[block_i][block_j])
            is_target = block_i == 3 and block_j == 3
            is_fit_cell = config.fit_fraction_min <= fraction <= config.fit_fraction_max
            if not (is_target or is_fit_cell):
                continue
            if not _is_reconstructed_fraction(fraction, config):
                continue
            lls_stencil[output_i][output_j] = reconstruct_lls_plic(
                _subblock(polygon_block, block_i, block_j, 1),
                _subblock(parker_young, block_i, block_j, 1),
                config,
            )
    return lls_stencil


def reconstruct_bare_pcic_cartesian_cell(
    polygon_block: Sequence[Sequence[object]],
    *,
    correction: VolumeCorrection,
    phase: Union[PCICPhase, PCICPhasePolicy] = "infer_from_plic",
    center_translation_root_policy: CenterTranslationRootPolicy = "nearest_bracket",
    config: PCICConfig = PCICConfig(),
) -> PCICCellFacet:
    """Run bare PCIC from volume fractions on a complete Cartesian halo."""

    plic_stencil = build_lls_parker_young_plic_stencil(polygon_block, config)
    polygon_stencil = _subblock(polygon_block, 3, 3, 1)
    return reconstruct_bare_pcic_cell(
        polygon_block[3][3],
        polygon_stencil,
        plic_stencil,
        correction=correction,
        phase=phase,
        center_translation_root_policy=center_translation_root_policy,
        config=config,
    )


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
    phase: Union[PCICPhase, PCICPhasePolicy] = "infer_from_plic",
    center_translation_root_policy: CenterTranslationRootPolicy = "nearest_bracket",
    config: PCICConfig = PCICConfig(),
) -> PCICCellFacet:
    """Reconstruct one static cell through the published bare-PCIC sequence.

    ``correction`` is mandatory because the paper documents two alternatives
    but does not identify which one produced every reported bare-PCIC result.
    """

    if correction not in ("translate_center", "adjust_radius"):
        raise ValueError(f"Unknown PCIC volume correction: {correction!r}")
    if phase not in ("disk", "complement", "infer_from_plic"):
        raise ValueError(f"Unknown PCIC phase convention: {phase!r}")

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

    resolved_phase = (
        infer_phase_from_plic(fit_center, central_plic)
        if phase == "infer_from_plic"
        else phase
    )
    sign = 1.0 if resolved_phase == "disk" else -1.0
    if fit_radius < cell_diagonal:
        fit_radius = cell_diagonal
        fit_center = _center_from_plic_chord(central_plic, fit_radius, sign)

    if correction == "adjust_radius":
        corrected_center = fit_center
        corrected_radius = _correct_radius(
            target_polygon, fit_center, sign, fit_radius, cell_diagonal, config
        )
    else:
        _, fitted_intersections = getCircleIntersectArea(
            fit_center, sign * fit_radius, points
        )
        direction = _fitted_chord_perpendicular(
            fit_center,
            fitted_intersections,
            config,
            central_plic=central_plic,
            polygon_points=points,
        )
        corrected_center = _correct_center(
            target_polygon,
            fit_center,
            sign * fit_radius,
            direction,
            cell_width,
            center_translation_root_policy,
            config,
        )
        corrected_radius = sign * fit_radius

    _, intersections = getCircleIntersectArea(
        corrected_center, corrected_radius, points
    )
    unique_intersections = _deduplicate_points(
        intersections, config.cartesian_tolerance * max(1.0, cell_width)
    )
    components, pairing_status = _pair_circle_components(
        corrected_center,
        corrected_radius,
        unique_intersections,
        points,
        config,
    )
    return PCICCircle(
        center=(float(corrected_center[0]), float(corrected_center[1])),
        radius=float(corrected_radius),
        intersections=tuple(
            (float(point[0]), float(point[1])) for point in unique_intersections
        ),
        source_center=(float(fit_center[0]), float(fit_center[1])),
        source_radius=float(sign * fit_radius),
        correction=correction,
        phase=resolved_phase,
        components=components,
        component_pairing_status=pairing_status,
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
    root_policy: CenterTranslationRootPolicy,
    config: PCICConfig,
) -> list[float]:
    if root_policy != "nearest_bracket":
        raise ValueError(f"Unknown center-translation root policy: {root_policy!r}")
    target = _polygon_fraction(polygon)
    points = _polygon_points(polygon)

    def candidate(offset: float) -> list[float]:
        return [center[0] + offset * direction[0], center[1] + offset * direction[1]]

    def residual(offset: float) -> float:
        area, _ = getCircleIntersectArea(candidate(offset), signed_radius, points)
        return area / abs(getArea(points)) - target

    if config.center_translation_samples_per_cell < 2:
        raise ValueError(
            "Center-translation root search needs at least two samples per cell"
        )
    samples = [(0.0, residual(0.0))]
    previous_extent = 0.0
    for expansion in range(config.max_bracket_expansions):
        extent = cell_width * (2.0**expansion)
        shell_width = extent - previous_extent
        shell_step = shell_width / config.center_translation_samples_per_cell
        positive = [
            previous_extent + index * shell_step
            for index in range(1, config.center_translation_samples_per_cell + 1)
        ]
        offsets = [-offset for offset in reversed(positive)] + positive
        samples.extend((offset, residual(offset)) for offset in offsets)
        samples.sort(key=lambda pair: pair[0])
        brackets = _all_brackets(samples)
        if brackets:
            roots = [
                (
                    bracket[0]
                    if bracket[0] == bracket[1]
                    else _bisect(residual, bracket[0], bracket[1], target, config)
                )
                for bracket in brackets
            ]
            return candidate(min(roots, key=lambda root: (abs(root), root)))
        previous_extent = extent
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


def _all_brackets(samples: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    brackets: list[tuple[float, float]] = []
    for left, right in zip(samples, samples[1:]):
        if left[1] == 0.0:
            brackets.append((left[0], left[0]))
        if left[1] * right[1] <= 0.0:
            brackets.append((left[0], right[0]))
    if samples and samples[-1][1] == 0.0:
        brackets.append((samples[-1][0], samples[-1][0]))
    unique: list[tuple[float, float]] = []
    for bracket in brackets:
        if bracket not in unique:
            unique.append(bracket)
    return unique


def _fitted_chord_perpendicular(
    center: Sequence[float],
    intersections: Sequence[Sequence[float]],
    config: PCICConfig,
    *,
    central_plic: Optional[LinearFacet] = None,
    polygon_points: Optional[Sequence[Sequence[float]]] = None,
) -> list[float]:
    """Return the fitted chord's perpendicular-bisector direction."""

    unique = _deduplicate_points(intersections, config.cartesian_tolerance)
    if len(unique) == 2:
        chord_points = unique
    elif (
        len(unique) > 2
        and len(unique) % 2 == 0
        and central_plic is not None
        and polygon_points is not None
    ):
        chord_points = _select_principal_chord(
            center, unique, polygon_points, central_plic, config
        )
    else:
        raise PCICAmbiguousSourceChoice(
            "The published center-translation correction defines one fitted "
            "arc chord, but the fitted target-cell circle has "
            f"{len(unique)} boundary crossings"
        )
    chord = [
        chord_points[1][0] - chord_points[0][0],
        chord_points[1][1] - chord_points[0][1],
    ]
    magnitude = math.hypot(chord[0], chord[1])
    if magnitude <= np.finfo(float).eps:
        raise PCICUnsupportedGeometry("The fitted target-cell chord is degenerate")
    direction = [-chord[1] / magnitude, chord[0] / magnitude]
    midpoint = [
        0.5 * (chord_points[0][0] + chord_points[1][0]),
        0.5 * (chord_points[0][1] + chord_points[1][1]),
    ]
    center_side = [center[0] - midpoint[0], center[1] - midpoint[1]]
    if np.dot(direction, center_side) < 0.0:
        direction = [-direction[0], -direction[1]]
    return direction


def _select_principal_chord(
    center: Sequence[float],
    intersections: Sequence[Sequence[float]],
    polygon_points: Sequence[Sequence[float]],
    central_plic: LinearFacet,
    config: PCICConfig,
) -> list[list[float]]:
    """Select the in-cell arc chord most consistent with the central PLIC."""

    components, status = _pair_circle_components(
        center,
        getDistance(center, intersections[0]),
        intersections,
        polygon_points,
        config,
    )
    if status != "paired" or not components:
        raise PCICAmbiguousSourceChoice("Could not pair the fitted target-cell chords")
    plic_midpoint = _facet_midpoint(central_plic)
    plic_direction = [
        central_plic.pRight[0] - central_plic.pLeft[0],
        central_plic.pRight[1] - central_plic.pLeft[1],
    ]
    plic_magnitude = math.hypot(*plic_direction)
    if plic_magnitude <= np.finfo(float).eps:
        raise PCICError("Degenerate central PLIC facet")
    plic_direction = [value / plic_magnitude for value in plic_direction]

    def score(component: PCICArcComponent) -> tuple[float, float, float, float]:
        midpoint = [
            0.5 * (component.p_start[0] + component.p_end[0]),
            0.5 * (component.p_start[1] + component.p_end[1]),
        ]
        chord = [
            component.p_end[0] - component.p_start[0],
            component.p_end[1] - component.p_start[1],
        ]
        magnitude = math.hypot(*chord)
        alignment = abs(
            (chord[0] * plic_direction[0] + chord[1] * plic_direction[1]) / magnitude
        )
        return (
            getDistance(midpoint, plic_midpoint),
            1.0 - alignment,
            midpoint[0],
            midpoint[1],
        )

    selected = min(components, key=score)
    return [list(selected.p_start), list(selected.p_end)]


def infer_phase_from_plic(
    fit_center: Sequence[float], central_plic: LinearFacet
) -> PCICPhase:
    """Orient an unsigned circle from the reconstructed side of the PLIC."""

    midpoint = _facet_midpoint(central_plic)
    normal = _plic_normal(central_plic)
    side = (fit_center[0] - midpoint[0]) * normal[0] + (
        fit_center[1] - midpoint[1]
    ) * normal[1]
    scale = max(1.0, getDistance(fit_center, midpoint))
    if abs(side) <= np.finfo(float).eps * scale:
        raise PCICAmbiguousSourceChoice(
            "The fitted circle center lies on the central PLIC and cannot orient the phase"
        )
    return "disk" if side > 0.0 else "complement"


def _pair_circle_components(
    center: Sequence[float],
    signed_radius: float,
    intersections: Sequence[Sequence[float]],
    polygon_points: Sequence[Sequence[float]],
    config: PCICConfig,
) -> tuple[tuple[PCICArcComponent, ...], Literal["paired", "unresolved"]]:
    """Pair every crossing into the connected circle arcs inside a convex cell."""

    magnitude = abs(signed_radius)
    if magnitude <= np.finfo(float).eps:
        return (), "unresolved"
    if len(intersections) == 0:
        probe = [center[0] + magnitude, center[1]]
        if not pointInPoly(probe, polygon_points):
            return (), "unresolved"
        sweep = 2.0 * math.pi if signed_radius > 0.0 else -2.0 * math.pi
        point = (float(probe[0]), float(probe[1]))
        return (
            PCICArcComponent(
                center=(float(center[0]), float(center[1])),
                radius=float(signed_radius),
                p_start=point,
                p_end=point,
                start_angle=0.0,
                sweep_angle=sweep,
                closed=True,
            ),
        ), "paired"
    if len(intersections) % 2 != 0:
        return (), "unresolved"

    angular_points = sorted(
        (
            math.atan2(point[1] - center[1], point[0] - center[0]) % (2.0 * math.pi),
            (float(point[0]), float(point[1])),
        )
        for point in intersections
    )
    components: list[PCICArcComponent] = []
    for index, (start_angle, start_point) in enumerate(angular_points):
        end_angle, end_point = angular_points[(index + 1) % len(angular_points)]
        counterclockwise_sweep = (end_angle - start_angle) % (2.0 * math.pi)
        if counterclockwise_sweep <= np.finfo(float).eps:
            continue
        middle_angle = start_angle + 0.5 * counterclockwise_sweep
        probe = [
            center[0] + magnitude * math.cos(middle_angle),
            center[1] + magnitude * math.sin(middle_angle),
        ]
        if not pointInPoly(probe, polygon_points):
            continue
        if signed_radius > 0.0:
            component_start = start_point
            component_end = end_point
            oriented_start_angle = start_angle
            oriented_sweep = counterclockwise_sweep
        else:
            component_start = end_point
            component_end = start_point
            oriented_start_angle = end_angle
            oriented_sweep = -counterclockwise_sweep
        components.append(
            PCICArcComponent(
                center=(float(center[0]), float(center[1])),
                radius=float(signed_radius),
                p_start=component_start,
                p_end=component_end,
                start_angle=oriented_start_angle,
                sweep_angle=oriented_sweep,
            )
        )
    if not components:
        return (), "unresolved"
    return tuple(components), "paired"


def _deduplicate_points(
    points: Sequence[Sequence[float]], tolerance: float
) -> list[list[float]]:
    unique: list[list[float]] = []
    for point in points:
        candidate = [float(point[0]), float(point[1])]
        if not any(getDistance(candidate, prior) <= tolerance for prior in unique):
            unique.append(candidate)
    return unique


def _ordinary_line_fit(
    independent: np.ndarray, dependent: np.ndarray
) -> tuple[float, float]:
    """Solve the cited ordinary 2-by-2 least-squares normal equations."""

    matrix = np.array(
        [
            [float(np.dot(independent, independent)), float(np.sum(independent))],
            [float(np.sum(independent)), float(independent.size)],
        ]
    )
    right_hand_side = -np.array(
        [float(np.dot(independent, dependent)), float(np.sum(dependent))]
    )
    try:
        slope, intercept = np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise PCICDegenerateFit("The LLS 2-by-2 system is singular") from error
    if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)):
        raise PCICDegenerateFit("The LLS 2-by-2 solution is not finite")
    return float(slope), float(intercept)


def _facet_midpoint(facet: LinearFacet) -> list[float]:
    return [
        0.5 * (facet.pLeft[0] + facet.pRight[0]),
        0.5 * (facet.pLeft[1] + facet.pRight[1]),
    ]


def _subblock(
    block: Sequence[Sequence[object]], center_i: int, center_j: int, radius: int
) -> list[list[object]]:
    return [
        [block[i][j] for j in range(center_j - radius, center_j + radius + 1)]
        for i in range(center_i - radius, center_i + radius + 1)
    ]


def _validate_square_stencil(
    stencil: Sequence[Sequence[object]], size: int, label: str
) -> None:
    if len(stencil) != size or any(len(row) != size for row in stencil):
        raise ValueError(f"PCIC requires a {size}-by-{size} {label} stencil")


def _validate_cartesian_block(
    block: Sequence[Sequence[object]], size: int, config: PCICConfig
) -> tuple[float, float, float, float]:
    _validate_square_stencil(block, size, "Cartesian polygon")
    cell_bounds: list[list[tuple[float, float, float, float]]] = [
        [None for _ in range(size)] for _ in range(size)  # type: ignore[list-item]
    ]
    reference_width: Optional[float] = None
    for i in range(size):
        for j in range(size):
            polygon = block[i][j]
            if polygon is None:
                raise PCICUnsupportedGeometry(
                    "The published Cartesian predictor requires a complete halo"
                )
            points = _polygon_points(polygon)
            if len(points) != 4:
                raise PCICUnsupportedGeometry("PCIC requires square Cartesian cells")
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            width = x_max - x_min
            height = y_max - y_min
            scale = max(1.0, abs(x_min), abs(x_max), abs(y_min), abs(y_max))
            tolerance = config.cartesian_tolerance * scale
            expected_corners = {
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            }
            actual_corners = {(point[0], point[1]) for point in points}
            if actual_corners != expected_corners or width <= 0.0 or height <= 0.0:
                raise PCICUnsupportedGeometry("PCIC requires axis-aligned square cells")
            if abs(width - height) > tolerance:
                raise PCICUnsupportedGeometry("PCIC requires square Cartesian cells")
            if reference_width is None:
                reference_width = width
            elif abs(width - reference_width) > tolerance:
                raise PCICUnsupportedGeometry("PCIC requires a uniform Cartesian grid")
            fraction = _polygon_fraction(polygon)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("PCIC volume fractions must lie in [0, 1]")
            cell_bounds[i][j] = (x_min, x_max, y_min, y_max)

    assert reference_width is not None
    origin_x = cell_bounds[0][0][0]
    origin_y = cell_bounds[0][0][2]
    alignment_scale = max(
        1.0,
        abs(origin_x),
        abs(origin_y),
        size * reference_width,
    )
    alignment_tolerance = config.cartesian_tolerance * alignment_scale
    for i in range(size):
        for j in range(size):
            x_min, _, y_min, _ = cell_bounds[i][j]
            if (
                abs(x_min - (origin_x + i * reference_width)) > alignment_tolerance
                or abs(y_min - (origin_y + j * reference_width)) > alignment_tolerance
            ):
                raise PCICUnsupportedGeometry(
                    "PCIC requires an ordered, uniform Cartesian block"
                )
    return (
        origin_x,
        origin_x + size * reference_width,
        origin_y,
        origin_y + size * reference_width,
    )


def _is_reconstructed_fraction(fraction: float, config: PCICConfig) -> bool:
    return (
        config.reconstruction_fraction_min
        < fraction
        < 1.0 - config.reconstruction_fraction_min
    )


def _center_from_plic_chord(
    facet: LinearFacet,
    radius: float,
    sign: float,
) -> list[float]:
    """Apply the paper's minimum-radius reset using the PLIC chord."""

    chord = getDistance(facet.pLeft, facet.pRight)
    if chord > 2.0 * radius:
        raise PCICUnsupportedGeometry(
            "The reset radius is smaller than half of the central PLIC chord"
        )
    midpoint = [
        0.5 * (facet.pLeft[0] + facet.pRight[0]),
        0.5 * (facet.pLeft[1] + facet.pRight[1]),
    ]
    normal = _plic_normal(facet)
    distance = math.sqrt(max(0.0, radius * radius - 0.25 * chord * chord))
    return [
        midpoint[0] + sign * distance * normal[0],
        midpoint[1] + sign * distance * normal[1],
    ]


def _plic_normal(facet: LinearFacet) -> list[float]:
    dx = facet.pRight[0] - facet.pLeft[0]
    dy = facet.pRight[1] - facet.pLeft[1]
    magnitude = math.hypot(dx, dy)
    if magnitude == 0.0:
        raise PCICError("Degenerate PLIC facet")
    return [-dy / magnitude, dx / magnitude]


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
    "CENTER_TRANSLATION_VARIANT",
    "CenterTranslationRootPolicy",
    "PCICCellFacet",
    "PCICArcComponent",
    "PCICAmbiguousSourceChoice",
    "PCICCircle",
    "PCICConfig",
    "PCICConvergenceError",
    "PCICDegenerateFit",
    "PCICError",
    "PCICPhase",
    "PCICPhasePolicy",
    "PCICUnsupportedGeometry",
    "RADIUS_ADJUSTMENT_VARIANT",
    "build_lls_parker_young_plic_stencil",
    "collect_stencil_samples",
    "fit_riemann_sphere",
    "infer_phase_from_plic",
    "reconstruct_bare_pcic_cell",
    "reconstruct_bare_pcic_cartesian_cell",
    "reconstruct_lls_plic",
    "reconstruct_parker_young_plic",
    "parker_young_normal",
    "sample_plic_segment",
    "source_variant_for_correction",
]
