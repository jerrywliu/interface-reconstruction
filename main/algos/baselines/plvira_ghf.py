"""Cartesian generalized-height-function curvature used by PLVIRA.

This module implements the two-dimensional, uniform-Cartesian specialization
of Algorithms 4--7 in Popinet (2009), which Remmerswaal and Veldman cite for
the curvature supplied to PLVIRA.  Fractions are indexed ``[row][column]``
with both row (y) and column (x) increasing in the positive coordinate
direction.  The represented liquid is the phase with volume fraction one.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq


Index = Tuple[int, int]
Point = Tuple[float, float]

_NOT_ZERO = 1.0e-30


class GHFStencilError(RuntimeError):
    """Raised when the source algorithm needs data outside the supplied grid."""


@dataclass(frozen=True)
class CartesianGHFCurvature:
    """One source-traceable Cartesian GHF curvature result."""

    curvature: float
    method: str
    normal: Point
    direction: Optional[str]
    height_points: int
    independent_points: int
    fit_points: int
    diagnostic: str


def _fraction_array(fractions: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(fractions, dtype=float)
    if array.ndim != 2 or min(array.shape) < 3:
        raise ValueError("Cartesian GHF requires a rectangular two-dimensional grid")
    if not np.all(np.isfinite(array)):
        raise ValueError("Cartesian GHF fractions must be finite")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("Cartesian GHF fractions must lie in [0, 1]")
    return array


def _is_interfacial(fraction: float) -> bool:
    # The source geometrical VOF field is bounded and clipped.  Neither the
    # cited algorithm nor its implementation applies a mixed-cell tolerance.
    return 0.0 < fraction < 1.0


def _require_neighborhood(array: np.ndarray, target: Index, radius: int) -> None:
    row, column = target
    if not (0 <= row < array.shape[0] and 0 <= column < array.shape[1]):
        raise ValueError("GHF target index lies outside the fraction grid")
    if (
        row - radius < 0
        or row + radius >= array.shape[0]
        or column - radius < 0
        or column + radius >= array.shape[1]
    ):
        raise GHFStencilError(
            "the Cartesian GHF source algorithm has no boundary-stencil policy; "
            "supply the required neighboring cells"
        )


def cartesian_myc_normal(fractions: Sequence[Sequence[float]], target: Index) -> Point:
    """Return the outward MYC normal used by Popinet's GHF algorithm.

    This is the two-dimensional Mixed-Youngs-Centered algebra used by the
    primary implementation.  The returned vector is Euclidean-normalized;
    the source MYC routine itself uses an L1 normalization, which does not
    change its direction.
    """

    array = _fraction_array(fractions)
    _require_neighborhood(array, target, 1)
    row, column = target

    c_top = float(np.sum(array[row + 1, column - 1 : column + 2]))
    c_bottom = float(np.sum(array[row - 1, column - 1 : column + 2]))
    c_right = float(np.sum(array[row - 1 : row + 2, column + 1]))
    c_left = float(np.sum(array[row - 1 : row + 2, column - 1]))

    centered_x = 0.5 * (c_left - c_right)
    centered_y = 0.5 * (c_bottom - c_top)

    if abs(centered_x) <= abs(centered_y):
        centered_y = 1.0 if centered_y > 0.0 else -1.0
        centered_dominant_y = True
    else:
        centered_x = 1.0 if centered_x > 0.0 else -1.0
        centered_dominant_y = False

    youngs_x = (
        array[row - 1, column - 1]
        + 2.0 * array[row, column - 1]
        + array[row + 1, column - 1]
        - array[row - 1, column + 1]
        - 2.0 * array[row, column + 1]
        - array[row + 1, column + 1]
        + _NOT_ZERO
    )
    youngs_y = (
        array[row - 1, column - 1]
        + 2.0 * array[row - 1, column]
        + array[row - 1, column + 1]
        - array[row + 1, column - 1]
        - 2.0 * array[row + 1, column]
        - array[row + 1, column + 1]
        + _NOT_ZERO
    )

    if centered_dominant_y:
        if abs(youngs_x) / abs(youngs_y) > abs(centered_x):
            centered_x, centered_y = float(youngs_x), float(youngs_y)
    elif abs(youngs_y) / abs(youngs_x) > abs(centered_y):
        centered_x, centered_y = float(youngs_x), float(youngs_y)

    magnitude = math.hypot(centered_x, centered_y)
    if magnitude == 0.0:
        raise RuntimeError("the source MYC normal is undefined for this stencil")
    return centered_x / magnitude, centered_y / magnitude


def _move(index: Index, axis: int, step: int) -> Index:
    row, column = index
    if axis == 0:
        return row, column + step
    return row + step, column


def _in_bounds(array: np.ndarray, index: Index) -> bool:
    return 0 <= index[0] < array.shape[0] and 0 <= index[1] < array.shape[1]


def _relative_center(index: Index, target: Index, cell_size: float) -> Point:
    return (
        (index[1] - target[1]) * cell_size,
        (index[0] - target[0]) * cell_size,
    )


def _interface_height_point(
    array: np.ndarray,
    start: Index,
    target: Index,
    axis: int,
    empty_direction: int,
    cell_size: float,
) -> Optional[Point]:
    """Apply Popinet Algorithm 4 and return its interface position."""

    total = float(array[start])

    top = start
    top_fraction = float(array[top])
    found_interface = top_fraction < 1.0
    while not found_interface or _is_interfacial(top_fraction):
        top = _move(top, axis, empty_direction)
        if not _in_bounds(array, top):
            return None
        top_fraction = float(array[top])
        total += top_fraction
        if _is_interfacial(top_fraction):
            found_interface = True
    if top_fraction != 0.0:
        return None

    bottom = start
    bottom_fraction = float(array[bottom])
    found_interface = bottom_fraction > 0.0
    while not found_interface or _is_interfacial(bottom_fraction):
        bottom = _move(bottom, axis, -empty_direction)
        if not _in_bounds(array, bottom):
            return None
        bottom_fraction = float(array[bottom])
        total += bottom_fraction
        if _is_interfacial(bottom_fraction):
            found_interface = True
    if bottom_fraction != 1.0:
        return None

    # Remmerswaal--Veldman state this LHF validity condition explicitly:
    # fractions must decrease monotonically from the full terminal cell to
    # the empty terminal cell.  Popinet's Algorithm 4 supplies the terminal
    # search but does not repeat this check in its pseudocode.
    current = bottom
    previous_fraction = bottom_fraction
    while current != top:
        current = _move(current, axis, empty_direction)
        current_fraction = float(array[current])
        if current_fraction > previous_fraction:
            return None
        previous_fraction = current_fraction

    point = list(_relative_center(start, target, cell_size))
    bottom_center = _relative_center(bottom, target, cell_size)[axis]
    oriented_bottom_center = empty_direction * bottom_center
    oriented_height = oriented_bottom_center - 0.5 * cell_size + total * cell_size
    point[axis] = empty_direction * oriented_height
    return float(point[0]), float(point[1])


def _independent_count(points: Sequence[Point], cell_size: float) -> int:
    """Count independent points using Popinet's published one-cell rule."""

    if len(points) < 2:
        return len(points)
    count = 1
    threshold_squared = cell_size * cell_size
    for index in range(1, len(points)):
        dependent = False
        for previous in range(index):
            dx = points[index][0] - points[previous][0]
            dy = points[index][1] - points[previous][1]
            if dx * dx + dy * dy < threshold_squared:
                dependent = True
                break
        if not dependent:
            count += 1
    return count


def _clip_square(center: Point, cell_size: float, normal: Point, alpha: float) -> list:
    half = 0.5 * cell_size
    polygon = [
        (center[0] - half, center[1] - half),
        (center[0] + half, center[1] - half),
        (center[0] + half, center[1] + half),
        (center[0] - half, center[1] + half),
    ]
    clipped = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        start_value = normal[0] * start[0] + normal[1] * start[1] - alpha
        end_value = normal[0] * end[0] + normal[1] * end[1] - alpha
        start_inside = start_value <= 0.0
        end_inside = end_value <= 0.0
        if start_inside:
            clipped.append(start)
        if start_inside != end_inside:
            parameter = start_value / (start_value - end_value)
            clipped.append(
                (
                    start[0] + parameter * (end[0] - start[0]),
                    start[1] + parameter * (end[1] - start[1]),
                )
            )
    return clipped


def _polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[index][1] * polygon[(index + 1) % len(polygon)][0]
            for index in range(len(polygon))
        )
    )


def _plic_fragment_center(
    center: Point, cell_size: float, fraction: float, normal: Point
) -> Point:
    """Return the barycentre of the MYC PLIC fragment in one square cell."""

    half = 0.5 * cell_size
    corners = [
        (center[0] - half, center[1] - half),
        (center[0] + half, center[1] - half),
        (center[0] + half, center[1] + half),
        (center[0] - half, center[1] + half),
    ]
    projections = [normal[0] * point[0] + normal[1] * point[1] for point in corners]
    lower, upper = min(projections), max(projections)
    target_area = fraction * cell_size * cell_size

    def residual(alpha: float) -> float:
        return (
            _polygon_area(_clip_square(center, cell_size, normal, alpha)) - target_area
        )

    alpha = float(
        brentq(
            residual,
            lower,
            upper,
            xtol=max(np.finfo(float).eps * cell_size, 1.0e-15),
            rtol=4.0 * np.finfo(float).eps,
            maxiter=100,
        )
    )
    clipped = _clip_square(center, cell_size, normal, alpha)
    tolerance = 64.0 * np.finfo(float).eps * max(cell_size, 1.0)
    on_fragment = []
    for point in clipped:
        if abs(normal[0] * point[0] + normal[1] * point[1] - alpha) <= tolerance:
            if not any(
                math.hypot(point[0] - other[0], point[1] - other[1]) <= tolerance
                for other in on_fragment
            ):
                on_fragment.append(point)
    if len(on_fragment) < 2:
        raise RuntimeError("could not recover the source MYC PLIC fragment")
    tangent = (-normal[1], normal[0])
    on_fragment.sort(key=lambda point: tangent[0] * point[0] + tangent[1] * point[1])
    left, right = on_fragment[0], on_fragment[-1]
    return 0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1])


def _fit_curvature(
    points: Sequence[Point], origin: Point, inward_normal: Point
) -> Optional[float]:
    tangent = (-inward_normal[1], inward_normal[0])
    design = []
    heights = []
    for point in points:
        dx = point[0] - origin[0]
        dy = point[1] - origin[1]
        coordinate = tangent[0] * dx + tangent[1] * dy
        height = inward_normal[0] * dx + inward_normal[1] * dy
        design.append((coordinate * coordinate, coordinate, 1.0))
        heights.append(height)
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.asarray(design, dtype=float), np.asarray(heights, dtype=float), rcond=None
    )
    if rank < 3:
        return None
    quadratic, slope, _ = coefficients
    return float(2.0 * quadratic / (1.0 + slope * slope) ** 1.5)


def _centroid_fallback_points(
    array: np.ndarray, target: Index, cell_size: float
) -> list:
    _require_neighborhood(array, target, 2)
    points = []
    for row in range(target[0] - 1, target[0] + 2):
        for column in range(target[1] - 1, target[1] + 2):
            fraction = float(array[row, column])
            if not _is_interfacial(fraction):
                continue
            normal = cartesian_myc_normal(array, (row, column))
            center = _relative_center((row, column), target, cell_size)
            points.append(_plic_fragment_center(center, cell_size, fraction, normal))
    return points


def cartesian_ghf_curvature(
    fractions: Sequence[Sequence[float]],
    target: Index,
    cell_size: float,
) -> CartesianGHFCurvature:
    """Compute source-paper GHF curvature for one Cartesian mixed cell.

    The hierarchy is Popinet (2009), Algorithms 4--7: try complete height
    functions in decreasing MYC-normal alignment, fit the combined consistent
    heights if both directions fail, replace them by 3x3 MYC-PLIC fragment
    barycentres if necessary, and return zero for a degenerate final fit.
    Curvature sign follows the Remmerswaal--Veldman parabola convention: a
    convex liquid domain has positive curvature.
    """

    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("Cartesian GHF cell_size must be a positive finite scalar")
    array = _fraction_array(fractions)
    _require_neighborhood(array, target, 1)
    if not _is_interfacial(float(array[target])):
        raise ValueError("Cartesian GHF target must be a mixed cell")

    normal = cartesian_myc_normal(array, target)
    axes = sorted((0, 1), key=lambda axis: (-abs(normal[axis]), axis))
    height_points = []

    for axis in axes:
        # The source papers order directions by alignment but do not specify
        # the zero-component sign.  Positive grid direction is the stable
        # coordinate-order tie break also used for equal alignments.
        empty_direction = 1 if normal[axis] >= 0.0 else -1
        perpendicular_axis = 1 - axis
        points_by_offset = {}
        for offset in (0, 1, -1):
            start = _move(target, perpendicular_axis, offset)
            if not _in_bounds(array, start):
                point = None
            else:
                point = _interface_height_point(
                    array,
                    start,
                    target,
                    axis,
                    empty_direction,
                    cell_size,
                )
            points_by_offset[offset] = point
            if point is not None:
                height_points.append(point)

        if all(points_by_offset[offset] is not None for offset in (-1, 0, 1)):
            # Heights are measured along the inward coordinate, matching the
            # sign of kappa in q = eta.x - phi + kappa*tau.x^2/2.
            inward_direction = -empty_direction
            heights = {
                offset: inward_direction * points_by_offset[offset][axis]
                for offset in (-1, 0, 1)
            }
            first = (heights[1] - heights[-1]) / (2.0 * cell_size)
            second = (heights[1] - 2.0 * heights[0] + heights[-1]) / (
                cell_size * cell_size
            )
            curvature = second / (1.0 + first * first) ** 1.5
            return CartesianGHFCurvature(
                curvature=float(curvature),
                method="height_function",
                normal=normal,
                direction="x" if axis == 0 else "y",
                height_points=3,
                independent_points=3,
                fit_points=0,
                diagnostic="complete three-column height function",
            )

    independent_heights = _independent_count(height_points, cell_size)
    target_center = (0.0, 0.0)
    target_fragment_center = _plic_fragment_center(
        target_center, cell_size, float(array[target]), normal
    )
    inward_normal = (-normal[0], -normal[1])

    if independent_heights >= 3:
        curvature = _fit_curvature(height_points, target_fragment_center, inward_normal)
        if curvature is not None:
            return CartesianGHFCurvature(
                curvature=curvature,
                method="mixed_height_parabola",
                normal=normal,
                direction=None,
                height_points=len(height_points),
                independent_points=independent_heights,
                fit_points=len(height_points),
                diagnostic="parabola fit to consistent heights from both directions",
            )

    centroid_points = _centroid_fallback_points(array, target, cell_size)
    independent_centroids = _independent_count(centroid_points, cell_size)
    if independent_centroids < 3:
        return CartesianGHFCurvature(
            curvature=0.0,
            method="degenerate_zero",
            normal=normal,
            direction=None,
            height_points=len(height_points),
            independent_points=independent_centroids,
            fit_points=len(centroid_points),
            diagnostic="fewer than three independent 3x3 PLIC fragment centroids",
        )

    curvature = _fit_curvature(centroid_points, target_fragment_center, inward_normal)
    if curvature is None:
        return CartesianGHFCurvature(
            curvature=0.0,
            method="degenerate_zero",
            normal=normal,
            direction=None,
            height_points=len(height_points),
            independent_points=independent_centroids,
            fit_points=len(centroid_points),
            diagnostic="singular 3x3 PLIC fragment-centroid parabola fit",
        )
    return CartesianGHFCurvature(
        curvature=curvature,
        method="plic_centroid_parabola",
        normal=normal,
        direction=None,
        height_points=len(height_points),
        independent_points=independent_centroids,
        fit_points=len(centroid_points),
        diagnostic="parabola fit to 3x3 MYC PLIC fragment centroids",
    )
