"""Static PLVIRA reconstruction following Remmerswaal and Veldman (2022)."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize

from main.algos.baselines.plvira_ghf import (
    CartesianGHFCurvature,
    cartesian_ghf_curvature,
)
from main.geoms.geoms import getArea, getCentroid


Point = Sequence[float]
Polygon = Sequence[Point]

_ROOT_TOLERANCE = 1.0e-12
_GEOMETRY_TOLERANCE = 1.0e-12


def _dot(a: Point, b: Point) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _local_point(
    point: Point, center: Point, normal: Point, tangent: Point
) -> Tuple[float, float]:
    displacement = (point[0] - center[0], point[1] - center[1])
    return _dot(tangent, displacement), _dot(normal, displacement)


def _level_set_local(s: float, n: float, curvature: float, shift: float) -> float:
    return n - shift + 0.5 * curvature * s * s


def _quadratic_roots_on_unit_interval(a: float, b: float, c: float) -> list:
    scale = max(abs(a), abs(b), abs(c), 1.0)
    roots = []
    if abs(a) <= _ROOT_TOLERANCE * scale:
        if abs(b) > _ROOT_TOLERANCE * scale:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        discriminant_scale = max(b * b, abs(4.0 * a * c), 1.0)
        if discriminant >= -_ROOT_TOLERANCE * discriminant_scale:
            discriminant = max(discriminant, 0.0)
            root_discriminant = math.sqrt(discriminant)
            roots.extend(
                (
                    (-b - root_discriminant) / (2.0 * a),
                    (-b + root_discriminant) / (2.0 * a),
                )
            )

    clipped = []
    for root in sorted(roots):
        if -_ROOT_TOLERANCE <= root <= 1.0 + _ROOT_TOLERANCE:
            root = min(max(root, 0.0), 1.0)
            if not clipped or abs(root - clipped[-1]) > _ROOT_TOLERANCE:
                clipped.append(root)
    return clipped


def _edge_root_parameters(
    start: Tuple[float, float],
    end: Tuple[float, float],
    curvature: float,
    shift: float,
) -> list:
    s0, n0 = start
    ds = end[0] - s0
    dn = end[1] - n0
    return _quadratic_roots_on_unit_interval(
        0.5 * curvature * ds * ds,
        dn + curvature * s0 * ds,
        _level_set_local(s0, n0, curvature, shift),
    )


def _lerp_local(
    start: Tuple[float, float], end: Tuple[float, float], t: float
) -> Tuple[float, float]:
    return (
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
    )


def _parabola_line_integral(
    s_start: float, s_end: float, curvature: float, shift: float
) -> float:
    """Return one half of integral(s dn - n ds) along q=0."""
    return -0.5 * shift * (s_end - s_start) - curvature * (s_end**3 - s_start**3) / 12.0


def _polygon_area_local(points: Sequence[Tuple[float, float]]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[i][1] * points[(i + 1) % len(points)][0]
        for i in range(len(points))
    )


def _split_boundary_pieces(
    local_polygon: Sequence[Tuple[float, float]], curvature: float, shift: float
) -> list:
    pieces = []
    for index, start in enumerate(local_polygon):
        end = local_polygon[(index + 1) % len(local_polygon)]
        parameters = [0.0]
        parameters.extend(_edge_root_parameters(start, end, curvature, shift))
        parameters.append(1.0)
        parameters = sorted(set(round(value, 15) for value in parameters))
        for left, right in zip(parameters[:-1], parameters[1:]):
            if right - left <= _ROOT_TOLERANCE:
                continue
            piece_start = _lerp_local(start, end, left)
            piece_end = _lerp_local(start, end, right)
            midpoint = _lerp_local(start, end, 0.5 * (left + right))
            inside = _level_set_local(*midpoint, curvature, shift) <= 0.0
            pieces.append((piece_start, piece_end, inside))
    return pieces


def parabolic_polygon_area(
    polygon: Polygon,
    center: Point,
    angle: float,
    curvature: float,
    shift: float,
) -> float:
    """Compute exactly the area of ``polygon intersect {q <= 0}``.

    The edge splitting and analytic parabolic correction follow Section 5.4
    of Remmerswaal and Veldman (2022).
    """
    if len(polygon) < 3:
        raise ValueError("PLVIRA requires polygons with at least three vertices")

    normal = (math.cos(angle), math.sin(angle))
    tangent = (-normal[1], normal[0])
    local_polygon = [_local_point(point, center, normal, tangent) for point in polygon]
    if _polygon_area_local(local_polygon) < 0.0:
        local_polygon.reverse()

    pieces = _split_boundary_pieces(local_polygon, curvature, shift)
    if not pieces:
        return 0.0

    inside_count = sum(piece[2] for piece in pieces)
    polygon_area = abs(_polygon_area_local(local_polygon))
    if inside_count == len(pieces):
        return polygon_area
    if inside_count == 0:
        return 0.0

    boundary_integral = 0.0
    entries = []
    exits = []
    count = len(pieces)
    for index, (start, end, inside) in enumerate(pieces):
        if inside:
            boundary_integral += 0.5 * (start[0] * end[1] - start[1] * end[0])
        previous_inside = pieces[index - 1][2]
        if inside and not previous_inside:
            entries.append(start)
        if inside and not pieces[(index + 1) % count][2]:
            exits.append(end)

    if len(entries) != len(exits):
        raise RuntimeError("Could not pair polygon/parabola boundary crossings")

    unused_entries = list(entries)
    for exit_point in exits:
        candidates = [
            entry
            for entry in unused_entries
            if entry[0] <= exit_point[0] + _ROOT_TOLERANCE
        ]
        if not candidates:
            candidates = unused_entries
        entry_point = max(candidates, key=lambda point: point[0])
        unused_entries.remove(entry_point)
        boundary_integral += _parabola_line_integral(
            exit_point[0], entry_point[0], curvature, shift
        )

    return min(max(abs(boundary_integral), 0.0), polygon_area)


def _level_set_extrema_on_polygon(
    polygon: Polygon,
    center: Point,
    angle: float,
    curvature: float,
) -> Tuple[float, float]:
    normal = (math.cos(angle), math.sin(angle))
    tangent = (-normal[1], normal[0])
    local_polygon = [_local_point(point, center, normal, tangent) for point in polygon]
    values = []
    for index, start in enumerate(local_polygon):
        end = local_polygon[(index + 1) % len(local_polygon)]
        s0, n0 = start
        ds = end[0] - s0
        dn = end[1] - n0
        a = 0.5 * curvature * ds * ds
        b = dn + curvature * s0 * ds
        c = n0 + 0.5 * curvature * s0 * s0
        parameters = [0.0, 1.0]
        if abs(a) > _ROOT_TOLERANCE and 0.0 < -b / (2.0 * a) < 1.0:
            parameters.append(-b / (2.0 * a))
        values.extend(a * t * t + b * t + c for t in parameters)
    return min(values), max(values)


def solve_volume_shift(
    polygon: Polygon,
    target_fraction: float,
    center: Point,
    angle: float,
    curvature: float,
    root_tolerance: float = 1.0e-13,
) -> float:
    """Solve the central-volume constraint, equation (23), with Brent's method."""
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in [0, 1]")
    polygon_area = abs(getArea(polygon))
    if polygon_area <= 0.0:
        raise ValueError("PLVIRA requires a polygon with positive area")

    lower, upper = _level_set_extrema_on_polygon(polygon, center, angle, curvature)
    if target_fraction == 0.0:
        return lower
    if target_fraction == 1.0:
        return upper
    target_area = target_fraction * polygon_area

    def residual(candidate_shift: float) -> float:
        return (
            parabolic_polygon_area(polygon, center, angle, curvature, candidate_shift)
            - target_area
        )

    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=root_tolerance,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=100,
        )
    )


def _point_in_convex_polygon(
    point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]
) -> bool:
    signs = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
            point[0] - start[0]
        )
        if abs(cross) > _GEOMETRY_TOLERANCE:
            signs.append(cross > 0.0)
    return not signs or all(sign == signs[0] for sign in signs)


def _interface_intervals(
    polygon: Polygon,
    center: Point,
    angle: float,
    curvature: float,
    shift: float,
) -> list:
    normal = (math.cos(angle), math.sin(angle))
    tangent = (-normal[1], normal[0])
    local_polygon = [_local_point(point, center, normal, tangent) for point in polygon]
    intersections = []
    for index, start in enumerate(local_polygon):
        end = local_polygon[(index + 1) % len(local_polygon)]
        for parameter in _edge_root_parameters(start, end, curvature, shift):
            intersections.append(_lerp_local(start, end, parameter)[0])
    intersections.sort()
    unique = []
    for value in intersections:
        if not unique or abs(value - unique[-1]) > _ROOT_TOLERANCE:
            unique.append(value)

    intervals = []
    for left, right in zip(unique[:-1], unique[1:]):
        if right - left <= _ROOT_TOLERANCE:
            continue
        midpoint = 0.5 * (left + right)
        curve_point = (midpoint, shift - 0.5 * curvature * midpoint * midpoint)
        if _point_in_convex_polygon(curve_point, local_polygon):
            intervals.append((left, right))
    return intervals


def _shift_angle_derivative(
    intervals: Sequence[Tuple[float, float]], curvature: float, shift: float
) -> float:
    total_length = sum(right - left for left, right in intervals)
    if total_length <= _GEOMETRY_TOLERANCE:
        raise RuntimeError("The PLVIRA interface does not cross the center cell")
    numerator = sum(
        0.5 * (curvature * shift - 1.0) * (right**2 - left**2)
        - 0.125 * curvature * curvature * (right**4 - left**4)
        for left, right in intervals
    )
    return -numerator / total_length


def _volume_angle_derivative(
    intervals: Sequence[Tuple[float, float]],
    curvature: float,
    shift: float,
    shift_derivative: float,
) -> float:
    return sum(
        shift_derivative * (right - left)
        + 0.5 * (curvature * shift - 1.0) * (right**2 - left**2)
        - 0.125 * curvature * curvature * (right**4 - left**4)
        for left, right in intervals
    )


def _validate_stencil(
    polygons: Sequence[Sequence[Polygon]], fractions: Sequence[Sequence[float]]
) -> None:
    if len(polygons) != 3 or any(len(row) != 3 for row in polygons):
        raise ValueError("PLVIRA requires a complete 3 x 3 polygon stencil")
    if len(fractions) != 3 or any(len(row) != 3 for row in fractions):
        raise ValueError("PLVIRA requires a complete 3 x 3 fraction stencil")
    for row in range(3):
        for column in range(3):
            if polygons[row][column] is None:
                raise ValueError(
                    "PLVIRA does not define a missing-cell stencil fallback"
                )
            if not 0.0 <= fractions[row][column] <= 1.0:
                raise ValueError("all PLVIRA reference fractions must lie in [0, 1]")
            if abs(getArea(polygons[row][column])) <= 0.0:
                raise ValueError("all PLVIRA stencil polygons must have positive area")


def _validate_cartesian_stencil(
    polygons: Sequence[Sequence[Polygon]], cell_size: float
) -> None:
    """Reject mesh geometry outside the source method's Cartesian scope."""
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("PLVIRA cell_size must be a positive finite scalar")
    def bounding_box_center(polygon: Polygon) -> Point:
        return (
            0.5
            * (min(point[0] for point in polygon) + max(point[0] for point in polygon)),
            0.5
            * (min(point[1] for point in polygon) + max(point[1] for point in polygon)),
        )

    # Polygon-centroid formulas lose accuracy through cancellation when small
    # cells have large absolute coordinates.  Axis-aligned bounding boxes give
    # the exact geometric center needed by this Cartesian-scope check.
    center = bounding_box_center(polygons[1][1])
    tolerance = (
        128.0
        * np.finfo(float).eps
        * max(abs(center[0]), abs(center[1]), cell_size, 1.0)
    )
    half = 0.5 * cell_size
    for row in range(3):
        for column in range(3):
            polygon = polygons[row][column]
            expected_center = (
                center[0] + (column - 1) * cell_size,
                center[1] + (row - 1) * cell_size,
            )
            actual_center = bounding_box_center(polygon)
            if (
                math.hypot(
                    actual_center[0] - expected_center[0],
                    actual_center[1] - expected_center[1],
                )
                > tolerance
            ):
                raise ValueError(
                    "PLVIRA is restricted to a uniform Cartesian 3 x 3 stencil"
                )
            expected_vertices = [
                (expected_center[0] - half, expected_center[1] - half),
                (expected_center[0] + half, expected_center[1] - half),
                (expected_center[0] + half, expected_center[1] + half),
                (expected_center[0] - half, expected_center[1] + half),
            ]
            if len(polygon) != 4 or any(
                not any(
                    math.hypot(point[0] - expected[0], point[1] - expected[1])
                    <= tolerance
                    for expected in expected_vertices
                )
                for point in polygon
            ):
                raise ValueError(
                    "PLVIRA is restricted to axis-aligned square Cartesian cells"
                )


def plvira_objective_and_gradient(
    polygons: Sequence[Sequence[Polygon]],
    fractions: Sequence[Sequence[float]],
    angle: float,
    curvature: float,
    root_tolerance: float = 1.0e-13,
) -> Tuple[float, float, float]:
    """Return ``(f_L2**2, d(f_L2**2)/dtheta, phi)`` for one trial angle."""
    _validate_stencil(polygons, fractions)
    center_polygon = polygons[1][1]
    center = getCentroid(center_polygon)
    shift = solve_volume_shift(
        center_polygon,
        fractions[1][1],
        center,
        angle,
        curvature,
        root_tolerance=root_tolerance,
    )
    center_intervals = _interface_intervals(
        center_polygon, center, angle, curvature, shift
    )
    shift_derivative = _shift_angle_derivative(center_intervals, curvature, shift)

    objective = 0.0
    gradient = 0.0
    for row in range(3):
        for column in range(3):
            if row == 1 and column == 1:
                continue
            polygon = polygons[row][column]
            cell_area = abs(getArea(polygon))
            reconstructed_area = parabolic_polygon_area(
                polygon, center, angle, curvature, shift
            )
            residual = reconstructed_area / cell_area - fractions[row][column]
            objective += residual * residual
            intervals = _interface_intervals(polygon, center, angle, curvature, shift)
            if intervals:
                area_derivative = _volume_angle_derivative(
                    intervals, curvature, shift, shift_derivative
                )
                gradient += 2.0 * residual * area_derivative / cell_area
    return objective, gradient, shift


def rectilinear_lvira_angle_guess(
    fractions: Sequence[Sequence[float]],
    cell_widths: Sequence[float],
    cell_heights: Sequence[float],
) -> float:
    """Return the LVIRA angle guess used by the paper-linked implementation."""
    if len(fractions) != 3 or any(len(row) != 3 for row in fractions):
        raise ValueError("the LVIRA angle guess requires a 3 x 3 fraction stencil")
    if len(cell_widths) != 3 or len(cell_heights) != 3:
        raise ValueError("cell_widths and cell_heights must each contain three values")
    if any(width <= 0.0 for width in cell_widths) or any(
        height <= 0.0 for height in cell_heights
    ):
        raise ValueError("rectilinear cell dimensions must be positive")

    normal_x = (fractions[1][0] - fractions[1][2]) / (
        cell_widths[0] + 2.0 * cell_widths[1] + cell_widths[2]
    )
    normal_y = (fractions[0][1] - fractions[2][1]) / (
        cell_heights[0] + 2.0 * cell_heights[1] + cell_heights[2]
    )
    if math.hypot(normal_x, normal_y) <= _GEOMETRY_TOLERANCE:
        raise RuntimeError(
            "the paper-associated LVIRA angle guess is undefined for this stencil"
        )
    return math.atan2(normal_y, normal_x)


@dataclass(frozen=True)
class ParabolicInterface:
    center: Tuple[float, float]
    angle: float
    curvature: float
    shift: float
    objective: float
    optimizer_success: bool
    optimizer_message: str
    curvature_source: str
    ghf_diagnostics: Optional[CartesianGHFCurvature]

    @property
    def normal(self) -> Tuple[float, float]:
        return math.cos(self.angle), math.sin(self.angle)

    @property
    def tangent(self) -> Tuple[float, float]:
        normal = self.normal
        return -normal[1], normal[0]

    def level_set(self, point: Point) -> float:
        s, n = _local_point(point, self.center, self.normal, self.tangent)
        return _level_set_local(s, n, self.curvature, self.shift)

    def intersect_area(self, polygon: Polygon) -> float:
        return parabolic_polygon_area(
            polygon, self.center, self.angle, self.curvature, self.shift
        )

    def intervals_in_polygon(self, polygon: Polygon) -> Tuple[Tuple[float, float], ...]:
        """Return every retained tangent-coordinate interval in ``polygon``."""

        return tuple(
            _interface_intervals(
                polygon, self.center, self.angle, self.curvature, self.shift
            )
        )


def _reconstruct_plvira_with_curvature(
    polygons: Sequence[Sequence[Polygon]],
    fractions: Sequence[Sequence[float]],
    curvature: float,
    *,
    curvature_source: str,
    ghf_diagnostics: Optional[CartesianGHFCurvature],
    cell_size: float,
    initial_angle: Optional[float] = None,
    gradient_tolerance: float = 1.0e-8,
    root_tolerance: float = 1.0e-13,
    max_iterations: int = 100,
) -> ParabolicInterface:
    """Shared fixed-curvature search; callers name the curvature source."""
    _validate_stencil(polygons, fractions)
    _validate_cartesian_stencil(polygons, cell_size)
    if not math.isfinite(curvature):
        raise ValueError("PLVIRA requires a finite curvature")
    if initial_angle is None:
        initial_angle = rectilinear_lvira_angle_guess(
            fractions,
            (cell_size, cell_size, cell_size),
            (cell_size, cell_size, cell_size),
        )

    latest_shift = None

    def objective(candidate: np.ndarray) -> Tuple[float, np.ndarray]:
        nonlocal latest_shift
        value, derivative, latest_shift = plvira_objective_and_gradient(
            polygons,
            fractions,
            float(candidate[0]),
            curvature,
            root_tolerance=root_tolerance,
        )
        return value, np.asarray([derivative], dtype=float)

    result = minimize(
        objective,
        np.asarray([initial_angle], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"gtol": gradient_tolerance, "maxiter": max_iterations, "maxls": 20},
    )
    final_angle = float(result.x[0])
    final_objective, _, final_shift = plvira_objective_and_gradient(
        polygons,
        fractions,
        final_angle,
        curvature,
        root_tolerance=root_tolerance,
    )
    center = getCentroid(polygons[1][1])
    return ParabolicInterface(
        center=(float(center[0]), float(center[1])),
        angle=final_angle,
        curvature=float(curvature),
        shift=final_shift,
        objective=final_objective,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        curvature_source=curvature_source,
        ghf_diagnostics=ghf_diagnostics,
    )


def reconstruct_plvira(
    polygons: Sequence[Sequence[Polygon]],
    fractions: Sequence[Sequence[float]],
    *,
    cartesian_fractions: Sequence[Sequence[float]],
    target_index: Tuple[int, int],
    cell_size: float,
    initial_angle: Optional[float] = None,
    gradient_tolerance: float = 1.0e-8,
    root_tolerance: float = 1.0e-13,
    max_iterations: int = 100,
) -> ParabolicInterface:
    """Run operational Cartesian PLVIRA with source-paper GHF curvature."""
    _validate_stencil(polygons, fractions)
    _validate_cartesian_stencil(polygons, cell_size)
    grid = np.asarray(cartesian_fractions, dtype=float)
    if grid.ndim != 2:
        raise ValueError("cartesian_fractions must be a two-dimensional grid")
    target_row, target_column = target_index
    if (
        target_row - 1 < 0
        or target_row + 1 >= grid.shape[0]
        or target_column - 1 < 0
        or target_column + 1 >= grid.shape[1]
    ):
        raise ValueError("target_index does not contain the PLVIRA 3 x 3 stencil")
    local_grid = grid[
        target_row - 1 : target_row + 2,
        target_column - 1 : target_column + 2,
    ]
    local_fractions = np.asarray(fractions, dtype=float)
    if not np.array_equal(local_grid, local_fractions):
        raise ValueError(
            "PLVIRA 3 x 3 fractions must exactly match cartesian_fractions "
            "around target_index"
        )
    ghf = cartesian_ghf_curvature(cartesian_fractions, target_index, cell_size)
    return _reconstruct_plvira_with_curvature(
        polygons,
        fractions,
        ghf.curvature,
        curvature_source="cartesian-ghf",
        ghf_diagnostics=ghf,
        cell_size=cell_size,
        initial_angle=initial_angle,
        gradient_tolerance=gradient_tolerance,
        root_tolerance=root_tolerance,
        max_iterations=max_iterations,
    )


def reconstruct_plvira_exact_curvature_oracle(
    polygons: Sequence[Sequence[Polygon]],
    fractions: Sequence[Sequence[float]],
    exact_curvature: float,
    *,
    cell_size: float,
    initial_angle: Optional[float] = None,
    gradient_tolerance: float = 1.0e-8,
    root_tolerance: float = 1.0e-13,
    max_iterations: int = 100,
) -> ParabolicInterface:
    """Run the explicitly named exact-curvature diagnostic oracle mode."""
    return _reconstruct_plvira_with_curvature(
        polygons,
        fractions,
        exact_curvature,
        curvature_source="exact-curvature-oracle",
        ghf_diagnostics=None,
        cell_size=cell_size,
        initial_angle=initial_angle,
        gradient_tolerance=gradient_tolerance,
        root_tolerance=root_tolerance,
        max_iterations=max_iterations,
    )
