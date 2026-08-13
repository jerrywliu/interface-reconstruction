import math

import numpy as np
import pytest

from main.algos.baselines.plvira import (
    parabolic_polygon_area,
    plvira_objective_and_gradient,
    reconstruct_plvira,
    reconstruct_plvira_exact_curvature_oracle,
    solve_volume_shift,
)


UNIT_SQUARE = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]


def _uniform_stencil():
    polygons = []
    for row in range(3):
        polygon_row = []
        for column in range(3):
            x0 = column - 1.5
            y0 = row - 1.5
            polygon_row.append(
                [(x0, y0), (x0 + 1.0, y0), (x0 + 1.0, y0 + 1.0), (x0, y0 + 1.0)]
            )
        polygons.append(polygon_row)
    return polygons


def _exact_fractions(polygons, angle, curvature, shift):
    return [
        [
            parabolic_polygon_area(poly, (0.0, 0.0), angle, curvature, shift)
            for poly in row
        ]
        for row in polygons
    ]


def _exact_cartesian_grid(size, angle, curvature, shift):
    center = size // 2
    fractions = []
    for row in range(size):
        fraction_row = []
        for column in range(size):
            x = column - center
            y = row - center
            polygon = [
                (x - 0.5, y - 0.5),
                (x + 0.5, y - 0.5),
                (x + 0.5, y + 0.5),
                (x - 0.5, y + 0.5),
            ]
            fraction_row.append(
                parabolic_polygon_area(polygon, (0.0, 0.0), angle, curvature, shift)
            )
        fractions.append(fraction_row)
    return fractions


def _angle_error(actual, expected):
    return abs(math.atan2(math.sin(actual - expected), math.cos(actual - expected)))


def test_parabolic_polygon_area_reduces_to_horizontal_line_cut():
    assert parabolic_polygon_area(
        UNIT_SQUARE, (0.0, 0.0), math.pi / 2.0, 0.0, 0.0
    ) == pytest.approx(0.5)


def test_parabolic_polygon_area_matches_analytic_square_integral():
    curvature = 0.6
    shift = 0.1
    expected = shift + 0.5 - curvature / 24.0
    assert parabolic_polygon_area(
        UNIT_SQUARE, (0.0, 0.0), 0.0, curvature, shift
    ) == pytest.approx(expected, abs=1.0e-12)


def test_volume_shift_enforces_center_fraction():
    angle = 0.37
    curvature = -0.4
    target_fraction = 0.63
    shift = solve_volume_shift(
        UNIT_SQUARE, target_fraction, (0.0, 0.0), angle, curvature
    )
    area = parabolic_polygon_area(UNIT_SQUARE, (0.0, 0.0), angle, curvature, shift)
    assert area == pytest.approx(target_fraction, abs=2.0e-12)


def test_objective_angle_derivative_matches_centered_difference():
    polygons = _uniform_stencil()
    exact_angle = 0.41
    curvature = 0.22
    fractions = _exact_fractions(polygons, exact_angle, curvature, 0.08)
    trial_angle = exact_angle + 0.09
    value, derivative, _ = plvira_objective_and_gradient(
        polygons, fractions, trial_angle, curvature
    )
    step = 2.0e-6
    value_plus, _, _ = plvira_objective_and_gradient(
        polygons, fractions, trial_angle + step, curvature
    )
    value_minus, _, _ = plvira_objective_and_gradient(
        polygons, fractions, trial_angle - step, curvature
    )
    finite_difference = (value_plus - value_minus) / (2.0 * step)
    assert value > 0.0
    assert derivative == pytest.approx(finite_difference, rel=2.0e-5, abs=2.0e-8)


def test_plvira_recovers_identifiable_parabola_from_exact_stencil():
    polygons = _uniform_stencil()
    exact_angle = 0.43
    curvature = 0.2
    exact_shift = 0.07
    fractions = _exact_fractions(polygons, exact_angle, curvature, exact_shift)
    reconstruction = reconstruct_plvira_exact_curvature_oracle(
        polygons,
        fractions,
        curvature,
        cell_size=1.0,
        gradient_tolerance=1.0e-11,
    )
    assert reconstruction.optimizer_success, reconstruction.optimizer_message
    assert _angle_error(reconstruction.angle, exact_angle) < 2.0e-7
    assert reconstruction.shift == pytest.approx(exact_shift, abs=2.0e-7)
    assert reconstruction.objective < 1.0e-13
    assert reconstruction.curvature_source == "exact-curvature-oracle"
    assert reconstruction.ghf_diagnostics is None


def test_operational_plvira_uses_cartesian_ghf_curvature():
    polygons = _uniform_stencil()
    curvature = 0.2
    exact_shift = 0.07
    grid = _exact_cartesian_grid(15, 0.0, curvature, exact_shift)
    center = len(grid) // 2
    fractions = [row[center - 1 : center + 2] for row in grid[center - 1 : center + 2]]
    reconstruction = reconstruct_plvira(
        polygons,
        fractions,
        cartesian_fractions=grid,
        target_index=(center, center),
        cell_size=1.0,
        gradient_tolerance=1.0e-11,
    )
    assert reconstruction.optimizer_success, reconstruction.optimizer_message
    assert reconstruction.curvature == pytest.approx(curvature, abs=2.0e-13)
    assert _angle_error(reconstruction.angle, 0.0) < 2.0e-7
    assert reconstruction.shift == pytest.approx(exact_shift, abs=2.0e-7)
    assert reconstruction.objective < 1.0e-13
    assert reconstruction.curvature_source == "cartesian-ghf"
    assert reconstruction.ghf_diagnostics.method == "height_function"


def test_operational_plvira_rejects_mismatched_local_and_ghf_fractions():
    polygons = _uniform_stencil()
    grid = _exact_cartesian_grid(15, 0.0, 0.2, 0.07)
    center = len(grid) // 2
    fractions = np.asarray(
        [row[center - 1 : center + 2] for row in grid[center - 1 : center + 2]]
    )
    fractions[0, 1] += 1.0e-15
    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_plvira(
            polygons,
            fractions.tolist(),
            cartesian_fractions=grid,
            target_index=(center, center),
            cell_size=1.0,
        )


def test_oracle_mode_rejects_perturbed_cells():
    polygons = _uniform_stencil()
    polygons[1][1][0] = (-0.45, -0.5)
    fractions = np.full((3, 3), 0.5).tolist()
    with pytest.raises(ValueError, match="Cartesian"):
        reconstruct_plvira_exact_curvature_oracle(
            polygons, fractions, 0.0, cell_size=1.0, initial_angle=0.0
        )
