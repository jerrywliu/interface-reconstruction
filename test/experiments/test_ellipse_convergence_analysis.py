import math

import pytest

from experiments.submission.analyze_ellipse_convergence import (
    fit_power_law,
    numerical_floor,
    select_fit_points,
)


def test_fit_power_law_reports_positive_order_against_cell_width():
    h_values = [3.0, 2.0, 1.0, 0.5]
    errors = [4.25 * h**2.75 for h in h_values]

    fit = fit_power_law(h_values, errors)

    assert fit["order"] == pytest.approx(2.75, abs=1.0e-12)
    assert fit["intercept_log_e"] == pytest.approx(math.log(4.25), abs=1.0e-12)
    assert fit["prefactor"] == pytest.approx(4.25, abs=1.0e-12)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1.0e-12)


def test_floor_points_are_excluded_before_finest_window_selection():
    points = [
        {
            "N": n,
            "median_error": error,
            "numerical_floor": numerical_floor("hausdorff", 100.0 / n),
        }
        for n, error in (
            (32, 1.0e-3),
            (50, 5.0e-4),
            (64, 2.0e-4),
            (100, 8.0e-5),
            (128, 4.0e-5),
            (150, 1.0e-12),
        )
    ]

    selected, excluded = select_fit_points(points, max_points=4)

    assert [point["N"] for point in selected] == [50, 64, 100, 128]
    assert excluded == [
        {"N": 32, "reason": "outside_fit_window"},
        {"N": 150, "reason": "numerical_floor"},
    ]
