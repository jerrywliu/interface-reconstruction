import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from experiments.plotting import add_convergence_order_triangle


def _log_axes():
    figure, axis = plt.subplots()
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(16.0, 256.0)
    axis.set_ylim(1.0e-8, 1.0e-1)
    return figure, axis


def test_decreasing_triangle_encodes_requested_log_log_order():
    figure, axis = _log_axes()
    triangle = add_convergence_order_triangle(axis, 2.5)

    (x_low, y_base), (x_high, _), (_, y_high), _ = triangle.vertices
    log_run = abs(math.log(x_high / x_low))
    log_rise = math.log(y_high / y_base)

    assert log_rise / log_run == pytest.approx(2.5)
    assert triangle.order_text.get_text() == "2.5"
    assert triangle.run_text is not None
    assert triangle.run_text.get_text() == "1"
    plt.close(figure)


def test_increasing_triangle_moves_vertical_leg_to_the_high_x_side():
    figure, axis = _log_axes()
    triangle = add_convergence_order_triangle(axis, 1.75, trend="increasing")

    (x_low, y_base), (x_high, _), (x_vertical, y_high), _ = triangle.vertices

    assert x_vertical == pytest.approx(x_high)
    assert math.log(y_high / y_base) / math.log(x_high / x_low) == pytest.approx(1.75)
    plt.close(figure)


def test_triangle_preserves_limits_and_rejects_linear_axes():
    figure, axis = _log_axes()
    original_limits = (axis.get_xlim(), axis.get_ylim())

    add_convergence_order_triangle(axis, 2.0, order_label=r"$p=2$")

    assert axis.get_xlim() == pytest.approx(original_limits[0])
    assert axis.get_ylim() == pytest.approx(original_limits[1])
    plt.close(figure)

    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="log-log"):
        add_convergence_order_triangle(axis, 2.0)
    plt.close(figure)
