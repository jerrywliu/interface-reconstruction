import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plotting import (
    add_convergence_order_triangle,
    add_shared_vertical_metric_label,
    align_axes_below_figure_legend,
    contiguous_true_runs,
    paper_metric_panel_title,
    plot_series_in_y_window,
    readable_resolution_ticks,
)


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


def test_contiguous_true_runs_returns_half_open_ranges():
    assert contiguous_true_runs([True, True, False, True, False]) == (
        (0, 2),
        (3, 4),
    )


def test_readable_resolution_ticks_thins_dense_labels_without_moving_endpoints():
    assert readable_resolution_ticks((32, 50, 64, 100, 128, 150, 256, 300)) == (
        32.0,
        50.0,
        100.0,
        150.0,
        300.0,
    )
    assert readable_resolution_ticks((256, 300, 512)) == (256.0, 512.0)
    assert readable_resolution_ticks((50, 64, 100, 128, 150)) == (
        50.0,
        64.0,
        100.0,
        150.0,
    )


def test_metric_titles_use_canonical_manuscript_labels():
    assert paper_metric_panel_title("hausdorff") == "Hausdorff error"
    assert paper_metric_panel_title("facet_gap") == "Facet-gap error"
    assert (
        paper_metric_panel_title("facet_gap", x_phrase="cells per side")
        == "Facet-gap error vs cells per side"
    )


def test_shared_metric_label_uses_requested_visual_gap():
    figure, axes = plt.subplots(2, 1, figsize=(4.0, 4.0))
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylim(1.0e-10, 1.0e-2)
        axis.set_ylabel("")

    label = add_shared_vertical_metric_label(
        figure,
        axes,
        "hausdorff",
        fontsize=10.0,
        gap_points=5.0,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tick_left = min(
        tick.get_window_extent(renderer).x0
        for axis in axes
        for tick in axis.get_yticklabels()
        if tick.get_visible() and tick.get_text()
    )
    label_right = label.get_window_extent(renderer).x1

    assert label.get_text() == "Hausdorff error"
    assert tick_left - label_right == pytest.approx(5.0 * figure.dpi / 72.0)
    plt.close(figure)


def test_figure_legend_uses_requested_visual_gap_above_titles():
    figure, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    for axis in axes:
        axis.plot([0.0, 1.0], [0.0, 1.0], label="method")
        axis.set_title("Panel title")
    legend = figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center")
    figure.subplots_adjust(top=0.95)

    align_axes_below_figure_legend(
        figure,
        legend,
        axes,
        gap_points=7.0,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title_top = max(axis.title.get_window_extent(renderer).y1 for axis in axes)
    legend_bottom = legend.get_window_extent(renderer).y0

    assert legend_bottom - title_top == pytest.approx(
        7.0 * figure.dpi / 72.0,
        abs=1.0,
    )
    plt.close(figure)


def test_windowed_series_does_not_connect_across_omitted_range():
    figure, axis = plt.subplots()
    values = np.asarray([1.0e-9, 1.0e-4, 1.0e-3, 2.0e-9])
    plot_series_in_y_window(
        axis,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        values,
        values,
        values,
        y_window=(1.0e-10, 1.0e-8),
        label="method",
        line_kwargs={"color": "black", "marker": "o"},
        fill_kwargs={"alpha": 0.0},
    )

    assert len(axis.lines) == 2
    assert [line.get_xdata().tolist() for line in axis.lines] == [[1.0], [4.0]]
    assert [line.get_label() for line in axis.lines] == ["method", "_nolegend_"]
    plt.close(figure)


def test_windowed_iqr_does_not_interpolate_to_an_out_of_window_sample():
    figure, axis = plt.subplots()
    plot_series_in_y_window(
        axis,
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0e-3, 1.0e-3, 1.0e-9]),
        np.asarray([1.0e-3, 5.0e-4, 5.0e-10]),
        np.asarray([3.0e-3, 2.0e-3, 2.0e-9]),
        y_window=(1.0e-4, 1.0e-2),
        label="method",
        line_kwargs={"color": "black", "marker": "o"},
        fill_kwargs={"alpha": 0.1, "color": "black"},
    )

    ribbon_paths = axis.collections[0].get_paths()
    assert ribbon_paths
    assert max(vertex[0] for path in ribbon_paths for vertex in path.vertices) == 2.0
    plt.close(figure)


def test_iqr_crossing_omitted_range_is_visible_in_both_retained_bands():
    x_values = np.asarray([64.0])
    medians = np.asarray([2.3e-1])
    q1_values = np.asarray([8.4e-9])
    q3_values = np.asarray([4.0e-1])
    figure, (upper, lower) = plt.subplots(2, 1)

    for axis, y_window in ((upper, (3.0e-3, 1.0)), (lower, (1.0e-10, 4.0e-8))):
        plot_series_in_y_window(
            axis,
            x_values,
            medians,
            q1_values,
            q3_values,
            y_window=y_window,
            label="method",
            line_kwargs={"color": "black", "marker": "o"},
            fill_kwargs={"alpha": 0.1, "color": "black"},
        )

    assert len(upper.collections) == 1
    assert len(lower.collections) == 1
    plt.close(figure)
