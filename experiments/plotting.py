"""Shared plotting helpers for experiment figures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullFormatter


PAPER_SERIF_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.8,
    "font.weight": "normal",
    "axes.labelsize": 9.8,
    "axes.labelweight": "normal",
    "axes.titlesize": 10.2,
    "axes.titleweight": "normal",
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

PAPER_METHOD_MARKERS = {
    "Youngs": "o",
    "ELVIRA": "s",
    "LVIRA": "D",
    "safe_linear": "^",
    "linear": "v",
    "linear+C0": "P",
    "linear+corner": "X",
    "safe_circle": "^",
    "circular": "v",
    "circular+C0": "P",
    "circular+corner": "X",
    "circular+corner+C0": "P",
}

# Approved paper-wide method encoding. Colors identify method families, while
# markers and line styles distinguish closely related variants within a family.
PAPER_METHOD_COLORS = {
    "Youngs": "#D94F9D",
    "ELVIRA": "#00A6C8",
    "LVIRA": "#84B547",
    "safe_linear": "#74A9CF",
    "linear": "#2F6FA3",
    "linear+C0": "#2F6FA3",
    "linear+corner": "#173F73",
    "safe_circle": "#E6AB02",
    "circular": "#D55E00",
    "circular+C0": "#D55E00",
    "circular+corner": "#B91C1C",
    "circular+corner+C0": "#B91C1C",
}

PAPER_METHOD_LINESTYLES = {
    "Youngs": "-",
    "ELVIRA": "--",
    "LVIRA": "-.",
    "safe_linear": "--",
    "linear": "-",
    "linear+C0": ":",
    "linear+corner": "-",
    "safe_circle": "--",
    "circular": "-",
    "circular+C0": ":",
    "circular+corner": "-",
    "circular+corner+C0": ":",
}

PAPER_HIGH_ORDER_COLORS = {
    "plvira": "#2D7D64",
    "pcic_center": "#7C5AA6",
    "quasi": "#4B5563",
    "ours_per_cell": "#E6AB02",
    "ours_graph": "#D55E00",
    "ours_c0": "#D55E00",
}

PAPER_METRIC_LABELS = {
    "hausdorff": "Hausdorff error",
    "facet_gap": "Facet-gap error",
    "curvature_error": "Curvature MAE",
    "tangent_error": "Tangent error",
}


def apply_paper_serif_style() -> None:
    """Apply the serif typography used by the approved Appendix B panels."""

    mpl.rcParams.update(PAPER_SERIF_RCPARAMS)


def paper_markers_by_label(display_labels: Mapping[str, str]) -> dict[str, str]:
    """Map method markers to the public labels used in figure legends."""

    return {
        display_labels.get(method, method): marker
        for method, marker in PAPER_METHOD_MARKERS.items()
    }


def readable_resolution_ticks(
    values: Sequence[float],
    *,
    max_ticks: int = 5,
) -> tuple[float, ...]:
    """Choose evenly spaced log-axis labels without changing plotted samples."""

    ticks = tuple(sorted({float(value) for value in values if value > 0.0}))
    if len(ticks) > max_ticks:
        log_min = math.log(ticks[0])
        log_max = math.log(ticks[-1])
        targets = (
            log_min + index * (log_max - log_min) / (max_ticks - 1)
            for index in range(max_ticks)
        )
        selected = []
        for target in targets:
            nearest = min(ticks, key=lambda value: abs(math.log(value) - target))
            if nearest not in selected:
                selected.append(nearest)
    else:
        selected = list(ticks)

    separated = [selected[0]]
    for value in selected[1:]:
        if value / separated[-1] < 1.25:
            if len(separated) > 1:
                separated[-1] = value
            continue
        separated.append(value)
    return tuple(separated)


def apply_paper_metric_axis_style(
    axis: Axes,
    metric: str,
    x_mode: str,
    *,
    markers_by_label: Mapping[str, str],
) -> None:
    """Apply the approved Appendix B axis language without changing curve data."""

    resolution_ticks = (
        [value for value in axis.get_xticks() if value > 0.0]
        if x_mode == "resolution"
        else []
    )
    for line in axis.get_lines():
        marker = markers_by_label.get(line.get_label())
        if marker is None:
            line.set_linewidth(0.9)
            continue
        line.set_marker(marker)
        line.set_markersize(3.6)
        line.set_linewidth(1.2)
    for collection in axis.collections:
        collection.set_alpha(0.10)
        collection.set_linewidth(0.0)

    axis.set_ylabel(
        PAPER_METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
        labelpad=9.0,
    )
    axis.grid(False)
    axis.grid(True, which="major", color="#d1d5db", linewidth=0.45)
    axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=8.5,
        width=0.6,
        length=3.0,
    )
    axis.tick_params(axis="both", which="minor", width=0.45, length=1.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    if x_mode == "resolution":
        resolution_ticks = readable_resolution_ticks(resolution_ticks)
        axis.set_xscale("log")
        axis.set_xticks(
            resolution_ticks,
            labels=[str(int(round(value))) for value in resolution_ticks],
        )
        axis.xaxis.set_minor_formatter(NullFormatter())


def contiguous_true_runs(mask: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    """Return half-open index ranges for contiguous true values."""

    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return tuple(runs)


def plot_series_in_y_window(
    axis: Axes,
    x_values,
    median,
    q25,
    q75,
    *,
    y_window: tuple[float, float],
    label: str,
    line_kwargs: Mapping,
    fill_kwargs: Mapping,
) -> None:
    """Draw one series without connecting samples across an omitted y-range."""

    import numpy as np

    x_values = np.asarray(x_values, dtype=float)
    median = np.asarray(median, dtype=float)
    q25 = np.asarray(q25, dtype=float)
    q75 = np.asarray(q75, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(median)
    lower, upper = sorted(y_window)
    visible = finite & (median >= lower) & (median <= upper)
    labelled = False
    for start, stop in contiguous_true_runs(visible):
        axis.plot(
            x_values[start:stop],
            median[start:stop],
            label=label if not labelled else "_nolegend_",
            **line_kwargs,
        )
        labelled = True

    finite_band = finite & np.isfinite(q25) & np.isfinite(q75)
    visible_band = finite_band & (q75 >= lower) & (q25 <= upper)
    clipped_q25 = np.maximum(q25, lower)
    clipped_q75 = np.minimum(q75, upper)
    for start, stop in contiguous_true_runs(visible_band):
        if stop - start == 1:
            bar_kwargs = {
                key: value
                for key, value in fill_kwargs.items()
                if key in {"alpha", "color", "zorder"}
            }
            axis.vlines(
                x_values[start],
                clipped_q25[start],
                clipped_q75[start],
                linewidth=2.0,
                **bar_kwargs,
            )
            continue
        axis.fill_between(
            x_values[start:stop],
            clipped_q25[start:stop],
            clipped_q75[start:stop],
            interpolate=False,
            **fill_kwargs,
        )


def draw_log_axis_break_marks(
    upper: Axes,
    lower: Axes,
    *,
    color: str = "#111827",
    linewidth: float = 0.8,
    size: float = 0.014,
) -> None:
    """Style paired axes and add conventional diagonal omission marks."""

    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    lower.tick_params(axis="x", which="both", top=False)
    style = {"color": color, "clip_on": False, "linewidth": linewidth}
    for x_coordinate in (0.0, 1.0):
        upper.plot(
            (x_coordinate - size, x_coordinate + size),
            (-size, size),
            transform=upper.transAxes,
            **style,
        )
        lower.plot(
            (x_coordinate - size, x_coordinate + size),
            (1.0 - size, 1.0 + size),
            transform=lower.transAxes,
            **style,
        )


@dataclass(frozen=True)
class ConvergenceOrderTriangle:
    """Artists and data-space vertices for a convergence-order marker."""

    order: float
    vertices: Tuple[Tuple[float, float], ...]
    line: Line2D
    order_text: Text
    run_text: Optional[Text]


def _log_interpolate(bounds: Tuple[float, float], fraction: float) -> float:
    lower, upper = bounds
    if lower <= 0.0 or upper <= 0.0:
        raise ValueError("convergence triangles require positive axis limits")
    return math.exp(math.log(lower) + fraction * (math.log(upper) - math.log(lower)))


def add_convergence_order_triangle(
    axis: Axes,
    order: float,
    *,
    anchor: Tuple[float, float] = (0.76, 0.72),
    width: float = 0.13,
    trend: Literal["decreasing", "increasing"] = "decreasing",
    order_label: Optional[str] = None,
    run_label: Optional[str] = "1",
    color: str = "#4b5563",
    linewidth: float = 0.9,
    fontsize: float = 8.6,
    zorder: float = 5.0,
) -> ConvergenceOrderTriangle:
    """Add a publication-style order triangle to log-log axes.

    ``anchor`` and ``width`` are axes fractions, so the marker remains compact
    when plot ranges change. The triangle itself is constructed in data space:
    its rise-to-run ratio in logarithmic coordinates is exactly ``order``.
    ``trend="decreasing"`` represents errors proportional to ``x**(-order)``;
    use ``"increasing"`` for errors proportional to ``x**order``.

    The helper preserves the existing axis limits and returns the artists and
    vertices to support later styling or quantitative tests.
    """

    if axis.get_xscale() != "log" or axis.get_yscale() != "log":
        raise ValueError("convergence triangles require log-log axes")
    if not math.isfinite(order) or order <= 0.0:
        raise ValueError(f"order must be positive and finite, got {order}")
    if trend not in {"decreasing", "increasing"}:
        raise ValueError(f"unsupported trend: {trend}")
    if len(anchor) != 2 or not all(math.isfinite(value) for value in anchor):
        raise ValueError(f"anchor must contain two finite fractions, got {anchor}")
    if not 0.0 < width < 1.0:
        raise ValueError(f"width must lie in (0, 1), got {width}")
    if not 0.0 <= anchor[0] < anchor[0] + width <= 1.0:
        raise ValueError("anchor x-coordinate and width must remain inside the axes")
    if not 0.0 <= anchor[1] <= 1.0:
        raise ValueError("anchor y-coordinate must remain inside the axes")

    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    x_left = _log_interpolate(x_limits, anchor[0])
    x_right = _log_interpolate(x_limits, anchor[0] + width)
    y_base = _log_interpolate(y_limits, anchor[1])

    exponent = -order if trend == "decreasing" else order
    log_y_change = exponent * math.log(x_right / x_left)
    x_high = x_left if log_y_change < 0.0 else x_right
    x_low = x_right if log_y_change < 0.0 else x_left
    y_high = y_base * math.exp(abs(log_y_change))
    y_min, y_max = sorted(y_limits)
    if not y_min <= y_high <= y_max:
        raise ValueError(
            "triangle extends beyond the y-axis; lower the anchor or reduce the width"
        )

    vertices = (
        (x_low, y_base),
        (x_high, y_base),
        (x_high, y_high),
        (x_low, y_base),
    )
    (line,) = axis.plot(
        [point[0] for point in vertices],
        [point[1] for point in vertices],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=zorder,
    )

    vertical_offset = -4.0 if x_high == x_left else 4.0
    horizontal_alignment = "right" if x_high == x_left else "left"
    order_text = axis.annotate(
        order_label if order_label is not None else f"{order:g}",
        (x_high, math.sqrt(y_base * y_high)),
        xytext=(vertical_offset, 0.0),
        textcoords="offset points",
        ha=horizontal_alignment,
        va="center",
        color=color,
        fontsize=fontsize,
        zorder=zorder,
    )
    run_text = None
    if run_label is not None:
        run_text = axis.annotate(
            run_label,
            (math.sqrt(x_left * x_right), y_base),
            xytext=(0.0, -3.0),
            textcoords="offset points",
            ha="center",
            va="top",
            color=color,
            fontsize=fontsize,
            zorder=zorder,
        )

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    return ConvergenceOrderTriangle(
        order=order,
        vertices=vertices,
        line=line,
        order_text=order_text,
        run_text=run_text,
    )
