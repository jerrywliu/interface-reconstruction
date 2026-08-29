"""Shared plotting helpers for experiment figures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Optional, Tuple

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullFormatter


PAPER_SERIF_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "font.weight": "normal",
    "axes.labelsize": 8.5,
    "axes.labelweight": "normal",
    "axes.titlesize": 9.0,
    "axes.titleweight": "normal",
    "legend.fontsize": 7.2,
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
    "linear+corner": "P",
    "safe_circle": "<",
    "circular": ">",
    "circular+corner": "X",
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

    axis.set_ylabel(PAPER_METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
    axis.grid(False)
    axis.grid(True, which="major", color="#d1d5db", linewidth=0.45)
    axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=7.5,
        width=0.6,
        length=3.0,
    )
    axis.tick_params(axis="both", which="minor", width=0.45, length=1.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    if x_mode == "resolution":
        axis.set_xscale("log")
        axis.set_xticks(
            resolution_ticks,
            labels=[str(int(round(value))) for value in resolution_ticks],
        )
        axis.xaxis.set_minor_formatter(NullFormatter())


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
    fontsize: float = 7.0,
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
