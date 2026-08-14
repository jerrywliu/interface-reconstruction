"""Shared plotting helpers for experiment figures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text


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
