#!/usr/bin/env python3
"""
Generate an appendix figure illustrating Youngs / ELVIRA / LVIRA on a perturbed
3x3 Cartesian stencil using a real regression case from the codebase.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from main.geoms.geoms import getPolyLineArea
from main.structs.polys.base_polygon import BasePolygon


WHITE = (1.0, 1.0, 1.0)
TOPOfull = tuple(v / 255.0 for v in (171, 201, 234))
EDGE = (0.37, 0.40, 0.45)
TRUE_LINE = (0.35, 0.35, 0.35)
RECON_LINE = (0.08, 0.08, 0.08)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{txfonts}",
    }
)


def _make_perturbed_stencil(seed: int) -> list[list[BasePolygon]]:
    rng = random.Random(seed)
    vertices = []
    for i in range(4):
        column = []
        for j in range(4):
            x = float(i)
            y = float(j)
            if 0 < i < 3 and 0 < j < 3:
                x += rng.uniform(-0.18, 0.18)
                y += rng.uniform(-0.18, 0.18)
            column.append([x, y])
        vertices.append(column)

    return [
        [
            BasePolygon(
                [
                    vertices[i][j],
                    vertices[i + 1][j],
                    vertices[i + 1][j + 1],
                    vertices[i][j + 1],
                ]
            )
            for j in range(3)
        ]
        for i in range(3)
    ]


def _line_points(theta: float, offset: float, ref=(1.5, 1.5), span: float = 10.0):
    normal = [math.cos(theta), math.sin(theta)]
    tangent = [-normal[1], normal[0]]
    point = [ref[0] + offset * normal[0], ref[1] + offset * normal[1]]
    l1 = [point[0] - span * tangent[0], point[1] - span * tangent[1]]
    l2 = [point[0] + span * tangent[0], point[1] + span * tangent[1]]
    return l1, l2


def _mix_with_white(color, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple((1.0 - t) * WHITE[i] + t * color[i] for i in range(3))


def _full_line_from_segment(p_left, p_right, bounds):
    xmin, xmax, ymin, ymax = bounds
    mx = 0.5 * (p_left[0] + p_right[0])
    my = 0.5 * (p_left[1] + p_right[1])
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    norm = math.hypot(dx, dy)
    dx /= norm
    dy /= norm
    span = 2.5 * math.hypot(xmax - xmin, ymax - ymin)
    return ([mx - span * dx, my - span * dy], [mx + span * dx, my + span * dy])


def _plot_panel(ax, stencil, title, bounds, true_line=None, recon_line=None):
    xmin, xmax, ymin, ymax = bounds
    for i in range(3):
        for j in range(3):
            poly = stencil[i][j]
            fill = _mix_with_white(TOPOfull, poly.getFraction())
            ax.add_patch(
                MplPolygon(
                    poly.points,
                    closed=True,
                    facecolor=fill,
                    edgecolor=EDGE,
                    linewidth=1.0,
                    joinstyle="round",
                )
            )

    center_poly = stencil[1][1]
    ax.add_patch(
        MplPolygon(
            center_poly.points,
            closed=True,
            facecolor="none",
            edgecolor=EDGE,
            linewidth=1.9,
            joinstyle="round",
            zorder=3,
        )
    )

    if true_line is not None:
        ax.plot(
            [true_line[0][0], true_line[1][0]],
            [true_line[0][1], true_line[1][1]],
            linestyle=(0, (2.2, 2.2)),
            color=TRUE_LINE,
            linewidth=1.25,
            zorder=4,
        )
    if recon_line is not None:
        ax.plot(
            [recon_line[0][0], recon_line[1][0]],
            [recon_line[0][1], recon_line[1][1]],
            linestyle="-",
            color=RECON_LINE,
            linewidth=1.35,
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(rf"\textbf{{{title}}}", fontsize=10.5, pad=6)


def build_figure(out_base: Path):
    # Chosen from a real perturbed-stencil search so the methods separate visibly
    # while the center cell remains meaningfully mixed.
    seed = 2
    theta = 2.0828357186043696
    offset = 0.017113478981178043

    stencil = _make_perturbed_stencil(seed=seed)
    true_l1, true_l2 = _line_points(theta=theta, offset=offset)
    for row in stencil:
        for poly in row:
            fraction = getPolyLineArea(poly.points, true_l1, true_l2) / poly.getMaxArea()
            poly.setFraction(fraction)

    center = stencil[1][1]
    center.set3x3Stencil(stencil)
    youngs = center.runYoungs(ret=True)
    elvira = center.runELVIRA(ret=True)
    lvira = center.runLVIRA(ret=True)

    xs = [p[0] for row in stencil for poly in row for p in poly.points]
    ys = [p[1] for row in stencil for poly in row for p in poly.points]
    pad = 0.16
    bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    youngs_line = _full_line_from_segment(youngs.pLeft, youngs.pRight, bounds)
    elvira_line = _full_line_from_segment(elvira.pLeft, elvira.pRight, bounds)
    lvira_line = _full_line_from_segment(lvira.pLeft, lvira.pRight, bounds)

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.75), constrained_layout=True)
    _plot_panel(
        axes[0],
        stencil,
        "Youngs",
        bounds,
        true_line=(true_l1, true_l2),
        recon_line=(youngs.pLeft, youngs.pRight),
    )
    _plot_panel(
        axes[1],
        stencil,
        "ELVIRA",
        bounds,
        true_line=(true_l1, true_l2),
        recon_line=(elvira.pLeft, elvira.pRight),
    )
    _plot_panel(
        axes[2],
        stencil,
        "LVIRA",
        bounds,
        true_line=(true_l1, true_l2),
        recon_line=(lvira.pLeft, lvira.pRight),
    )

    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Appendix A PLIC baseline stencil figure.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output base path without suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(args.out)
    print(f"Wrote {args.out.with_suffix('.pdf')}")
    print(f"Wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
