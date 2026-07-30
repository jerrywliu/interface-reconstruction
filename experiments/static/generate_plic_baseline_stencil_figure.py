#!/usr/bin/env python3
"""Generate the perfect-reconstruction PLIC comparison on a real line stencil."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from experiments.static.lines import RANDOM_SEED
from main.geoms.linear_facet import getPolyLineIntersects
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.polys.base_polygon import BasePolygon
from util.config import read_yaml
from util.initialize.areas import initializeLine
from util.initialize.mesh_factory import apply_mesh_overrides, make_points_from_config
from util.metrics.metrics import hausdorffFacets


WHITE = (1.0, 1.0, 1.0)
TOPO_FULL = tuple(value / 255.0 for value in (171, 201, 234))
EDGE = (0.37, 0.40, 0.45)
TRUE_LINE = (0.30, 0.30, 0.30)
METHOD_COLORS = {
    "Youngs": "#B14E5E",
    "ELVIRA": "#B3811B",
    "LVIRA": "#2D7D64",
}

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{txfonts}",
        "svg.fonttype": "none",
    }
)


def _line_case_params(case_index: int, num_cases: int = 25):
    rng = np.random.default_rng(RANDOM_SEED)
    angles = np.linspace(0.0, 2.0 * math.pi, num_cases + 1)[:-1]
    for index, angle in enumerate(angles):
        x1, y1 = rng.uniform(50, 51), rng.uniform(50, 51)
        x2 = x1 + 0.2
        y2 = y1 + math.tan(angle) * (x2 - x1)
        if index == case_index:
            return [float(x1), float(y1)], [float(x2), float(y2)], float(angle)
    raise ValueError(f"case_index={case_index} is outside [0, {num_cases - 1}]")


def _extract_stencil(
    *,
    case_index: int,
    cell_x: int,
    cell_y: int,
    resolution: float,
    wiggle: float,
    seed: int,
):
    config = read_yaml("config/static/line.yaml")
    mesh_config = apply_mesh_overrides(
        config["MESH"],
        resolution=resolution,
        mesh_type="perturbed_cartesian",
        perturb_wiggle=wiggle,
        perturb_seed=seed,
    )
    mesh = MergeMesh(
        make_points_from_config(mesh_config), config["GEOMS"]["THRESHOLD"]
    )
    p1, p2, angle = _line_case_params(case_index)
    mesh.initializeFractions(initializeLine(mesh, p1, p2))

    if not (0 < cell_x < len(mesh.polys) - 1):
        raise ValueError(f"cell_x={cell_x} does not admit a 3x3 stencil")
    if not (0 < cell_y < len(mesh.polys[0]) - 1):
        raise ValueError(f"cell_y={cell_y} does not admit a 3x3 stencil")

    center = mesh.polys[cell_x][cell_y]
    if not center.isMixed():
        raise ValueError(f"selected center cell ({cell_x}, {cell_y}) is not mixed")
    stencil = mesh.get3x3Stencil(cell_x, cell_y)
    center.set3x3Stencil(stencil)
    return stencil, center, p1, p2, angle


def _mix_with_white(color, fraction: float):
    fraction = max(0.0, min(1.0, float(fraction)))
    return tuple(
        (1.0 - fraction) * WHITE[index] + fraction * color[index]
        for index in range(3)
    )


def _full_line_from_segment(p_left, p_right, bounds):
    xmin, xmax, ymin, ymax = bounds
    midpoint = [
        0.5 * (p_left[0] + p_right[0]),
        0.5 * (p_left[1] + p_right[1]),
    ]
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    norm = math.hypot(dx, dy)
    dx /= norm
    dy /= norm
    span = 2.5 * math.hypot(xmax - xmin, ymax - ymin)
    return (
        [midpoint[0] - span * dx, midpoint[1] - span * dy],
        [midpoint[0] + span * dx, midpoint[1] + span * dy],
    )


def _scientific_tex(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    return rf"{mantissa:.1f}\times 10^{{{exponent}}}"


def _plot_panel(
    ax,
    stencil: list[list[BasePolygon]],
    panel_label: str,
    method: str,
    bounds,
    true_line,
    recon_line,
    normalized_error: float,
):
    xmin, xmax, ymin, ymax = bounds
    for i in range(3):
        for j in range(3):
            poly = stencil[i][j]
            ax.add_patch(
                MplPolygon(
                    poly.points,
                    closed=True,
                    facecolor=_mix_with_white(TOPO_FULL, poly.getFraction()),
                    edgecolor=EDGE,
                    linewidth=0.9,
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
            linewidth=2.0,
            joinstyle="round",
            zorder=3,
        )
    )

    ax.plot(
        [true_line[0][0], true_line[1][0]],
        [true_line[0][1], true_line[1][1]],
        linestyle=(0, (2.4, 2.0)),
        color=TRUE_LINE,
        linewidth=1.25,
        zorder=4,
    )
    ax.plot(
        [recon_line[0][0], recon_line[1][0]],
        [recon_line[0][1], recon_line[1][1]],
        linestyle="-",
        color=METHOD_COLORS[method],
        linewidth=1.75,
        solid_capstyle="round",
        zorder=5,
    )

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        rf"\textbf{{({panel_label}) {method}}}"
        + "\n"
        + rf"$d_H/h={_scientific_tex(normalized_error)}$",
        fontsize=10.5,
        pad=5,
    )


def build_figure(
    out_base: Path,
    *,
    case_index: int,
    cell_x: int,
    cell_y: int,
    resolution: float,
    wiggle: float,
    seed: int,
):
    stencil, center, p1, p2, angle = _extract_stencil(
        case_index=case_index,
        cell_x=cell_x,
        cell_y=cell_y,
        resolution=resolution,
        wiggle=wiggle,
        seed=seed,
    )
    facets = {
        "Youngs": center.runYoungs(ret=True),
        "ELVIRA": center.runELVIRA(ret=True),
        "LVIRA": center.runLVIRA(ret=True),
    }

    true_intersections = getPolyLineIntersects(center.points, p1, p2)
    if len(true_intersections) < 2:
        raise RuntimeError("true line does not cross the selected center cell twice")
    true_facet = LinearFacet(true_intersections[0], true_intersections[-1])
    errors = {
        method: float(hausdorffFacets(true_facet, facet))
        for method, facet in facets.items()
    }
    nominal_h = 1.0 / resolution
    normalized_errors = {
        method: error / nominal_h for method, error in errors.items()
    }

    xs = [point[0] for row in stencil for poly in row for point in poly.points]
    ys = [point[1] for row in stencil for poly in row for point in poly.points]
    pad = 0.055 * max(max(xs) - min(xs), max(ys) - min(ys))
    bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    true_line = _full_line_from_segment(p1, p2, bounds)

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.05))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.82, bottom=0.17, wspace=0.07)
    for ax, panel_label, method in zip(axes, "abc", facets):
        facet = facets[method]
        _plot_panel(
            ax,
            stencil,
            panel_label,
            method,
            bounds,
            true_line=true_line,
            recon_line=(facet.pLeft, facet.pRight),
            normalized_error=normalized_errors[method],
        )

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TRUE_LINE,
                linewidth=1.25,
                linestyle=(0, (2.4, 2.0)),
                label="true interface",
            ),
            Line2D(
                [0],
                [0],
                color=EDGE,
                linewidth=1.75,
                linestyle="-",
                label="reconstructed center-cell facet",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.8,
        columnspacing=1.8,
        fontsize=8.5,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source": "extracted from the perturbed Cartesian line benchmark",
        "case_index": case_index,
        "center_cell": [cell_x, cell_y],
        "resolution": resolution,
        "cells_per_side": int(round(100 * resolution)),
        "perturbation_magnitude": wiggle,
        "mesh_seed": seed,
        "line_angle_radians": angle,
        "line_angle_degrees": angle * 180.0 / math.pi,
        "line_points": [p1, p2],
        "center_fraction": float(center.getFraction()),
        "stencil_fractions": [
            [float(stencil[i][j].getFraction()) for j in range(3)]
            for i in range(3)
        ],
        "center_cell_hausdorff": errors,
        "center_cell_hausdorff_over_h": normalized_errors,
    }
    with out_base.with_name(f"{out_base.name}_data.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the perturbed-grid perfect-reconstruction comparison."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=4)
    parser.add_argument("--cell-x", type=int, default=14)
    parser.add_argument("--cell-y", type=int, default=13)
    parser.add_argument("--resolution", type=float, default=0.32)
    parser.add_argument("--wiggle", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = build_figure(
        args.out,
        case_index=args.case_index,
        cell_x=args.cell_x,
        cell_y=args.cell_y,
        resolution=args.resolution,
        wiggle=args.wiggle,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))
    for suffix in (".svg", ".pdf", ".png"):
        print(f"Wrote {args.out.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
