#!/usr/bin/env python3
"""Generate a data-driven staged reconstruction figure from a Zalesak case."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Patch, Polygon
from matplotlib.collections import PatchCollection

from experiments.static.zalesak import RANDOM_SEED, initialize_zalesak
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh
from util.config import read_yaml
from util.initialize.mesh_factory import apply_mesh_overrides, make_points_from_config


STAGE_ORDER = [
    "volume_fractions",
    "topology",
    "linear",
    "linear_corners",
    "circular",
    "final",
]

STAGE_TITLES = {
    "volume_fractions": "(a) Volume fractions",
    "topology": "(b) Oriented topology",
    "linear": "(c) Linear facets",
    "linear_corners": "(d) Linear corners",
    "circular": "(e) Circular arcs",
    "final": "(f) Curved corners",
}

STROKE_COLORS = {
    "linear": "#356D9A",
    "linear_corner": "#B14E5E",
    "circular": "#2D7D64",
    "curved_corner": "#80558C",
    "fallback": "#B3811B",
}

FILL_COLORS = {
    "linear": "#DCE8F1",
    "linear_corner": "#F2DDE1",
    "circular": "#DCECE5",
    "curved_corner": "#E9DFEC",
    "fallback": "#F4E7C7",
}

EMPTY_FILL = "#FBFBFA"
FULL_FILL = "#E8ECEB"
MIXED_FILL = "#617078"
MESH_EDGE = "#C6CCCA"
TOPOLOGY_COLOR = "#263F4A"
MERGE_COLOR = "#D1843D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, default=22)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--wiggle", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--slot-width", type=float, default=5.0)
    parser.add_argument("--slot-top-rel", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/static/method_figures/staged_reconstruction_v1"),
    )
    parser.add_argument("--prefix", default="staged_reconstruction_zalesak")
    return parser.parse_args()


def case_parameters(case_index: int) -> tuple[list[float], float]:
    rng = np.random.default_rng(RANDOM_SEED)
    center = None
    theta = None
    for _ in range(case_index + 1):
        center = [float(rng.uniform(50, 51)), float(rng.uniform(50, 51))]
        theta = float(rng.uniform(0, np.pi / 2))
    return center, theta


def polygon_center(points: list[list[float]]) -> list[float]:
    points_array = np.asarray(points, dtype=float)
    return [float(points_array[:, 0].mean()), float(points_array[:, 1].mean())]


def facet_category(facet, is_fallback: bool) -> str:
    if is_fallback:
        return "fallback"
    if isinstance(facet, CornerFacet):
        if facet.radiusLeft is None and facet.radiusRight is None:
            return "linear_corner"
        return "curved_corner"
    if isinstance(facet, ArcFacet):
        return "circular"
    if isinstance(facet, LinearFacet):
        return "linear"
    return "fallback"


def serialize_facet(merge_id: int, facet, fallback_ids: set[int]) -> dict:
    sample_count = 2 if isinstance(facet, LinearFacet) else 81
    sampled = facet.sample(sample_count)
    data = {
        "merge_id": int(merge_id),
        "name": str(getattr(facet, "name", type(facet).__name__)),
        "category": facet_category(facet, merge_id in fallback_ids),
        "points": [[float(p[0]), float(p[1])] for p in sampled],
        "p_left": [float(facet.pLeft[0]), float(facet.pLeft[1])],
        "p_right": [float(facet.pRight[0]), float(facet.pRight[1])],
    }
    if isinstance(facet, CornerFacet):
        data["corner"] = [float(facet.corner[0]), float(facet.corner[1])]
        data["radius_left"] = facet.radiusLeft
        data["radius_right"] = facet.radiusRight
    if isinstance(facet, ArcFacet):
        data["center"] = [float(facet.center[0]), float(facet.center[1])]
        data["radius"] = float(facet.radius)
    return data


def serialize_cells(mesh: MergeMesh) -> list[dict]:
    cells = []
    for ix, row in enumerate(mesh.polys):
        for iy, poly in enumerate(row):
            fraction = float(poly.getFraction())
            if poly.isMixed():
                state = "mixed"
            elif poly.isFull():
                state = "full"
            else:
                state = "empty"
            cells.append(
                {
                    "coords": [ix, iy],
                    "points": [[float(p[0]), float(p[1])] for p in poly.points],
                    "fraction": fraction,
                    "state": state,
                    "merge_id": mesh._get_merge_id(ix, iy),
                }
            )
    return cells


def serialize_snapshot(stage: str, mesh: MergeMesh, merge_ids=()) -> dict:
    active_ids = [int(merge_id) for merge_id in merge_ids]
    object_to_merge_id = {
        id(mesh.merged_polys[merge_id]): merge_id
        for merge_id in active_ids
        if merge_id in mesh.merged_polys
    }
    fallback_ids = {
        int(record["merge_id"]) for record in mesh.plic_fallback_records
    }

    merge_groups = []
    facets = []
    orientation_links = []
    for merge_id in active_ids:
        merged_poly = mesh.merged_polys[merge_id]
        coords = mesh._get_merge_coords(merge_id)
        left = object_to_merge_id.get(id(merged_poly.getLeftNeighbor()))
        right = object_to_merge_id.get(id(merged_poly.getRightNeighbor()))
        merge_groups.append(
            {
                "merge_id": merge_id,
                "coords": [[int(x), int(y)] for x, y in coords],
                "points": [
                    [float(point[0]), float(point[1])] for point in merged_poly.points
                ],
                "fraction": float(merged_poly.getFraction()),
                "left": left,
                "right": right,
                "center": polygon_center(merged_poly.points),
            }
        )
        if right is not None and right != merge_id:
            orientation_links.append([merge_id, int(right)])
        if merged_poly.hasFacet():
            facets.append(
                serialize_facet(merge_id, merged_poly.getFacet(), fallback_ids)
            )

    return {
        "stage": stage,
        "cells": serialize_cells(mesh),
        "merge_groups": merge_groups,
        "orientation_links": orientation_links,
        "facets": facets,
        "counts": dict(Counter(facet["category"] for facet in facets)),
    }


def crop_bounds(snapshot: dict, resolution: float) -> tuple[float, float, float, float]:
    mixed_points = [
        point
        for cell in snapshot["cells"]
        if cell["state"] == "mixed"
        for point in cell["points"]
    ]
    points = np.asarray(mixed_points, dtype=float)
    pad = 2.5 / resolution
    return (
        float(points[:, 0].min() - pad),
        float(points[:, 0].max() + pad),
        float(points[:, 1].min() - pad),
        float(points[:, 1].max() + pad),
    )


def visible(cell: dict, bounds: tuple[float, float, float, float]) -> bool:
    points = np.asarray(cell["points"], dtype=float)
    x0, x1, y0, y1 = bounds
    return not (
        points[:, 0].max() < x0
        or points[:, 0].min() > x1
        or points[:, 1].max() < y0
        or points[:, 1].min() > y1
    )


def stage_cell_category(snapshot: dict) -> dict[int, str]:
    return {
        int(facet["merge_id"]): facet["category"] for facet in snapshot["facets"]
    }


def draw_cells(
    ax,
    snapshot: dict,
    bounds: tuple[float, float, float, float],
    fraction_cmap,
) -> None:
    category_by_merge_id = stage_cell_category(snapshot)
    patches = []
    facecolors = []
    for cell in snapshot["cells"]:
        if not visible(cell, bounds):
            continue
        patches.append(Polygon(cell["points"], closed=True))
        if cell["state"] == "empty":
            facecolors.append(EMPTY_FILL)
        elif cell["state"] == "full":
            facecolors.append(FULL_FILL)
        elif snapshot["stage"] == "volume_fractions":
            facecolors.append(fraction_cmap(cell["fraction"]))
        elif snapshot["stage"] == "topology":
            facecolors.append(MIXED_FILL)
        else:
            category = category_by_merge_id.get(cell["merge_id"])
            facecolors.append(FILL_COLORS.get(category, MIXED_FILL))

    collection = PatchCollection(
        patches,
        facecolor=facecolors,
        edgecolor=MESH_EDGE,
        linewidth=0.22,
        antialiased=True,
        zorder=1,
    )
    ax.add_collection(collection)


def draw_topology(ax, snapshot: dict) -> None:
    group_by_id = {
        int(group["merge_id"]): group for group in snapshot["merge_groups"]
    }
    for link_index, (source_id, target_id) in enumerate(snapshot["orientation_links"]):
        if source_id not in group_by_id or target_id not in group_by_id:
            continue
        source = np.asarray(group_by_id[source_id]["center"], dtype=float)
        target = np.asarray(group_by_id[target_id]["center"], dtype=float)
        delta = target - source
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            continue
        start = source + 0.13 * delta
        end = target - min(0.28, 0.22 / distance) * delta
        if link_index % 4 == 0:
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.7,
                color=TOPOLOGY_COLOR,
                alpha=0.95,
                shrinkA=0,
                shrinkB=0,
                zorder=4,
            )
            ax.add_patch(arrow)
        else:
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=TOPOLOGY_COLOR,
                linewidth=0.55,
                alpha=0.72,
                solid_capstyle="round",
                zorder=4,
            )

    for group in snapshot["merge_groups"]:
        if len(group["coords"]) <= 1:
            continue
        outline = Polygon(
            group["points"],
            closed=True,
            facecolor="#F3DFC8",
            edgecolor=MERGE_COLOR,
            linewidth=1.15,
            linestyle=(0, (3, 1.5)),
            zorder=5,
        )
        ax.add_patch(outline)


def draw_facets(ax, snapshot: dict) -> None:
    for facet in snapshot["facets"]:
        category = facet["category"]
        color = STROKE_COLORS[category]
        points = np.asarray(facet["points"], dtype=float)
        linestyle = (0, (2.4, 1.5)) if category == "fallback" else "solid"
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=1.45,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=6,
        )
        if category in {"linear_corner", "curved_corner"} and "corner" in facet:
            ax.scatter(
                facet["corner"][0],
                facet["corner"][1],
                marker="D",
                s=7,
                facecolor=color,
                edgecolor=EMPTY_FILL,
                linewidth=0.35,
                zorder=7,
            )


def add_fraction_key(ax, fraction_cmap) -> None:
    inset = ax.inset_axes([0.61, 0.055, 0.32, 0.027])
    scalar = ScalarMappable(norm=Normalize(0, 1), cmap=fraction_cmap)
    colorbar = plt.colorbar(scalar, cax=inset, orientation="horizontal")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=6.5, length=1.6, pad=1)
    colorbar.outline.set_linewidth(0.45)
    inset.text(
        -0.12,
        0.5,
        r"$C$",
        ha="right",
        va="center",
        fontsize=7,
        transform=inset.transAxes,
    )


def build_figure(
    snapshots: dict[str, dict],
    bounds: tuple[float, float, float, float],
    output_base: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "svg.fonttype": "none",
        }
    )
    fraction_cmap = LinearSegmentedColormap.from_list(
        "mixed_fraction", ["#A7CEC8", "#4C958C", "#0D5E6A"]
    )

    fig, axes = plt.subplots(3, 2, figsize=(7.25, 9.35), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.975, bottom=0.085, hspace=0.13, wspace=0.055)

    for ax, stage in zip(axes.flat, STAGE_ORDER):
        snapshot = snapshots[stage]
        draw_cells(ax, snapshot, bounds, fraction_cmap)
        if stage == "topology":
            draw_topology(ax, snapshot)
        elif stage not in {"volume_fractions"}:
            draw_facets(ax, snapshot)
        if stage == "volume_fractions":
            add_fraction_key(ax, fraction_cmap)

        ax.set_title(STAGE_TITLES[stage], loc="left", pad=3.0)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=MIXED_FILL, edgecolor="none", label="unresolved mixed cell"),
        Patch(facecolor=FILL_COLORS["linear"], edgecolor=STROKE_COLORS["linear"], label="linear"),
        Patch(facecolor=FILL_COLORS["linear_corner"], edgecolor=STROKE_COLORS["linear_corner"], label="linear corner"),
        Patch(facecolor=FILL_COLORS["circular"], edgecolor=STROKE_COLORS["circular"], label="circular arc"),
        Patch(facecolor=FILL_COLORS["curved_corner"], edgecolor=STROKE_COLORS["curved_corner"], label="curved corner"),
        Patch(facecolor="none", edgecolor=MERGE_COLOR, linestyle="--", label="merged region"),
    ]
    final_categories = set(snapshots["final"]["counts"])
    if "fallback" in final_categories:
        legend_handles.insert(
            -1,
            Patch(
                facecolor=FILL_COLORS["fallback"],
                edgecolor=STROKE_COLORS["fallback"],
                linestyle="--",
                label="fallback",
            ),
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        handlelength=1.8,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=8.2,
    )

    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 240},
    }.items():
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml("config/static/zalesak.yaml")
    mesh_config = apply_mesh_overrides(
        config["MESH"],
        resolution=args.resolution,
        mesh_type="perturbed_cartesian",
        perturb_wiggle=args.wiggle,
        perturb_seed=args.seed,
    )
    points = make_points_from_config(mesh_config)
    mesh = MergeMesh(points, config["GEOMS"]["THRESHOLD"])
    center, theta = case_parameters(args.case_index)
    fractions = initialize_zalesak(
        mesh,
        center,
        args.radius,
        args.slot_width,
        y_top_rel=args.slot_top_rel,
        theta=theta,
    )
    mesh.initializeFractions(fractions)

    snapshots = {
        "volume_fractions": serialize_snapshot("volume_fractions", mesh),
    }
    mesh.merge1Neighbors()
    merge_ids = mesh.findOrientations()
    snapshots["topology"] = serialize_snapshot("topology", mesh, merge_ids)

    def record_stage(stage, stage_mesh, stage_merge_ids):
        snapshots[stage] = serialize_snapshot(stage, stage_mesh, stage_merge_ids)

    mesh.fitFacets(
        merge_ids,
        setting="circular+corner",
        plic_fallback="LVIRA",
        stage_callback=record_stage,
    )

    missing = [stage for stage in STAGE_ORDER if stage not in snapshots]
    if missing:
        raise RuntimeError(f"Missing stage snapshots: {missing}")

    bounds = crop_bounds(snapshots["volume_fractions"], args.resolution)
    metadata = {
        "source": "perturbed Zalesak reconstruction",
        "case_index": args.case_index,
        "resolution": args.resolution,
        "wiggle": args.wiggle,
        "seed": args.seed,
        "center": center,
        "theta": theta,
        "radius": args.radius,
        "slot_width": args.slot_width,
        "slot_top_rel": args.slot_top_rel,
        "bounds": bounds,
        "stage_counts": {
            stage: snapshots[stage]["counts"] for stage in STAGE_ORDER
        },
        "merged_regions": sum(
            len(group["coords"]) > 1 for group in snapshots["topology"]["merge_groups"]
        ),
    }
    export_snapshots = {
        stage: {
            key: value
            for key, value in snapshots[stage].items()
            if key != "cells"
        }
        for stage in STAGE_ORDER
    }
    export_cells = [
        cell
        for cell in snapshots["volume_fractions"]["cells"]
        if visible(cell, bounds)
    ]
    data_path = args.output_dir / f"{args.prefix}_data.json"
    with data_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": metadata,
                "mesh_cells_in_figure": export_cells,
                "snapshots": export_snapshots,
            },
            handle,
            indent=2,
        )

    output_base = args.output_dir / args.prefix
    build_figure(snapshots, bounds, output_base)
    print(json.dumps(metadata, indent=2))
    print(f"SVG: {output_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
