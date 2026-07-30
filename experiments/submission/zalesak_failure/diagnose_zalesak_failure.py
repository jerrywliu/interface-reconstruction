#!/usr/bin/env python3
"""Diagnose the frozen July Zalesak case-23 reconstruction tail.

The script is deliberately archive-only: it reads frozen metrics and geometric
metadata, reconstructs the recorded support/intersection evidence, and writes a
vector diagnostic plus machine-readable provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
import numpy as np
import vtk

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.geoms.corner_facet import getPolyCornerArea
from main.geoms.geoms import (
    getArea,
    getCentroid,
    getDistance,
    getPolyLineIntersects,
)


DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "results/static/static_paper_simplified_default_20260717_212413"
)
DEFAULT_PREVIOUS_ROOT = (
    REPO_ROOT / "results/static/static_paper_affected_diagnostics_20260714_102206"
)
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "results/submission/zalesak_failure_case23"
DEFAULT_PDF_PATH = REPO_ROOT / "output/pdf/zalesak_failure_case23_diagnostic.pdf"

TARGET = {
    "experiment": "zalesak",
    "algo": "circular+corner",
    "resolution": 1.5,
    "wiggle": 0.2,
    "seed": 0,
    "case_index": 23,
}

COLORS = {
    "truth": "#111827",
    "reconstruction": "#5b7fa3",
    "failure": "#c43d35",
    "left_support": "#21856f",
    "right_support": "#d17a00",
    "target_cell": "#f6d5d2",
    "left_cell": "#d7eee7",
    "right_cell": "#f8e7c9",
    "mesh": "#cbd5e1",
}


@dataclass(frozen=True)
class SupportEvidence:
    side: str
    row: dict
    primitive: dict
    attachment: list[float]
    finite_segment: list[list[float]]
    candidate_count: int


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _same_case(row: dict, spec: dict = TARGET) -> bool:
    return (
        row.get("experiment") == spec["experiment"]
        and row.get("algo") == spec["algo"]
        and math.isclose(float(row.get("resolution", "nan")), spec["resolution"])
        and math.isclose(float(row.get("wiggle", "nan")), spec["wiggle"])
        and int(row.get("seed", -1)) == spec["seed"]
        and int(row.get("case_index", -1)) == spec["case_index"]
    )


def _load_case_geometry(path: Path, save_name: str, case_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("save_name") == save_name and row.get("case_index") == case_index:
                return row
    raise RuntimeError(f"Missing case geometry for {save_name}, case {case_index}")


def _primitive_records(geometry: dict | None) -> list[dict]:
    if not geometry:
        return []
    primitive_class = geometry.get("class")
    if primitive_class == "linear":
        return [
            {
                "kind": "line",
                "source_name": geometry.get("name", "linear"),
                "p_left": geometry["p_left"],
                "p_right": geometry["p_right"],
            }
        ]
    if primitive_class == "circular":
        return [
            {
                "kind": "arc",
                "source_name": geometry.get("name", "arc"),
                "p_left": geometry["p_left"],
                "p_right": geometry["p_right"],
                "center": geometry["center"],
                "radius": float(geometry["radius"]),
            }
        ]
    if primitive_class in {"linear_corner", "curved_corner"}:
        return _primitive_records(geometry.get("left_branch")) + _primitive_records(
            geometry.get("right_branch")
        )
    return []


def _sample_record(record: dict, max_spacing: float = 0.03) -> np.ndarray:
    p_left = np.asarray(record["p_left"], dtype=float)
    p_right = np.asarray(record["p_right"], dtype=float)
    if record.get("kind") != "arc":
        length = float(np.linalg.norm(p_right - p_left))
        count = max(2, int(math.ceil(length / max_spacing)) + 1)
        return np.linspace(p_left, p_right, count)

    center = np.asarray(record["center"], dtype=float)
    radius = abs(float(record["radius"]))
    start = math.atan2(p_left[1] - center[1], p_left[0] - center[0])
    signed_delta = record.get("signed_delta")
    if signed_delta is None:
        end = math.atan2(p_right[1] - center[1], p_right[0] - center[0])
        signed_delta = (end - start) % (2 * math.pi)
        if float(record["radius"]) < 0:
            signed_delta -= 2 * math.pi
    signed_delta = float(signed_delta)
    count = max(2, int(math.ceil(abs(signed_delta) * radius / max_spacing)) + 1)
    angles = np.linspace(start, start + signed_delta, count)
    return np.column_stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
    )


def _distance_to_record(point: Iterable[float], record: dict) -> float:
    point = np.asarray(point, dtype=float)
    p_left = np.asarray(record["p_left"], dtype=float)
    p_right = np.asarray(record["p_right"], dtype=float)
    if record.get("kind") != "arc":
        direction = p_right - p_left
        norm_sq = float(direction @ direction)
        if norm_sq < 1e-24:
            return float(np.linalg.norm(point - p_left))
        t = float(np.clip(((point - p_left) @ direction) / norm_sq, 0.0, 1.0))
        return float(np.linalg.norm(point - (p_left + t * direction)))

    center = np.asarray(record["center"], dtype=float)
    radius = abs(float(record["radius"]))
    start = math.atan2(p_left[1] - center[1], p_left[0] - center[0])
    theta = math.atan2(point[1] - center[1], point[0] - center[0])
    signed_delta = float(record["signed_delta"])
    relative = (
        (theta - start) % (2 * math.pi)
        if signed_delta >= 0
        else (start - theta) % (2 * math.pi)
    )
    if relative <= abs(signed_delta) + 1e-10:
        return abs(float(np.linalg.norm(point - center)) - radius)
    return min(float(np.linalg.norm(point - p_left)), float(np.linalg.norm(point - p_right)))


def _one_sided_max_distance(record: dict, targets: list[dict]) -> float:
    samples = _sample_record(record, max_spacing=0.02)
    return max(min(_distance_to_record(point, target) for target in targets) for point in samples)


def _line_point_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-14:
        return float("inf")
    return abs(float(np.cross(direction, point - start))) / length


def _point_segment_parameter(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    norm_sq = float(direction @ direction)
    if norm_sq < 1e-24:
        return float("inf")
    return float(((point - start) @ direction) / norm_sq)


def _parallel_sine(first: np.ndarray, second: np.ndarray) -> float:
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < 1e-24:
        return float("inf")
    return abs(float(np.cross(first, second))) / denom


def _find_support(
    rows: list[dict],
    stage_rows: list[dict],
    target_merge_id: int,
    side: str,
    attachment: list[float],
    apex: list[float],
) -> SupportEvidence:
    attachment_arr = np.asarray(attachment, dtype=float)
    apex_arr = np.asarray(apex, dtype=float)
    branch_direction = apex_arr - attachment_arr
    branch_direction /= np.linalg.norm(branch_direction)
    candidates = []

    for stage_row in stage_rows:
        if int(stage_row["merge_id"]) == target_merge_id:
            continue
        geometry = json.loads(stage_row["facet_geometry_json"])
        for primitive in _primitive_records(geometry):
            if primitive["kind"] != "line":
                continue
            start = np.asarray(primitive["p_left"], dtype=float)
            end = np.asarray(primitive["p_right"], dtype=float)
            segment_direction = end - start
            if _parallel_sine(branch_direction, segment_direction) > 1e-6:
                continue
            if _line_point_distance(attachment_arr, start, end) > 1e-7:
                continue
            parameter = _point_segment_parameter(attachment_arr, start, end)
            # The two archived methods serialize tiny near-full-cell line
            # supports with a few nanometers of endpoint drift.
            if parameter < -1e-5 or parameter > 1.0 + 1e-5:
                continue
            projections = [
                float((start - attachment_arr) @ branch_direction),
                float((end - attachment_arr) @ branch_direction),
            ]
            negative = [
                (projection, point)
                for projection, point in zip(projections, [start, end])
                if projection < -1e-8
            ]
            if not negative:
                continue
            support_endpoint = min(negative, key=lambda item: item[0])[1]
            production_rows = [
                row
                for row in rows
                if row["merge_id"] == stage_row["merge_id"]
                and row["cell_id"] == stage_row["cell_id"]
            ]
            if len(production_rows) != 1:
                continue
            candidates.append(
                (
                    abs(_line_point_distance(attachment_arr, start, end)),
                    production_rows[0],
                    primitive,
                    [support_endpoint.tolist(), attachment_arr.tolist()],
                )
            )

    if not candidates:
        raise RuntimeError(f"Could not recover {side} support for attachment {attachment}")
    _, row, primitive, finite_segment = min(candidates, key=lambda item: item[0])
    return SupportEvidence(
        side=side,
        row=row,
        primitive=primitive,
        attachment=list(attachment),
        finite_segment=finite_segment,
        candidate_count=len(candidates),
    )


def _primitive_matches(first: dict, second: dict, tolerance: float = 1e-8) -> bool:
    if first.get("kind") != second.get("kind"):
        return False
    first_points = [np.asarray(first["p_left"]), np.asarray(first["p_right"])]
    second_points = [np.asarray(second["p_left"]), np.asarray(second["p_right"])]
    direct = max(np.linalg.norm(first_points[i] - second_points[i]) for i in range(2))
    reverse = max(np.linalg.norm(first_points[i] - second_points[1 - i]) for i in range(2))
    return min(direct, reverse) <= tolerance


def _find_owner(rows: list[dict], primitive: dict) -> dict:
    owners = []
    for row in rows:
        geometry = json.loads(row["facet_geometry_json"])
        if any(_primitive_matches(candidate, primitive) for candidate in _primitive_records(geometry)):
            owners.append(row)
    if len(owners) != 1:
        raise RuntimeError(f"Expected one owner for worst primitive, found {len(owners)}")
    return owners[0]


def _read_grid(mesh_path: Path):
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    grid = reader.GetOutput()
    if grid.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Could not read structured mesh {mesh_path}")
    return grid


def _cell_polygon(grid, cell_id: str | list[int]) -> list[list[float]]:
    if isinstance(cell_id, str):
        x, y = [int(value) for value in cell_id.split(",")]
    else:
        x, y = cell_id
    nx, ny, _ = grid.GetDimensions()
    if not (0 <= x < nx - 1 and 0 <= y < ny - 1):
        raise ValueError(f"Cell {(x, y)} lies outside grid {(nx, ny)}")
    point_ids = [x * ny + y, (x + 1) * ny + y, (x + 1) * ny + y + 1, x * ny + y + 1]
    return [[float(value) for value in grid.GetPoint(point_id)[:2]] for point_id in point_ids]


def _plot_records(ax, records: list[dict], **kwargs):
    for record in records:
        points = _sample_record(record)
        ax.plot(points[:, 0], points[:, 1], **kwargs)


def _plot_cell(ax, polygon, facecolor, edgecolor, linewidth=1.5, alpha=0.75):
    patch = MplPolygon(
        polygon,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=4,
    )
    ax.add_patch(patch)


def _set_clean_axis(ax, bounds):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#64748b")
        spine.set_linewidth(0.8)


def _build_figure(
    pdf_path: Path,
    svg_path: Path,
    png_path: Path,
    true_records: list[dict],
    reconstructed_records: list[dict],
    corner_geometry: dict,
    true_junction: list[float],
    cell_polygons: dict[str, list[list[float]]],
    supports: dict[str, SupportEvidence],
    values: dict,
    grid,
):
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )
    fig = plt.figure(figsize=(14.2, 5.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.82, 0.82], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    apex = np.asarray(corner_geometry["corner"], dtype=float)
    p_left = np.asarray(corner_geometry["p_left"], dtype=float)
    p_right = np.asarray(corner_geometry["p_right"], dtype=float)
    bad_path = np.vstack([p_left, apex, p_right])

    all_truth = np.concatenate([_sample_record(record) for record in true_records])
    truth_bounds = (
        float(all_truth[:, 0].min() - 2.0),
        float(max(all_truth[:, 0].max(), apex[0]) + 2.0),
        float(min(all_truth[:, 1].min(), apex[1]) - 2.0),
        float(all_truth[:, 1].max() + 2.0),
    )
    mechanism_points = np.vstack(
        [
            apex,
            p_left,
            p_right,
            np.asarray(true_junction),
            *[np.asarray(point) for polygon in cell_polygons.values() for point in polygon],
        ]
    )
    mechanism_bounds = (
        float(mechanism_points[:, 0].min() - 0.65),
        float(mechanism_points[:, 0].max() + 0.65),
        float(mechanism_points[:, 1].min() - 0.65),
        float(mechanism_points[:, 1].max() + 0.65),
    )

    ax = axes[0]
    _plot_records(ax, reconstructed_records, color=COLORS["reconstruction"], linewidth=0.9, alpha=0.72, zorder=2)
    _plot_records(ax, true_records, color=COLORS["truth"], linewidth=2.0, linestyle=(0, (4, 2)), zorder=3)
    ax.plot(bad_path[:, 0], bad_path[:, 1], color=COLORS["failure"], linewidth=2.4, zorder=6)
    for key, polygon in cell_polygons.items():
        _plot_cell(
            ax,
            polygon,
            COLORS[f"{key}_cell"],
            COLORS[f"{key}_support"] if key != "target" else COLORS["failure"],
            linewidth=1.2,
            alpha=0.9,
        )
    ax.scatter(*np.asarray(true_junction), marker="D", s=32, facecolor="white", edgecolor=COLORS["truth"], linewidth=1.2, zorder=8)
    ax.scatter(*apex, marker="x", s=58, color=COLORS["failure"], linewidth=1.8, zorder=8)
    rectangle = Rectangle(
        (mechanism_bounds[0], mechanism_bounds[2]),
        mechanism_bounds[1] - mechanism_bounds[0],
        mechanism_bounds[3] - mechanism_bounds[2],
        fill=False,
        edgecolor="#475569",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=7,
    )
    ax.add_patch(rectangle)
    _set_clean_axis(ax, truth_bounds)
    ax.set_title("(a) Full reconstructed interface", loc="left", fontweight="semibold")

    ax = axes[1]
    _plot_records(ax, reconstructed_records, color=COLORS["reconstruction"], linewidth=0.75, alpha=0.42, zorder=1)
    _plot_records(ax, true_records, color=COLORS["truth"], linewidth=1.9, linestyle=(0, (4, 2)), zorder=2)
    for key, polygon in cell_polygons.items():
        _plot_cell(
            ax,
            polygon,
            COLORS[f"{key}_cell"],
            COLORS[f"{key}_support"] if key != "target" else COLORS["failure"],
            linewidth=1.5,
            alpha=0.95,
        )
        centroid = getCentroid(polygon)
        label = {"left": "L", "right": "R", "target": "T"}[key]
        ax.text(
            centroid[0],
            centroid[1],
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111827",
            zorder=9,
        )
    for side, color in [("left", COLORS["left_support"]), ("right", COLORS["right_support"])]:
        segment = np.asarray(supports[side].finite_segment)
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=4.0, solid_capstyle="round", zorder=7)
        attach = np.asarray(supports[side].attachment)
        ax.plot([attach[0], apex[0]], [attach[1], apex[1]], color=color, linewidth=1.5, linestyle=(0, (3, 2)), zorder=6)
    ax.plot(bad_path[:, 0], bad_path[:, 1], color=COLORS["failure"], linewidth=2.0, alpha=0.9, zorder=5)
    ax.scatter(*np.asarray(true_junction), marker="D", s=42, facecolor="white", edgecolor=COLORS["truth"], linewidth=1.3, zorder=9)
    ax.scatter(*apex, marker="x", s=68, color=COLORS["failure"], linewidth=2.0, zorder=9)
    ax.annotate(
        "selected intersection",
        xy=apex,
        xytext=(apex[0] - 0.15, apex[1] + 0.85),
        ha="right",
        va="bottom",
        color=COLORS["failure"],
        arrowprops={"arrowstyle": "-", "color": COLORS["failure"], "lw": 0.8},
        fontsize=8,
    )
    ax.annotate(
        "true line-arc junction",
        xy=np.asarray(true_junction),
        xytext=(true_junction[0] - 0.2, true_junction[1] + 1.45),
        ha="right",
        color=COLORS["truth"],
        arrowprops={"arrowstyle": "-", "color": COLORS["truth"], "lw": 0.8},
        fontsize=8,
    )
    _set_clean_axis(ax, mechanism_bounds)
    ax.set_title("(b) Nonlocal support intersection", loc="left", fontweight="semibold")

    ax = axes[2]
    target_x, target_y = [int(value) for value in supports["target_row"]["cell_id"].split(",")]
    for x in range(target_x - 1, target_x + 2):
        for y in range(target_y - 1, target_y + 2):
            polygon = _cell_polygon(grid, [x, y])
            _plot_cell(ax, polygon, "white", COLORS["mesh"], linewidth=0.7, alpha=1.0)
    _plot_cell(ax, cell_polygons["target"], COLORS["target_cell"], COLORS["failure"], linewidth=1.7, alpha=0.9)
    _plot_records(ax, true_records, color=COLORS["truth"], linewidth=1.8, linestyle=(0, (4, 2)), zorder=3)
    _plot_records(ax, reconstructed_records, color=COLORS["reconstruction"], linewidth=0.8, alpha=0.45, zorder=2)
    ax.plot(bad_path[:, 0], bad_path[:, 1], color=COLORS["failure"], linewidth=2.2, zorder=6)
    intersections = np.asarray(values["right_branch_intersections"], dtype=float)
    if len(intersections):
        ax.scatter(intersections[:, 0], intersections[:, 1], s=30, facecolor="white", edgecolor=COLORS["failure"], linewidth=1.3, zorder=8)
    target_polygon = np.asarray(cell_polygons["target"])
    close_bounds = (
        float(target_polygon[:, 0].min() - 0.25),
        float(target_polygon[:, 0].max() + 0.25),
        float(target_polygon[:, 1].min() - 0.22),
        float(target_polygon[:, 1].max() + 0.22),
    )
    ax.text(
        0.03,
        0.04,
        "left-branch crossings: 0\n"
        f"right-branch crossings: {len(intersections)}\n"
        f"area-fraction residual: {values['area_fraction_residual']:.2e}\n"
        f"apex distance: {values['apex_cell_radius_ratio']:.2f} cell radii",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#111827",
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.94},
        zorder=10,
    )
    _set_clean_axis(ax, close_bounds)
    ax.set_title("(c) Target-cell acceptance", loc="left", fontweight="semibold")

    legend = [
        Line2D([0], [0], color=COLORS["truth"], lw=2.0, ls=(0, (4, 2)), label="true interface"),
        Line2D([0], [0], color=COLORS["reconstruction"], lw=1.4, label="reconstruction"),
        Line2D([0], [0], color=COLORS["failure"], lw=2.2, label="selected corner"),
        Line2D([0], [0], color=COLORS["left_support"], lw=3.5, label="left line support"),
        Line2D([0], [0], color=COLORS["right_support"], lw=3.5, label="right line support"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.12, top=0.96, left=0.025, right=0.985)
    for path in [pdf_path, svg_path, png_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def _candidate_comparison(case_rows: list[dict], plots_root: Path) -> list[dict]:
    candidates = [
        row
        for row in case_rows
        if row["experiment"] == "zalesak"
        and row["algo"] == "circular+corner"
        and float(row["hausdorff"]) > 1.0
    ]
    output = []
    for row in sorted(candidates, key=lambda item: float(item["hausdorff"]), reverse=True):
        metadata_path = (
            plots_root
            / row["save_name"]
            / "vtk/reconstructed/facets"
            / f"{row['case_index']}.facet_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        truth_path = (
            plots_root
            / row["save_name"]
            / "vtk/true"
            / f"true_zalesak{row['case_index']}.facet_metadata.json"
        )
        truth_records = json.loads(truth_path.read_text(encoding="utf-8"))["primitives"]
        _, dominant_record = max(
            (
                (_one_sided_max_distance(record, truth_records), record)
                for record in metadata["primitives"]
            ),
            key=lambda item: item[0],
        )
        corner_lines = [
            record
            for record in metadata["primitives"]
            if record.get("source_name") == "corner" and record.get("kind") == "line"
        ]
        output.append(
            {
                "resolution": row["resolution"],
                "N": int(round(100 * float(row["resolution"]))),
                "wiggle": row["wiggle"],
                "seed": row["seed"],
                "case_index": row["case_index"],
                "hausdorff": row["hausdorff"],
                "facet_gap": row["facet_gap"],
                "area_error": row["area_error"],
                "plic_fallback_cells": row["num_plic_fallback_cells"],
                "linear_corner_cells": row["num_final_linear_corner_cells"],
                "curved_corner_cells": row["num_final_curved_corner_cells"],
                "contains_corner_line_primitives": int(bool(corner_lines)),
                "dominant_primitive_source": dominant_record.get("source_name", ""),
                "dominant_primitive_kind": dominant_record.get("kind", ""),
                "dominant_facet_index": dominant_record.get("facet_index", ""),
                "dominant_primitive_index": dominant_record.get("primitive_index", ""),
                "mechanism_family": (
                    "linear-corner dominated"
                    if dominant_record.get("source_name") == "corner"
                    and dominant_record.get("kind") == "line"
                    else "coarse arc dominated"
                ),
                "selected_for_diagnostic": int(int(row["case_index"]) == TARGET["case_index"] and math.isclose(float(row["resolution"]), TARGET["resolution"]) and math.isclose(float(row["wiggle"]), TARGET["wiggle"])),
                "save_name": row["save_name"],
            }
        )
    return output


def _result_readme(summary: dict) -> str:
    values = summary["evidence"]["acceptance_values"]
    metrics = summary["target_case"]["metrics"]
    comparison = summary["candidate_comparison"]
    corner_count = sum(row["mechanism_family"] == "linear-corner dominated" for row in comparison)
    return f"""# Zalesak Case-23 Failure Diagnostic

## Finding

The July production case is a fully oriented, non-fallback failure caused by a
nonlocal line-support pairing. Target merge `286` (base cell `[98,76]`) combines
a line from the slot edge with a nearly degenerate line from an almost-full
outer-circle cell. Extending those lines selects an apex at
`{summary['evidence']['selected_apex']}`. The apex is
`{values['apex_cell_radius_ratio']:.2f}` target-cell radii away, while only the
right branch crosses the target cell. The resulting local area-fraction residual
is nevertheless `{values['area_fraction_residual']:.3e}`, below the archived
`1e-4` corner-acceptance threshold.

The selected apex is `{values['selected_apex_to_true_junction']:.12g}` from the
true line-arc junction, exactly reproducing the reported Hausdorff error
`{float(metrics['hausdorff']):.12g}`. This is therefore a support-selection and
extrapolated-intersection failure, not insufficient resolution, unresolved
orientation, LVIRA fallback, arc fitting, or a VTK rendering defect.

## Why this case

The July sweep contains `{len(comparison)}` Zalesak `circular+corner` cases with
Hausdorff error above one. `{corner_count}/{len(comparison)}` are dominated by a
linear corner primitive, and case 23 has the largest reported error. It is the
clearest example because one recorded corner branch and one selected apex account
for the full Hausdorff witness, with no fallback event or ambiguous missing facet.
The coarse Cartesian case (`N=64`, `w=0`) is arc-dominated and represents a
different mechanism.

## Artifacts

- `zalesak_failure_case23_diagnostic.pdf`: vector diagnostic
- `zalesak_failure_case23_diagnostic.svg`: editable vector diagnostic
- `zalesak_failure_case23_diagnostic.png`: rendered preview
- `zalesak_failure_case23_provenance.json`: complete input and inference record
- `zalesak_failure_case23_entities.csv`: implicated cells, supports, and points
- `zalesak_tail_candidate_comparison.csv`: all July Zalesak cases with Hausdorff above one

## Journal-suitable factual interpretation

For the illustrated perturbed Zalesak case, the dominant error is caused by a
fully oriented cell that pairs a straight support from the slot boundary with a
near-degenerate linear support on the circular boundary. Their extrapolated
intersection lies far outside the target cell; although only one branch crosses
the cell, the resulting corner satisfies the local volume-fraction tolerance and
is retained. The distance from this spurious apex to the true line-arc junction
equals the global Hausdorff error. Thus the outlier reflects nonlocal support
selection at a line-to-arc transition rather than unresolved orientation, PLIC
fallback, or inadequate mesh resolution.

## Provenance caveat

The July event log records the target corner assignment and final per-cell
geometry, but not support IDs at the instant of assignment. Support owners are
recovered by collinear containment of the two accepted branch attachment points.
The left support is later upgraded to a curved corner; its finite line segment is
cross-checked against the archived same-case `circular` sibling, which shares the
initial line-fitting stage.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--previous-root", type=Path, default=DEFAULT_PREVIOUS_ROOT)
    parser.add_argument("--plots-root", type=Path, default=REPO_ROOT / "plots")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--pdf-path", type=Path, default=DEFAULT_PDF_PATH)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    plots_root = args.plots_root.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    case_rows = _read_csv(run_root / "diagnostics/case_metrics.csv")
    target_case_rows = [row for row in case_rows if _same_case(row)]
    if len(target_case_rows) != 1:
        raise RuntimeError(f"Expected one target case row, found {len(target_case_rows)}")
    target_case = target_case_rows[0]
    save_name = target_case["save_name"]
    case_index = TARGET["case_index"]
    run_bundle = plots_root / save_name

    all_cell_rows = _read_csv(run_root / "diagnostics/cell_metrics.csv")
    cell_rows = [
        row
        for row in all_cell_rows
        if row["save_name"] == save_name and int(row["case_index"]) == case_index
    ]
    circular_sibling_cases = [
        row
        for row in case_rows
        if row["experiment"] == "zalesak"
        and row["algo"] == "circular"
        and math.isclose(float(row["resolution"]), TARGET["resolution"])
        and math.isclose(float(row["wiggle"]), TARGET["wiggle"])
        and int(row["seed"]) == TARGET["seed"]
        and int(row["case_index"]) == case_index
    ]
    if len(circular_sibling_cases) != 1:
        raise RuntimeError(
            f"Expected one circular sibling case, found {len(circular_sibling_cases)}"
        )
    circular_sibling = circular_sibling_cases[0]
    circular_sibling_rows = [
        row
        for row in all_cell_rows
        if row["save_name"] == circular_sibling["save_name"]
        and int(row["case_index"]) == case_index
    ]
    merge_events = [
        row
        for row in _read_csv(run_root / "diagnostics/merge_events.csv")
        if row["save_name"] == save_name and int(row["case_index"]) == case_index
    ]
    fallback_rows = [
        row
        for row in _read_csv(run_root / "diagnostics/unresolved_plic_fallbacks.csv")
        if row["save_name"] == save_name and int(row["case_index"]) == case_index
    ]
    case_geometry = _load_case_geometry(
        run_root / "diagnostics/case_geometry.jsonl", save_name, case_index
    )
    source_state = json.loads(
        (run_root / "diagnostics/source_state.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads((run_bundle / "run_manifest.json").read_text(encoding="utf-8"))

    reconstructed_metadata_path = (
        run_bundle / "vtk/reconstructed/facets" / f"{case_index}.facet_metadata.json"
    )
    true_metadata_path = run_bundle / "vtk/true" / f"true_zalesak{case_index}.facet_metadata.json"
    reconstructed_metadata = json.loads(reconstructed_metadata_path.read_text(encoding="utf-8"))
    true_metadata = json.loads(true_metadata_path.read_text(encoding="utf-8"))
    reconstructed_records = reconstructed_metadata["primitives"]
    true_records = true_metadata["primitives"]

    ranked = sorted(
        (
            (_one_sided_max_distance(record, true_records), record)
            for record in reconstructed_records
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    worst_distance, worst_record = ranked[0]
    target_row = _find_owner(cell_rows, worst_record)
    target_geometry = json.loads(target_row["facet_geometry_json"])
    if target_geometry.get("class") != "linear_corner":
        raise RuntimeError(f"Expected linear-corner owner, got {target_geometry.get('class')}")
    target_merge_id = int(target_row["merge_id"])
    apex = target_geometry["corner"]

    supports = {
        "left": _find_support(
            cell_rows,
            circular_sibling_rows,
            target_merge_id,
            "left",
            target_geometry["p_left"],
            apex,
        ),
        "right": _find_support(
            cell_rows,
            circular_sibling_rows,
            target_merge_id,
            "right",
            target_geometry["p_right"],
            apex,
        ),
    }
    supports["target_row"] = target_row

    grid = _read_grid(run_bundle / "vtk/mesh.vtk")
    cell_polygons = {
        "left": _cell_polygon(grid, supports["left"].row["cell_id"]),
        "right": _cell_polygon(grid, supports["right"].row["cell_id"]),
        "target": _cell_polygon(grid, target_row["cell_id"]),
    }
    target_polygon = cell_polygons["target"]

    true_points = []
    for record in true_records:
        true_points.extend([record["p_left"], record["p_right"]])
    unique_true_points = []
    for point in true_points:
        if not any(getDistance(point, existing) < 1e-8 for existing in unique_true_points):
            unique_true_points.append(point)
    true_junction = min(unique_true_points, key=lambda point: getDistance(point, apex))

    target_fraction = float(target_row["cell_fraction"])
    reconstructed_fraction = getPolyCornerArea(
        target_polygon,
        target_geometry["p_left"],
        apex,
        target_geometry["p_right"],
    ) / getArea(target_polygon)
    left_intersections = getPolyLineIntersects(
        target_polygon, target_geometry["p_left"], apex
    )
    right_intersections = getPolyLineIntersects(
        target_polygon, apex, target_geometry["p_right"]
    )
    centroid = getCentroid(target_polygon)
    cell_radius = max(getDistance(centroid, point) for point in target_polygon)
    left_support_length = getDistance(*supports["left"].finite_segment)
    right_support_length = getDistance(*supports["right"].finite_segment)

    acceptance_values = {
        "target_fraction": target_fraction,
        "reconstructed_fraction": reconstructed_fraction,
        "area_fraction_residual": abs(reconstructed_fraction - target_fraction),
        "linear_corner_area_threshold": 1e-4,
        "left_branch_intersections": left_intersections,
        "right_branch_intersections": right_intersections,
        "left_support_length": left_support_length,
        "right_support_length": right_support_length,
        "left_support_extrapolation_ratio": getDistance(target_geometry["p_left"], apex) / left_support_length,
        "right_support_extrapolation_ratio": getDistance(target_geometry["p_right"], apex) / right_support_length,
        "target_cell_radius": cell_radius,
        "apex_to_target_cell_centroid": getDistance(centroid, apex),
        "apex_cell_radius_ratio": getDistance(centroid, apex) / cell_radius,
        "selected_apex_to_true_junction": getDistance(apex, true_junction),
        "worst_primitive_one_sided_distance": worst_distance,
    }

    target_events = [
        row
        for row in merge_events
        if int(row["merge_id"]) == target_merge_id
        and row["event_kind"] == "facet_assignment"
    ]
    assignment_events = [row for row in target_events if row["stage"] == "linear_corners"]
    if len(assignment_events) != 1:
        raise RuntimeError(f"Expected one target linear-corner assignment, found {len(assignment_events)}")

    previous_rows = []
    previous_case = None
    previous_path = args.previous_root.resolve() / "diagnostics/case_metrics.csv"
    if previous_path.exists():
        previous_rows = _read_csv(previous_path)
        matches = [row for row in previous_rows if _same_case(row)]
        if len(matches) == 1:
            previous_case = matches[0]

    candidate_rows = _candidate_comparison(case_rows, plots_root)
    comparison_path = artifact_root / "zalesak_tail_candidate_comparison.csv"
    _write_csv(comparison_path, candidate_rows)

    entity_rows = [
        {
            "role": "target_cell",
            "merge_id": target_row["merge_id"],
            "cell_id": target_row["cell_id"],
            "cell_fraction": target_row["cell_fraction"],
            "facet_class": target_row["final_facet_class"],
            "point": json.dumps(apex),
            "derivation": "exact archived cell geometry",
        },
        {
            "role": "left_support",
            "merge_id": supports["left"].row["merge_id"],
            "cell_id": supports["left"].row["cell_id"],
            "cell_fraction": supports["left"].row["cell_fraction"],
            "facet_class": supports["left"].row["final_facet_class"],
            "point": json.dumps(supports["left"].finite_segment),
            "derivation": "collinear branch containment; finite endpoint cross-checked with circular sibling",
        },
        {
            "role": "right_support",
            "merge_id": supports["right"].row["merge_id"],
            "cell_id": supports["right"].row["cell_id"],
            "cell_fraction": supports["right"].row["cell_fraction"],
            "facet_class": supports["right"].row["final_facet_class"],
            "point": json.dumps(supports["right"].finite_segment),
            "derivation": "exact endpoint and collinear branch containment",
        },
        {
            "role": "selected_apex",
            "merge_id": target_row["merge_id"],
            "cell_id": target_row["cell_id"],
            "cell_fraction": "",
            "facet_class": "linear_corner",
            "point": json.dumps(apex),
            "derivation": "exact archived cell geometry",
        },
        {
            "role": "nearest_true_junction",
            "merge_id": "",
            "cell_id": "",
            "cell_fraction": "",
            "facet_class": "truth",
            "point": json.dumps(true_junction),
            "derivation": "exact archived truth primitive endpoint",
        },
    ]
    entity_path = artifact_root / "zalesak_failure_case23_entities.csv"
    _write_csv(entity_path, entity_rows)

    pdf_path = artifact_root / "zalesak_failure_case23_diagnostic.pdf"
    svg_path = artifact_root / "zalesak_failure_case23_diagnostic.svg"
    png_path = artifact_root / "zalesak_failure_case23_diagnostic.png"
    _build_figure(
        pdf_path,
        svg_path,
        png_path,
        true_records,
        reconstructed_records,
        target_geometry,
        true_junction,
        cell_polygons,
        supports,
        {**acceptance_values, "right_branch_intersections": right_intersections},
        grid,
    )
    args.pdf_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, args.pdf_path.resolve())

    summary = {
        "schema_version": 1,
        "diagnosis": {
            "mechanism": "nonlocal cross-regime line-support pairing followed by an extrapolated line-line intersection",
            "classification": "support selection and selected intersection",
            "ruled_out": [
                "insufficient resolution",
                "unresolved orientation",
                "LVIRA fallback",
                "failed high-order fit fallback",
                "arc rendering",
                "missing final facet",
            ],
            "confidence": "high",
        },
        "target_case": {
            "spec": TARGET,
            "save_name": save_name,
            "metrics": {
                key: target_case[key]
                for key in [
                    "hausdorff",
                    "facet_gap",
                    "area_error",
                    "num_mixed_cells",
                    "num_merge_components",
                    "num_merged_cells",
                    "num_plic_fallback_cells",
                    "num_final_linear_corner_cells",
                    "num_final_curved_corner_cells",
                ]
            },
            "geometry": case_geometry,
            "run_manifest": run_manifest,
            "previous_configuration_metrics": (
                {
                    key: previous_case.get(key, "")
                    for key in [
                        "save_name",
                        "rescue_profile",
                        "corner_behavior_profile",
                        "hausdorff",
                        "facet_gap",
                        "area_error",
                        "num_final_linear_corner_cells",
                    ]
                }
                if previous_case
                else None
            ),
        },
        "evidence": {
            "target_cell": {
                key: target_row[key]
                for key in [
                    "merge_id",
                    "cell_id",
                    "cell_fraction",
                    "is_merged",
                    "orientation_status",
                    "has_3x3_stencil",
                    "final_facet_class",
                    "final_facet_name",
                    "construction_path",
                    "event_count",
                ]
            },
            "target_assignment_event": assignment_events[0],
            "unresolved_fallback_events": fallback_rows,
            "selected_corner_geometry": target_geometry,
            "selected_apex": apex,
            "nearest_true_junction": true_junction,
            "worst_primitive": worst_record,
            "supports": {
                side: {
                    "merge_id": support.row["merge_id"],
                    "cell_id": support.row["cell_id"],
                    "cell_fraction": support.row["cell_fraction"],
                    "final_facet_class": support.row["final_facet_class"],
                    "attachment": support.attachment,
                    "finite_segment": support.finite_segment,
                    "accepted_line_primitive": support.primitive,
                    "candidate_count": support.candidate_count,
                    "support_owner_source": circular_sibling["save_name"],
                    "recovery_method": "unique collinear branch attachment in the archived circular sibling's identical initial line-fitting stage",
                }
                for side, support in supports.items()
                if side in {"left", "right"}
            },
            "acceptance_values": acceptance_values,
            "archived_profile_flags": {
                "linear_corner_locality_guard": False,
                "require_both_linear_corner_branches": False,
                "corner_branch_propagation": False,
            },
        },
        "candidate_comparison": candidate_rows,
        "provenance": {
            "run_root": str(run_root),
            "run_bundle": str(run_bundle),
            "source_state": source_state,
            "source_files": {
                "case_metrics": str(run_root / "diagnostics/case_metrics.csv"),
                "cell_metrics": str(run_root / "diagnostics/cell_metrics.csv"),
                "merge_events": str(run_root / "diagnostics/merge_events.csv"),
                "fallback_events": str(run_root / "diagnostics/unresolved_plic_fallbacks.csv"),
                "case_geometry": str(run_root / "diagnostics/case_geometry.jsonl"),
                "reconstructed_metadata": str(reconstructed_metadata_path),
                "truth_metadata": str(true_metadata_path),
                "mesh": str(run_bundle / "vtk/mesh.vtk"),
            },
            "support_inference_caveat": "The archived event log omits support IDs and stage-time support geometry. Owners are recovered from exact/collinear branch geometry; the left finite endpoint is cross-checked against the circular sibling.",
        },
        "artifacts": {
            "pdf": str(pdf_path),
            "pdf_mirror": str(args.pdf_path.resolve()),
            "svg": str(svg_path),
            "png": str(png_path),
            "entities_csv": str(entity_path),
            "candidate_comparison_csv": str(comparison_path),
        },
    }

    if not math.isclose(
        acceptance_values["selected_apex_to_true_junction"],
        float(target_case["hausdorff"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise AssertionError("Selected-apex witness does not reproduce archived Hausdorff")
    if target_row["orientation_status"] != "oriented" or fallback_rows:
        raise AssertionError("Target case is not the expected fully oriented, no-fallback witness")
    if left_intersections or len(right_intersections) != 2:
        raise AssertionError("Archived one-branch intersection evidence changed")
    if acceptance_values["area_fraction_residual"] >= 1e-4:
        raise AssertionError("Archived corner should satisfy the local area threshold")
    if int(target_row["is_merged"]) != 0:
        raise AssertionError("Target witness unexpectedly belongs to a merged component")

    provenance_path = artifact_root / "zalesak_failure_case23_provenance.json"
    provenance_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    readme_path = artifact_root / "README.md"
    readme_path.write_text(_result_readme(summary), encoding="utf-8")

    print(f"Artifact root: {artifact_root}")
    print(f"Vector PDF: {pdf_path}")
    print(f"Vector SVG: {svg_path}")
    print(f"Provenance: {provenance_path}")
    print(
        "Mechanism: fully oriented target merge "
        f"{target_merge_id}, no fallback, apex {acceptance_values['apex_cell_radius_ratio']:.2f} "
        "cell radii away, one branch crossing, accepted area residual "
        f"{acceptance_values['area_fraction_residual']:.3e}"
    )


if __name__ == "__main__":
    main()
