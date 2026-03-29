#!/usr/bin/env python3
"""
Generate the Section 6 main-text static figures.

This script creates:
- compact quantitative 2x2 panels for each static experiment
- representative reconstruction-comparison figures for each experiment

It reuses the merged Section 6 CSV for summary metrics, with a small tangent-error
backfill for the circle sweep from saved run directories when those rows are
missing from the merged CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import vtk
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as PolygonPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static.circles import RANDOM_SEED as CIRCLE_RANDOM_SEED
from experiments.static.ellipses import RANDOM_SEED as ELLIPSE_RANDOM_SEED
from experiments.static.lines import RANDOM_SEED as LINE_RANDOM_SEED
from experiments.static.squares import RANDOM_SEED as SQUARE_RANDOM_SEED
from experiments.static.run_perturbed_sweeps import (
    DISPLAY_LABELS,
    METHOD_STYLES,
    PERTURBATION_AXIS_LABEL,
    RESOLUTION_AXIS_LABEL,
    _build_method_curves,
    _build_method_curves_by_resolution,
    _build_metric_index,
    _draw_method_curves,
    _load_sweep_rows,
    _make_save_name,
)
from experiments.static.zalesak import (
    RANDOM_SEED as ZALESAK_RANDOM_SEED,
    build_true_reference_zalesak,
    rotate_point_around_center,
)
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet


PLOTS_ROOT = REPO_ROOT / "plots"
DEFAULT_CSV = (
    REPO_ROOT
    / "results"
    / "static"
    / "camera_ready"
    / "static_cameraready_plotrefresh_elvira_lvira_backfill_20260327"
    / "csv"
    / "section6_plotrefresh_merged.csv"
)

MAINTEXT_METHODS = {
    "lines": ["Youngs", "ELVIRA", "LVIRA", "linear"],
    "squares": ["ELVIRA", "LVIRA", "linear", "linear+corner"],
    "circles": ["ELVIRA", "LVIRA", "linear", "circular"],
    "ellipses": ["ELVIRA", "LVIRA", "linear", "circular"],
    "zalesak": ["ELVIRA", "LVIRA", "circular", "circular+corner"],
}

QUANT_SPECS = {
    "lines": {"metrics": ("hausdorff", "facet_gap")},
    "squares": {"metrics": ("hausdorff", "facet_gap")},
    "circles": {"metrics": ("hausdorff", "tangent_error")},
    "ellipses": {"metrics": ("hausdorff", "tangent_error")},
    "zalesak": {"metrics": ("hausdorff", "facet_gap")},
}

REPRESENTATIVE_CASES = {
    "lines": {
        "resolution": 0.32,
        "wiggle": 0.30,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("Youngs", "Youngs"),
            ("ELVIRA", "ELVIRA"),
            ("LVIRA", "LVIRA"),
            ("linear", "Ours (linear)"),
        ],
        "min_span": 100.0,
        "margin_frac": 0.00,
        "inset": None,
    },
    "squares": {
        "resolution": 0.50,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("ELVIRA", "ELVIRA"),
            ("LVIRA", "LVIRA"),
            ("linear", "Ours (linear)"),
            ("linear+corner", "Ours (linear+corner)"),
        ],
        "min_span": 42.0,
        "margin_frac": 0.10,
        "inset": {"kind": "square_corner", "zoom": 2.8},
    },
    "circles": {
        "resolution": 0.32,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("ELVIRA", "ELVIRA"),
            ("LVIRA", "LVIRA"),
            ("linear", "Ours (linear)"),
            ("circular", "Ours (circular)"),
        ],
        "min_span": 26.0,
        "margin_frac": 0.14,
        "inset": None,
    },
    "ellipses": {
        "resolution": 0.32,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("ELVIRA", "ELVIRA"),
            ("LVIRA", "LVIRA"),
            ("linear", "Ours (linear)"),
            ("circular", "Ours (circular)"),
        ],
        "min_span": 66.0,
        "margin_frac": 0.12,
        "inset": None,
    },
    "zalesak": {
        "resolution": 1.00,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("ELVIRA", "ELVIRA"),
            ("LVIRA", "LVIRA"),
            ("circular", "Ours (circular)"),
            ("circular+corner", "Ours (circular+corner)"),
        ],
        "min_span": 42.0,
        "margin_frac": 0.12,
        "inset": {"kind": "zalesak_corner", "zoom": 3.0},
    },
}

APPENDIX_BEST_METHODS = {
    "lines": {
        "method": ("linear", "Ours (linear)"),
        "resolutions": [0.32, 0.64, 1.0, 1.5],
        "wiggle": 0.30,
        "seed": 0,
        "case_index": 12,
        "min_span": 100.0,
        "margin_frac": 0.00,
    },
    "squares": {
        "method": ("linear+corner", "Ours (linear+corner)"),
        "resolutions": [0.50, 0.64, 1.0, 1.5],
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "min_span": 42.0,
        "margin_frac": 0.10,
    },
    "circles": {
        "method": ("circular", "Ours (circular)"),
        "resolutions": [0.32, 0.64, 1.0, 1.5],
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "min_span": 26.0,
        "margin_frac": 0.14,
    },
    "ellipses": {
        "method": ("circular", "Ours (circular)"),
        "resolutions": [0.32, 0.64, 1.0, 1.5],
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "min_span": 66.0,
        "margin_frac": 0.12,
    },
    "zalesak": {
        "method": ("circular+corner", "Ours (circular+corner)"),
        "resolutions": [0.50, 0.64, 1.0, 1.5],
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "min_span": 42.0,
        "margin_frac": 0.12,
    },
}

APPENDIX_CARTESIAN_CASES = {
    "lines": {**REPRESENTATIVE_CASES["lines"], "wiggle": 0.0},
    "squares": {**REPRESENTATIVE_CASES["squares"], "wiggle": 0.0},
    "circles": {**REPRESENTATIVE_CASES["circles"], "wiggle": 0.0},
    "ellipses": {**REPRESENTATIVE_CASES["ellipses"], "wiggle": 0.0},
    "zalesak": {**REPRESENTATIVE_CASES["zalesak"], "wiggle": 0.0},
}

TRUE_COLOR = "#111827"
TRUE_STYLE = (0, (3.0, 2.2))
MESH_COLOR = "#d1d5db"
MESH_ALPHA = 0.65
FLUID_FILL_COLOR = "#bfdbfe"
FLUID_FILL_ALPHA = 0.30
ENDPOINT_MARKER_SIZE = 10


def _read_polydata(path: Path):
    if path.suffix.lower() == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput()
    if path.suffix.lower() == ".vtk":
        reader = vtk.vtkStructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        extract = vtk.vtkExtractEdges()
        extract.SetInputData(reader.GetOutput())
        extract.Update()
        return extract.GetOutput()
    raise ValueError(f"Unsupported polydata format: {path}")


def _read_metric_values(path: Path) -> list[float]:
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(float(line))
    return values


def _metric_stats(metric_name: str, values: list[float]) -> list[dict]:
    arr = np.asarray(values, dtype=float)
    return [
        {"metric_key": f"{metric_name}_mean", "metric_value": float(np.mean(arr))},
        {"metric_key": f"{metric_name}_median", "metric_value": float(np.median(arr))},
        {
            "metric_key": f"{metric_name}_p25",
            "metric_value": float(np.percentile(arr, 25)),
        },
        {
            "metric_key": f"{metric_name}_p75",
            "metric_value": float(np.percentile(arr, 75)),
        },
    ]


def _backfill_circle_tangent_rows(rows: list[dict]) -> list[dict]:
    existing = {
        (
            row["experiment"],
            row["algo"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row.get("seed", 0)),
            row["metric_key"],
        )
        for row in rows
    }
    circle_algos_present = {
        row["algo"] for row in rows if row.get("experiment") == "circles"
    }
    backfill_algos = ["Youngs", "ELVIRA", "safe_linear", "linear", "safe_circle", "circular"]
    if "LVIRA" in circle_algos_present:
        backfill_algos.insert(2, "LVIRA")
    backfilled = 0
    for algo in backfill_algos:
        for resolution in [0.32, 0.64, 1.28]:
            for wiggle in [0.0, 0.05, 0.1, 0.2, 0.3]:
                seed = 0
                save_name = _make_save_name("circles", algo, resolution, wiggle, seed)
                metrics_path = PLOTS_ROOT / save_name / "metrics" / "tangent_error.txt"
                normalized_algo = algo
                if algo == "ELVIRA" and not metrics_path.exists():
                    legacy_save_name = _make_save_name("circles", "LVIRA", resolution, wiggle, seed)
                    metrics_path = PLOTS_ROOT / legacy_save_name / "metrics" / "tangent_error.txt"
                if not metrics_path.exists():
                    continue
                values = _read_metric_values(metrics_path)
                if not values:
                    continue
                for entry in _metric_stats("tangent_error", values):
                    key = (
                        "circles",
                        normalized_algo,
                        float(resolution),
                        float(wiggle),
                        seed,
                        entry["metric_key"],
                    )
                    if key in existing:
                        continue
                    rows.append(
                        {
                            "experiment": "circles",
                            "algo": normalized_algo,
                            "resolution": resolution,
                            "wiggle": wiggle,
                            "seed": seed,
                            "metric_key": entry["metric_key"],
                            "metric_value": entry["metric_value"],
                            "save_name": save_name,
                        }
                    )
                    existing.add(key)
                    backfilled += 1
    return rows


def _iter_lines(poly) -> list[np.ndarray]:
    lines = []
    for cell_id in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_id)
        pts = []
        for i in range(cell.GetNumberOfPoints()):
            point = cell.GetPoints().GetPoint(i)
            pts.append([point[0], point[1]])
        if len(pts) >= 2:
            lines.append(np.asarray(pts, dtype=float))
    return lines


def _segments_from_polydata(poly) -> np.ndarray:
    chunks = []
    for line in _iter_lines(poly):
        if len(line) < 2:
            continue
        chunks.append(np.stack([line[:-1], line[1:]], axis=1))
    if not chunks:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


def _facet_endpoints_from_polydata(poly) -> np.ndarray:
    endpoints = []
    for line in _iter_lines(poly):
        if len(line) < 2:
            continue
        endpoints.append(line[0])
        endpoints.append(line[-1])
    if not endpoints:
        return np.empty((0, 2), dtype=float)
    return np.asarray(endpoints, dtype=float)


def _mesh_segments(mesh_path: Path) -> np.ndarray:
    return _segments_from_polydata(_read_polydata(mesh_path))


def _true_vtp_path(exp_name: str, save_name: str, case_index: int) -> Path:
    stem = {
        "lines": "true_line",
        "squares": "true_square",
        "circles": "true_circle",
        "zalesak": "true_zalesak",
    }[exp_name]
    return PLOTS_ROOT / save_name / "vtk" / "true" / f"{stem}{case_index}.vtp"


def _ellipse_case_params(case_index: int) -> dict:
    rng = np.random.default_rng(ELLIPSE_RANDOM_SEED)
    aspect_ratios = np.linspace(1.5, 3.0, 25)
    major_axis = 30.0
    for i, aspect_ratio in enumerate(aspect_ratios):
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        theta = rng.uniform(0, math.pi / 2)
        if i == case_index:
            minor_axis = major_axis / aspect_ratio
            return {
                "center": np.asarray(center, dtype=float),
                "theta": float(theta),
                "major_axis": major_axis,
                "minor_axis": float(minor_axis),
            }
    raise ValueError(f"Invalid ellipse case index: {case_index}")


def _line_case_params(case_index: int) -> dict:
    rng = np.random.default_rng(LINE_RANDOM_SEED)
    angles = np.linspace(0.0, 2.0 * math.pi, 25 + 1)[:-1]
    for i, angle in enumerate(angles):
        x1, y1 = rng.uniform(50, 51), rng.uniform(50, 51)
        x2 = x1 + 0.2
        y2 = y1 + math.tan(angle) * (x2 - x1)
        if i == case_index:
            return {
                "p1": np.asarray([x1, y1], dtype=float),
                "p2": np.asarray([x2, y2], dtype=float),
            }
    raise ValueError(f"Invalid line case index: {case_index}")


def _line_true_segments(case_index: int, bounds: tuple[float, float, float, float]) -> np.ndarray:
    params = _line_case_params(case_index)
    p1 = params["p1"]
    p2 = params["p2"]
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = direction / norm
    x0, x1, y0, y1 = bounds
    center = 0.5 * (p1 + p2)
    span = max(x1 - x0, y1 - y0)
    half_length = 0.9 * math.sqrt(2.0) * span
    a = center - half_length * direction
    b = center + half_length * direction
    return np.asarray([[a, b]], dtype=float)


def _line_fill_polygon(case_index: int, bounds: tuple[float, float, float, float]) -> np.ndarray:
    params = _line_case_params(case_index)
    p1 = params["p1"]
    p2 = params["p2"]
    rect = np.asarray(
        [
            [bounds[0], bounds[2]],
            [bounds[1], bounds[2]],
            [bounds[1], bounds[3]],
            [bounds[0], bounds[3]],
        ],
        dtype=float,
    )

    def _cross(point):
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

    def _intersect(start, end):
        s_val = _cross(start)
        e_val = _cross(end)
        denom = s_val - e_val
        if abs(denom) < 1e-14:
            return end
        t = s_val / denom
        return start + t * (end - start)

    clipped = []
    for start, end in zip(rect, np.roll(rect, -1, axis=0)):
        start_inside = _cross(start) >= 0
        end_inside = _cross(end) >= 0
        if start_inside and end_inside:
            clipped.append(end)
        elif start_inside and not end_inside:
            clipped.append(_intersect(start, end))
        elif (not start_inside) and end_inside:
            clipped.append(_intersect(start, end))
            clipped.append(end)
    if not clipped:
        return np.empty((0, 2), dtype=float)
    return np.asarray(clipped, dtype=float)


def _ellipse_true_segments(case_index: int, sample_count: int = 720) -> np.ndarray:
    params = _ellipse_case_params(case_index)
    center = params["center"]
    theta = params["theta"]
    a = params["major_axis"]
    b = params["minor_axis"]
    ts = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    pts = np.zeros((sample_count, 2), dtype=float)
    c = math.cos(theta)
    s = math.sin(theta)
    for i, t in enumerate(ts):
        x_local = a * math.cos(t)
        y_local = b * math.sin(t)
        pts[i, 0] = center[0] + c * x_local - s * y_local
        pts[i, 1] = center[1] + s * x_local + c * y_local
    pts = np.vstack([pts, pts[0]])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _circle_case_params(case_index: int, radius: float = 10.0) -> dict:
    rng = np.random.default_rng(CIRCLE_RANDOM_SEED)
    for i in range(25):
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        if i == case_index:
            return {"center": np.asarray(center, dtype=float), "radius": float(radius)}
    raise ValueError(f"Invalid circle case index: {case_index}")


def _circle_boundary_points(case_index: int, sample_count: int = 720) -> np.ndarray:
    params = _circle_case_params(case_index)
    ts = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    pts = np.zeros((sample_count, 2), dtype=float)
    for i, t in enumerate(ts):
        pts[i, 0] = params["center"][0] + params["radius"] * math.cos(t)
        pts[i, 1] = params["center"][1] + params["radius"] * math.sin(t)
    return pts


def _circle_true_segments(case_index: int, sample_count: int = 720) -> np.ndarray:
    pts = _circle_boundary_points(case_index, sample_count=sample_count)
    pts = np.vstack([pts, pts[0]])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _square_case_params(case_index: int) -> dict:
    rng = np.random.default_rng(SQUARE_RANDOM_SEED)
    side_lengths = np.linspace(10, 30, 25)
    for i, side_length in enumerate(side_lengths):
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        theta = rng.uniform(0, math.pi / 2)
        if i == case_index:
            half_side = side_length / 2
            square = [
                [-half_side, -half_side],
                [half_side, -half_side],
                [half_side, half_side],
                [-half_side, half_side],
            ]
            rotated_square = []
            for point in square:
                x = point[0] * math.cos(theta) - point[1] * math.sin(theta)
                y = point[0] * math.sin(theta) + point[1] * math.cos(theta)
                rotated_square.append([x + center[0], y + center[1]])
            return {
                "center": np.asarray(center, dtype=float),
                "theta": float(theta),
                "side_length": float(side_length),
                "polygon": np.asarray(rotated_square, dtype=float),
            }
    raise ValueError(f"Invalid square case index: {case_index}")


def _square_true_segments(case_index: int) -> np.ndarray:
    pts = _square_case_params(case_index)["polygon"]
    pts = np.vstack([pts, pts[0]])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _facet_segments(facet, *, arc_samples: int = 256) -> np.ndarray:
    if isinstance(facet, (ArcFacet, CornerFacet)):
        points = np.asarray(facet.sample(arc_samples), dtype=float)
    else:
        points = np.asarray(facet.sample(2), dtype=float)
    if len(points) < 2:
        return np.empty((0, 2, 2), dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)


def _concat_facet_points(facets, *, arc_samples: int = 256) -> np.ndarray:
    points = []
    for facet in facets:
        sample_count = arc_samples if isinstance(facet, (ArcFacet, CornerFacet)) else 2
        sampled = np.asarray(facet.sample(sample_count), dtype=float)
        if len(sampled) == 0:
            continue
        if not points:
            points.extend(sampled.tolist())
            continue
        if np.allclose(points[-1], sampled[0], atol=1e-8):
            points.extend(sampled[1:].tolist())
        else:
            points.extend(sampled.tolist())
    return np.asarray(points, dtype=float)


def _zalesak_case_params(
    case_index: int,
    radius: float = 15.0,
    slot_width: float = 5.0,
    slot_top_rel: float = 10.0,
) -> dict:
    rng = np.random.default_rng(ZALESAK_RANDOM_SEED)
    for i in range(25):
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        theta = rng.uniform(0, math.pi / 2)
        if i == case_index:
            cx, cy = center
            half_w = slot_width * 0.5
            y_bottom = cy - radius - 1.0e-6
            y_top = cy + slot_top_rel
            rect = [
                [cx - half_w, y_bottom],
                [cx + half_w, y_bottom],
                [cx + half_w, y_top],
                [cx - half_w, y_top],
            ]
            rect = [rotate_point_around_center(point, center, theta) for point in rect]
            return {
                "center": np.asarray(center, dtype=float),
                "theta": float(theta),
                "slot_rect": np.asarray(rect, dtype=float),
                "radius": float(radius),
                "slot_width": float(slot_width),
                "slot_top_rel": float(slot_top_rel),
            }
    raise ValueError(f"Invalid zalesak case index: {case_index}")


def _zalesak_true_facets(case_index: int):
    params = _zalesak_case_params(case_index)
    return build_true_reference_zalesak(
        params["center"].tolist(),
        params["radius"],
        params["slot_rect"].tolist(),
        params["theta"],
    )["facets"]


def _zalesak_true_segments(case_index: int) -> np.ndarray:
    chunks = [_facet_segments(facet) for facet in _zalesak_true_facets(case_index)]
    chunks = [chunk for chunk in chunks if len(chunk)]
    if not chunks:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


def _load_true_segments(exp_name: str, save_name: str, case_index: int) -> np.ndarray:
    if exp_name == "squares":
        return _square_true_segments(case_index)
    if exp_name == "circles":
        return _circle_true_segments(case_index)
    if exp_name == "ellipses":
        return _ellipse_true_segments(case_index)
    if exp_name == "zalesak":
        return _zalesak_true_segments(case_index)
    true_path = _true_vtp_path(exp_name, save_name, case_index)
    return _segments_from_polydata(_read_polydata(true_path))


def _load_reconstructed_segments(save_name: str, case_index: int) -> np.ndarray:
    facet_path = (
        PLOTS_ROOT / save_name / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    )
    return _segments_from_polydata(_read_polydata(facet_path))


def _load_reconstructed_segments_and_endpoints(
    save_name: str, case_index: int
) -> tuple[np.ndarray, np.ndarray]:
    facet_path = (
        PLOTS_ROOT / save_name / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    )
    poly = _read_polydata(facet_path)
    return _segments_from_polydata(poly), _facet_endpoints_from_polydata(poly)


def _segments_bounds(segments: np.ndarray) -> tuple[float, float, float, float]:
    if len(segments) == 0:
        return (0.0, 1.0, 0.0, 1.0)
    pts = segments.reshape(-1, 2)
    return (
        float(np.min(pts[:, 0])),
        float(np.max(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 1])),
    )


def _compute_view_bounds(
    segments: np.ndarray,
    *,
    min_span: float,
    margin_frac: float,
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = _segments_bounds(segments)
    width = max(xmax - xmin, min_span)
    height = max(ymax - ymin, min_span)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    width *= 1.0 + 2.0 * margin_frac
    height *= 1.0 + 2.0 * margin_frac
    return (
        cx - width / 2.0,
        cx + width / 2.0,
        cy - height / 2.0,
        cy + height / 2.0,
    )


def _add_segments(ax, segments: np.ndarray, *, color: str, linewidth: float, alpha: float = 1.0, linestyle: str | tuple = "-", zorder: int = 1):
    if len(segments) == 0:
        return
    coll = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
        capstyle="round",
    )
    coll.set_rasterized(True)
    ax.add_collection(coll)


def _add_fill_patch(ax, vertices: np.ndarray, *, facecolor: str = FLUID_FILL_COLOR, alpha: float = FLUID_FILL_ALPHA, zorder: int = 0):
    if len(vertices) < 3:
        return
    patch = PolygonPatch(
        vertices,
        closed=True,
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    patch.set_rasterized(True)
    ax.add_patch(patch)


def _add_true_region_fill(
    ax,
    exp_name: str,
    spec: dict,
    bounds: tuple[float, float, float, float],
):
    case_index = spec["case_index"]
    if exp_name == "lines":
        vertices = _line_fill_polygon(case_index, bounds)
        _add_fill_patch(ax, vertices)
        return
    if exp_name == "squares":
        _add_fill_patch(ax, _square_case_params(case_index)["polygon"])
        return
    if exp_name == "circles":
        _add_fill_patch(ax, _circle_boundary_points(case_index))
        return
    if exp_name == "ellipses":
        ellipse_segments = _ellipse_true_segments(case_index)
        pts = ellipse_segments[:, 0, :]
        _add_fill_patch(ax, pts)
        return
    if exp_name == "zalesak":
        _add_fill_patch(ax, _concat_facet_points(_zalesak_true_facets(case_index)))
        return


def _generate_quantitative_panel(exp_name: str, exp_data: dict, methods: list[str], metrics: tuple[str, str], out_path: Path):
    metric_left, metric_right = metrics
    filtered = {algo: exp_data[algo] for algo in methods if algo in exp_data}
    wiggle_curves = {}
    resolution_curves = {}
    for metric in metrics:
        curves_w = _build_method_curves(filtered, metric)
        if curves_w:
            wiggle_curves[metric] = curves_w
        curves_r = _build_method_curves_by_resolution(filtered, metric)
        if curves_r:
            resolution_curves[metric] = curves_r

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8))
    subplot_defs = [
        (0, 0, metric_left, wiggle_curves, PERTURBATION_AXIS_LABEL, "perturbation"),
        (0, 1, metric_left, resolution_curves, RESOLUTION_AXIS_LABEL, "resolution"),
        (1, 0, metric_right, wiggle_curves, PERTURBATION_AXIS_LABEL, "perturbation"),
        (1, 1, metric_right, resolution_curves, RESOLUTION_AXIS_LABEL, "resolution"),
    ]
    legend_entries = {}
    for row, col, metric, curve_map, xlabel, x_mode in subplot_defs:
        ax = axes[row][col]
        curves = curve_map.get(metric)
        if not curves:
            ax.set_axis_off()
            continue
        _draw_method_curves(
            ax,
            curves,
            metric,
            x_label=xlabel,
            x_mode=x_mode,
            exp_name=exp_name,
        )
        axis_phrase = (
            PERTURBATION_AXIS_LABEL.lower()
            if x_mode == "perturbation"
            else "cells per side"
        )
        ax.set_title(
            f"{metric.replace('_', ' ').title()} vs {axis_phrase}",
            fontsize=11.5,
            fontweight="bold",
        )
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in legend_entries:
                legend_entries[label] = handle

    for row in range(2):
        row_axes = [axes[row][col] for col in range(2) if axes[row][col].axison]
        if row_axes:
            ymin = min(ax.get_ylim()[0] for ax in row_axes)
            ymax = max(ax.get_ylim()[1] for ax in row_axes)
            for ax in row_axes:
                ax.set_ylim(ymin, ymax)

    for col in range(2):
        col_axes = [axes[row][col] for row in range(2) if axes[row][col].axison]
        if col_axes:
            xmin = min(ax.get_xlim()[0] for ax in col_axes)
            xmax = max(ax.get_xlim()[1] for ax in col_axes)
            for ax in col_axes:
                ax.set_xlim(xmin, xmax)

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(5, len(legend_entries)),
            fontsize=9.5,
            frameon=True,
            bbox_to_anchor=(0.5, -0.005),
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _inset_bounds(exp_name: str, spec: dict) -> tuple[float, float, float, float] | None:
    inset_spec = spec.get("inset")
    if not inset_spec:
        return None
    case_index = spec["case_index"]
    if inset_spec["kind"] == "square_corner":
        polygon = _square_case_params(case_index)["polygon"]
        corner = polygon[np.argmax(np.sum(polygon, axis=1))]
        half_span = 4.0
        return (
            float(corner[0] - half_span),
            float(corner[0] + half_span),
            float(corner[1] - half_span),
            float(corner[1] + half_span),
        )
    if inset_spec["kind"] == "zalesak_corner":
        slot_rect = _zalesak_case_params(case_index)["slot_rect"]
        corner = slot_rect[np.argmax(slot_rect[:, 0] + slot_rect[:, 1])]
        half_span = 4.5
        return (
            float(corner[0] - half_span),
            float(corner[0] + half_span),
            float(corner[1] - half_span),
            float(corner[1] + half_span),
        )
    return None


def _plot_panel(
    ax,
    *,
    exp_name: str,
    spec: dict,
    algo: str,
    mesh_segments: np.ndarray,
    true_segments: np.ndarray,
    recon_segments: np.ndarray,
    endpoint_points: np.ndarray,
    title: str,
    bounds: tuple[float, float, float, float],
):
    x0, x1, y0, y1 = bounds
    _add_true_region_fill(ax, exp_name, spec, bounds)
    mesh_linewidth = 0.42 if exp_name == "lines" else 0.32
    mesh_alpha = 0.72 if exp_name == "lines" else 0.58
    _add_segments(
        ax,
        mesh_segments,
        color=MESH_COLOR,
        linewidth=mesh_linewidth,
        alpha=mesh_alpha,
        zorder=1,
    )
    _add_segments(
        ax,
        true_segments,
        color=TRUE_COLOR,
        linewidth=0.95,
        alpha=0.90,
        linestyle=TRUE_STYLE,
        zorder=2,
    )

    color = METHOD_STYLES.get(algo, {}).get("color", "#1f77b4")
    _add_segments(
        ax,
        recon_segments,
        color=color,
        linewidth=1.55,
        alpha=1.0,
        linestyle="-",
        zorder=3,
    )
    if len(endpoint_points):
        ax.scatter(
            endpoint_points[:, 0],
            endpoint_points[:, 1],
            s=ENDPOINT_MARKER_SIZE,
            facecolors="white",
            edgecolors=color,
            alpha=0.95,
            zorder=4,
            linewidths=0.65,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11.0, fontweight="bold")

    inset_bounds = _inset_bounds(exp_name, spec)
    if inset_bounds is not None:
        ix0, ix1, iy0, iy1 = inset_bounds
        inset = ax.inset_axes([0.56, 0.05, 0.39, 0.39])
        _add_true_region_fill(inset, exp_name, spec, inset_bounds)
        _add_segments(
            inset,
            mesh_segments,
            color=MESH_COLOR,
            linewidth=max(0.22, mesh_linewidth - 0.08),
            alpha=min(0.7, mesh_alpha),
            zorder=1,
        )
        _add_segments(
            inset,
            true_segments,
            color=TRUE_COLOR,
            linewidth=0.85,
            alpha=0.90,
            linestyle=TRUE_STYLE,
            zorder=2,
        )
        _add_segments(
            inset,
            recon_segments,
            color=color,
            linewidth=1.20,
            alpha=1.0,
            linestyle="-",
            zorder=3,
        )
        if len(endpoint_points):
            inset.scatter(
                endpoint_points[:, 0],
                endpoint_points[:, 1],
                s=ENDPOINT_MARKER_SIZE * 0.8,
                facecolors="white",
                edgecolors=color,
                alpha=0.95,
                zorder=4,
                linewidths=0.55,
            )
        inset.set_xlim(ix0, ix1)
        inset.set_ylim(iy0, iy1)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_aspect("equal", adjustable="box")
        inset.set_facecolor("white")
        ax.indicate_inset_zoom(inset, edgecolor="#374151", alpha=0.8)


def _generate_representative_figure(exp_name: str, spec: dict, out_path: Path):
    base_method = spec["methods"][0][0]
    base_save_name = _make_save_name(
        exp_name,
        base_method,
        spec["resolution"],
        spec["wiggle"],
        spec["seed"],
    )
    mesh_path = PLOTS_ROOT / base_save_name / "vtk" / "mesh.vtk"
    mesh_segments = _mesh_segments(mesh_path)
    if exp_name == "lines":
        x0, x1, y0, y1 = _segments_bounds(mesh_segments)
        true_segments = _line_true_segments(spec["case_index"], (x0, x1, y0, y1))
    else:
        true_segments = _load_true_segments(exp_name, base_save_name, spec["case_index"])
        x0, x1, y0, y1 = _compute_view_bounds(
            true_segments,
            min_span=spec["min_span"],
            margin_frac=spec["margin_frac"],
        )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6))
    flat_axes = axes.ravel()
    for ax, (algo, title) in zip(flat_axes, spec["methods"]):
        save_name = _make_save_name(
            exp_name,
            algo,
            spec["resolution"],
            spec["wiggle"],
            spec["seed"],
        )
        recon_segments, endpoint_points = _load_reconstructed_segments_and_endpoints(
            save_name, spec["case_index"]
        )
        _plot_panel(
            ax,
            exp_name=exp_name,
            spec=spec,
            algo=algo,
            mesh_segments=mesh_segments,
            true_segments=true_segments,
            recon_segments=recon_segments,
            endpoint_points=endpoint_points,
            title=title,
            bounds=(x0, x1, y0, y1),
        )

    for ax in flat_axes[len(spec["methods"]) :]:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _generate_resolution_strip(exp_name: str, spec: dict, out_path: Path):
    method, title = spec["method"]
    ncols = len(spec["resolutions"])
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.3))
    if ncols == 1:
        axes = [axes]

    for ax, resolution in zip(axes, spec["resolutions"]):
        save_name = _make_save_name(
            exp_name,
            method,
            resolution,
            spec["wiggle"],
            spec["seed"],
        )
        mesh_path = PLOTS_ROOT / save_name / "vtk" / "mesh.vtk"
        mesh_segments = _mesh_segments(mesh_path)
        if exp_name == "lines":
            x0m, x1m, y0m, y1m = _segments_bounds(mesh_segments)
            true_segments = _line_true_segments(spec["case_index"], (x0m, x1m, y0m, y1m))
        else:
            true_segments = _load_true_segments(exp_name, save_name, spec["case_index"])
        bounds = _compute_view_bounds(
            true_segments,
            min_span=spec["min_span"],
            margin_frac=spec["margin_frac"],
        )
        recon_segments, endpoint_points = _load_reconstructed_segments_and_endpoints(
            save_name, spec["case_index"]
        )
        _plot_panel(
            ax,
            exp_name=exp_name,
            spec={"case_index": spec["case_index"], "inset": None},
            algo=method,
            mesh_segments=mesh_segments,
            true_segments=true_segments,
            recon_segments=recon_segments,
            endpoint_points=endpoint_points,
            title=f"N={int(round(resolution * 100))}",
            bounds=bounds,
        )

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Section 6 main-text figures.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Merged Section 6 CSV to use as the quantitative source.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "static"
            / "camera_ready"
            / "static_cameraready_maintext_20260319"
        ),
        help="Output directory for generated main-text figures.",
    )
    args = parser.parse_args()

    rows = _load_sweep_rows(args.csv)
    rows = _backfill_circle_tangent_rows(rows)
    metric_index = _build_metric_index(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.out_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = args.out_dir / "representative_cases"
    compare_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir = args.out_dir / "appendix_cases"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "quantitative": {},
        "representative": {},
        "appendix_resolutions": {},
        "appendix_cartesian": {},
        "cases": REPRESENTATIVE_CASES,
    }
    for exp_name, methods in MAINTEXT_METHODS.items():
        out_name = f"{exp_name}_maintext_metrics.png"
        out_path = summary_dir / out_name
        _generate_quantitative_panel(
            exp_name=exp_name,
            exp_data=metric_index.get(exp_name, {}),
            methods=methods,
            metrics=QUANT_SPECS[exp_name]["metrics"],
            out_path=out_path,
        )
        outputs["quantitative"][exp_name] = str(out_path)

    for exp_name, spec in REPRESENTATIVE_CASES.items():
        out_name = f"{exp_name}_maintext_representative.png"
        out_path = compare_dir / out_name
        _generate_representative_figure(exp_name, spec, out_path)
        outputs["representative"][exp_name] = str(out_path)

    for exp_name, spec in APPENDIX_BEST_METHODS.items():
        out_name = f"{exp_name}_best_by_resolution.png"
        out_path = appendix_dir / out_name
        _generate_resolution_strip(exp_name, spec, out_path)
        outputs["appendix_resolutions"][exp_name] = str(out_path)

    for exp_name, spec in APPENDIX_CARTESIAN_CASES.items():
        out_name = f"{exp_name}_cartesian_representative.png"
        out_path = appendix_dir / out_name
        _generate_representative_figure(exp_name, spec, out_path)
        outputs["appendix_cartesian"][exp_name] = str(out_path)

    manifest_path = args.out_dir / "maintext_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
