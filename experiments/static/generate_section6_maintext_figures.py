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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import vtk
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


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
from main.structs.interface_geometry import ArcPrimitive, LinePrimitive


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
        "case_index": 6,
        "methods": [
            ("Youngs", "Youngs"),
            ("ELVIRA", "ELVIRA"),
            ("LVIRA", "LVIRA"),
            ("linear", "Ours (linear)"),
        ],
        "min_span": 100.0,
        "margin_frac": 0.00,
        "inset": {"kind": "line_fit", "half_span": 4.0},
    },
    "squares": {
        "resolution": 0.50,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 24,
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
        "case_index": 0,
        "min_span": 100.0,
        "margin_frac": 0.00,
    },
    "squares": {
        "method": ("linear+corner", "Ours (linear+corner)"),
        "resolutions": [0.50, 0.64, 1.0, 1.5],
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 22,
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
        "case_index": 20,
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
ENDPOINT_MARKER_SIZE_MAIN = 5.5
ENDPOINT_MARKER_SIZE_INSET = 8.0
CORNER_TIP_MARKER_SIZE_MAIN = 12.0
CORNER_TIP_MARKER_SIZE_INSET = 18.0
CORNER_CROSSING_MARKER_SIZE_MAIN = 10.0
CORNER_CROSSING_MARKER_SIZE_INSET = 15.0
CORNER_TIP_CLUSTER_FACTOR = 2.0
CORNER_TIP_CLUSTER_MAX_BY_EXPERIMENT = {
    "zalesak": 2.4,
}
SPYGLASS_OUTER_SIZE = 0.32
SPYGLASS_OUTER_GAP = 0.06
SPYGLASS_OUTER_BOTTOM = 0.04
SPYGLASS_FRAME_COLOR = "#7e22ce"
FIGURE_GROUPS = {
    "quantitative",
    "representative",
    "appendix_resolutions",
    "appendix_cartesian",
}
ENDPOINT_VARIANT_MODES = {"annotated", "clean", "paired"}


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


def _read_corner_tip_metadata(facet_path: Path) -> np.ndarray:
    metadata_path = facet_path.with_suffix(".corner_tips.json")
    if not metadata_path.exists():
        return np.empty((0, 2), dtype=float)
    data = json.loads(metadata_path.read_text())
    points = []
    for entry in data.get("corner_tips", []):
        if entry.get("kind") != "corner":
            continue
        point = entry.get("point")
        if point is None or len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _read_facet_metadata(facet_path: Path) -> dict | None:
    metadata_path = facet_path.with_suffix(".facet_metadata.json")
    if not metadata_path.exists():
        return None
    data = json.loads(metadata_path.read_text())
    if data.get("schema_version", 0) < 2:
        return None
    return data


def _primitive_from_metadata(record: dict):
    if record.get("kind") == "arc":
        return ArcPrimitive(
            center=record["center"],
            radius=float(record["radius"]),
            pLeft=record["p_left"],
            pRight=record["p_right"],
            source_name=record.get("source_name", "arc"),
        )
    return LinePrimitive(
        pLeft=record["p_left"],
        pRight=record["p_right"],
        source_name=record.get("source_name", "linear"),
    )


def _metadata_plot_geometry(
    metadata: dict,
    mesh_segments: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild smooth plot geometry from exact primitive records when present."""
    step = _mesh_step_from_segments(mesh_segments) if mesh_segments is not None else None
    max_spacing = max((step or 1.0) / 4.0, 1.0e-3)
    segments = []
    endpoints = []
    for record in metadata.get("primitives", []):
        primitive = _primitive_from_metadata(record)
        points = np.asarray(primitive.sample_by_max_spacing(max_spacing), dtype=float)
        if len(points) >= 2:
            segments.append(np.stack([points[:-1], points[1:]], axis=1))
        endpoints.extend([record["p_left"], record["p_right"]])
    if not segments:
        segment_array = np.empty((0, 2, 2), dtype=float)
    else:
        segment_array = np.concatenate(segments, axis=0)
    endpoint_array = (
        np.asarray(endpoints, dtype=float)
        if endpoints
        else np.empty((0, 2), dtype=float)
    )
    return segment_array, endpoint_array


def _cross2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _line_segment_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> np.ndarray | None:
    direction = p1 - p0
    boundary = q1 - q0
    denominator = _cross2d(direction, boundary)
    tolerance = 1.0e-10
    if abs(denominator) <= tolerance:
        return None
    offset = q0 - p0
    t = _cross2d(offset, boundary) / denominator
    u = _cross2d(offset, direction) / denominator
    if -tolerance <= t <= 1.0 + tolerance and -tolerance <= u <= 1.0 + tolerance:
        return p0 + np.clip(t, 0.0, 1.0) * direction
    return None


def _arc_segment_intersections(record: dict, q0: np.ndarray, q1: np.ndarray) -> list[np.ndarray]:
    center = np.asarray(record["center"], dtype=float)
    radius = abs(float(record["radius"]))
    direction = q1 - q0
    quadratic_a = float(np.dot(direction, direction))
    if quadratic_a <= 1.0e-24 or radius <= 1.0e-14:
        return []
    offset = q0 - center
    quadratic_b = 2.0 * float(np.dot(offset, direction))
    quadratic_c = float(np.dot(offset, offset)) - radius * radius
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
    discriminant_tolerance = 1.0e-10 * max(
        quadratic_b * quadratic_b,
        abs(4.0 * quadratic_a * quadratic_c),
        1.0,
    )
    if discriminant < -discriminant_tolerance:
        return []
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    primitive = _primitive_from_metadata(record)
    intersections = []
    for t in {
        (-quadratic_b - root) / (2.0 * quadratic_a),
        (-quadratic_b + root) / (2.0 * quadratic_a),
    }:
        if -1.0e-10 <= t <= 1.0 + 1.0e-10:
            point = q0 + np.clip(t, 0.0, 1.0) * direction
            if primitive._facet.pointInArcRange(point.tolist()):
                intersections.append(point)
    return intersections


def _deduplicate_points(points: list[np.ndarray], tolerance: float) -> np.ndarray:
    unique = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    if not unique:
        return np.empty((0, 2), dtype=float)
    return np.asarray(unique, dtype=float)


def _corner_boundary_crossings(
    metadata: dict,
    mesh_segments: np.ndarray | None,
) -> np.ndarray:
    """Find interior crossings of each saved corner side with mesh boundaries."""
    if mesh_segments is None or len(mesh_segments) == 0:
        return np.empty((0, 2), dtype=float)
    step = _mesh_step_from_segments(mesh_segments) or 1.0
    tolerance = max(1.0e-9, 1.0e-8 * step)
    crossings = []
    for corner in metadata.get("corners", []):
        for key in ("left_primitive", "right_primitive"):
            record = corner.get(key)
            if record is None:
                continue
            p_left = np.asarray(record["p_left"], dtype=float)
            p_right = np.asarray(record["p_right"], dtype=float)
            for boundary in mesh_segments:
                q_left = np.asarray(boundary[0], dtype=float)
                q_right = np.asarray(boundary[1], dtype=float)
                if record.get("kind") == "arc":
                    candidates = _arc_segment_intersections(record, q_left, q_right)
                else:
                    candidate = _line_segment_intersection(
                        p_left, p_right, q_left, q_right
                    )
                    candidates = [] if candidate is None else [candidate]
                for point in candidates:
                    if min(
                        np.linalg.norm(point - p_left),
                        np.linalg.norm(point - p_right),
                    ) <= tolerance:
                        continue
                    crossings.append(point)
    return _deduplicate_points(crossings, tolerance)


def _corner_tip_points_from_metadata(metadata: dict) -> np.ndarray:
    points = []
    for corner in metadata.get("corners", []):
        apex = corner.get("apex")
        if apex is not None and len(apex) >= 2:
            points.append([float(apex[0]), float(apex[1])])
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _mesh_step_from_segments(mesh_segments: np.ndarray) -> float | None:
    if len(mesh_segments) == 0:
        return None
    lengths = np.linalg.norm(mesh_segments[:, 1, :] - mesh_segments[:, 0, :], axis=1)
    lengths = lengths[lengths > 1.0e-10]
    if len(lengths) == 0:
        return None
    return float(np.median(lengths))


def _corner_tip_cluster_tolerance(
    exp_name: str | None,
    mesh_segments: np.ndarray | None,
) -> float:
    step = _mesh_step_from_segments(mesh_segments) if mesh_segments is not None else None
    if step is None:
        return 1.0e-6
    tol = CORNER_TIP_CLUSTER_FACTOR * step
    if exp_name in CORNER_TIP_CLUSTER_MAX_BY_EXPERIMENT:
        tol = min(tol, CORNER_TIP_CLUSTER_MAX_BY_EXPERIMENT[exp_name])
    return max(float(tol), 1.0e-6)


def _cluster_corner_tip_points(points: np.ndarray, tol: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)

    n = len(points)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        indices = []
        while stack:
            current = stack.pop()
            indices.append(current)
            distances = np.linalg.norm(points - points[current], axis=1)
            neighbors = np.where((distances <= tol) & (~visited))[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                stack.append(int(neighbor))
        clusters.append(points[indices])

    centers = [np.mean(cluster, axis=0) for cluster in clusters]
    return np.asarray(centers, dtype=float)


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


def _load_reconstructed_plot_geometry(
    save_name: str,
    case_index: int,
    *,
    exp_name: str | None = None,
    mesh_segments: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    facet_path = (
        PLOTS_ROOT / save_name / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    )
    poly = _read_polydata(facet_path)
    metadata = _read_facet_metadata(facet_path)
    if metadata is not None:
        recon_segments, endpoint_points = _metadata_plot_geometry(
            metadata,
            mesh_segments,
        )
        raw_corner_tip_points = _corner_tip_points_from_metadata(metadata)
        corner_boundary_points = _corner_boundary_crossings(metadata, mesh_segments)
    else:
        recon_segments = _segments_from_polydata(poly)
        endpoint_points = _facet_endpoints_from_polydata(poly)
        raw_corner_tip_points = _read_corner_tip_metadata(facet_path)
        corner_boundary_points = np.empty((0, 2), dtype=float)
    corner_tip_points = _cluster_corner_tip_points(
        raw_corner_tip_points,
        _corner_tip_cluster_tolerance(exp_name, mesh_segments),
    )
    return (
        recon_segments,
        endpoint_points,
        corner_tip_points,
        corner_boundary_points,
    )


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
    ax.add_collection(coll)


def _add_corner_markers(
    ax,
    *,
    color: str,
    corner_tip_points: np.ndarray | None,
    corner_boundary_points: np.ndarray | None,
    tip_size: float,
    crossing_size: float,
):
    if corner_boundary_points is not None and len(corner_boundary_points):
        ax.scatter(
            corner_boundary_points[:, 0],
            corner_boundary_points[:, 1],
            s=crossing_size,
            marker="o",
            facecolors=color,
            edgecolors="white",
            alpha=0.98,
            zorder=4.5,
            linewidths=0.55,
        )
    if corner_tip_points is not None and len(corner_tip_points):
        ax.scatter(
            corner_tip_points[:, 0],
            corner_tip_points[:, 1],
            s=tip_size,
            marker="D",
            facecolors=color,
            edgecolors="#111827",
            alpha=0.98,
            zorder=5,
            linewidths=0.45,
        )


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
    ax.add_patch(patch)


def _save_figure(fig, out_path: Path):
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def _prefixed_save_name(
    exp_name: str,
    algo: str,
    resolution: float,
    wiggle: float,
    seed: int,
    *,
    save_prefix: str | None = None,
) -> str:
    base = _make_save_name(exp_name, algo, resolution, wiggle, seed)
    return f"{save_prefix}_{base}" if save_prefix else base


def _parse_figure_groups(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(FIGURE_GROUPS)
    groups = {part.strip() for part in raw.split(",") if part.strip()}
    if "appendix" in groups:
        groups.remove("appendix")
        groups.update({"appendix_resolutions", "appendix_cartesian"})
    unknown = sorted(groups - FIGURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown figure groups requested: {', '.join(unknown)}")
    return groups


def _parse_case_overrides(raw: str | None) -> dict[str, int]:
    if raw is None or not raw.strip():
        return {}
    overrides = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "Case overrides must look like experiment=index, e.g. squares=24"
            )
        exp_name, value = part.split("=", 1)
        overrides[exp_name.strip()] = int(value.strip())
    unknown = sorted(set(overrides) - set(MAINTEXT_METHODS))
    if unknown:
        raise ValueError(f"Unknown case override experiments: {', '.join(unknown)}")
    return overrides


def _endpoint_variant_specs(mode: str) -> list[tuple[str, str, bool]]:
    """Return (variant name, filename suffix, main-endpoint visibility)."""
    if mode == "annotated":
        return [("with_endpoints", "", True)]
    if mode == "clean":
        return [("clean", "", False)]
    if mode == "paired":
        return [
            ("with_endpoints", "_with_endpoints", True),
            ("clean", "_clean", False),
        ]
    raise ValueError(f"Unsupported endpoint variant mode: {mode}")


def _endpoint_visibility_spec(spec: dict, *, show_main_endpoints: bool) -> dict:
    variant = dict(spec)
    variant["show_main_endpoints"] = show_main_endpoints
    variant["show_inset_endpoints"] = True
    return variant


def _resolution_panel_spec(spec: dict) -> dict:
    return {
        "case_index": spec["case_index"],
        "inset": None,
        "show_main_endpoints": spec.get("show_main_endpoints", True),
        "show_inset_endpoints": spec.get("show_inset_endpoints", True),
    }


def _add_true_region_fill(
    ax,
    exp_name: str,
    spec: dict,
    bounds: tuple[float, float, float, float],
):
    override_vertices = spec.get("true_fill_vertices")
    if override_vertices is not None:
        _add_fill_patch(ax, np.asarray(override_vertices, dtype=float))
        return
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
    _save_figure(fig, out_path)
    plt.close(fig)


def _inset_bounds(exp_name: str, spec: dict) -> tuple[float, float, float, float] | None:
    override_bounds = spec.get("inset_bounds")
    if override_bounds is not None:
        return tuple(float(v) for v in override_bounds)
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
    if inset_spec["kind"] == "line_fit":
        params = _line_case_params(case_index)
        center = 0.5 * (params["p1"] + params["p2"])
        half_span = float(inset_spec.get("half_span", 4.0))
        return (
            float(center[0] - half_span),
            float(center[0] + half_span),
            float(center[1] - half_span),
            float(center[1] + half_span),
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


def _outer_spyglass_axes(
    side: str,
    *,
    size: float = SPYGLASS_OUTER_SIZE,
    gap: float = SPYGLASS_OUTER_GAP,
    bottom: float = SPYGLASS_OUTER_BOTTOM,
) -> list[float]:
    """Return inset-axes coordinates wholly outside the main data axes."""
    if side == "left":
        left = -gap - size
    elif side == "right":
        left = 1.0 + gap
    else:
        raise ValueError(f"Unsupported spyglass side: {side}")
    return [left, bottom, size, size]


def _panel_spyglass_spec(spec: dict, column: int) -> dict:
    """Place panel spyglasses in the reserved outer margin for their column."""
    panel_spec = dict(spec)
    if not panel_spec.get("inset"):
        return panel_spec
    panel_spec.pop("inset_axes", None)
    panel_spec["inset_side"] = "left" if column == 0 else "right"
    panel_spec.setdefault("inset_connector", "frame")
    return panel_spec


def _resolve_spyglass_axes(spec: dict) -> list[float]:
    explicit = spec.get("inset_axes")
    if explicit is not None:
        return list(explicit)
    side = spec.get("inset_side")
    if side is not None:
        return _outer_spyglass_axes(
            side,
            size=float(spec.get("inset_size", SPYGLASS_OUTER_SIZE)),
            gap=float(spec.get("inset_gap", SPYGLASS_OUTER_GAP)),
            bottom=float(spec.get("inset_bottom", SPYGLASS_OUTER_BOTTOM)),
        )
    return [0.56, 0.05, 0.39, 0.39]


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
    corner_tip_points: np.ndarray | None = None,
    corner_boundary_points: np.ndarray | None = None,
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
    show_main_endpoints = spec.get("show_main_endpoints", True)
    if show_main_endpoints and len(endpoint_points):
        ax.scatter(
            endpoint_points[:, 0],
            endpoint_points[:, 1],
            s=ENDPOINT_MARKER_SIZE_MAIN,
            facecolors="white",
            edgecolors=color,
            alpha=0.95,
            zorder=4,
            linewidths=0.65,
        )
    _add_corner_markers(
        ax,
        color=color,
        corner_tip_points=corner_tip_points,
        corner_boundary_points=(
            corner_boundary_points if show_main_endpoints else None
        ),
        tip_size=CORNER_TIP_MARKER_SIZE_MAIN,
        crossing_size=CORNER_CROSSING_MARKER_SIZE_MAIN,
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
        inset = ax.inset_axes(_resolve_spyglass_axes(spec))
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
        show_inset_endpoints = spec.get("show_inset_endpoints", True)
        if show_inset_endpoints and len(endpoint_points):
            inset.scatter(
                endpoint_points[:, 0],
                endpoint_points[:, 1],
                s=ENDPOINT_MARKER_SIZE_INSET,
                facecolors="white",
                edgecolors=color,
                alpha=0.95,
                zorder=4,
                linewidths=0.55,
            )
        _add_corner_markers(
            inset,
            color=color,
            corner_tip_points=corner_tip_points,
            corner_boundary_points=(
                corner_boundary_points if show_inset_endpoints else None
            ),
            tip_size=CORNER_TIP_MARKER_SIZE_INSET,
            crossing_size=CORNER_CROSSING_MARKER_SIZE_INSET,
        )
        inset.set_xlim(ix0, ix1)
        inset.set_ylim(iy0, iy1)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_aspect("equal", adjustable="box")
        inset.set_facecolor("white")
        connector_style = spec.get("inset_connector", "leaders")
        frame_color = spec.get("inset_frame_color", SPYGLASS_FRAME_COLOR)
        for spine in inset.spines.values():
            spine.set_color(frame_color if connector_style == "frame" else "#374151")
            spine.set_linewidth(1.15 if connector_style == "frame" else 0.9)
        if connector_style == "frame":
            source_box = Rectangle(
                (ix0, iy0),
                ix1 - ix0,
                iy1 - iy0,
                fill=False,
                edgecolor=frame_color,
                linewidth=1.05,
                linestyle=(0, (3.0, 2.0)),
                zorder=6,
            )
            ax.add_patch(source_box)
        else:
            source_box, connectors = ax.indicate_inset_zoom(
                inset,
                edgecolor="#374151",
                alpha=0.75,
            )
            source_box.set_linewidth(0.8)
            for connector in connectors:
                connector.set_linewidth(0.65)
                connector.set_clip_on(False)


def _generate_representative_figure(
    exp_name: str,
    spec: dict,
    out_path: Path,
    *,
    save_prefix: str | None = None,
):
    base_method = spec["methods"][0][0]
    base_save_name = _prefixed_save_name(
        exp_name,
        base_method,
        spec["resolution"],
        spec["wiggle"],
        spec["seed"],
        save_prefix=save_prefix,
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

    has_spyglass = bool(spec.get("inset"))
    fig_width = 10.4 if has_spyglass else 8.2
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 7.6))
    flat_axes = axes.ravel()
    for panel_index, (ax, (algo, title)) in enumerate(
        zip(flat_axes, spec["methods"])
    ):
        panel_spec = _panel_spyglass_spec(spec, panel_index % 2)
        save_name = _prefixed_save_name(
            exp_name,
            algo,
            spec["resolution"],
            spec["wiggle"],
            spec["seed"],
            save_prefix=save_prefix,
        )
        (
            recon_segments,
            endpoint_points,
            corner_tip_points,
            corner_boundary_points,
        ) = _load_reconstructed_plot_geometry(
            save_name,
            spec["case_index"],
            exp_name=exp_name,
            mesh_segments=mesh_segments,
        )
        _plot_panel(
            ax,
            exp_name=exp_name,
            spec=panel_spec,
            algo=algo,
            mesh_segments=mesh_segments,
            true_segments=true_segments,
            recon_segments=recon_segments,
            endpoint_points=endpoint_points,
            corner_tip_points=corner_tip_points,
            corner_boundary_points=corner_boundary_points,
            title=title,
            bounds=(x0, x1, y0, y1),
        )

    for ax in flat_axes[len(spec["methods"]) :]:
        ax.set_axis_off()

    if has_spyglass:
        fig.subplots_adjust(
            left=0.16,
            right=0.84,
            bottom=0.06,
            top=0.97,
            wspace=0.24,
            hspace=0.18,
        )
    else:
        fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _generate_resolution_strip(
    exp_name: str,
    spec: dict,
    out_path: Path,
    *,
    save_prefix: str | None = None,
):
    method, title = spec["method"]
    ncols = len(spec["resolutions"])
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.3))
    if ncols == 1:
        axes = [axes]

    for ax, resolution in zip(axes, spec["resolutions"]):
        save_name = _prefixed_save_name(
            exp_name,
            method,
            resolution,
            spec["wiggle"],
            spec["seed"],
            save_prefix=save_prefix,
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
        (
            recon_segments,
            endpoint_points,
            corner_tip_points,
            corner_boundary_points,
        ) = _load_reconstructed_plot_geometry(
            save_name,
            spec["case_index"],
            exp_name=exp_name,
            mesh_segments=mesh_segments,
        )
        _plot_panel(
            ax,
            exp_name=exp_name,
            spec=_resolution_panel_spec(spec),
            algo=method,
            mesh_segments=mesh_segments,
            true_segments=true_segments,
            recon_segments=recon_segments,
            endpoint_points=endpoint_points,
            corner_tip_points=corner_tip_points,
            corner_boundary_points=corner_boundary_points,
            title=f"N={int(round(resolution * 100))}",
            bounds=bounds,
        )

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, out_path)
    plt.close(fig)


def main():
    global PLOTS_ROOT

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
    parser.add_argument(
        "--plots_root",
        type=Path,
        default=PLOTS_ROOT,
        help="Root containing saved per-run plot artifacts (default: repo plots/).",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated experiment names to regenerate (default: all).",
    )
    parser.add_argument(
        "--plot_save_prefix",
        type=str,
        default=None,
        help="Optional prefix prepended to plot save directories for non-clobbering previews.",
    )
    parser.add_argument(
        "--figure_groups",
        type=str,
        default="all",
        help=(
            "Comma-separated figure groups to regenerate: quantitative, representative, "
            "appendix_resolutions, appendix_cartesian, or appendix/all."
        ),
    )
    parser.add_argument(
        "--case_overrides",
        type=str,
        default=None,
        help="Optional representative case overrides, e.g. squares=24,zalesak=12.",
    )
    parser.add_argument(
        "--endpoint_variants",
        choices=sorted(ENDPOINT_VARIANT_MODES),
        default="annotated",
        help=(
            "Qualitative endpoint-marker exports: annotated, clean main panels, "
            "or paired (default: annotated). Spyglass endpoints are always retained."
        ),
    )
    args = parser.parse_args()
    PLOTS_ROOT = args.plots_root

    figure_groups = _parse_figure_groups(args.figure_groups)
    case_overrides = _parse_case_overrides(args.case_overrides)
    if "quantitative" in figure_groups:
        rows = _load_sweep_rows(args.csv)
        rows = _backfill_circle_tangent_rows(rows)
        metric_index = _build_metric_index(rows)
    else:
        metric_index = {}

    if args.experiments.strip().lower() == "all":
        selected_experiments = list(MAINTEXT_METHODS.keys())
    else:
        selected_experiments = [
            name.strip()
            for name in args.experiments.split(",")
            if name.strip()
        ]
        unknown = sorted(set(selected_experiments) - set(MAINTEXT_METHODS.keys()))
        if unknown:
            raise ValueError(f"Unknown experiments requested: {', '.join(unknown)}")

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
        "specs": {
            "representative": {},
            "appendix_resolutions": {},
            "appendix_cartesian": {},
        },
    }
    for exp_name, methods in MAINTEXT_METHODS.items():
        if "quantitative" not in figure_groups:
            continue
        if exp_name not in selected_experiments:
            continue
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
        if "representative" not in figure_groups:
            continue
        if exp_name not in selected_experiments:
            continue
        spec = dict(spec)
        if exp_name in case_overrides:
            spec["case_index"] = case_overrides[exp_name]
        variant_outputs = {}
        for variant_name, suffix, show_main_endpoints in _endpoint_variant_specs(
            args.endpoint_variants
        ):
            out_name = f"{exp_name}_maintext_representative{suffix}.png"
            out_path = compare_dir / out_name
            _generate_representative_figure(
                exp_name,
                _endpoint_visibility_spec(
                    spec,
                    show_main_endpoints=show_main_endpoints,
                ),
                out_path,
                save_prefix=args.plot_save_prefix,
            )
            variant_outputs[variant_name] = str(out_path)
        outputs["representative"][exp_name] = (
            variant_outputs
            if args.endpoint_variants == "paired"
            else next(iter(variant_outputs.values()))
        )
        outputs["specs"]["representative"][exp_name] = spec

    for exp_name, spec in APPENDIX_BEST_METHODS.items():
        if "appendix_resolutions" not in figure_groups:
            continue
        if exp_name not in selected_experiments:
            continue
        variant_outputs = {}
        for variant_name, suffix, show_main_endpoints in _endpoint_variant_specs(
            args.endpoint_variants
        ):
            out_name = f"{exp_name}_best_by_resolution{suffix}.png"
            out_path = appendix_dir / out_name
            _generate_resolution_strip(
                exp_name,
                _endpoint_visibility_spec(
                    spec,
                    show_main_endpoints=show_main_endpoints,
                ),
                out_path,
                save_prefix=args.plot_save_prefix,
            )
            variant_outputs[variant_name] = str(out_path)
        outputs["appendix_resolutions"][exp_name] = (
            variant_outputs
            if args.endpoint_variants == "paired"
            else next(iter(variant_outputs.values()))
        )
        outputs["specs"]["appendix_resolutions"][exp_name] = spec

    for exp_name, spec in APPENDIX_CARTESIAN_CASES.items():
        if "appendix_cartesian" not in figure_groups:
            continue
        if exp_name not in selected_experiments:
            continue
        variant_outputs = {}
        for variant_name, suffix, show_main_endpoints in _endpoint_variant_specs(
            args.endpoint_variants
        ):
            out_name = f"{exp_name}_cartesian_representative{suffix}.png"
            out_path = appendix_dir / out_name
            _generate_representative_figure(
                exp_name,
                _endpoint_visibility_spec(
                    spec,
                    show_main_endpoints=show_main_endpoints,
                ),
                out_path,
                save_prefix=args.plot_save_prefix,
            )
            variant_outputs[variant_name] = str(out_path)
        outputs["appendix_cartesian"][exp_name] = (
            variant_outputs
            if args.endpoint_variants == "paired"
            else next(iter(variant_outputs.values()))
        )
        outputs["specs"]["appendix_cartesian"][exp_name] = spec

    manifest_path = args.out_dir / "maintext_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
