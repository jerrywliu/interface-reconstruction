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
import pyvista as pv
from matplotlib.collections import LineCollection


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static.ellipses import RANDOM_SEED as ELLIPSE_RANDOM_SEED
from experiments.static.lines import RANDOM_SEED as LINE_RANDOM_SEED
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


PLOTS_ROOT = REPO_ROOT / "plots"
DEFAULT_CSV = (
    REPO_ROOT
    / "results"
    / "static"
    / "camera_ready"
    / "static_cameraready_plotrefresh_20260319"
    / "csv"
    / "section6_plotrefresh_merged.csv"
)

MAINTEXT_METHODS = {
    "lines": ["Youngs", "LVIRA", "linear"],
    "squares": ["LVIRA", "linear", "linear+corner"],
    "circles": ["LVIRA", "linear", "circular"],
    "ellipses": ["LVIRA", "linear", "circular"],
    "zalesak": ["LVIRA", "circular", "circular+corner"],
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
            ("LVIRA", "ELVIRA"),
            ("linear", "ours (linear)"),
        ],
        "true_title": "true",
        "min_span": 100.0,
        "margin_frac": 0.00,
    },
    "squares": {
        "resolution": 0.50,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("LVIRA", "ELVIRA"),
            ("linear", "linear"),
            ("linear+corner", "ours (linear+corner)"),
        ],
        "true_title": "true",
        "min_span": 42.0,
        "margin_frac": 0.10,
    },
    "circles": {
        "resolution": 0.32,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("LVIRA", "ELVIRA"),
            ("linear", "linear"),
            ("circular", "ours (circular)"),
        ],
        "true_title": "true",
        "min_span": 26.0,
        "margin_frac": 0.14,
    },
    "ellipses": {
        "resolution": 0.32,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("LVIRA", "ELVIRA"),
            ("linear", "linear"),
            ("circular", "ours (circular)"),
        ],
        "true_title": "true",
        "min_span": 66.0,
        "margin_frac": 0.12,
    },
    "zalesak": {
        "resolution": 1.00,
        "wiggle": 0.10,
        "seed": 0,
        "case_index": 12,
        "methods": [
            ("LVIRA", "ELVIRA"),
            ("circular", "circular"),
            ("circular+corner", "ours (circular+corner)"),
        ],
        "true_title": "true",
        "min_span": 42.0,
        "margin_frac": 0.12,
    },
}

TRUE_COLOR = "#111827"
TRUE_STYLE = (0, (3.0, 2.2))
MESH_COLOR = "#d1d5db"
MESH_ALPHA = 0.65


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
    backfilled = 0
    for algo in ["Youngs", "LVIRA", "safe_linear", "linear", "safe_circle", "circular"]:
        for resolution in [0.32, 0.64, 1.28]:
            for wiggle in [0.0, 0.05, 0.1, 0.2, 0.3]:
                seed = 0
                save_name = _make_save_name("circles", algo, resolution, wiggle, seed)
                metrics_path = PLOTS_ROOT / save_name / "metrics" / "tangent_error.txt"
                if not metrics_path.exists():
                    continue
                values = _read_metric_values(metrics_path)
                if not values:
                    continue
                for entry in _metric_stats("tangent_error", values):
                    key = (
                        "circles",
                        algo,
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
                            "algo": algo,
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


def _iter_lines(poly: pv.PolyData) -> list[np.ndarray]:
    lines = poly.lines
    if lines is None or len(lines) == 0:
        return []
    segments = []
    idx = 0
    while idx < len(lines):
        n = lines[idx]
        pts = lines[idx + 1 : idx + 1 + n]
        idx += n + 1
        segments.append(poly.points[pts][:, :2].copy())
    return segments


def _segments_from_polydata(poly: pv.PolyData) -> np.ndarray:
    chunks = []
    for line in _iter_lines(poly):
        if len(line) < 2:
            continue
        chunks.append(np.stack([line[:-1], line[1:]], axis=1))
    if not chunks:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


def _mesh_segments(mesh_path: Path) -> np.ndarray:
    mesh = pv.read(mesh_path)
    return _segments_from_polydata(mesh.extract_all_edges())


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


def _load_true_segments(exp_name: str, save_name: str, case_index: int) -> np.ndarray:
    if exp_name == "ellipses":
        return _ellipse_true_segments(case_index)
    true_path = _true_vtp_path(exp_name, save_name, case_index)
    return _segments_from_polydata(pv.read(true_path))


def _load_reconstructed_segments(save_name: str, case_index: int) -> np.ndarray:
    facet_path = (
        PLOTS_ROOT / save_name / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    )
    return _segments_from_polydata(pv.read(facet_path))


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
        (0, 1, metric_right, wiggle_curves, PERTURBATION_AXIS_LABEL, "perturbation"),
        (1, 0, metric_left, resolution_curves, RESOLUTION_AXIS_LABEL, "resolution"),
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
        if row == 0:
            ax.set_title(
                f"{metric.replace('_', ' ').title()} vs {PERTURBATION_AXIS_LABEL.lower()}",
                fontsize=11.5,
                fontweight="bold",
            )
        else:
            ax.set_title(
                f"{metric.replace('_', ' ').title()} vs cells per side",
                fontsize=11.5,
                fontweight="bold",
            )
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in legend_entries:
                legend_entries[label] = handle

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

    ncols = 1 + len(spec["methods"])
    fig, axes = plt.subplots(1, ncols, figsize=(3.25 * ncols, 3.5))
    if ncols == 1:
        axes = [axes]

    panels = [("true", None)] + spec["methods"]
    for ax, (algo_or_true, title) in zip(axes, panels):
        mesh_linewidth = 0.6 if exp_name == "lines" else 0.45
        mesh_alpha = 0.8 if exp_name == "lines" else MESH_ALPHA
        _add_segments(
            ax,
            mesh_segments,
            color=MESH_COLOR,
            linewidth=mesh_linewidth,
            alpha=mesh_alpha,
            zorder=1,
        )
        if algo_or_true == "true":
            _add_segments(
                ax,
                true_segments,
                color=TRUE_COLOR,
                linewidth=2.4,
                alpha=1.0,
                linestyle="-",
                zorder=3,
            )
            panel_title = spec["true_title"]
        else:
            algo = algo_or_true
            save_name = _make_save_name(
                exp_name,
                algo,
                spec["resolution"],
                spec["wiggle"],
                spec["seed"],
            )
            recon_segments = _load_reconstructed_segments(save_name, spec["case_index"])
            style = METHOD_STYLES.get(algo, {})
            _add_segments(
                ax,
                true_segments,
                color=TRUE_COLOR,
                linewidth=1.4,
                alpha=0.95,
                linestyle=TRUE_STYLE,
                zorder=2,
            )
            _add_segments(
                ax,
                recon_segments,
                color=style.get("color", "#1f77b4"),
                linewidth=max(2.1, style.get("linewidth", 2.1)),
                alpha=1.0,
                linestyle="-",
                zorder=3,
            )
            panel_title = title

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        ax.set_title(panel_title, fontsize=11.0, fontweight="bold")

        if exp_name == "lines" and algo_or_true != "true" and len(recon_segments):
            pts = recon_segments.reshape(-1, 2)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c=style.get("color", "#1f77b4"),
                alpha=0.9,
                zorder=4,
                linewidths=0.0,
            )

    fig.tight_layout()
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

    outputs = {"quantitative": {}, "representative": {}, "cases": REPRESENTATIVE_CASES}
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

    manifest_path = args.out_dir / "maintext_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
