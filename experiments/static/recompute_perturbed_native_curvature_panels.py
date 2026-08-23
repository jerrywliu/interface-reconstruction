#!/usr/bin/env python3
"""Rebuild curved-interface sweep panels with a common native curvature MAE.

The sealed static package stores exact schema-v2 line/arc metadata for every
case.  This utility reads that geometry in place, evaluates the common
arc-length-weighted curvature observable, and writes a separate analysis
bundle.  It never mutates the archived runs or their aggregate CSV.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import add_convergence_order_triangle
from experiments.static.run_perturbed_sweeps import (
    PERTURBATION_AXIS_LABEL,
    RESOLUTION_AXIS_LABEL,
    _build_method_curves,
    _build_method_curves_by_resolution,
    _build_metric_index,
    _draw_method_curves,
    _load_sweep_rows,
    _merge_legend_entries,
    _metric_label,
)
from main.algos.baselines.external_geometry import ExternalEllipsePrimitive
from main.algos.baselines.external_metrics import geometric_curvature_error_external
from main.algos.baselines.project_facet_adapter import (
    external_primitives_from_facet_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / (
    "results/static/submission_static_20260731_012430_505aefa45432.sealed"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "results/submission/perturbed_native_curvature_panels_20260822"
)
EXPERIMENTS = ("circles", "ellipses")
METRICS = ("hausdorff", "facet_gap", "curvature_error", "tangent_error")
STATISTICS = ("mean", "median", "p25", "p75")

# Reference slopes supported by the corresponding resolution studies.
ORDER_TRIANGLES = {
    "circles": {"hausdorff": 2.0, "facet_gap": 2.0},
    "ellipses": {"facet_gap": 3.0, "curvature_error": 1.0},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[int(row["case_index"])] = row
    return rows


def _load_case_metric(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw = row.get("curvature_error", "")
            if raw not in (None, ""):
                values[int(row["case_index"])] = float(raw)
    return values


def truth_primitive(geometry: Mapping[str, Any]) -> ExternalEllipsePrimitive:
    geometry_type = str(geometry["geometry_type"])
    if geometry_type == "circle":
        major_axis = minor_axis = float(geometry["radius"])
        angle = 0.0
    elif geometry_type == "ellipse":
        major_axis = float(geometry["major_axis"])
        minor_axis = float(geometry["minor_axis"])
        angle = float(geometry["theta"])
    else:
        raise ValueError(f"unsupported truth geometry: {geometry_type!r}")
    return ExternalEllipsePrimitive(
        center=geometry["center"],
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle=angle,
        metadata={"geometry_type": geometry_type},
    )


def evaluate_run(run_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    experiment = str(manifest["experiment"])
    if experiment not in EXPERIMENTS:
        return []
    parameters = manifest["parameters"]
    geometries = _load_jsonl(run_dir / "metrics" / "case_geometry.jsonl")
    old_values = _load_case_metric(run_dir / "metrics" / "case_metrics.csv")
    metadata_dir = run_dir / "vtk" / "reconstructed" / "facets"
    rows = []
    for case_index, geometry in sorted(geometries.items()):
        metadata_path = metadata_dir / f"{case_index}.facet_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing native geometry: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        reconstruction = external_primitives_from_facet_metadata(payload)
        metric = geometric_curvature_error_external(
            reconstruction, (truth_primitive(geometry),)
        )
        value = float(metric["mean_absolute_error"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite native curvature MAE: {metadata_path}")
        rows.append(
            {
                "experiment": experiment,
                "algo": str(parameters["facet_algo"]),
                "resolution": float(parameters["resolution"]),
                "wiggle": float(parameters["perturb_wiggle"]),
                "seed": int(parameters["perturb_seed"]),
                "case_index": case_index,
                "primitive_count": len(reconstruction),
                "reconstructed_length": float(metric["reconstructed_length"]),
                "old_curvature_error": old_values.get(case_index, math.nan),
                "native_curvature_mae": value,
                "source_run": run_dir.name,
                "source_commit": str(manifest["source_commit"]),
            }
        )
    if set(geometries) != set(old_values):
        raise ValueError(f"case geometry/metric mismatch in {run_dir}")
    return rows


def discover_runs(source: Path) -> list[Path]:
    raw_root = source / "raw_runs"
    runs = []
    for experiment in EXPERIMENTS:
        runs.extend(raw_root.glob(f"*_perturb_sweep_{experiment}_*"))
    return sorted(path for path in runs if path.is_dir())


def aggregate_run_rows(case_rows: Sequence[Mapping[str, Any]]) -> dict[tuple, dict[str, float]]:
    grouped: dict[tuple, list[float]] = {}
    for row in case_rows:
        key = (
            row["experiment"],
            row["algo"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(float(row["native_curvature_mae"]))
    result = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=float)
        result[key] = {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p25": float(np.percentile(array, 25)),
            "p75": float(np.percentile(array, 75)),
        }
    return result


def patch_sweep_rows(
    source_rows: Sequence[Mapping[str, str]],
    aggregates: Mapping[tuple, Mapping[str, float]],
) -> list[dict[str, str]]:
    patched = [dict(row) for row in source_rows]
    replaced: set[tuple] = set()
    for row in patched:
        experiment = row.get("experiment")
        metric_key = row.get("metric_key", "")
        if experiment not in EXPERIMENTS or not metric_key.startswith("curvature_error_"):
            continue
        statistic = metric_key.removeprefix("curvature_error_")
        if statistic not in STATISTICS:
            continue
        key = (
            experiment,
            row["algo"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
        )
        if key not in aggregates:
            raise KeyError(f"missing native curvature aggregate for {key}")
        row["metric_value"] = f"{aggregates[key][statistic]:.17g}"
        replaced.add((key, statistic))
    expected = {(key, statistic) for key in aggregates for statistic in STATISTICS}
    if replaced != expected:
        missing = sorted(expected - replaced)
        extra = sorted(replaced - expected)
        raise ValueError(f"aggregate replacement mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return patched


def _plot_panel(data: Mapping[str, Any], experiment: str, output: Path) -> None:
    exp_data = data[experiment]
    wiggle_curves = {metric: _build_method_curves(exp_data, metric) for metric in METRICS}
    resolution_curves = {
        metric: _build_method_curves_by_resolution(exp_data, metric) for metric in METRICS
    }
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(14, 16.8))
    legend_entries: dict[str, Any] = {}
    for row_index, metric in enumerate(METRICS):
        label = "Curvature MAE" if metric == "curvature_error" else _metric_label(metric)
        for column, (curves, x_label, mode) in enumerate(
            (
                (wiggle_curves[metric], PERTURBATION_AXIS_LABEL, "perturbation"),
                (resolution_curves[metric], RESOLUTION_AXIS_LABEL, "resolution"),
            )
        ):
            axis = axes[row_index, column]
            _draw_method_curves(
                axis,
                curves,
                metric,
                x_label=x_label,
                x_mode=mode,
                exp_name=experiment,
            )
            axis.set_title(
                f"{label} vs {'Perturbation Magnitude' if column == 0 else 'Cells per Side'}",
                fontsize=11.5,
                fontweight="bold",
            )
            if metric == "curvature_error":
                axis.set_ylabel("Curvature MAE", fontsize=11)
            _merge_legend_entries(legend_entries, axis)

        left, right = axes[row_index]
        y_min = min(left.get_ylim()[0], right.get_ylim()[0])
        y_max = max(left.get_ylim()[1], right.get_ylim()[1])
        left.set_ylim(y_min, y_max)
        right.set_ylim(y_min, y_max)

        order = ORDER_TRIANGLES.get(experiment, {}).get(metric)
        if order is not None:
            right.set_xscale("log")
            add_convergence_order_triangle(
                right,
                order,
                anchor=(0.72, 0.68),
                width=0.11,
                trend="decreasing",
                fontsize=7.5,
            )

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(4, len(legend_entries)),
            fontsize=8.5,
            frameon=True,
            bbox_to_anchor=(0.5, -0.004),
        )
    singular = "Circle" if experiment == "circles" else "Ellipse"
    figure_title = fig.suptitle(
        f"{singular} Reconstruction on Perturbed Cartesian Meshes",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.89], h_pad=1.8, w_pad=1.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_title_box = figure_title.get_window_extent(renderer=renderer)
    top_title_boxes = [
        axis.title.get_window_extent(renderer=renderer) for axis in axes[0]
    ]
    minimum_clearance = 12.0
    if max(box.y1 for box in top_title_boxes) + minimum_clearance >= figure_title_box.y0:
        raise ValueError(
            f"{experiment} top-row titles overlap the figure-title clearance band"
        )
    if figure_title_box.y1 > fig.bbox.y1:
        raise ValueError(f"{experiment} figure title is clipped by the canvas")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_summary(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in case_rows:
        old = float(row["old_curvature_error"])
        new = float(row["native_curvature_mae"])
        grouped.setdefault((str(row["experiment"]), str(row["algo"])), []).append((old, new))
    summaries = []
    for (experiment, algo), values in sorted(grouped.items()):
        old = np.asarray([value[0] for value in values], dtype=float)
        new = np.asarray([value[1] for value in values], dtype=float)
        summaries.append(
            {
                "experiment": experiment,
                "algo": algo,
                "case_count": len(values),
                "old_median": float(np.median(old)),
                "native_median": float(np.median(new)),
                "median_absolute_change": float(np.median(np.abs(new - old))),
                "median_ratio_new_to_old": float(np.median(new / np.maximum(old, 1.0e-300))),
                "max_absolute_change": float(np.max(np.abs(new - old))),
            }
        )
    return summaries


def run(
    source: Path,
    output: Path,
    *,
    workers: int = 1,
    reuse_case_metrics: bool = False,
) -> dict[str, Any]:
    runs = discover_runs(source)
    if len(runs) != 420:
        raise ValueError(f"expected 420 curved-interface runs, found {len(runs)}")
    case_csv = output / "case_curvature_comparison.csv"
    if reuse_case_metrics:
        if not case_csv.is_file():
            raise FileNotFoundError(f"cannot reuse missing case metrics: {case_csv}")
        case_rows = _read_csv(case_csv)
    else:
        if workers < 1:
            raise ValueError("workers must be positive")
        if workers == 1:
            evaluated_runs: Iterable[list[dict[str, Any]]] = map(evaluate_run, runs)
            case_rows = [row for run_rows in evaluated_runs for row in run_rows]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                evaluated_runs = executor.map(evaluate_run, runs, chunksize=1)
                case_rows = [row for run_rows in evaluated_runs for row in run_rows]
    if len(case_rows) != 10_500:
        raise ValueError(f"expected 10,500 case rows, found {len(case_rows)}")

    aggregates = aggregate_run_rows(case_rows)
    source_csv = source / "perturbed_sweep.csv"
    patched_rows = patch_sweep_rows(_read_csv(source_csv), aggregates)
    output.mkdir(parents=True, exist_ok=True)
    summary_csv = output / "old_vs_native_summary.csv"
    patched_csv = output / "perturbed_sweep_native_curvature.csv"
    if not reuse_case_metrics:
        _write_csv(case_csv, case_rows)
    _write_csv(summary_csv, _difference_summary(case_rows))
    _write_csv(patched_csv, patched_rows)

    mpl.rcParams.update(
        {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    )
    data = _build_metric_index(_load_sweep_rows(patched_csv))
    artifacts = []
    for experiment in EXPERIMENTS:
        singular = "circle" if experiment == "circles" else "ellipse"
        panel = output / f"{singular}_reconstruction_perturbed_all_methods_5x2_axes.pdf"
        _plot_panel(data, experiment, panel)
        artifacts.extend((panel, panel.with_suffix(".png")))

    manifest = {
        "schema_version": 1,
        "source_package": str(source.resolve()),
        "source_sweep_csv": str(source_csv.resolve()),
        "run_count": len(runs),
        "case_count": len(case_rows),
        "workers": workers,
        "reused_case_metrics": reuse_case_metrics,
        "observable": "arc-length-weighted native curvature MAE against nearest analytic truth",
        "quadrature_order": 16,
        "artifacts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (case_csv, summary_csv, patched_csv, *artifacts)
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reuse-case-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        args.source.resolve(),
        args.output.resolve(),
        workers=args.workers,
        reuse_case_metrics=args.reuse_case_metrics,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
