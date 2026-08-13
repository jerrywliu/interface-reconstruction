#!/usr/bin/env python3
"""Test whether ellipse curvature convergence depends on metric conventions.

The diagnostic reads exact primitive metadata from the sealed Cartesian
ellipse runs.  It never reruns or mutates reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
from scipy.integrate import quad


DEFAULT_SOURCE = Path(
    "results/static/submission_static_20260731_012430_505aefa45432.sealed/raw_runs"
)
DEFAULT_OUTPUT = Path(
    "results/submission/ellipse_curvature_metric_sensitivity_20260813"
)
TARGETS = ("chord_midpoint", "native_arc_midpoint", "interval_average")
AGGREGATIONS = ("equal_facet", "arc_length_weighted")


@dataclass(frozen=True)
class CaseMetric:
    source_run: str
    source_commit: str
    resolution: int
    cell_size: float
    case_index: int
    primitive_count: int
    arc_count: int
    non_arc_count: int
    fallback_non_arc_count: int
    total_true_interval_length: float
    chord_midpoint_equal_facet: float
    chord_midpoint_arc_length_weighted: float
    native_arc_midpoint_equal_facet: float
    native_arc_midpoint_arc_length_weighted: float
    interval_average_equal_facet: float
    interval_average_arc_length_weighted: float


def _load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["case_index"])] = row
    return rows


def ellipse_parameter_projection(point: Sequence[float], geometry: dict) -> float:
    """Map a point to the parameter of its normalized-radial ellipse projection."""

    x = float(point[0]) - float(geometry["center"][0])
    y = float(point[1]) - float(geometry["center"][1])
    theta = float(geometry["theta"])
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_local = cos_theta * x + sin_theta * y
    y_local = -sin_theta * x + cos_theta * y
    return math.atan2(
        y_local / float(geometry["minor_axis"]),
        x_local / float(geometry["major_axis"]),
    )


def ellipse_curvature(geometry: dict, parameter: float) -> float:
    a = float(geometry["major_axis"])
    b = float(geometry["minor_axis"])
    denominator = (
        a * a * math.sin(parameter) ** 2 + b * b * math.cos(parameter) ** 2
    ) ** 1.5
    return a * b / denominator


def ellipse_speed(geometry: dict, parameter: float) -> float:
    a = float(geometry["major_axis"])
    b = float(geometry["minor_axis"])
    return math.sqrt(
        a * a * math.sin(parameter) ** 2 + b * b * math.cos(parameter) ** 2
    )


def _short_periodic_delta(start: float, end: float) -> float:
    delta = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    if abs(delta) <= 1.0e-14:
        raise ValueError("Projected primitive endpoints define a zero-length interval")
    return delta


def true_interval_statistics(primitive: dict, geometry: dict) -> tuple[float, float]:
    """Return true ellipse arc length and mean curvature over endpoint interval."""

    start = ellipse_parameter_projection(primitive["p_left"], geometry)
    end = ellipse_parameter_projection(primitive["p_right"], geometry)
    stop = start + _short_periodic_delta(start, end)
    lower, upper = sorted((start, stop))
    arc_length = quad(
        lambda parameter: ellipse_speed(geometry, parameter),
        lower,
        upper,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=100,
    )[0]
    integrated_curvature = quad(
        lambda parameter: (
            ellipse_curvature(geometry, parameter) * ellipse_speed(geometry, parameter)
        ),
        lower,
        upper,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=100,
    )[0]
    if not math.isfinite(arc_length) or arc_length <= 0.0:
        raise ValueError(f"Invalid projected ellipse interval length: {arc_length!r}")
    return arc_length, integrated_curvature / arc_length


def chord_midpoint(primitive: dict) -> list[float]:
    return [
        0.5 * (float(primitive["p_left"][axis]) + float(primitive["p_right"][axis]))
        for axis in (0, 1)
    ]


def native_primitive_midpoint(primitive: dict) -> list[float]:
    """Return the angular midpoint of an arc, or chord midpoint for a line."""

    if primitive["kind"] == "line":
        return chord_midpoint(primitive)
    if primitive["kind"] != "arc":
        raise ValueError(f"Unsupported primitive kind: {primitive['kind']!r}")
    center = [float(value) for value in primitive["center"]]
    radius = abs(float(primitive["radius"]))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Invalid serialized arc radius: {radius!r}")
    angle = math.atan2(
        float(primitive["p_left"][1]) - center[1],
        float(primitive["p_left"][0]) - center[0],
    )
    angle += 0.5 * float(primitive["signed_delta"])
    return [
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
    ]


def reconstructed_unsigned_curvature(primitive: dict) -> float:
    if primitive["kind"] == "line":
        return 0.0
    if primitive["kind"] != "arc":
        raise ValueError(f"Unsupported primitive kind: {primitive['kind']!r}")
    radius = abs(float(primitive["radius"]))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Invalid serialized arc radius: {radius!r}")
    return 1.0 / radius


def primitive_diagnostic(
    primitive: dict, geometry: dict
) -> tuple[dict[str, float], float]:
    reconstructed = reconstructed_unsigned_curvature(primitive)
    length, average = true_interval_statistics(primitive, geometry)
    targets = {
        "chord_midpoint": ellipse_curvature(
            geometry,
            ellipse_parameter_projection(chord_midpoint(primitive), geometry),
        ),
        "native_arc_midpoint": ellipse_curvature(
            geometry,
            ellipse_parameter_projection(
                native_primitive_midpoint(primitive), geometry
            ),
        ),
        "interval_average": average,
    }
    return (
        {name: abs(reconstructed - value) for name, value in targets.items()},
        length,
    )


def discover_cartesian_runs(source_root: Path) -> list[Path]:
    runs = []
    for run_dir in source_root.glob("*_perturb_sweep_ellipses_circular_r*_w0p0_s0"):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        parameters = json.loads(manifest_path.read_text())["parameters"]
        if float(parameters["perturb_wiggle"]) != 0.0:
            continue
        if parameters["facet_algo"] != "circular":
            continue
        runs.append(run_dir)
    return sorted(
        runs,
        key=lambda path: float(
            json.loads((path / "run_manifest.json").read_text())["parameters"][
                "resolution"
            ]
        ),
    )


def _aggregate(
    errors: Sequence[float], weights: Sequence[float]
) -> tuple[float, float]:
    values = np.asarray(errors, dtype=float)
    lengths = np.asarray(weights, dtype=float)
    return float(np.mean(values)), float(np.average(values, weights=lengths))


def compute_case_metrics(run_dirs: Iterable[Path]) -> list[CaseMetric]:
    rows: list[CaseMetric] = []
    seen_resolutions: set[int] = set()
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        parameters = manifest["parameters"]
        resolution = round(100.0 * float(parameters["resolution"]))
        if resolution in seen_resolutions:
            raise ValueError(f"Duplicate Cartesian ellipse run at N={resolution}")
        seen_resolutions.add(resolution)
        geometries = _load_jsonl(run_dir / "metrics" / "case_geometry.jsonl")
        metadata_paths = sorted(
            (run_dir / "vtk" / "reconstructed" / "facets").glob(
                "*.facet_metadata.json"
            ),
            key=lambda path: int(path.name.split(".", 1)[0]),
        )
        if len(metadata_paths) != len(geometries):
            raise ValueError(
                f"Expected one primitive metadata file per case in {run_dir}"
            )
        for metadata_path in metadata_paths:
            case_index = int(metadata_path.name.split(".", 1)[0])
            primitives = json.loads(metadata_path.read_text())["primitives"]
            if not primitives:
                raise ValueError(f"No active primitives in {metadata_path}")
            errors = {target: [] for target in TARGETS}
            lengths = []
            for primitive in primitives:
                primitive_errors, length = primitive_diagnostic(
                    primitive, geometries[case_index]
                )
                lengths.append(length)
                for target, error in primitive_errors.items():
                    errors[target].append(error)
            aggregations = {
                target: _aggregate(values, lengths) for target, values in errors.items()
            }
            kinds = [primitive["kind"] for primitive in primitives]
            non_arcs = [
                primitive for primitive in primitives if primitive["kind"] != "arc"
            ]
            fallback_non_arcs = [
                primitive
                for primitive in non_arcs
                if primitive.get("source_name") not in (None, "arc")
            ]
            rows.append(
                CaseMetric(
                    source_run=run_dir.name,
                    source_commit=manifest["source_commit"],
                    resolution=resolution,
                    cell_size=100.0 / resolution,
                    case_index=case_index,
                    primitive_count=len(primitives),
                    arc_count=kinds.count("arc"),
                    non_arc_count=len(non_arcs),
                    fallback_non_arc_count=len(fallback_non_arcs),
                    total_true_interval_length=float(sum(lengths)),
                    chord_midpoint_equal_facet=aggregations["chord_midpoint"][0],
                    chord_midpoint_arc_length_weighted=aggregations["chord_midpoint"][
                        1
                    ],
                    native_arc_midpoint_equal_facet=aggregations["native_arc_midpoint"][
                        0
                    ],
                    native_arc_midpoint_arc_length_weighted=aggregations[
                        "native_arc_midpoint"
                    ][1],
                    interval_average_equal_facet=aggregations["interval_average"][0],
                    interval_average_arc_length_weighted=aggregations[
                        "interval_average"
                    ][1],
                )
            )
    return rows


def _metric_column(target: str, aggregation: str) -> str:
    return f"{target}_{aggregation}"


def fit_order(summary_rows: Sequence[dict], fine_count: int | None = None) -> float:
    selected = list(summary_rows if fine_count is None else summary_rows[-fine_count:])
    if len(selected) < 3:
        raise ValueError("At least three resolutions are required for an order fit")
    slope, _ = np.polyfit(
        np.log([row["cell_size"] for row in selected]),
        np.log([row["median_case_error"] for row in selected]),
        1,
    )
    return float(slope)


def summarize(rows: Sequence[CaseMetric]) -> list[dict]:
    summaries = []
    for target in TARGETS:
        for aggregation in AGGREGATIONS:
            metric = _metric_column(target, aggregation)
            group = []
            for resolution in sorted({row.resolution for row in rows}):
                selected = [row for row in rows if row.resolution == resolution]
                values = np.array([getattr(row, metric) for row in selected])
                primitive_count = sum(row.primitive_count for row in selected)
                non_arc_count = sum(row.non_arc_count for row in selected)
                group.append(
                    {
                        "target": target,
                        "aggregation": aggregation,
                        "resolution": resolution,
                        "cell_size": 100.0 / resolution,
                        "case_count": len(values),
                        "primitive_count": primitive_count,
                        "arc_count": sum(row.arc_count for row in selected),
                        "non_arc_count": non_arc_count,
                        "fallback_non_arc_count": sum(
                            row.fallback_non_arc_count for row in selected
                        ),
                        "non_arc_fraction": non_arc_count / primitive_count,
                        "median_case_error": float(np.median(values)),
                        "q25_case_error": float(np.quantile(values, 0.25)),
                        "q75_case_error": float(np.quantile(values, 0.75)),
                        "mean_case_error": float(np.mean(values)),
                    }
                )
            order_all = fit_order(group)
            order_fine = fit_order(group, fine_count=4)
            for summary in group:
                summary["order_all_resolutions"] = order_all
                summary["order_finest_four"] = order_fine
            summaries.extend(group)
    return summaries


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _series(summaries: Sequence[dict], target: str, aggregation: str) -> list[dict]:
    return [
        row
        for row in summaries
        if row["target"] == target and row["aggregation"] == aggregation
    ]


def plot_comparison(summaries: Sequence[dict], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), constrained_layout=True)
    colors = {
        "chord_midpoint": "#C55A11",
        "native_arc_midpoint": "#276FBF",
        "interval_average": "#2F7D32",
    }
    labels = {
        "chord_midpoint": "Projected chord midpoint",
        "native_arc_midpoint": "Projected native-arc midpoint",
        "interval_average": "True-interval mean curvature",
    }
    for target in TARGETS:
        series = _series(summaries, target, "equal_facet")
        order = series[0]["order_all_resolutions"]
        fine = series[0]["order_finest_four"]
        axes[0].loglog(
            [row["resolution"] for row in series],
            [row["median_case_error"] for row in series],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=colors[target],
            label=f"{labels[target]} ({order:.2f}/{fine:.2f})",
        )
    for aggregation, marker, linestyle, label in (
        ("equal_facet", "o", "-", "Equal facet"),
        ("arc_length_weighted", "s", "--", "True arc-length weighted"),
    ):
        series = _series(summaries, "interval_average", aggregation)
        order = series[0]["order_all_resolutions"]
        fine = series[0]["order_finest_four"]
        axes[1].loglog(
            [row["resolution"] for row in series],
            [row["median_case_error"] for row in series],
            marker=marker,
            markersize=3.5,
            linestyle=linestyle,
            linewidth=1.5,
            color="#2F7D32" if aggregation == "equal_facet" else "#704C9F",
            label=f"{label} ({order:.2f}/{fine:.2f})",
        )
    reference_series = _series(summaries, "interval_average", "equal_facet")
    resolutions = np.array([row["resolution"] for row in reference_series])
    reference = (
        reference_series[2]["median_case_error"] * (resolutions / resolutions[2]) ** -2
    )
    for axis in axes:
        axis.loglog(
            resolutions,
            reference,
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            label=r"Second-order reference, $h^2$",
        )
        axis.set_xlabel("Cells per side, $N$")
        axis.xaxis.set_major_locator(FixedLocator(resolutions))
        axis.set_xticklabels([str(int(value)) for value in resolutions])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="major", color="#D4D4D4", linewidth=0.45)
        axis.grid(True, which="minor", color="#ECECEC", linewidth=0.3)
        axis.legend(
            frameon=False,
            loc="upper right",
            title="Observed order: all / finest four",
            title_fontsize=7.3,
        )
    axes[0].set_ylabel(r"Median mean absolute curvature error, $e_\kappa$")
    axes[0].set_title("Reference-location sensitivity")
    axes[1].set_title("Aggregation sensitivity (interval mean)")
    fig.savefig(output_dir / "ellipse_curvature_metric_sensitivity.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_case_metrics(discover_cartesian_runs(args.source_root))
    summaries = summarize(rows)
    _write_csv(args.output_dir / "case_metrics.csv", [asdict(row) for row in rows])
    _write_csv(args.output_dir / "metric_sensitivity.csv", summaries)
    plot_comparison(summaries, args.output_dir)
    provenance = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "source_runs": sorted({row.source_run for row in rows}),
        "source_commits": sorted({row.source_commit for row in rows}),
        "method": "graph-coordinated circular reconstruction",
        "mesh": "Cartesian (perturbation magnitude w=0)",
        "resolutions": sorted({row.resolution for row in rows}),
        "case_indices": sorted({row.case_index for row in rows}),
        "targets": list(TARGETS),
        "aggregations": list(AGGREGATIONS),
        "fine_order_fit": "four finest resolutions",
        "quadrature": "scipy.integrate.quad with epsabs=epsrel=1e-13",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"runs={len({row.source_run for row in rows})}")
    print(f"cases={len(rows)}")
    print(f"non_arc_primitives={sum(row.non_arc_count for row in rows)}")
    for target in TARGETS:
        for aggregation in AGGREGATIONS:
            series = _series(summaries, target, aggregation)
            print(
                f"{target}/{aggregation}: "
                f"order_all={series[0]['order_all_resolutions']:.6f}, "
                f"order_fine={series[0]['order_finest_four']:.6f}"
            )
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
