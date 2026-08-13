#!/usr/bin/env python3
"""Recompute ellipse curvature error from exact saved primitive metadata.

This audit intentionally leaves the sealed submission runs untouched.  It uses
the exact arc radii and endpoints emitted by ``writeFacets`` rather than the
legacy signed in-memory curvature metric stored in ``case_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np


DEFAULT_SOURCE = Path(
    "results/static/submission_static_20260731_012430_505aefa45432.sealed/raw_runs"
)
DEFAULT_OUTPUT = Path(
    "results/submission/ellipse_curvature_convergence_20260813"
)


@dataclass(frozen=True)
class CaseMetric:
    source_run: str
    source_commit: str
    resolution: int
    cell_size: float
    perturbation: float
    case_index: int
    primitive_count: int
    arc_count: int
    line_count: int
    mean_absolute_curvature_error: float


def _load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["case_index"])] = row
    return rows


def ellipse_parameter(point: Sequence[float], geometry: dict) -> float:
    """Return the ellipse parameter associated with a physical-space point."""

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
        a * a * math.sin(parameter) ** 2
        + b * b * math.cos(parameter) ** 2
    ) ** 1.5
    return a * b / denominator


def primitive_curvature(primitive: dict) -> float:
    """Return unsigned local curvature for a serialized line or circular arc."""

    kind = primitive["kind"]
    if kind == "line":
        return 0.0
    if kind != "arc":
        raise ValueError(f"Unsupported primitive kind for this audit: {kind!r}")
    radius = abs(float(primitive["radius"]))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Invalid serialized arc radius: {radius!r}")
    return 1.0 / radius


def primitive_error(primitive: dict, geometry: dict) -> float:
    midpoint = [
        0.5 * (float(primitive["p_left"][0]) + float(primitive["p_right"][0])),
        0.5 * (float(primitive["p_left"][1]) + float(primitive["p_right"][1])),
    ]
    reference = ellipse_curvature(geometry, ellipse_parameter(midpoint, geometry))
    return abs(primitive_curvature(primitive) - reference)


def discover_cartesian_runs(source_root: Path) -> list[Path]:
    runs = []
    for run_dir in source_root.glob("*_perturb_sweep_ellipses_circular_r*_w0p0_s0"):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parameters = manifest["parameters"]
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
            errors = [
                primitive_error(primitive, geometries[case_index])
                for primitive in primitives
            ]
            kinds = [primitive["kind"] for primitive in primitives]
            rows.append(
                CaseMetric(
                    source_run=run_dir.name,
                    source_commit=manifest["source_commit"],
                    resolution=resolution,
                    cell_size=100.0 / resolution,
                    perturbation=float(parameters["perturb_wiggle"]),
                    case_index=case_index,
                    primitive_count=len(primitives),
                    arc_count=kinds.count("arc"),
                    line_count=kinds.count("line"),
                    mean_absolute_curvature_error=float(np.mean(errors)),
                )
            )
    return rows


def summarize(rows: Sequence[CaseMetric]) -> tuple[list[dict], float]:
    summaries = []
    for resolution in sorted({row.resolution for row in rows}):
        values = np.array(
            [
                row.mean_absolute_curvature_error
                for row in rows
                if row.resolution == resolution
            ]
        )
        summaries.append(
            {
                "resolution": resolution,
                "cell_size": 100.0 / resolution,
                "case_count": len(values),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "mean": float(np.mean(values)),
            }
        )
    if len(summaries) < 3:
        raise ValueError("At least three resolutions are required for an order fit")
    slope, _ = np.polyfit(
        np.log([row["cell_size"] for row in summaries]),
        np.log([row["median"] for row in summaries]),
        1,
    )
    return summaries, float(slope)


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summaries: Sequence[dict], order: float, output_dir: Path) -> None:
    resolutions = np.array([row["resolution"] for row in summaries], dtype=float)
    medians = np.array([row["median"] for row in summaries])
    q25 = np.array([row["q25"] for row in summaries])
    q75 = np.array([row["q75"] for row in summaries])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.9, 3.25), constrained_layout=True)
    color = "#C55A11"
    ax.fill_between(resolutions, q25, q75, color=color, alpha=0.16, linewidth=0)
    ax.loglog(
        resolutions,
        medians,
        color=color,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=rf"Our circular method (fit: $h^{{{order:.2f}}}$)",
    )

    reference = medians[1] * (resolutions / resolutions[1]) ** -2
    ax.loglog(
        resolutions,
        reference,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label=r"Second-order reference, $h^2$",
    )
    ax.set_xlabel("Cells per side, $N$")
    ax.set_ylabel(r"Mean absolute curvature error, $e_\kappa$")
    ax.set_title("Ellipse curvature-estimate convergence (Cartesian mesh)")
    ax.xaxis.set_major_locator(FixedLocator(resolutions))
    ax.set_xticklabels([str(int(value)) for value in resolutions])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#D0D0D0", linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", color="#E8E8E8", linewidth=0.35, alpha=0.55)
    ax.legend(frameon=False, loc="upper right")
    for suffix in ("pdf", "svg"):
        fig.savefig(output_dir / f"ellipse_curvature_convergence.{suffix}")
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
    summaries, order = summarize(rows)
    case_dicts = [row.__dict__ for row in rows]
    _write_csv(
        args.output_dir / "ellipse_curvature_case_metrics.csv",
        case_dicts,
        list(CaseMetric.__dataclass_fields__),
    )
    _write_csv(
        args.output_dir / "ellipse_curvature_summary.csv",
        summaries,
        ["resolution", "cell_size", "case_count", "median", "q25", "q75", "mean"],
    )
    provenance = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "source_runs": sorted({row.source_run for row in rows}),
        "source_commits": sorted({row.source_commit for row in rows}),
        "method": "graph-coordinated circular reconstruction",
        "mesh": "Cartesian (perturbation magnitude w=0)",
        "resolutions": sorted({row.resolution for row in rows}),
        "case_indices": sorted({row.case_index for row in rows}),
        "metric": (
            "per-case arithmetic mean over every active serialized primitive "
            "of abs(unsigned reconstructed curvature - analytic ellipse "
            "curvature at the primitive chord midpoint)"
        ),
        "aggregation": "median and interquartile range over 25 cases",
        "order_fit": "ordinary least squares of log(median error) on log(h)",
        "observed_order": order,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    plot_summary(summaries, order, args.output_dir)
    print(f"runs={len({row.source_run for row in rows})}")
    print(f"cases={len(rows)}")
    print(f"median_order={order:.6f}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
