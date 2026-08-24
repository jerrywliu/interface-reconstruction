#!/usr/bin/env python3
"""Run a matched few-seed smoke study of whole-chain G1 refinement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from experiments.baselines.project_benchmarks import (
    DOMAIN_SIZE,
    canonical_benchmark_cases,
)
from experiments.baselines.run_ellipse_circular_variant_metrics import (
    signed_arc_diagnostics,
)
from main.algos.baselines.external_metrics import (
    directed_hausdorff_external,
    geometric_curvature_error_external,
)
from main.algos.baselines.project_facet_adapter import (
    external_primitives_from_facet_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("results/submission/g1_chain_smoke_20260823")
DEFAULT_PREFIX = "g1_chain_signal_20260823"
MODES = (
    ("joint", "Current joint $C^0$"),
    ("g1_chain", "Whole-chain $G^1$"),
)
BENCHMARKS = {
    "circles": {
        "module": "experiments.static.circles",
        "config": "static/circle",
        "count_flag": "--num_circles",
        "case_indices": (0, 1, 2),
        "extra_args": ("--radius", "10"),
    },
    "ellipses": {
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "count_flag": "--num_ellipses",
        "case_indices": (0, 10, 24),
        "extra_args": (),
    },
}
SETTINGS = (
    ("cartesian", 0.0, 0),
    ("perturbed-s0", 0.1, 0),
    ("perturbed-s1", 0.1, 1),
    ("perturbed-s2", 0.1, 2),
)
CASE_FIELDS = (
    "benchmark",
    "setting",
    "wiggle",
    "seed",
    "mode",
    "case_index",
    "cells_per_side",
    "source_run",
    "run_seconds",
    "num_mixed_cells",
    "num_g1_bad_joins_before_refinement",
    "num_g1_bad_joins_after_refinement",
    "mean_g1_tangent_angle_before_radians",
    "mean_g1_tangent_angle_after_radians",
    "max_g1_tangent_angle_before_radians",
    "max_g1_tangent_angle_after_radians",
    "production_facet_gap",
    "max_c0_relative_area_residual",
    "native_symmetric_hausdorff",
    "geometric_curvature_mean_absolute_error",
    "geometric_curvature_relative_l1_error",
    "concave_arc_count",
    "concave_arc_fraction",
    "concave_arc_length_fraction",
    "signed_curvature_arc_length_mean",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--skip-runs", action="store_true")
    return parser.parse_args()


def _read_case_metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["case_index"]): row for row in csv.DictReader(stream)}


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else math.nan


def _run_name(
    prefix: str, benchmark: str, setting: str, mode: str, resolution: int
) -> str:
    return f"{prefix}_{benchmark}_{setting}_{mode}_n{resolution}"


def _command(
    benchmark: str,
    setting: str,
    wiggle: float,
    seed: int,
    mode: str,
    resolution: int,
    max_nfev: int,
    prefix: str,
) -> tuple[str, list[str]]:
    spec = BENCHMARKS[benchmark]
    name = _run_name(prefix, benchmark, setting, mode, resolution)
    command = [
        sys.executable,
        "-m",
        str(spec["module"]),
        "--config",
        str(spec["config"]),
        "--resolution",
        f"{resolution / DOMAIN_SIZE:g}",
        "--facet_algo",
        "circular",
        "--save_name",
        name,
        "--mesh_type",
        "perturbed_quads",
        "--perturb_wiggle",
        str(wiggle),
        "--perturb_seed",
        str(seed),
        "--perturb_fix_boundary",
        "1",
        "--do_c0",
        "1",
        "--c0_mode",
        mode,
        "--c0_joint_max_nfev",
        str(max_nfev),
        str(spec["count_flag"]),
        "25",
        "--case_indices",
        ",".join(str(index) for index in spec["case_indices"]),
        "--plic_fallback",
        "LVIRA",
        "--corner_behavior_profile",
        "pre_f8_corner",
        *spec["extra_args"],
    ]
    return name, command


def _run_matrix(args: argparse.Namespace) -> dict[str, float]:
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_seconds: dict[str, float] = {}
    jobs = [
        (benchmark, setting, wiggle, seed, mode)
        for benchmark in BENCHMARKS
        for setting, wiggle, seed in SETTINGS
        for mode, _ in MODES
    ]
    for job_index, (benchmark, setting, wiggle, seed, mode) in enumerate(jobs, 1):
        name, command = _command(
            benchmark,
            setting,
            wiggle,
            seed,
            mode,
            args.resolution,
            args.max_nfev,
            args.prefix,
        )
        metrics_path = REPO_ROOT / "plots" / name / "metrics" / "case_metrics.csv"
        if metrics_path.exists():
            print(f"[{job_index}/{len(jobs)}] reuse {name}", flush=True)
            run_seconds[name] = math.nan
            continue
        print(f"[{job_index}/{len(jobs)}] run {name}", flush=True)
        started = time.monotonic()
        environment = os.environ.copy()
        environment.setdefault("MPLBACKEND", "Agg")
        with (logs / f"{name}.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run_seconds[name] = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"{name} failed with exit code {completed.returncode}; "
                f"see {logs / f'{name}.log'}"
            )
    return run_seconds


def _evaluate_case(
    benchmark: str,
    setting: str,
    wiggle: float,
    seed: int,
    mode: str,
    case_index: int,
    resolution: int,
    prefix: str,
    run_seconds: Mapping[str, float],
) -> dict[str, Any]:
    name = _run_name(prefix, benchmark, setting, mode, resolution)
    run_dir = REPO_ROOT / "plots" / name
    diagnostics = _read_case_metrics(run_dir / "metrics" / "case_metrics.csv")[
        case_index
    ]
    metadata_path = (
        run_dir
        / "vtk"
        / "reconstructed"
        / "facets"
        / f"{case_index}.facet_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reconstruction = external_primitives_from_facet_metadata(metadata)
    truth = canonical_benchmark_cases(benchmark, (case_index,))[0].truth_primitives()
    reconstruction_to_truth = directed_hausdorff_external(reconstruction, truth)
    truth_to_reconstruction = directed_hausdorff_external(truth, reconstruction)
    curvature = geometric_curvature_error_external(reconstruction, truth)
    return {
        "benchmark": benchmark,
        "setting": setting,
        "wiggle": wiggle,
        "seed": seed,
        "mode": mode,
        "case_index": case_index,
        "cells_per_side": resolution,
        "source_run": name,
        "run_seconds": run_seconds.get(name, math.nan),
        "num_mixed_cells": int(diagnostics["num_mixed_cells"]),
        "num_g1_bad_joins_before_refinement": int(
            diagnostics.get("num_g1_bad_joins_before_refinement") or 0
        ),
        "num_g1_bad_joins_after_refinement": int(
            diagnostics.get("num_g1_bad_joins_after_refinement") or 0
        ),
        "mean_g1_tangent_angle_before_radians": _float(
            diagnostics, "mean_g1_tangent_angle_before_radians"
        ),
        "mean_g1_tangent_angle_after_radians": _float(
            diagnostics, "mean_g1_tangent_angle_after_radians"
        ),
        "max_g1_tangent_angle_before_radians": _float(
            diagnostics, "max_g1_tangent_angle_before_radians"
        ),
        "max_g1_tangent_angle_after_radians": _float(
            diagnostics, "max_g1_tangent_angle_after_radians"
        ),
        "production_facet_gap": _float(diagnostics, "facet_gap"),
        "max_c0_relative_area_residual": _float(
            diagnostics, "max_c0_relative_area_residual"
        ),
        "native_symmetric_hausdorff": max(
            reconstruction_to_truth, truth_to_reconstruction
        ),
        "geometric_curvature_mean_absolute_error": curvature["mean_absolute_error"],
        "geometric_curvature_relative_l1_error": curvature["relative_l1_error"],
        **signed_arc_diagnostics(metadata),
    }


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]
) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["benchmark"], row["setting"], row["case_index"], row["mode"]): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        for setting, _, _ in SETTINGS:
            for case_index in BENCHMARKS[benchmark]["case_indices"]:
                joint = indexed[(benchmark, setting, case_index, "joint")]
                chain = indexed[(benchmark, setting, case_index, "g1_chain")]
                record: dict[str, Any] = {
                    "benchmark": benchmark,
                    "setting": setting,
                    "case_index": case_index,
                }
                for metric in (
                    "native_symmetric_hausdorff",
                    "geometric_curvature_mean_absolute_error",
                    "production_facet_gap",
                    "max_c0_relative_area_residual",
                    "concave_arc_length_fraction",
                ):
                    old = float(joint[metric])
                    new = float(chain[metric])
                    record[f"joint_{metric}"] = old
                    record[f"g1_chain_{metric}"] = new
                    record[f"ratio_{metric}"] = new / old if old > 0.0 else math.nan
                record["joint_bad_g1_after"] = joint[
                    "num_g1_bad_joins_after_refinement"
                ]
                record["g1_chain_bad_g1_after"] = chain[
                    "num_g1_bad_joins_after_refinement"
                ]
                paired.append(record)
    return paired


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        for setting_group, predicate in (
            ("cartesian", lambda row: row["setting"] == "cartesian"),
            ("perturbed", lambda row: row["setting"].startswith("perturbed")),
        ):
            for mode, _ in MODES:
                selected = [
                    row
                    for row in rows
                    if row["benchmark"] == benchmark
                    and row["mode"] == mode
                    and predicate(row)
                ]
                summaries.append(
                    {
                        "benchmark": benchmark,
                        "setting_group": setting_group,
                        "mode": mode,
                        "num_cases": len(selected),
                        "total_bad_g1_after": sum(
                            int(row["num_g1_bad_joins_after_refinement"])
                            for row in selected
                        ),
                        "max_tangent_angle_after_radians": max(
                            float(row["max_g1_tangent_angle_after_radians"])
                            for row in selected
                        ),
                        "median_hausdorff": float(
                            np.median(
                                [row["native_symmetric_hausdorff"] for row in selected]
                            )
                        ),
                        "median_curvature_mae": float(
                            np.median(
                                [
                                    row["geometric_curvature_mean_absolute_error"]
                                    for row in selected
                                ]
                            )
                        ),
                        "max_relative_area_residual": max(
                            float(row["max_c0_relative_area_residual"])
                            for row in selected
                        ),
                        "mean_concave_arc_length_fraction": float(
                            np.mean(
                                [row["concave_arc_length_fraction"] for row in selected]
                            )
                        ),
                    }
                )
    return summaries


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"joint": "#4C78A8", "g1_chain": "#D1495B"}
    markers = {"joint": "o", "g1_chain": "D"}
    labels = dict(MODES)
    metrics = (
        ("max_g1_tangent_angle_after_radians", "Max tangent jump (rad)"),
        ("production_facet_gap", "Facet gap"),
        ("native_symmetric_hausdorff", "Hausdorff error"),
        ("geometric_curvature_mean_absolute_error", "Curvature MAE"),
        ("max_c0_relative_area_residual", "Max relative area residual"),
        ("concave_arc_length_fraction", "Concave arc-length fraction"),
    )
    groups = (
        ("circles", "cartesian", "Circle\nCartesian"),
        ("circles", "perturbed", "Circle\nperturbed"),
        ("ellipses", "cartesian", "Ellipse\nCartesian"),
        ("ellipses", "perturbed", "Ellipse\nperturbed"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(8.1, 5.3), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for group_index, (benchmark, setting_group, _) in enumerate(groups):
            for mode_index, (mode, _) in enumerate(MODES):
                values = [
                    max(float(row[metric]), 1.0e-16)
                    for row in rows
                    if row["benchmark"] == benchmark
                    and row["mode"] == mode
                    and (
                        (setting_group == "cartesian" and row["setting"] == "cartesian")
                        or (
                            setting_group == "perturbed"
                            and row["setting"].startswith("perturbed")
                        )
                    )
                ]
                x = group_index + (-0.14 if mode_index == 0 else 0.14)
                jitter = (
                    np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else [0]
                )
                axis.scatter(
                    x + np.asarray(jitter),
                    values,
                    color=colors[mode],
                    marker=markers[mode],
                    s=16,
                    alpha=0.7,
                    linewidths=0.4,
                    edgecolors="white",
                    label=labels[mode] if group_index == 0 else None,
                    zorder=3,
                )
                axis.plot(
                    [x - 0.07, x + 0.07],
                    [np.median(values), np.median(values)],
                    color=colors[mode],
                    linewidth=1.5,
                    zorder=4,
                )
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels([group[2] for group in groups])
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.tick_params(axis="x", labelsize=7.5)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    figure.savefig(output / "g1_chain_smoke_all_methods.pdf", bbox_inches="tight")
    figure.savefig(
        output / "g1_chain_smoke_all_methods.png", dpi=240, bbox_inches="tight"
    )
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_seconds = {} if args.skip_runs else _run_matrix(args)
    rows = [
        _evaluate_case(
            benchmark,
            setting,
            wiggle,
            seed,
            mode,
            case_index,
            args.resolution,
            args.prefix,
            run_seconds,
        )
        for benchmark in BENCHMARKS
        for setting, wiggle, seed in SETTINGS
        for mode, _ in MODES
        for case_index in BENCHMARKS[benchmark]["case_indices"]
    ]
    paired = _paired_rows(rows)
    summaries = _summary(rows)
    _write_csv(args.output / "case_results.csv", rows, CASE_FIELDS)
    _write_csv(args.output / "paired_results.csv", paired, paired[0].keys())
    _write_csv(args.output / "summary.csv", summaries, summaries[0].keys())
    _plot(rows, args.output)
    manifest = {
        "resolution": args.resolution,
        "modes": [mode for mode, _ in MODES],
        "benchmarks": {
            key: list(value["case_indices"]) for key, value in BENCHMARKS.items()
        },
        "settings": [
            {"name": name, "wiggle": wiggle, "seed": seed}
            for name, wiggle, seed in SETTINGS
        ],
        "max_nfev": args.max_nfev,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output / "g1_chain_smoke_all_methods.pdf")


if __name__ == "__main__":
    main()
