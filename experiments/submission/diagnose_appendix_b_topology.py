"""Diagnose the topology stories behind Appendix Figures B.16 and B.20.

This script is read-only with respect to the sealed submission sweep. It uses
the saved facet geometry to apply the manuscript's shared-vertex phase-label
test, then joins those diagnostics to the sealed case-level errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.submission.topology_consistency_diagnostics import (
    evaluate_case,
    read_structured_mesh,
)


DEFAULT_SEALED_ROOT = (
    REPO_ROOT
    / "results/static/submission_static_20260731_012430_505aefa45432.sealed"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results/submission/revision_diagnostics_20260813"
)
PAIR_DEFINITIONS = {
    "linear": ("safe_linear", "linear"),
    "circular": ("safe_circle", "circular"),
}
ZALASAK_HIGH_ERROR = 1.0e-5
LINE_HIGH_ERROR = 1.0e-7


def _load_inputs(sealed_root: Path):
    diagnostics = sealed_root / "diagnostics"
    case_metrics = pd.read_csv(diagnostics / "case_metrics.csv", low_memory=False)
    inventory = pd.read_csv(diagnostics / "run_inventory.csv")
    selected_algorithms = {
        algorithm
        for pair in PAIR_DEFINITIONS.values()
        for algorithm in pair
    } | {"circular+corner"}
    selected_saves = set(
        inventory.loc[inventory["algo"].isin(selected_algorithms), "save_name"]
    )
    cell_columns = [
        "experiment",
        "algo",
        "resolution",
        "wiggle",
        "seed",
        "save_name",
        "case_index",
        "cell_id",
        "cell_x",
        "cell_y",
        "merge_id",
        "orientation_status",
        "construction_path",
        "fallback_policy",
        "facet_geometry_json",
    ]
    cell_metrics = pd.read_csv(
        diagnostics / "cell_metrics.csv",
        usecols=cell_columns,
        low_memory=False,
    )
    cell_metrics = cell_metrics[cell_metrics["save_name"].isin(selected_saves)]
    return case_metrics, inventory, cell_metrics


def _evaluate_saved_topology(
    sealed_root: Path, inventory: pd.DataFrame, cell_metrics: pd.DataFrame
) -> pd.DataFrame:
    inventory_by_save = inventory.set_index("save_name").to_dict("index")
    mesh_cache = {}
    rows = []
    for (save_name, case_index), group in cell_metrics.groupby(
        ["save_name", "case_index"], sort=False
    ):
        run = inventory_by_save[save_name]
        mesh_path = sealed_root / run["run_bundle"] / "vtk/mesh.vtk"
        cache_key = str(mesh_path)
        if cache_key not in mesh_cache:
            mesh_cache[cache_key] = read_structured_mesh(mesh_path)
        records = group.fillna("").astype(str).to_dict("records")
        summary, _ = evaluate_case(records, mesh_cache[cache_key])
        rows.append(
            {
                "experiment": run["experiment"],
                "algo": run["algo"],
                "resolution": float(run["resolution"]),
                "wiggle": float(run["wiggle"]),
                "seed": int(run["seed"]),
                "save_name": save_name,
                "case_index": int(case_index),
                "evaluated_shared_vertices": summary[
                    "complete_evaluated_shared_vertices"
                ],
                "conflict_vertices": summary["complete_conflict_vertices"],
                "conflict_rate": summary["complete_conflict_rate"],
                "has_conflict": int(summary["complete_conflict_vertices"] > 0),
            }
        )
    return pd.DataFrame(rows)


def _topology_by_setting(joined: pd.DataFrame) -> pd.DataFrame:
    selected = joined[joined["algo"].isin({a for p in PAIR_DEFINITIONS.values() for a in p})]
    grouped = selected.groupby(
        ["experiment", "algo", "resolution", "wiggle"], as_index=False
    ).agg(
        cases=("case_index", "size"),
        median_hausdorff=("hausdorff", "median"),
        median_facet_gap=("facet_gap", "median"),
        cases_with_conflict=("has_conflict", "sum"),
        case_conflict_fraction=("has_conflict", "mean"),
        conflict_vertices=("conflict_vertices", "sum"),
        evaluated_shared_vertices=("evaluated_shared_vertices", "sum"),
    )
    grouped["shared_vertex_conflict_rate"] = (
        grouped["conflict_vertices"] / grouped["evaluated_shared_vertices"]
    )
    return grouped


def _paired_summary(joined: pd.DataFrame) -> pd.DataFrame:
    identity = ["experiment", "resolution", "wiggle", "seed", "case_index"]
    outputs = []
    for family, (per_cell, coordinated) in PAIR_DEFINITIONS.items():
        left = joined[joined["algo"] == per_cell].copy()
        right = joined[joined["algo"] == coordinated].copy()
        if left.empty or right.empty:
            continue
        pair = left.merge(right, on=identity, suffixes=("_per_cell", "_coordinated"))
        pair["family"] = family
        pair["facet_gap_ratio"] = pair["facet_gap_per_cell"] / np.maximum(
            pair["facet_gap_coordinated"], 1.0e-14
        )
        pair["hausdorff_ratio"] = pair["hausdorff_per_cell"] / np.maximum(
            pair["hausdorff_coordinated"], 1.0e-14
        )
        outputs.append(pair)
    paired = pd.concat(outputs, ignore_index=True)
    summary = paired.groupby(["family", "experiment", "wiggle"], as_index=False).agg(
        paired_cases=("case_index", "size"),
        per_cell_case_conflict_fraction=("has_conflict_per_cell", "mean"),
        coordinated_case_conflict_fraction=("has_conflict_coordinated", "mean"),
        median_facet_gap_ratio=("facet_gap_ratio", "median"),
        p75_facet_gap_ratio=("facet_gap_ratio", lambda values: values.quantile(0.75)),
        median_hausdorff_ratio=("hausdorff_ratio", "median"),
    )
    return paired, summary


def _zalesak_summary(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = joined[
        (joined["experiment"] == "zalesak")
        & (joined["algo"] == "circular+corner")
    ].copy()
    data["has_linear_corner_cells"] = data["num_final_linear_corner_cells"] > 0
    data["high_facet_gap"] = data["facet_gap"] > ZALASAK_HIGH_ERROR
    setting = data.groupby(["resolution", "wiggle"], as_index=False).agg(
        cases=("case_index", "size"),
        median_hausdorff=("hausdorff", "median"),
        median_facet_gap=("facet_gap", "median"),
        p75_case_facet_gap=("facet_gap", lambda values: values.quantile(0.75)),
        high_error_case_fraction=("high_facet_gap", "mean"),
        no_linear_corner_case_fraction=(
            "has_linear_corner_cells",
            lambda values: (~values).mean(),
        ),
        shared_vertex_conflict_case_fraction=("has_conflict", "mean"),
        median_linear_corner_cells=("num_final_linear_corner_cells", "median"),
        median_curved_corner_cells=("num_final_curved_corner_cells", "median"),
    )

    # Reproduce the B.20 perturbation-panel aggregation: each point is the
    # median of the resolution-level medians, and its band spans their IQR.
    plot_semantics = setting.groupby("wiggle", as_index=False).agg(
        plotted_median_facet_gap=("median_facet_gap", "median"),
        plotted_p25_facet_gap=("median_facet_gap", lambda values: values.quantile(0.25)),
        plotted_p75_facet_gap=("median_facet_gap", lambda values: values.quantile(0.75)),
        worst_resolution_median=("median_facet_gap", "max"),
    )
    return setting, plot_semantics


def _line_case_association(joined: pd.DataFrame) -> pd.DataFrame:
    data = joined[
        (joined["experiment"] == "lines")
        & (joined["algo"].isin(PAIR_DEFINITIONS["linear"]))
    ].copy()
    data["high_facet_gap"] = data["facet_gap"] > LINE_HIGH_ERROR
    rows = []
    for (algo, wiggle), group in data.groupby(["algo", "wiggle"]):
        conflicted = group[group["has_conflict"] == 1]
        consistent = group[group["has_conflict"] == 0]
        high = group[group["high_facet_gap"]]
        rows.append(
            {
                "algo": algo,
                "wiggle": wiggle,
                "cases": len(group),
                "high_error_cases": int(group["high_facet_gap"].sum()),
                "conflict_cases": int(group["has_conflict"].sum()),
                "high_error_fraction": group["high_facet_gap"].mean(),
                "conflict_fraction": group["has_conflict"].mean(),
                "high_error_given_conflict": (
                    conflicted["high_facet_gap"].mean() if len(conflicted) else np.nan
                ),
                "high_error_given_no_conflict": (
                    consistent["high_facet_gap"].mean() if len(consistent) else np.nan
                ),
                "conflict_given_high_error": (
                    high["has_conflict"].mean() if len(high) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _zalesak_contingency(joined: pd.DataFrame) -> pd.DataFrame:
    data = joined[
        (joined["experiment"] == "zalesak")
        & (joined["algo"] == "circular+corner")
    ].copy()
    data["has_linear_corner_cells"] = data["num_final_linear_corner_cells"] > 0
    data["high_facet_gap"] = data["facet_gap"] > ZALASAK_HIGH_ERROR
    rows = []
    for resolution, group in data.groupby("resolution"):
        no_corner = ~group["has_linear_corner_cells"]
        high = group["high_facet_gap"]
        rows.append(
            {
                "resolution": resolution,
                "cells_per_side": int(round(100 * resolution)),
                "cases": len(group),
                "high_error_no_linear_corner": int((high & no_corner).sum()),
                "high_error_with_linear_corner": int((high & ~no_corner).sum()),
                "low_error_no_linear_corner": int((~high & no_corner).sum()),
                "low_error_with_linear_corner": int((~high & ~no_corner).sum()),
                "high_error_explained_by_no_linear_corner": (
                    (high & no_corner).sum() / high.sum() if high.sum() else np.nan
                ),
                "high_error_given_no_linear_corner": (
                    (high & no_corner).sum() / no_corner.sum() if no_corner.sum() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_b16(setting: pd.DataFrame, output_dir: Path):
    data = setting[setting["experiment"] == "lines"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    labels = {"safe_linear": "Per-cell", "linear": "Graph-coordinated"}
    colors = {"safe_linear": "#b45309", "linear": "#2563eb"}
    for algo in ("safe_linear", "linear"):
        subset = data[data["algo"] == algo]
        curves = subset.groupby("wiggle", as_index=False).agg(
            facet_gap=("median_facet_gap", "median"),
            conflict_fraction=("case_conflict_fraction", "mean"),
        )
        axes[0].plot(
            curves["wiggle"], curves["facet_gap"], "o-", color=colors[algo], label=labels[algo]
        )
        axes[1].plot(
            curves["wiggle"], 100 * curves["conflict_fraction"], "o-", color=colors[algo], label=labels[algo]
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median facet gap")
    axes[1].set_ylabel("Cases with a shared-vertex conflict (%)")
    for axis in axes:
        axis.set_xlabel("Perturbation magnitude")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"b16_lines_topology_diagnostic.{suffix}", dpi=200)
    plt.close(fig)


def _plot_broad(summary: pd.DataFrame, output_dir: Path):
    linear = summary[summary["family"] == "linear"].copy()
    benchmarks = [name for name in ["lines", "squares", "circles", "ellipses", "zalesak"] if name in set(linear["experiment"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for benchmark in benchmarks:
        subset = linear[linear["experiment"] == benchmark]
        axes[0].plot(
            subset["wiggle"],
            subset["median_facet_gap_ratio"],
            marker="o",
            label=benchmark.title(),
        )
        axes[1].plot(
            subset["wiggle"],
            100 * subset["per_cell_case_conflict_fraction"],
            marker="o",
            label=benchmark.title(),
        )
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Per-cell / graph-coordinated facet gap")
    axes[1].set_ylabel("Per-cell cases with shared-vertex conflict (%)")
    for axis in axes:
        axis.set_xlabel("Perturbation magnitude")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"cross_benchmark_topology_diagnostic.{suffix}", dpi=200)
    plt.close(fig)


def _plot_b20(setting: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for wiggle in sorted(setting["wiggle"].unique()):
        subset = setting[setting["wiggle"] == wiggle]
        cells = 100 * subset["resolution"]
        axes[0].plot(cells, subset["median_facet_gap"], marker="o", label=f"w={wiggle:g}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Cells per side")
    axes[0].set_ylabel("Median facet gap")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    n64 = setting[np.isclose(setting["resolution"], 0.64)]
    axes[1].plot(
        n64["wiggle"],
        100 * n64["high_error_case_fraction"],
        "o-",
        label="High-error cases",
    )
    axes[1].plot(
        n64["wiggle"],
        100 * n64["no_linear_corner_case_fraction"],
        "s--",
        label="No line-line corner cells",
    )
    axes[1].plot(
        n64["wiggle"],
        100 * n64["shared_vertex_conflict_case_fraction"],
        "^:",
        label="Shared-vertex conflicts",
    )
    axes[1].set_xlabel("Perturbation magnitude")
    axes[1].set_ylabel("Cases at N=64 (%)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"b20_zalesak_cusp_diagnostic.{suffix}", dpi=200)
    plt.close(fig)


def run(sealed_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    case_metrics, inventory, cell_metrics = _load_inputs(sealed_root)
    topology = _evaluate_saved_topology(sealed_root, inventory, cell_metrics)
    joined = case_metrics.merge(
        topology,
        on=[
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "save_name",
            "case_index",
        ],
        how="inner",
        validate="one_to_one",
    )
    setting = _topology_by_setting(joined)
    _, paired_summary = _paired_summary(joined)
    zalesak_setting, zalesak_plot = _zalesak_summary(joined)
    line_association = _line_case_association(joined)
    zalesak_contingency = _zalesak_contingency(joined)

    setting.to_csv(output_dir / "topology_by_setting.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_variant_summary.csv", index=False)
    zalesak_setting.to_csv(output_dir / "zalesak_cusp_by_setting.csv", index=False)
    zalesak_plot.to_csv(output_dir / "zalesak_b20_plot_semantics.csv", index=False)
    line_association.to_csv(output_dir / "line_conflict_association.csv", index=False)
    zalesak_contingency.to_csv(output_dir / "zalesak_cusp_contingency.csv", index=False)
    _plot_b16(setting, output_dir)
    _plot_broad(paired_summary, output_dir)
    _plot_b20(zalesak_setting, output_dir)

    manifest = {
        "sealed_root": str(sealed_root.resolve()),
        "source_commit": sorted(case_metrics["source_commit"].dropna().unique()),
        "topology_definition": "Disagreement among phase labels assigned by saved incident facets at a shared mesh vertex.",
        "relative_geometric_tolerance": 1.0e-10,
        "absolute_geometric_tolerance": 1.0e-12,
        "zalesak_high_facet_gap_threshold": ZALASAK_HIGH_ERROR,
        "line_high_facet_gap_threshold": LINE_HIGH_ERROR,
        "case_rows_evaluated": int(len(joined)),
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.sealed_root, args.output_dir)


if __name__ == "__main__":
    main()
