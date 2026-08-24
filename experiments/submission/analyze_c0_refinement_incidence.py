#!/usr/bin/env python3
"""Summarize how often and how broadly joint C0 refinement is required."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("results/static/camera_ready/appendix_b5_joint_c0_20260814")
DEFAULT_OUTPUT = Path("results/submission/c0_refinement_incidence_20260823")
EXPERIMENTS = ("ellipses", "zalesak")
COLORS = {"ellipses": "#4C78A8", "zalesak": "#D1495B"}
LABELS = {"ellipses": "Ellipses", "zalesak": "Zalesak"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def analyze(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qa_rows = _read_csv(source / "csv" / "joint_c0_case_qa.csv")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    component_sizes = {experiment: [] for experiment in EXPERIMENTS}
    component_kinds = {experiment: Counter() for experiment in EXPERIMENTS}
    function_evaluations = {experiment: [] for experiment in EXPERIMENTS}
    participating_original_cells = Counter()

    for run in manifest["runs"]:
        experiment = run["experiment"]
        metrics = Path(run["run_dir"]) / "metrics"
        component_merge_ids: dict[int, set[str]] = defaultdict(set)
        for row in _read_csv(metrics / "c0_components.csv"):
            size = int(row["num_facets"])
            component_sizes[experiment].append(size)
            component_kinds[experiment][row["solution_kind"]] += 1
            if row["function_evaluations"]:
                function_evaluations[experiment].append(
                    int(row["function_evaluations"])
                )
            component_merge_ids[int(row["case_index"])].update(
                str(value) for value in json.loads(row["merge_ids_json"])
            )
        for row in _read_csv(metrics / "cell_metrics.csv"):
            case_index = int(row["case_index"])
            if row["merge_id"] in component_merge_ids.get(case_index, set()):
                participating_original_cells[experiment] += 1

    summaries: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    for experiment in (*EXPERIMENTS, "combined"):
        selected_qa = (
            qa_rows
            if experiment == "combined"
            else [row for row in qa_rows if row["experiment"] == experiment]
        )
        sizes = (
            component_sizes["ellipses"] + component_sizes["zalesak"]
            if experiment == "combined"
            else component_sizes[experiment]
        )
        kinds = (
            component_kinds["ellipses"] + component_kinds["zalesak"]
            if experiment == "combined"
            else component_kinds[experiment]
        )
        evaluations = (
            function_evaluations["ellipses"] + function_evaluations["zalesak"]
            if experiment == "combined"
            else function_evaluations[experiment]
        )
        eligible_joins = sum(int(row["num_c0_eligible_joins"]) for row in selected_qa)
        bad_before = sum(
            int(row["num_c0_bad_joins_before_joint"]) for row in selected_qa
        )
        bad_after = sum(int(row["num_c0_bad_joins_after_joint"]) for row in selected_qa)
        mixed_cells = sum(int(row["num_mixed_cells"]) for row in selected_qa)
        participating = (
            sum(participating_original_cells.values())
            if experiment == "combined"
            else participating_original_cells[experiment]
        )
        cases_requiring_joint = sum(
            int(row["num_c0_joint_components"]) > 0 for row in selected_qa
        )
        summaries.append(
            {
                "experiment": experiment,
                "cases": len(selected_qa),
                "cases_requiring_joint": cases_requiring_joint,
                "fraction_cases_requiring_joint": cases_requiring_joint
                / len(selected_qa),
                "mixed_cells": mixed_cells,
                "mixed_cells_in_joint_components": participating,
                "fraction_mixed_cells_in_joint_components": participating / mixed_cells,
                "eligible_joins": eligible_joins,
                "bad_joins_after_local": bad_before,
                "fraction_bad_joins_after_local": bad_before / eligible_joins,
                "bad_joins_after_joint": bad_after,
                "fraction_bad_joins_after_joint": bad_after / eligible_joins,
                "joint_components": len(sizes),
                "mean_facets_per_component": statistics.mean(sizes),
                "median_facets_per_component": statistics.median(sizes),
                "p90_facets_per_component": _percentile(sizes, 0.9),
                "max_facets_per_component": max(sizes),
                "exact_tangent_components": kinds["exact_c1"],
                "conservative_c0_components": kinds["c0_min_tangent"],
                "failed_components": kinds["failed"],
                "median_function_evaluations": statistics.median(evaluations),
                "p90_function_evaluations": _percentile(evaluations, 0.9),
                "max_function_evaluations": max(evaluations),
            }
        )
        counts = Counter(sizes)
        for size in sorted(counts):
            distributions.append(
                {
                    "experiment": experiment,
                    "num_facets": size,
                    "components": counts[size],
                    "fraction_components": counts[size] / len(sizes),
                }
            )
    return summaries, distributions


def plot(
    summaries: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    by_experiment = {row["experiment"]: row for row in summaries}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)

    x = np.arange(len(EXPERIMENTS))
    width = 0.32
    local_values = [
        100.0 * by_experiment[experiment]["fraction_bad_joins_after_local"]
        for experiment in EXPERIMENTS
    ]
    joint_values = [
        100.0 * by_experiment[experiment]["fraction_bad_joins_after_joint"]
        for experiment in EXPERIMENTS
    ]
    axes[0].bar(
        x - width / 2,
        local_values,
        width,
        color="#8DA0B6",
        label="After local pass",
    )
    axes[0].bar(
        x + width / 2,
        joint_values,
        width,
        color="#2F6B4F",
        label="After joint refinement",
    )
    axes[0].set_xticks(x, [LABELS[experiment] for experiment in EXPERIMENTS])
    axes[0].set_ylabel("Eligible joins with a gap (\%)")
    axes[0].set_title("Residual endpoint gaps")
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False)
    for positions, values in (
        (x - width / 2, local_values),
        (x + width / 2, joint_values),
    ):
        for position, value in zip(positions, values):
            axes[0].text(
                position,
                value + 0.35,
                f"{value:.2f}\%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    histogram_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11+"]
    histogram_x = np.arange(len(histogram_labels))
    grouped: dict[str, Counter[int]] = {}
    for experiment in EXPERIMENTS:
        counter: Counter[int] = Counter()
        for row in distributions:
            if row["experiment"] == experiment:
                size = int(row["num_facets"])
                counter[min(size, 11)] += int(row["components"])
        grouped[experiment] = counter
    for offset, experiment in zip((-width / 2, width / 2), EXPERIMENTS):
        total = sum(grouped[experiment].values())
        values = [100.0 * grouped[experiment][index] / total for index in range(1, 12)]
        axes[1].bar(
            histogram_x + offset,
            values,
            width,
            color=COLORS[experiment],
            label=LABELS[experiment],
        )
    axes[1].set_xticks(histogram_x, histogram_labels)
    axes[1].set_xlabel("Facets in jointly refined component")
    axes[1].set_ylabel("Components (\%)")
    axes[1].set_title("Joint-component size distribution")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False)

    figure.savefig(output / "c0_joint_refinement_incidence.pdf", bbox_inches="tight")
    figure.savefig(
        output / "c0_joint_refinement_incidence.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, distributions = analyze(args.source)
    _write_csv(args.output / "summary.csv", summaries, summaries[0].keys())
    _write_csv(
        args.output / "component_size_distribution.csv",
        distributions,
        distributions[0].keys(),
    )
    plot(summaries, distributions, args.output)
    print(args.output / "c0_joint_refinement_incidence.pdf")


if __name__ == "__main__":
    main()
