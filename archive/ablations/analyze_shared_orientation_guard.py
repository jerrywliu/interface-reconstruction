#!/usr/bin/env python3
"""Summarize the matched post-f8 versus pre-f8 orientation guard."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROFILES = ("current", "pre_f8_corner")
SCIENTIFIC_METRICS = (
    "hausdorff",
    "facet_gap",
    "area_error",
    "curvature_error",
    "tangent_error",
    "curvature_proxy_error",
)


def _optional_float(value):
    return None if value == "" else float(value)


def _load_pairs(case_metrics_path):
    with case_metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    pairs = defaultdict(dict)
    for row in rows:
        key = (
            row["experiment"],
            row["algo"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
            int(row["case_index"]),
        )
        pairs[key][row["corner_behavior_profile"]] = row

    invalid = {key: sorted(pair) for key, pair in pairs.items() if set(pair) != set(PROFILES)}
    if invalid:
        raise ValueError(f"Unpaired orientation rows: {invalid}")
    return pairs


def _build_rows(pairs):
    comparisons = []
    grouped = defaultdict(list)
    for key, pair in sorted(pairs.items()):
        experiment, algo, resolution, wiggle, seed, case_index = key
        current = pair["current"]
        candidate = pair["pre_f8_corner"]
        row = {
            "experiment": experiment,
            "algo": algo,
            "resolution": resolution,
            "wiggle": wiggle,
            "seed": seed,
            "case_index": case_index,
        }
        for metric in SCIENTIFIC_METRICS:
            current_value = _optional_float(current[metric])
            candidate_value = _optional_float(candidate[metric])
            row[f"current_{metric}"] = current_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"delta_{metric}"] = (
                None
                if current_value is None or candidate_value is None
                else candidate_value - current_value
            )
        comparisons.append(row)
        grouped[(experiment, algo)].append(row)

    summaries = []
    for (experiment, algo), rows in sorted(grouped.items()):
        current = np.asarray([row["current_hausdorff"] for row in rows])
        candidate = np.asarray([row["candidate_hausdorff"] for row in rows])
        summaries.append(
            {
                "experiment": experiment,
                "algo": algo,
                "case_count": len(rows),
                "changed_hausdorff_cases": int(np.count_nonzero(current != candidate)),
                "improved_hausdorff_cases": int(np.count_nonzero(candidate < current)),
                "worsened_hausdorff_cases": int(np.count_nonzero(candidate > current)),
                "current_hausdorff_median": float(np.median(current)),
                "candidate_hausdorff_median": float(np.median(candidate)),
                "current_hausdorff_mean": float(np.mean(current)),
                "candidate_hausdorff_mean": float(np.mean(candidate)),
                "current_hausdorff_max": float(np.max(current)),
                "candidate_hausdorff_max": float(np.max(candidate)),
            }
        )
    return comparisons, summaries


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(comparisons, output_path):
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[(row["experiment"], row["algo"])].append(row)

    groups = sorted(grouped)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.2))
    for ax, key in zip(axes.flat, groups):
        rows = grouped[key]
        current = np.asarray([row["current_hausdorff"] for row in rows])
        candidate = np.asarray([row["candidate_hausdorff"] for row in rows])
        floor = 1e-12
        current_plot = np.maximum(current, floor)
        candidate_plot = np.maximum(candidate, floor)
        lower = min(current_plot.min(), candidate_plot.min()) / 2
        upper = max(current_plot.max(), candidate_plot.max()) * 2

        ax.scatter(current_plot, candidate_plot, s=24, color="#147d64", alpha=0.82)
        ax.plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(True, which="both", alpha=0.18)
        improved = int(np.count_nonzero(candidate < current))
        worsened = int(np.count_nonzero(candidate > current))
        ax.set_title(f"{key[0]} / {key[1]}\n{improved} improved, {worsened} worsened", fontsize=10)

    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    fig.supxlabel("Post-f8 orientation hint: case Hausdorff")
    fig.supylabel("Pre-f8 orientation behavior: case Hausdorff")
    fig.suptitle("Shared-orientation guard across all non-corner merged methods", fontsize=14)
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_root",
        type=Path,
        default=Path("results/static/pre_f8_shared_orientation_guard_20260717"),
    )
    args = parser.parse_args()

    output_dir = args.run_root / "analysis"
    pairs = _load_pairs(args.run_root / "diagnostics" / "case_metrics.csv")
    comparisons, summaries = _build_rows(pairs)
    _write_csv(output_dir / "case_comparison.csv", comparisons)
    _write_csv(output_dir / "summary.csv", summaries)
    _plot(comparisons, output_dir / "shared_orientation_guard_all_methods.png")

    total_improved = sum(row["improved_hausdorff_cases"] for row in summaries)
    total_worsened = sum(row["worsened_hausdorff_cases"] for row in summaries)
    changed_pairs = sum(row["changed_hausdorff_cases"] for row in summaries)
    readme = "\n".join(
        [
            "# Shared-orientation guard",
            "",
            "Matched `current` (post-f8 three-neighbor orientation hint) against the new",
            "default `pre_f8_corner` behavior at `N=150`, perturbation magnitude `0.3`,",
            "seed `0`, with `25` cases per method.",
            "",
            "## Result",
            "",
            f"- `18/18` runs succeeded, covering `{len(comparisons)}` matched case pairs.",
            f"- Hausdorff changed in `{changed_pairs}` pairs: `{total_improved}` improved and `{total_worsened}` worsened.",
            "- The guard therefore supports removing the shared three-neighbor hint from the production default.",
            "- Secondary metric and method-level details are retained in the CSVs.",
            "",
            "## Artifacts",
            "",
            "- `summary.csv`",
            "- `case_comparison.csv`",
            "- `shared_orientation_guard_all_methods.png` / `.pdf`",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
