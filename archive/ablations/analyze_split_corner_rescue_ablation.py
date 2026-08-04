#!/usr/bin/env python3
"""Merge and summarize the targeted split corner-rescue ablation."""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/static/debug/split_corner_rescue_ablation_20260710/analysis"
SELECTED_CASES = [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 14, 18, 20, 22, 23, 24]
GROUPS = {
    "a": [0, 1, 2, 3, 5],
    "b": [6, 7, 8, 11, 12, 14],
    "c": [18, 20, 22, 23, 24],
}
PROFILES = [
    "full",
    "no_linear_corner_rescues",
    "no_curved_corner_rescues",
    "no_repeated_corner_rescues",
    "no_repeated_tiny_corner_rescues",
    "no_repeated_corner_component_rescues",
    "candidate_keep_12346_drop_9",
    "no_corner_rescues",
]
METRICS = ["hausdorff", "facet_gap", "area_error"]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_float_lines(path):
    return [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_full_and_no_rescues():
    full_rows = {
        int(row["case_index"]): row
        for row in read_rows(
            ROOT / "results/static/debug/fast_signal_lvira_default_20260709/case_metrics.csv"
        )
        if row["experiment"] == "zalesak"
        and row["algo"] == "circular+corner"
        and row["resolution"] == "1.5"
        and row["wiggle"] == "0.3"
        and row["policy"] == "default_LVIRA"
    }
    no_rescue_rows = {
        int(row["case_index"]): row
        for row in read_rows(
            ROOT
            / "results/static/debug/no_corner_rescue_ablation_20260710/analysis/case_comparison.csv"
        )
    }
    data = {profile: {} for profile in PROFILES}
    for case_index in SELECTED_CASES:
        full = full_rows[case_index]
        no_rescue = no_rescue_rows[case_index]
        for metric in METRICS:
            data["full"].setdefault(case_index, {})[metric] = float(full[metric])
            data["no_corner_rescues"].setdefault(case_index, {})[metric] = float(
                no_rescue[f"no_corner_rescues_{metric}"]
            )
    return data


def load_split_profile(data, profile):
    for group, cases in GROUPS.items():
        base = ROOT / "plots" / f"{profile}_{group}_r1p5_w0p3_s0" / "metrics"
        values = {metric: read_float_lines(base / f"{metric}.txt") for metric in METRICS}
        if any(len(values[metric]) != len(cases) for metric in METRICS):
            raise RuntimeError(f"Metric length mismatch for {profile}/{group}: {values}")
        for position, case_index in enumerate(cases):
            data[profile].setdefault(case_index, {})
            for metric in METRICS:
                data[profile][case_index][metric] = values[metric][position]


def fmt(value):
    return f"{value:.6e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_full_and_no_rescues()
    for profile in (
        "no_linear_corner_rescues",
        "no_curved_corner_rescues",
        "no_repeated_corner_rescues",
        "no_repeated_tiny_corner_rescues",
        "no_repeated_corner_component_rescues",
        "candidate_keep_12346_drop_9",
    ):
        load_split_profile(data, profile)

    for profile in PROFILES:
        missing = sorted(set(SELECTED_CASES) - set(data[profile]))
        if missing:
            raise RuntimeError(f"Missing cases for {profile}: {missing}")

    comparison_path = OUT / "case_comparison.csv"
    fieldnames = ["case_index"]
    for profile in PROFILES:
        for metric in METRICS:
            fieldnames.append(f"{profile}_{metric}")
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_index in SELECTED_CASES:
            row = {"case_index": case_index}
            for profile in PROFILES:
                for metric in METRICS:
                    row[f"{profile}_{metric}"] = data[profile][case_index][metric]
            writer.writerow(row)

    summary = []
    for profile in PROFILES:
        row = {"profile": profile, "n": len(SELECTED_CASES)}
        for metric in METRICS:
            values = np.array([data[profile][case][metric] for case in SELECTED_CASES])
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_max"] = float(np.max(values))
        hausdorff = np.array([data[profile][case]["hausdorff"] for case in SELECTED_CASES])
        row["hausdorff_below_1e-6"] = int(np.sum(hausdorff < 1e-6))
        row["hausdorff_above_0p5"] = int(np.sum(hausdorff > 0.5))
        summary.append(row)

    summary_csv = OUT / "summary.csv"
    summary_fields = ["profile", "n"]
    for metric in METRICS:
        summary_fields += [f"{metric}_median", f"{metric}_mean", f"{metric}_max"]
    summary_fields += ["hausdorff_below_1e-6", "hausdorff_above_0p5"]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    labels = {
        "full": "Full",
        "no_linear_corner_rescues": "No linear/branch",
        "no_curved_corner_rescues": "No curved loop/transition",
        "no_repeated_corner_rescues": "No repeated-corner rescues",
        "no_repeated_tiny_corner_rescues": "No tiny triplet rescue (#4)",
        "no_repeated_corner_component_rescues": "No repeated component rescue (#5)",
        "candidate_keep_12346_drop_9": "Candidate: keep 1,2,3,4,6",
        "no_corner_rescues": "No added rescues",
    }
    colors = {
        "full": "#1f77b4",
        "no_linear_corner_rescues": "#d62728",
        "no_curved_corner_rescues": "#2ca02c",
        "no_repeated_corner_rescues": "#ff7f0e",
        "no_repeated_tiny_corner_rescues": "#8c564b",
        "no_repeated_corner_component_rescues": "#e377c2",
        "candidate_keep_12346_drop_9": "#17becf",
        "no_corner_rescues": "#9467bd",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    x = np.arange(len(SELECTED_CASES))
    for axis, metric in zip(axes, METRICS):
        for profile in PROFILES:
            values = [data[profile][case][metric] for case in SELECTED_CASES]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=labels[profile],
                color=colors[profile],
            )
        axis.set_yscale("log")
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Case index")
        axis.set_xticks(x)
        axis.set_xticklabels(SELECTED_CASES, rotation=60, ha="right")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Metric value (log scale)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Zalesak split corner-rescue ablation\nN=150, perturbation=0.3, seed=0")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_md = OUT / "summary.md"
    with summary_md.open("w") as handle:
        handle.write("# Split corner-rescue ablation\n\n")
        handle.write(
            "Matched targeted Zalesak cases: `circular+corner`, perturbed quads, "
            "`N=150`, perturbation magnitude `0.3`, seed `0`, LVIRA fallback.\n\n"
        )
        handle.write(
            "Selected cases are intentionally failure-heavy and are not a representative "
            "full-suite distribution. Profiles differ only in the added rescue calls; "
            "ordinary fitting, arc-root fallback, dead-end handling, and final PLIC fallback "
            "remain enabled.\n\n"
        )
        handle.write("Selected cases: `" + ",".join(map(str, SELECTED_CASES)) + "`.\n\n")
        handle.write(
            "| Profile | H median | Facet-gap median | Area-error median | H < 1e-6 | H > 0.5 |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for row in summary:
            handle.write(
                f"| `{row['profile']}` | {fmt(row['hausdorff_median'])} | "
                f"{fmt(row['facet_gap_median'])} | {fmt(row['area_error_median'])} | "
                f"{row['hausdorff_below_1e-6']}/{row['n']} | "
                f"{row['hausdorff_above_0p5']}/{row['n']} |\n"
            )
        handle.write("\nArtifacts: `case_comparison.csv`, `summary.csv`, `comparison.png`.\n")

    print(summary_md)
    for row in summary:
        print(
            row["profile"],
            "H_median=", fmt(row["hausdorff_median"]),
            "gap_median=", fmt(row["facet_gap_median"]),
            "area_median=", fmt(row["area_error_median"]),
            "H<1e-6=", row["hausdorff_below_1e-6"],
            "H>0.5=", row["hausdorff_above_0p5"],
        )


if __name__ == "__main__":
    main()
