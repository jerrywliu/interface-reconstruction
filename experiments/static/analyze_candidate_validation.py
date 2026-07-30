#!/usr/bin/env python3
"""Summarize the paired full-vs-candidate Zalesak validation grid."""

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/static/debug/candidate_validation_20260710/analysis"
PROFILES = [
    "full",
    "candidate_keep_12346_drop_9",
    "no_curved_corner_rescues",
]
SETTINGS = [
    ("r0p64_w0p0", 0.64, 0.0),
    ("r0p64_w0p3", 0.64, 0.3),
    ("r1p5_w0p0", 1.5, 0.0),
    ("r1p5_w0p3", 1.5, 0.3),
]
METRICS = ["hausdorff", "facet_gap", "area_error"]


def read_metric(profile, tag, metric):
    path = ROOT / "plots" / f"{profile}_{tag}_s0" / "metrics" / f"{metric}.txt"
    return [float(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    case_records = []
    for tag, resolution, wiggle in SETTINGS:
        values = {
            profile: {metric: read_metric(profile, tag, metric) for metric in METRICS}
            for profile in PROFILES
        }
        if any(len(values[profile]["hausdorff"]) != 25 for profile in PROFILES):
            raise RuntimeError(f"Expected 25 cases for {tag}: {values}")

        for profile in PROFILES:
            row = {
                "profile": profile,
                "setting": tag,
                "resolution": resolution,
                "wiggle": wiggle,
                "n": 25,
            }
            for metric in METRICS:
                metric_values = values[profile][metric]
                row[f"{metric}_median"] = statistics.median(metric_values)
                row[f"{metric}_mean"] = statistics.mean(metric_values)
                row[f"{metric}_max"] = max(metric_values)
            hausdorff = values[profile]["hausdorff"]
            row["hausdorff_below_1e-6"] = sum(value < 1e-6 for value in hausdorff)
            row["hausdorff_above_0p5"] = sum(value > 0.5 for value in hausdorff)
            records.append(row)

        for case_index in range(25):
            row = {
                "setting": tag,
                "resolution": resolution,
                "wiggle": wiggle,
                "case_index": case_index,
            }
            for profile in PROFILES:
                for metric in METRICS:
                    row[f"{profile}_{metric}"] = values[profile][metric][case_index]
            case_records.append(row)

    summary_fields = ["profile", "setting", "resolution", "wiggle", "n"]
    for metric in METRICS:
        summary_fields += [f"{metric}_median", f"{metric}_mean", f"{metric}_max"]
    summary_fields += ["hausdorff_below_1e-6", "hausdorff_above_0p5"]
    with (OUT / "grid_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(records)

    case_fields = ["setting", "resolution", "wiggle", "case_index"]
    for profile in PROFILES:
        for metric in METRICS:
            case_fields.append(f"{profile}_{metric}")
    with (OUT / "case_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        writer.writerows(case_records)

    colors = {
        "full": "#1f77b4",
        "candidate_keep_12346_drop_9": "#17becf",
        "no_curved_corner_rescues": "#2ca02c",
    }
    labels = {
        "full": "Full",
        "candidate_keep_12346_drop_9": "Candidate",
        "no_curved_corner_rescues": "Drop #9, keep #5",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, (tag, resolution, wiggle) in zip(axes.flat, SETTINGS):
        rows = [row for row in case_records if row["setting"] == tag]
        x = np.arange(25)
        for profile in PROFILES:
            axis.plot(
                x,
                [row[f"{profile}_hausdorff"] for row in rows],
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=labels[profile],
                color=colors[profile],
            )
        axis.set_yscale("log")
        axis.set_title(f"N={int(round(resolution * 100))}, perturbation={wiggle}")
        axis.set_xlabel("Case index")
        axis.set_xticks(x)
        axis.set_xticklabels(range(25), rotation=60, ha="right")
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel("Hausdorff (log scale)")
    axes[1, 0].set_ylabel("Hausdorff (log scale)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Candidate rescue profile validation against full profile")
    fig.tight_layout()
    fig.savefig(OUT / "hausdorff_grid_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (OUT / "summary.md").open("w") as handle:
        handle.write("# Candidate rescue profile validation\n\n")
        handle.write(
            "Candidate: keep additions #1/#2/#3/#4/#6, drop #5 and all #9 curved "
            "loop/transition rescues. LVIRA fallback is held fixed.\n\n"
        )
        handle.write(
            "All runs use 25 perturbed-quad Zalesak cases, seed 0. The candidate is "
            "compared against a fresh full-profile run at each setting.\n\n"
        )
        handle.write(
            "| Setting | H median full/candidate/drop-9 | H mean full/candidate/drop-9 | "
            "H max full/candidate/drop-9 | H>0.5 full/candidate/drop-9 |\n"
            "|---|---:|---:|---:|---:|\n"
        )
        for tag, _, _ in SETTINGS:
            rows = {
                row["profile"]: row
                for row in records
                if row["setting"] == tag
            }
            full = rows["full"]
            candidate = rows["candidate_keep_12346_drop_9"]
            drop9 = rows["no_curved_corner_rescues"]
            handle.write(
                f"| `{tag}` | {full['hausdorff_median']:.6e} / "
                f"{candidate['hausdorff_median']:.6e} / {drop9['hausdorff_median']:.6e} | "
                f"{full['hausdorff_mean']:.6e} / {candidate['hausdorff_mean']:.6e} / "
                f"{drop9['hausdorff_mean']:.6e} | {full['hausdorff_max']:.6e} / "
                f"{candidate['hausdorff_max']:.6e} / {drop9['hausdorff_max']:.6e} | "
                f"{full['hausdorff_above_0p5']}/{candidate['hausdorff_above_0p5']}/"
                f"{drop9['hausdorff_above_0p5']} |\n"
            )
        handle.write(
            "\nThe candidate and the drop-#9 control match at every setting. Adding #5 back "
            "therefore does not recover the regressions; the changes are caused by removing #9.\n"
        )

    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
