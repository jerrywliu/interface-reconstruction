#!/usr/bin/env python3
"""Compare full vs drop-#9 Zalesak median metrics across resolution."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/static/debug/drop9_resolution_comparison_20260710/analysis"
INTERMEDIATE_CSV = ROOT / "results/static/debug/drop9_resolution_comparison_20260710/drop9_intermediate_sweep.csv"
RESOLUTIONS = [0.5, 0.64, 1.0, 1.28, 1.5]
WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]
ENDPOINT_WIGGLES = [0.0, 0.3]
METRICS = ["hausdorff", "facet_gap", "area_error"]
LABELS = {"full": "Full", "no_curved_corner_rescues": "Drop #9"}
COLORS = {"full": "#1f77b4", "no_curved_corner_rescues": "#d62728"}


def read_drop9(resolution, wiggle, metric):
    tag = f"r{str(resolution).replace('.', 'p')}_w{str(wiggle).replace('.', 'p')}"
    path = ROOT / "plots" / f"no_curved_corner_rescues_{tag}_s0" / "metrics" / f"{metric}.txt"
    if not path.exists():
        runner_tag = f"r{str(resolution).replace('.', 'p')}_w{str(wiggle).replace('.', 'p')}"
        path = (
            ROOT
            / "plots"
            / f"perturb_sweep_zalesak_circularpluscorner_{runner_tag}_s0_no_curved_corner_rescues"
            / "metrics"
            / f"{metric}.txt"
        )
    if not path.exists() and INTERMEDIATE_CSV.exists():
        with INTERMEDIATE_CSV.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["algo"] == "circular+corner"
                    and float(row["resolution"]) == resolution
                    and float(row["wiggle"]) == wiggle
                    and row["metric_key"] == f"{metric}_median"
                ):
                    return float(row["metric_value"])
        raise RuntimeError(f"Missing drop-#9 row for {resolution=} {wiggle=} {metric=}")
    values = [float(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != 25:
        raise RuntimeError(f"Expected 25 cases in {path}, found {len(values)}")
    return sorted(values)[12]


def read_full():
    path = ROOT / "results/static/lvira_default_affected_rerun_20260709/csv/zalesak.csv"
    full = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["algo"] != "circular+corner" or row["metric_key"] not in {
                "hausdorff_median",
                "facet_gap_median",
                "area_error_median",
            }:
                continue
            key = (float(row["resolution"]), float(row["wiggle"]), row["metric_key"][:-7])
            full[key] = float(row["metric_value"])
    return full


def aggregate(records, profile, metric, axis, value):
    if axis == "resolution":
        values = [
            record["median"]
            for record in records
            if record["profile"] == profile
            and record["metric"] == metric
            and record["resolution"] == value
        ]
    else:
        values = [
            record["median"]
            for record in records
            if record["profile"] == profile
            and record["metric"] == metric
            and record["wiggle"] == value
        ]
    return float(sorted(values)[len(values) // 2])


def write_paper_plots(records):
    resolution_records = []
    for metric in METRICS:
        for profile in LABELS:
            values = [aggregate(records, profile, metric, "resolution", resolution) for resolution in RESOLUTIONS]
            resolution_records.append({"profile": profile, "metric": metric, "values": values})

    wiggle_records = []
    for metric in METRICS:
        for profile in LABELS:
            values = [aggregate(records, profile, metric, "wiggle", wiggle) for wiggle in WIGGLES]
            wiggle_records.append({"profile": profile, "metric": metric, "values": values})

    for filename, x_values, grouped_records, x_label, title in [
        (
            "drop9_paper_resolution_medians.png",
            [int(round(100 * resolution)) for resolution in RESOLUTIONS],
            resolution_records,
            "Cells per side, N",
            "Zalesak paper-style resolution medians: full vs drop #9",
        ),
        (
            "drop9_paper_perturbation_medians.png",
            WIGGLES,
            wiggle_records,
            "Perturbation magnitude, w",
            "Zalesak paper-style perturbation medians: full vs drop #9",
        ),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
        for axis, metric in zip(axes, METRICS):
            for profile in LABELS:
                values = next(item["values"] for item in grouped_records if item["profile"] == profile and item["metric"] == metric)
                axis.plot(
                    x_values,
                    values,
                    marker="o",
                    linewidth=2.2,
                    label=LABELS[profile],
                    color=COLORS[profile],
                )
            axis.set_yscale("log")
            axis.set_title(metric.replace("_", " ").title())
            axis.set_xlabel(x_label)
            axis.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("Median metric")
        axes[0].legend(loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    with (OUT / "paper_aggregate_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["axis", "profile", "metric", "x", "median"])
        writer.writeheader()
        for axis, x_values, grouped_records in [
            ("resolution", RESOLUTIONS, resolution_records),
            ("wiggle", WIGGLES, wiggle_records),
        ]:
            for item in grouped_records:
                for x_value, median in zip(x_values, item["values"]):
                    writer.writerow(
                        {
                            "axis": axis,
                            "profile": item["profile"],
                            "metric": item["metric"],
                            "x": x_value,
                            "median": median,
                        }
                    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = read_full()
    records = []
    for wiggle in WIGGLES:
        for resolution in RESOLUTIONS:
            for metric in METRICS:
                records.append(
                    {
                        "profile": "full",
                        "resolution": resolution,
                        "wiggle": wiggle,
                        "metric": metric,
                        "median": full[(resolution, wiggle, metric)],
                    }
                )
                records.append(
                    {
                        "profile": "no_curved_corner_rescues",
                        "resolution": resolution,
                        "wiggle": wiggle,
                        "metric": metric,
                        "median": read_drop9(resolution, wiggle, metric),
                    }
                )

    with (OUT / "resolution_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["profile", "resolution", "wiggle", "metric", "median"])
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for row_index, wiggle in enumerate(ENDPOINT_WIGGLES):
        for col_index, metric in enumerate(METRICS):
            axis = axes[row_index, col_index]
            for profile in LABELS:
                values = [
                    next(
                        record["median"]
                        for record in records
                        if record["profile"] == profile
                        and record["resolution"] == resolution
                        and record["wiggle"] == wiggle
                        and record["metric"] == metric
                    )
                    for resolution in RESOLUTIONS
                ]
                axis.plot(
                    [int(round(100 * resolution)) for resolution in RESOLUTIONS],
                    values,
                    marker="o",
                    linewidth=2.2,
                    label=LABELS[profile],
                    color=COLORS[profile],
                )
            axis.set_yscale("log")
            axis.set_title(metric.replace("_", " ").title())
            axis.grid(True, which="both", alpha=0.25)
            axis.set_xlabel("Cells per side, N")
            if col_index == 0:
                axis.set_ylabel(f"Median, perturbation={wiggle}")
            if row_index == 0 and col_index == 0:
                axis.legend(loc="best")
    fig.suptitle("Zalesak median metrics vs resolution: full vs drop #9")
    fig.tight_layout()
    fig.savefig(OUT / "drop9_resolution_medians.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_paper_plots(records)

    with (OUT / "summary.md").open("w") as handle:
        handle.write("# Drop-#9 resolution comparison\n\n")
        handle.write(
            "Full profile versus `no_curved_corner_rescues` (keep #1--#6, drop only #9), "
            "25 perturbed-quad Zalesak cases, seed 0, LVIRA fallback.\n\n"
        )
        handle.write(
            "The paper-style plots aggregate the per-`w` medians over the full spectrum "
            "`w = 0, 0.05, 0.1, 0.2, 0.3` for the resolution axis, and aggregate over "
            "the five resolutions for the perturbation axis.\n\n"
        )
        handle.write("- `drop9_paper_resolution_medians.png`: exact resolution-axis aggregation.\n")
        handle.write("- `drop9_paper_perturbation_medians.png`: exact perturbation-axis aggregation.\n")
        handle.write("- `drop9_resolution_medians.png`: endpoint diagnostic for `w=0` and `w=0.3`.\n\n")
        for wiggle in WIGGLES:
            handle.write(f"## Perturbation {wiggle}\n\n")
            handle.write("| N | H full | H drop #9 | gap full | gap drop #9 | area full | area drop #9 |\n")
            handle.write("|---:|---:|---:|---:|---:|---:|---:|\n")
            for resolution in RESOLUTIONS:
                vals = {}
                for metric in METRICS:
                    for profile in LABELS:
                        vals[(metric, profile)] = next(
                            record["median"]
                            for record in records
                            if record["profile"] == profile
                            and record["resolution"] == resolution
                            and record["wiggle"] == wiggle
                            and record["metric"] == metric
                        )
                h_full = vals[("hausdorff", "full")]
                h_drop9 = vals[("hausdorff", "no_curved_corner_rescues")]
                gap_full = vals[("facet_gap", "full")]
                gap_drop9 = vals[("facet_gap", "no_curved_corner_rescues")]
                area_full = vals[("area_error", "full")]
                area_drop9 = vals[("area_error", "no_curved_corner_rescues")]
                handle.write(
                    f"| {int(round(100 * resolution))} | {h_full:.6e} | {h_drop9:.6e} | "
                    f"{gap_full:.6e} | {gap_drop9:.6e} | {area_full:.6e} | "
                    f"{area_drop9:.6e} |\n"
                )
            handle.write("\n")

    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
