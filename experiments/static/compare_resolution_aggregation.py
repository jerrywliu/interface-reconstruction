#!/usr/bin/env python3
"""Compare hierarchical and pooled aggregation for Section 6 resolution plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


mpl.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GRID_SIZE = 100
ERROR_FLOOR = 1.0e-14
METRICS = ("hausdorff", "facet_gap")
METHODS = {
    "lines": ("Youngs", "ELVIRA", "LVIRA", "linear"),
    "squares": ("ELVIRA", "LVIRA", "linear", "linear+corner"),
    "circles": ("ELVIRA", "LVIRA", "linear", "circular"),
    "ellipses": ("ELVIRA", "LVIRA", "linear", "circular"),
    "zalesak": ("ELVIRA", "LVIRA", "circular", "circular+corner"),
}
LABELS = {
    "Youngs": "Youngs",
    "ELVIRA": "ELVIRA",
    "LVIRA": "LVIRA",
    "linear": "Ours (linear)",
    "linear+corner": "Ours (linear+corner)",
    "circular": "Ours (circular)",
    "circular+corner": "Ours (circular+corner)",
}
COLORS = {
    "Youngs": "#6b7280",
    "ELVIRA": "#2563eb",
    "LVIRA": "#111827",
    "linear": "#0891b2",
    "linear+corner": "#059669",
    "circular": "#ea580c",
    "circular+corner": "#dc2626",
}
MARKERS = {
    "Youngs": "o",
    "ELVIRA": "s",
    "LVIRA": "^",
    "linear": "D",
    "linear+corner": "P",
    "circular": "v",
    "circular+corner": "X",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    default_root = (
        repo
        / "results"
        / "static"
        / "submission_static_20260731_012430_505aefa45432.sealed"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-metrics",
        type=Path,
        default=default_root / "diagnostics" / "case_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "output" / "pdf" / "aggregation_audit_20260828",
    )
    return parser.parse_args()


def load_values(path: Path):
    values = defaultdict(list)
    expected = {(exp, method) for exp, methods in METHODS.items() for method in methods}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            exp = row["experiment"]
            method = row["algo"]
            if (exp, method) not in expected:
                continue
            resolution = float(row["resolution"])
            wiggle = float(row["wiggle"])
            for metric in METRICS:
                raw = row.get(metric, "")
                if raw not in ("", None):
                    values[(exp, method, metric, resolution, wiggle)].append(float(raw))
    return values


def summarize(values):
    grouped = defaultdict(dict)
    for (exp, method, metric, resolution, wiggle), samples in values.items():
        grouped[(exp, method, metric, resolution)][wiggle] = np.asarray(samples)

    rows = []
    for (exp, method, metric, resolution), by_wiggle in sorted(grouped.items()):
        if len(by_wiggle) != 5:
            raise ValueError(
                f"Expected five perturbation levels for {exp}/{method}/{metric}/{resolution}, "
                f"found {sorted(by_wiggle)}"
            )
        counts = {len(samples) for samples in by_wiggle.values()}
        if counts != {25}:
            raise ValueError(
                f"Expected 25 cases per perturbation level for "
                f"{exp}/{method}/{metric}/{resolution}, found {sorted(counts)}"
            )

        perturbation_medians = np.asarray(
            [np.median(by_wiggle[w]) for w in sorted(by_wiggle)]
        )
        pooled = np.concatenate([by_wiggle[w] for w in sorted(by_wiggle)])
        current = np.percentile(perturbation_medians, (25, 50, 75))
        pooled_stats = np.percentile(pooled, (25, 50, 75))
        rows.append(
            {
                "experiment": exp,
                "method": method,
                "metric": metric,
                "resolution": resolution,
                "cells_per_side": int(round(GRID_SIZE * resolution)),
                "case_count": len(pooled),
                "current_p25": current[0],
                "current_median": current[1],
                "current_p75": current[2],
                "pooled_p25": pooled_stats[0],
                "pooled_median": pooled_stats[1],
                "pooled_p75": pooled_stats[2],
            }
        )
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_series(ax, rows, exp: str, metric: str, mode: str):
    prefix = "current" if mode == "current" else "pooled"
    for method in METHODS[exp]:
        selected = sorted(
            (
                row
                for row in rows
                if row["experiment"] == exp
                and row["method"] == method
                and row["metric"] == metric
            ),
            key=lambda row: row["cells_per_side"],
        )
        x = np.asarray([row["cells_per_side"] for row in selected])
        median = np.maximum(
            np.asarray([row[f"{prefix}_median"] for row in selected]), ERROR_FLOOR
        )
        p25 = np.maximum(
            np.asarray([row[f"{prefix}_p25"] for row in selected]), ERROR_FLOOR
        )
        p75 = np.maximum(
            np.asarray([row[f"{prefix}_p75"] for row in selected]), ERROR_FLOOR
        )
        ax.plot(
            x,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=3.8,
            linewidth=1.5,
            label=LABELS[method],
        )
        ax.fill_between(x, p25, p75, color=COLORS[method], alpha=0.10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("Cells per side, $N$")
    ax.set_ylabel("Hausdorff distance" if metric == "hausdorff" else "Facet gap")


def substantive_factor(a: float, b: float) -> float:
    if max(a, b) <= 1.0e-12:
        return 1.0
    a = max(a, ERROR_FLOOR)
    b = max(b, ERROR_FLOOR)
    return max(a / b, b / a)


def benchmark_diagnostics(rows, exp: str):
    selected = [row for row in rows if row["experiment"] == exp]
    median_factors = [
        substantive_factor(row["current_median"], row["pooled_median"])
        for row in selected
    ]
    wider = 0
    comparable = 0
    width_ratios = []
    for row in selected:
        current_width = max(row["current_p75"], ERROR_FLOOR) / max(
            row["current_p25"], ERROR_FLOOR
        )
        pooled_width = max(row["pooled_p75"], ERROR_FLOOR) / max(
            row["pooled_p25"], ERROR_FLOOR
        )
        if max(row["current_p75"], row["pooled_p75"]) > 1.0e-12:
            comparable += 1
            wider += pooled_width > current_width
            if current_width > 1.0:
                width_ratios.append(pooled_width / current_width)

    ranking_changes = 0
    ranking_total = 0
    keys = sorted({(row["metric"], row["resolution"]) for row in selected})
    for metric, resolution in keys:
        setting = [
            row
            for row in selected
            if row["metric"] == metric and row["resolution"] == resolution
        ]
        current_best = min(setting, key=lambda row: row["current_median"])["method"]
        pooled_best = min(setting, key=lambda row: row["pooled_median"])["method"]
        ranking_total += 1
        ranking_changes += current_best != pooled_best

    return {
        "max_median_factor": max(median_factors),
        "pooled_wider_fraction": wider / comparable if comparable else 0.0,
        "median_band_width_ratio": float(np.median(width_ratios)) if width_ratios else 1.0,
        "ranking_changes": ranking_changes,
        "ranking_total": ranking_total,
    }


def write_report(rows, path: Path):
    lines = [
        "# Resolution-Plot Aggregation Audit",
        "",
        "The current statistic first takes the median of 25 cases at each of five "
        "perturbation magnitudes, then reports the median and IQR of those five "
        "medians. The pooled statistic reports the median and IQR of all 125 "
        "case/perturbation observations at each resolution.",
        "",
        "| Benchmark | Max center-line change | Pooled band wider | Median band-width ratio | Best-method changes |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp in METHODS:
        diag = benchmark_diagnostics(rows, exp)
        lines.append(
            f"| {exp.title()} | {diag['max_median_factor']:.3g}x | "
            f"{100 * diag['pooled_wider_fraction']:.1f}% | "
            f"{diag['median_band_width_ratio']:.3g}x | "
            f"{diag['ranking_changes']}/{diag['ranking_total']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_pdf(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for exp in METHODS:
            fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex="col")
            for col, metric in enumerate(METRICS):
                plot_series(axes[0, col], rows, exp, metric, "current")
                plot_series(axes[1, col], rows, exp, metric, "pooled")
                ymin = min(axes[0, col].get_ylim()[0], axes[1, col].get_ylim()[0])
                ymax = max(axes[0, col].get_ylim()[1], axes[1, col].get_ylim()[1])
                axes[0, col].set_ylim(ymin, ymax)
                axes[1, col].set_ylim(ymin, ymax)
            axes[0, 0].set_title("Current: distribution across five fixed-$w$ medians")
            axes[0, 1].set_title("Current: distribution across five fixed-$w$ medians")
            axes[1, 0].set_title("Pooled: distribution across all 125 observations")
            axes[1, 1].set_title("Pooled: distribution across all 125 observations")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(labels),
                frameon=True,
                bbox_to_anchor=(0.5, 0.015),
            )
            diag = benchmark_diagnostics(rows, exp)
            fig.suptitle(
                f"{exp.title()} benchmark: aggregation comparison",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.055,
                "Pooled IQR is wider at "
                f"{100 * diag['pooled_wider_fraction']:.0f}% of non-floor points; "
                f"largest substantive median change is {diag['max_median_factor']:.2g}x; "
                f"best method changes at {diag['ranking_changes']}/{diag['ranking_total']} settings.",
                ha="center",
                fontsize=8.5,
            )
            fig.tight_layout(rect=(0, 0.09, 1, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    rows = summarize(load_values(args.case_metrics))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "aggregation_quantiles.csv")
    write_report(rows, args.out_dir / "README.md")
    write_pdf(rows, args.out_dir / "aggregation_comparison_all_benchmarks.pdf")
    print(args.out_dir / "aggregation_comparison_all_benchmarks.pdf")


if __name__ == "__main__":
    main()
