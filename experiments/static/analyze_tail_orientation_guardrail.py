#!/usr/bin/env python3
"""Compare a corner-method guardrail sweep to both paper sweep baselines."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASELINE = Path(
    "results/static/static_paper_simplified_default_20260717_212413"
)
DEFAULT_PRIOR = Path(
    "results/static/static_paper_affected_diagnostics_20260714_102206"
)
DEFAULT_CANDIDATE = Path(
    "results/static/tail_orientation_guardrail_20260729_1638"
)
FLOOR = 1e-6


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _key(row):
    return (
        row["experiment"],
        row["algo"],
        float(row["resolution"]),
        float(row["wiggle"]),
        int(row["seed"]),
        int(row["case_index"]),
    )


def _index(root):
    return {
        _key(row): row
        for row in _read_csv(Path(root) / "diagnostics" / "case_metrics.csv")
    }


def _material_outcome(baseline, candidate):
    threshold = max(1e-10, 0.01 * max(abs(baseline), abs(candidate)))
    if candidate < baseline - threshold:
        return "improved"
    if candidate > baseline + threshold:
        return "worsened"
    return "stable"


def _int(row, field):
    value = row.get(field, "")
    return 0 if value in (None, "") else int(float(value))


def _comparison_rows(baseline_root, prior_root, candidate_root):
    baseline = _index(baseline_root)
    prior = _index(prior_root)
    candidate = _index(candidate_root)
    missing_baseline = candidate.keys() - baseline.keys()
    missing_prior = candidate.keys() - prior.keys()
    if missing_baseline or missing_prior:
        raise ValueError(
            f"Unmatched candidate cases: baseline={list(missing_baseline)[:5]} "
            f"prior={list(missing_prior)[:5]}"
        )

    rows = []
    for key in sorted(candidate):
        source_rows = {
            "prior": prior[key],
            "baseline": baseline[key],
            "candidate": candidate[key],
        }
        values = {
            label: float(row["hausdorff"]) for label, row in source_rows.items()
        }
        rows.append(
            {
                "experiment": key[0],
                "algo": key[1],
                "resolution": key[2],
                "cells_per_side": int(round(100 * key[2])),
                "wiggle": key[3],
                "seed": key[4],
                "case_index": key[5],
                "prior_hausdorff": values["prior"],
                "baseline_hausdorff": values["baseline"],
                "candidate_hausdorff": values["candidate"],
                "candidate_minus_baseline": values["candidate"]
                - values["baseline"],
                "hausdorff_outcome": _material_outcome(
                    values["baseline"], values["candidate"]
                ),
                "prior_facet_gap": float(source_rows["prior"]["facet_gap"]),
                "baseline_facet_gap": float(source_rows["baseline"]["facet_gap"]),
                "candidate_facet_gap": float(source_rows["candidate"]["facet_gap"]),
                "prior_fallback_cells": _int(
                    source_rows["prior"], "num_plic_fallback_cells"
                ),
                "baseline_fallback_cells": _int(
                    source_rows["baseline"], "num_plic_fallback_cells"
                ),
                "candidate_fallback_cells": _int(
                    source_rows["candidate"], "num_plic_fallback_cells"
                ),
                "prior_joint_floor": int(
                    values["prior"] < FLOOR
                    and float(source_rows["prior"]["facet_gap"]) < FLOOR
                ),
                "baseline_joint_floor": int(
                    values["baseline"] < FLOOR
                    and float(source_rows["baseline"]["facet_gap"]) < FLOOR
                ),
                "candidate_joint_floor": int(
                    values["candidate"] < FLOOR
                    and float(source_rows["candidate"]["facet_gap"]) < FLOOR
                ),
            }
        )
    return rows


def _summary(rows, subset):
    arrays = {
        label: np.asarray([row[f"{label}_hausdorff"] for row in rows])
        for label in ("prior", "baseline", "candidate")
    }
    summary = {
        "subset": subset,
        "case_count": len(rows),
        "material_improved_cases": sum(
            row["hausdorff_outcome"] == "improved" for row in rows
        ),
        "material_worsened_cases": sum(
            row["hausdorff_outcome"] == "worsened" for row in rows
        ),
        "material_stable_cases": sum(
            row["hausdorff_outcome"] == "stable" for row in rows
        ),
    }
    for label, values in arrays.items():
        summary[f"{label}_hausdorff_median"] = float(np.median(values))
        summary[f"{label}_hausdorff_p95"] = float(np.quantile(values, 0.95))
        summary[f"{label}_hausdorff_max"] = float(np.max(values))
        summary[f"{label}_hausdorff_above_one"] = int(np.count_nonzero(values > 1))
        summary[f"{label}_hausdorff_floor_cases"] = int(
            np.count_nonzero(values < FLOOR)
        )
        summary[f"{label}_joint_floor_cases"] = sum(
            row[f"{label}_joint_floor"] for row in rows
        )
        summary[f"{label}_fallback_cells"] = sum(
            row[f"{label}_fallback_cells"] for row in rows
        )
    return summary


def _summaries(rows):
    summaries = [_summary(rows, "all")]
    for experiment in sorted({row["experiment"] for row in rows}):
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        summaries.append(_summary(experiment_rows, f"experiment:{experiment}"))
        for wiggle in sorted({row["wiggle"] for row in experiment_rows}):
            subset = [row for row in experiment_rows if row["wiggle"] == wiggle]
            summaries.append(_summary(subset, f"experiment:{experiment}:w={wiggle:g}"))
        for resolution in sorted({row["resolution"] for row in experiment_rows}):
            subset = [
                row for row in experiment_rows if row["resolution"] == resolution
            ]
            summaries.append(
                _summary(
                    subset,
                    f"experiment:{experiment}:N={int(round(100 * resolution))}",
                )
            )
    return summaries


def _setting_summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["experiment"],
                row["algo"],
                row["resolution"],
                row["wiggle"],
                row["seed"],
            )
        ].append(row)
    output = []
    for key, group in sorted(groups.items()):
        item = {
            "experiment": key[0],
            "algo": key[1],
            "resolution": key[2],
            "cells_per_side": int(round(100 * key[2])),
            "wiggle": key[3],
            "seed": key[4],
            "case_count": len(group),
        }
        for label in ("prior", "baseline", "candidate"):
            values = np.asarray([row[f"{label}_hausdorff"] for row in group])
            item[f"{label}_hausdorff_median"] = float(np.median(values))
            item[f"{label}_hausdorff_p95"] = float(np.quantile(values, 0.95))
            item[f"{label}_hausdorff_max"] = float(np.max(values))
            item[f"{label}_median_at_floor"] = int(np.median(values) < FLOOR)
        output.append(item)
    return output


def _plot(rows, setting_rows, output_dir):
    experiments = ("squares", "zalesak")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    palette = plt.get_cmap("viridis")
    for column, experiment in enumerate(experiments):
        ax = axes[0, column]
        experiment_settings = [
            row for row in setting_rows if row["experiment"] == experiment
        ]
        wiggles = sorted({row["wiggle"] for row in experiment_settings})
        for index, wiggle in enumerate(wiggles):
            group = sorted(
                (row for row in experiment_settings if row["wiggle"] == wiggle),
                key=lambda row: row["cells_per_side"],
            )
            ax.plot(
                [row["cells_per_side"] for row in group],
                [max(row["candidate_hausdorff_median"], 1e-12) for row in group],
                marker="o",
                linewidth=1.5,
                color=palette(index / max(len(wiggles) - 1, 1)),
                label=f"w={wiggle:g}",
            )
        ax.axhline(FLOOR, color="#777777", linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Cells per side, N")
        ax.set_ylabel("Candidate median Hausdorff")
        ax.set_title(experiment.capitalize())
        ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[1, column]
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        colors = [
            "#009E73" if row["hausdorff_outcome"] == "improved" else
            "#D55E00" if row["hausdorff_outcome"] == "worsened" else
            "#999999"
            for row in experiment_rows
        ]
        ax.scatter(
            [max(row["baseline_hausdorff"], 1e-12) for row in experiment_rows],
            [max(row["candidate_hausdorff"], 1e-12) for row in experiment_rows],
            s=14,
            color=colors,
            alpha=0.7,
            edgecolors="none",
        )
        bounds = [1e-12, 10]
        ax.plot(bounds, bounds, color="#555555", linewidth=1, linestyle="--")
        ax.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds)
        ax.set_xlabel("Simplified baseline Hausdorff")
        ax.set_ylabel("Greedy-retry Hausdorff")
        ax.set_title(f"{experiment.capitalize()} case comparison")

    fig.suptitle("Full corner-method guardrail: one greedy-orientation retry")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "orientation_retry_guardrail_all_methods.png"
    pdf = output_dir / "orientation_retry_guardrail_all_methods.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _readme(output_dir, summaries, setting_rows, plot_paths):
    lines = [
        "# Greedy-orientation retry guardrail",
        "",
        "Matched full paper grids for square linear+corner and Zalesak circular+corner.",
        "",
        "| Problem | Cases | Material improved / worsened | H>1 prior / baseline / candidate | Joint floor prior / baseline / candidate | Fallback cells baseline / candidate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in ("squares", "zalesak"):
        item = next(
            row for row in summaries if row["subset"] == f"experiment:{experiment}"
        )
        lines.append(
            f"| {experiment} | {item['case_count']} | "
            f"{item['material_improved_cases']} / {item['material_worsened_cases']} | "
            f"{item['prior_hausdorff_above_one']} / {item['baseline_hausdorff_above_one']} / {item['candidate_hausdorff_above_one']} | "
            f"{item['prior_joint_floor_cases']} / {item['baseline_joint_floor_cases']} / {item['candidate_joint_floor_cases']} | "
            f"{item['baseline_fallback_cells']} / {item['candidate_fallback_cells']} |"
        )
    lines += [
        "",
        "## Setting medians",
        "",
    ]
    for experiment in ("squares", "zalesak"):
        group = [row for row in setting_rows if row["experiment"] == experiment]
        lines.append(
            f"- {experiment}: prior/baseline/candidate medians at floor = "
            f"{sum(row['prior_median_at_floor'] for row in group)}/"
            f"{sum(row['baseline_median_at_floor'] for row in group)}/"
            f"{sum(row['candidate_median_at_floor'] for row in group)} of {len(group)} settings"
        )
    lines += [
        "",
        f"Plot: `{plot_paths[0].relative_to(output_dir)}`",
        "",
        "Artifacts: `case_comparison.csv`, `summary.csv`, and `setting_summary.csv`.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.candidate_root / "comparison"
    rows = _comparison_rows(args.baseline_root, args.prior_root, args.candidate_root)
    summaries = _summaries(rows)
    setting_rows = _setting_summaries(rows)
    _write_csv(output_dir / "case_comparison.csv", rows)
    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(output_dir / "setting_summary.csv", setting_rows)
    plot_paths = _plot(rows, setting_rows, output_dir / "plots")
    _readme(output_dir, summaries, setting_rows, plot_paths)
    print(f"Compared {len(rows)} cases: {output_dir}")


if __name__ == "__main__":
    main()
