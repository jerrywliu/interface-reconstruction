#!/usr/bin/env python3
"""Patch verified fallback rows into a derived extended-convergence summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from experiments.static.run_extended_convergence_smoke import _summaries


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "results/static/extended_convergence_smoke_full_cell_area_fix_20260813"
)
DEFAULT_REPLACEMENTS = (
    REPO_ROOT / "plots/extended_convergence_verified_ellipse_case13_e108128",
    REPO_ROOT / "plots/extended_convergence_verified_circle_case10_e108128",
    REPO_ROOT / "plots/extended_convergence_verified_circle_case0_e108128",
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/static/extended_convergence_corrected_e108128_20260813"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setting_key(run_root: Path, metric_row: dict[str, str]) -> tuple:
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    parameters = manifest["parameters"]
    return (
        manifest["experiment"],
        parameters["facet_algo"],
        int(round(float(parameters["resolution"]) * 100)),
        float(parameters["perturb_wiggle"]),
        int(metric_row["case_index"]),
    )


def convergence_order(values: dict[int, float]) -> float:
    xs = [math.log(1.0 / n) for n in sorted(values)]
    ys = [math.log(values[n]) for n in sorted(values)]
    x_bar = mean(xs)
    y_bar = mean(ys)
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sum(
        (x - x_bar) ** 2 for x in xs
    )


def ellipse_convergence(rows: list[dict[str, str]]) -> list[dict]:
    ellipse_rows = [
        row
        for row in rows
        if row["experiment"] == "ellipses" and row["method"] == "circular"
    ]
    output = []
    for wiggle in (0.0, 0.2):
        for metric in ("hausdorff", "facet_gap"):
            for aggregate, function in (("median", median), ("mean", mean)):
                values = {}
                for n in (256, 300, 512):
                    samples = [
                        float(row[metric])
                        for row in ellipse_rows
                        if int(row["cells_per_side"]) == n
                        and math.isclose(float(row["wiggle"]), wiggle)
                    ]
                    values[n] = function(samples)
                output.append(
                    {
                        "wiggle": wiggle,
                        "metric": metric,
                        "aggregate": aggregate,
                        "order": convergence_order(values),
                        "values": ";".join(f"{n}:{values[n]:.12g}" for n in sorted(values)),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replacement", type=Path, action="append", dest="replacements"
    )
    args = parser.parse_args()
    replacements = args.replacements or list(DEFAULT_REPLACEMENTS)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    rows = read_csv(args.source.resolve() / "case_metrics.csv")
    by_key = {
        (
            row["experiment"],
            row["method"],
            int(row["cells_per_side"]),
            float(row["wiggle"]),
            int(row["case_index"]),
        ): row
        for row in rows
    }
    corrections = []
    for replacement in replacements:
        replacement = replacement.resolve()
        metric_rows = read_csv(replacement / "metrics/case_metrics.csv")
        if len(metric_rows) != 1:
            raise RuntimeError(f"Expected one replacement row in {replacement}")
        replacement_row = metric_rows[0]
        key = setting_key(replacement, replacement_row)
        source_row = by_key[key]
        correction = {
            "experiment": key[0],
            "method": key[1],
            "cells_per_side": key[2],
            "wiggle": key[3],
            "case_index": key[4],
            "old_hausdorff": source_row["hausdorff"],
            "new_hausdorff": replacement_row["hausdorff"],
            "old_facet_gap": source_row["facet_gap"],
            "new_facet_gap": replacement_row["facet_gap"],
            "replacement_run": str(replacement.relative_to(REPO_ROOT)),
        }
        corrections.append(correction)
        source_row["hausdorff"] = replacement_row["hausdorff"]
        source_row["facet_gap"] = replacement_row["facet_gap"]
        floor = float(source_row["expected_fit_floor"])
        source_row["hausdorff_to_expected_floor"] = (
            float(replacement_row["hausdorff"]) / floor
        )
        source_row["facet_gap_to_expected_floor"] = (
            float(replacement_row["facet_gap"]) / floor
        )

    write_csv(output / "case_metrics.csv", rows)
    write_csv(output / "summary_metrics.csv", _summaries(rows))
    write_csv(output / "all_case_convergence_summary.csv", ellipse_convergence(rows))
    write_csv(output / "corrections.csv", corrections)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "derived_metric_correction",
                "source": str(args.source.resolve().relative_to(REPO_ROOT)),
                "replacements": [
                    str(path.resolve().relative_to(REPO_ROOT)) for path in replacements
                ],
                "correction_count": len(corrections),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
