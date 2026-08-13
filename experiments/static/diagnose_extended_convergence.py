#!/usr/bin/env python3
"""Diagnose circular-fit line selections in an extended convergence run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.submission.conservation_analyzer import load_run_grid
from main.geoms.geoms import getArea, getDistance
from main.geoms.linear_facet import getPolyLineArea


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO_ROOT / "results/static/extended_convergence_smoke_20260812_182921"
LINEARITY_THRESHOLD = 1.0e-6
RUN_PATTERN = re.compile(r"_n(?P<n>\d+)_w(?P<w>\d+p\d+)_s\d+$")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_coordinates(run_root: Path) -> tuple[int, float]:
    match = RUN_PATTERN.search(run_root.name)
    if match is None:
        raise ValueError(f"Cannot parse resolution and wiggle from {run_root.name}")
    return int(match.group("n")), float(match.group("w").replace("p", "."))


def _facet_endpoints(row: Mapping[str, str]) -> tuple[list[float], list[float]]:
    geometry = json.loads(row["facet_geometry_json"])
    return geometry["p_left"], geometry["p_right"]


def _minimum_endpoint_gap(
    first: Mapping[str, str], second: Mapping[str, str]
) -> float:
    first_points = _facet_endpoints(first)
    second_points = _facet_endpoints(second)
    return min(getDistance(p, q) for p in first_points for q in second_points)


def diagnose_linear_facets(run_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    raw_root = run_dir / "raw_runs"
    for run_root in sorted(raw_root.glob("*ellipses_circular*")):
        n, wiggle = _run_coordinates(run_root)
        grid = load_run_grid(run_root, repo_root=REPO_ROOT)
        rows = _read_csv(run_root / "metrics/cell_metrics.csv")
        by_case: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_case[int(row["case_index"])].append(row)

        for case_index, case_rows in sorted(by_case.items()):
            lookup = {
                (int(row["cell_x"]), int(row["cell_y"])): row
                for row in case_rows
            }
            for row in case_rows:
                if row["final_facet_class"] != "linear":
                    continue
                x, y = int(row["cell_x"]), int(row["cell_y"])
                polygon = grid.cell_polygon(x, y)
                cell_area = abs(getArea(polygon))
                fraction = float(row["cell_fraction"])
                p_left, p_right = _facet_endpoints(row)
                line_area = getPolyLineArea(polygon, p_left, p_right)
                correct_fraction = line_area / cell_area
                implemented_fraction = line_area / (fraction * cell_area)
                correct_error = abs(fraction - correct_fraction)
                implemented_error = abs(fraction - implemented_fraction)

                adjacent_gaps = []
                for coords in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    neighbor = lookup.get(coords)
                    if neighbor is not None:
                        adjacent_gaps.append(_minimum_endpoint_gap(row, neighbor))
                adjacent_gaps.sort()
                interface_gap_sum = sum(adjacent_gaps[:2])

                output.append(
                    {
                        "run": run_root.name,
                        "cells_per_side": n,
                        "wiggle": wiggle,
                        "case_index": case_index,
                        "cell_x": x,
                        "cell_y": y,
                        "cell_fraction": fraction,
                        "facet_name": row["final_facet_name"],
                        "line_fraction_over_cell_area": correct_fraction,
                        "line_fraction_over_fluid_area": implemented_fraction,
                        "correct_linearity_error": correct_error,
                        "implemented_linearity_error": implemented_error,
                        "false_precheck_candidate": int(
                            implemented_error < LINEARITY_THRESHOLD
                            and correct_error >= LINEARITY_THRESHOLD
                        ),
                        "absolute_area_residual": abs(
                            line_area - fraction * cell_area
                        ),
                        "two_smallest_cardinal_endpoint_gaps": interface_gap_sum,
                        "mixed_cell_count": len(case_rows),
                        "estimated_mean_gap_contribution": interface_gap_sum
                        / max(len(case_rows), 1),
                    }
                )
    return output


def convergence_windows(run_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv(run_dir / "case_metrics.csv")
    groups: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == "ellipses" and row["method"] == "circular":
            groups[(float(row["wiggle"]), int(row["case_index"]))].append(row)

    output = []
    for (wiggle, case_index), case_rows in sorted(groups.items()):
        case_rows.sort(key=lambda row: int(row["cells_per_side"]))
        n = np.asarray([float(row["cells_per_side"]) for row in case_rows])
        gap = np.asarray([float(row["facet_gap"]) for row in case_rows])
        hausdorff = np.asarray([float(row["hausdorff"]) for row in case_rows])

        def order(values: np.ndarray) -> float:
            return float(-np.polyfit(np.log(n), np.log(values), 1)[0])

        record: dict[str, Any] = {
            "wiggle": wiggle,
            "case_index": case_index,
            "hausdorff_three_point_order": order(hausdorff),
            "facet_gap_three_point_order": order(gap),
        }
        for index, row in enumerate(case_rows):
            record[f"n_{int(n[index])}_hausdorff"] = hausdorff[index]
            record[f"n_{int(n[index])}_facet_gap"] = gap[index]
            if index:
                left, right = int(n[index - 1]), int(n[index])
                record[f"hausdorff_order_{left}_{right}"] = -math.log(
                    hausdorff[index] / hausdorff[index - 1]
                ) / math.log(n[index] / n[index - 1])
                record[f"facet_gap_order_{left}_{right}"] = -math.log(
                    gap[index] / gap[index - 1]
                ) / math.log(n[index] / n[index - 1])
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / f"results/static/extended_convergence_diagnosis_{timestamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    line_rows = diagnose_linear_facets(run_dir)
    window_rows = convergence_windows(run_dir)
    _write_csv(output_dir / "linear_facet_diagnostics.csv", line_rows)
    _write_csv(output_dir / "convergence_windows.csv", window_rows)
    summary = {
        "source_run": str(run_dir),
        "linearity_threshold": LINEARITY_THRESHOLD,
        "linear_facet_count": len(line_rows),
        "false_precheck_candidate_count": sum(
            int(row["false_precheck_candidate"]) for row in line_rows
        ),
        "maximum_absolute_area_residual": max(
            (float(row["absolute_area_residual"]) for row in line_rows),
            default=0.0,
        ),
        "full_sweep_status": "blocked_pending_shared_algorithm_fix_and_smoke_rerun",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
