"""Diagnose the frozen QUASI port's sweep convergence without tuning it."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.baselines.project_benchmarks import (
    DEFAULT_BENCHMARKS,
    DEFAULT_CASE_INDICES,
    canonical_benchmark_cases,
)
from experiments.baselines.project_smoke import write_csv, write_json
from main.algos.baselines.quasi import QuasiPolicy, reconstruct_quasi


C1_DIAGNOSTIC_TOLERANCE = 1.0e-8
SWEEP_FIELDS = (
    "benchmark",
    "case_index",
    "cells_per_side",
    "sweep",
    "updates",
    "misses",
    "multiple_root_updates",
    "root_branch_switches",
    "max_update_displacement",
    "mean_update_displacement",
    "max_net_endpoint_displacement",
    "max_two_sweep_endpoint_displacement",
    "mean_c1_mismatch_before",
    "max_c1_mismatch_before",
    "mean_c1_mismatch_after",
    "max_c1_mismatch_after",
    "max_area_residual",
)
CASE_FIELDS = (
    "benchmark",
    "case_index",
    "cells_per_side",
    "mixed_cells",
    "joins",
    "runtime_seconds",
    "sweeps_completed",
    "converged",
    "c1_residual_below_1e_8",
    "diagnostic_converged",
    "initial_max_update_displacement",
    "final_max_update_displacement",
    "final_max_net_endpoint_displacement",
    "final_max_two_sweep_endpoint_displacement",
    "final_max_c1_mismatch",
    "tail_log10_slope_per_sweep",
    "tail_root_branch_switches",
    "tail_misses",
    "classification",
)


def _classify(trace: Sequence[Mapping[str, Any]], converged: bool) -> str:
    final = trace[-1]
    tail = trace[-min(20, len(trace)) :]
    tail_misses = sum(int(row["misses"]) for row in tail)
    if (
        float(final["max_c1_mismatch_after"]) > C1_DIAGNOSTIC_TOLERANCE
        and tail_misses > 0
    ):
        return "persistent inadmissible C1 joins"
    if converged:
        return "converged"
    if sum(int(row["root_branch_switches"]) for row in tail) > 0:
        return "persistent root-branch switching"
    net = float(final["max_net_endpoint_displacement"])
    two_sweep = float(final["max_two_sweep_endpoint_displacement"])
    if net > 0.0 and two_sweep < 0.1 * net:
        return "two-sweep oscillation"
    values = np.asarray(
        [float(row["max_update_displacement"]) for row in tail], dtype=float
    )
    positive = values[values > 0.0]
    if len(positive) >= 3:
        ratios = positive[1:] / positive[:-1]
        if float(np.median(ratios)) >= 0.98:
            return "plateau"
    return "slow decay"


def _tail_slope(trace: Sequence[Mapping[str, Any]]) -> float:
    tail = trace[-min(20, len(trace)) :]
    x = np.asarray([float(row["sweep"]) for row in tail])
    y = np.asarray([float(row["max_update_displacement"]) for row in tail])
    valid = y > 0.0
    if np.count_nonzero(valid) < 3:
        return math.nan
    return float(np.polyfit(x[valid], np.log10(y[valid]), 1)[0])


def _run_case(
    benchmark: str,
    case_index: int,
    cells_per_side: int,
    max_sweeps: int,
    output_text: str,
) -> Dict[str, Any]:
    output = Path(output_text)
    cache = (
        output / "cases" / f"{benchmark}_N{cells_per_side}_case{case_index:02d}.json"
    )
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    case = canonical_benchmark_cases(benchmark, (case_index,))[0]
    mesh = case.build_mesh(cells_per_side)
    case.initialize_fractions(mesh)
    started = time.perf_counter()
    result = reconstruct_quasi(
        mesh,
        policy=QuasiPolicy(max_sweeps=max_sweeps),
        trace_sweeps=True,
    )
    runtime = time.perf_counter() - started
    trace = [diagnostic.to_dict() for diagnostic in result.sweep_diagnostics]
    if trace:
        final = trace[-1]
        initial = trace[0]
    else:
        final = initial = {
            "max_update_displacement": 0.0,
            "max_net_endpoint_displacement": 0.0,
            "max_two_sweep_endpoint_displacement": 0.0,
            "max_c1_mismatch_after": 0.0,
        }
    tail = trace[-min(20, len(trace)) :]
    payload = {
        "case": {
            "benchmark": benchmark,
            "case_index": case_index,
            "cells_per_side": cells_per_side,
            "mixed_cells": len(result.facets),
            "joins": len(result.joins),
            "runtime_seconds": runtime,
            "sweeps_completed": result.sweeps_completed,
            "converged": result.converged,
            "c1_residual_below_1e_8": (
                float(final["max_c1_mismatch_after"]) <= C1_DIAGNOSTIC_TOLERANCE
            ),
            "diagnostic_converged": (
                result.converged
                and float(final["max_c1_mismatch_after"]) <= C1_DIAGNOSTIC_TOLERANCE
            ),
            "initial_max_update_displacement": initial["max_update_displacement"],
            "final_max_update_displacement": final["max_update_displacement"],
            "final_max_net_endpoint_displacement": final[
                "max_net_endpoint_displacement"
            ],
            "final_max_two_sweep_endpoint_displacement": final[
                "max_two_sweep_endpoint_displacement"
            ],
            "final_max_c1_mismatch": final["max_c1_mismatch_after"],
            "tail_log10_slope_per_sweep": _tail_slope(trace),
            "tail_root_branch_switches": sum(
                int(row["root_branch_switches"]) for row in tail
            ),
            "tail_misses": sum(int(row["misses"]) for row in tail),
            "classification": _classify(trace, result.converged),
        },
        "sweeps": trace,
        "unresolved": result.unresolved,
        "policy": result.policy,
    }
    write_json(cache, payload)
    return payload


def _refresh_case_diagnosis(payload: Dict[str, Any]) -> None:
    trace = payload["sweeps"]
    final = trace[-1]
    case = payload["case"]
    c1_satisfied = float(final["max_c1_mismatch_after"]) <= C1_DIAGNOSTIC_TOLERANCE
    case["c1_residual_below_1e_8"] = c1_satisfied
    case["diagnostic_converged"] = bool(case["converged"]) and c1_satisfied
    case["classification"] = _classify(trace, bool(case["converged"]))


def _aggregate_sweeps(
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["benchmark"]), int(row["sweep"])), []).append(row)
    summary = []
    for (benchmark, sweep), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "benchmark": benchmark,
            "sweep": sweep,
            "cases": len(group),
        }
        for field in (
            "max_update_displacement",
            "max_net_endpoint_displacement",
            "max_two_sweep_endpoint_displacement",
            "max_c1_mismatch_after",
            "max_area_residual",
        ):
            values = np.asarray([float(row[field]) for row in group])
            item[field + "_median"] = float(np.median(values))
            item[field + "_q25"] = float(np.quantile(values, 0.25))
            item[field + "_q75"] = float(np.quantile(values, 0.75))
        for field in ("misses", "root_branch_switches", "multiple_root_updates"):
            values = np.asarray([float(row[field]) for row in group])
            item[field + "_median"] = float(np.median(values))
            item[field + "_total"] = int(np.sum(values))
        summary.append(item)
    return summary


def _positive(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[result <= 0.0] = np.nan
    return result


def _plot(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(5, 3, figsize=(10.0, 11.0), sharex=True)
    for index, benchmark in enumerate(DEFAULT_BENCHMARKS):
        rows = sorted(
            (row for row in summary if row["benchmark"] == benchmark),
            key=lambda row: row["sweep"],
        )
        x = np.asarray([row["sweep"] for row in rows])
        for axis, field, color in (
            (axes[index, 0], "max_update_displacement", "#355c7d"),
            (axes[index, 1], "max_c1_mismatch_after", "#b24a3b"),
        ):
            median = _positive([row[field + "_median"] for row in rows])
            lower = _positive([row[field + "_q25"] for row in rows])
            upper = _positive([row[field + "_q75"] for row in rows])
            axis.plot(x, median, color=color)
            axis.fill_between(x, lower, upper, color=color, alpha=0.18)
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
        axes[index, 2].plot(
            x,
            [row["misses_total"] for row in rows],
            label="root misses",
            color="#d08c60",
        )
        axes[index, 2].plot(
            x,
            [row["root_branch_switches_total"] for row in rows],
            label="root-branch switches",
            color="#5b8c5a",
        )
        axes[index, 2].grid(True, alpha=0.25)
        axes[index, 0].set_ylabel(benchmark.capitalize())
        axes[index, 0].axhline(1.0e-11, color="#777777", linestyle=":", linewidth=0.8)
        axes[index, 1].axhline(
            C1_DIAGNOSTIC_TOLERANCE,
            color="#777777",
            linestyle=":",
            linewidth=0.8,
        )
    axes[0, 0].set_title("Maximum update displacement")
    axes[0, 1].set_title("Residual C1 mismatch")
    axes[0, 2].set_title("Discrete root events")
    axes[0, 2].legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Gauss-Seidel sweep")
    fig.suptitle("QUASI extended-sweep diagnostic (median and IQR over five cases)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tasks = [
        (benchmark, case_index)
        for benchmark in DEFAULT_BENCHMARKS
        for case_index in DEFAULT_CASE_INDICES
    ]
    payloads = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_case,
                benchmark,
                case_index,
                args.resolution,
                args.max_sweeps,
                str(args.output),
            ): (benchmark, case_index)
            for benchmark, case_index in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            payloads.append(future.result())
            print(f"completed {completed}/{len(futures)}", flush=True)

    for payload in payloads:
        _refresh_case_diagnosis(payload)
        case = payload["case"]
        write_json(
            args.output
            / "cases"
            / (
                f"{case['benchmark']}_N{case['cells_per_side']}_"
                f"case{case['case_index']:02d}.json"
            ),
            payload,
        )
    case_rows = sorted(
        (payload["case"] for payload in payloads),
        key=lambda row: (row["benchmark"], row["case_index"]),
    )
    sweep_rows = sorted(
        (
            {
                "benchmark": payload["case"]["benchmark"],
                "case_index": payload["case"]["case_index"],
                "cells_per_side": payload["case"]["cells_per_side"],
                **row,
            }
            for payload in payloads
            for row in payload["sweeps"]
        ),
        key=lambda row: (row["benchmark"], row["case_index"], row["sweep"]),
    )
    write_csv(args.output / "case_summary.csv", case_rows, CASE_FIELDS)
    write_csv(args.output / "sweep_results.csv", sweep_rows, SWEEP_FIELDS)
    summary = _aggregate_sweeps(sweep_rows)
    write_csv(args.output / "sweep_summary.csv", summary, tuple(summary[0]))
    write_json(
        args.output / "manifest.json",
        {
            "resolution": args.resolution,
            "case_indices": list(DEFAULT_CASE_INDICES),
            "benchmarks": list(DEFAULT_BENCHMARKS),
            "max_sweeps": args.max_sweeps,
            "purpose": (
                "diagnostic extension of the frozen QUASI iteration; no porting "
                "policy or published update equation was tuned"
            ),
        },
    )
    _plot(summary, args.output / "quasi_sweep_diagnostic.pdf")


if __name__ == "__main__":
    main()
