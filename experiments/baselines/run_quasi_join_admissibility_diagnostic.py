"""Check whether persistent QUASI C1 misses have an admissible edge root."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.baselines.project_smoke import write_csv, write_json
from main.algos.baselines.quasi import (
    QuasiPolicy,
    _CellState,
    _candidate_c1_roots,
    _edge_parameter,
    _join_mismatch,
    reconstruct_quasi,
)


DEFAULT_CASES = (
    ("circles", 2),
    ("ellipses", 3),
    ("lines", 3),
    ("squares", 4),
    ("zalesak", 2),
)
JOIN_FIELDS = (
    "benchmark",
    "case_index",
    "cells_per_side",
    "cell_a",
    "cell_b",
    "current_mismatch",
    "admissible_root_count",
    "dense_minimum_mismatch",
    "dense_minimum_parameter",
)


def _run_case(
    benchmark: str,
    case_index: int,
    resolution: int,
    max_sweeps: int,
    dense_samples: int,
) -> Dict[str, Any]:
    case = canonical_benchmark_cases(benchmark, (case_index,))[0]
    mesh = case.build_mesh(resolution)
    case.initialize_fractions(mesh)
    result = reconstruct_quasi(
        mesh,
        policy=QuasiPolicy(max_sweeps=max_sweeps),
        trace_sweeps=True,
    )
    states = {
        index: _CellState(
            index,
            mesh.polys[index[0]][index[1]],
            [list(facet.pLeft), list(facet.pRight)],
            facet,
        )
        for index, facet in result.facets.items()
    }
    rows = []
    for join in result.joins:
        if join.kind != "edge":
            continue
        point = states[join.cells[0]].endpoints[join.slots[0]]
        alpha = min(1.0, max(0.0, _edge_parameter(point, join.edge)))
        mismatch = abs(_join_mismatch(alpha, join, states))
        roots = _candidate_c1_roots(join, states, 1.0e-12)
        dense_minimum = 0.0
        dense_parameter = alpha
        if not roots:
            candidates = [(mismatch, alpha)]
            for parameter in np.linspace(0.0, 1.0, dense_samples):
                value = _join_mismatch(float(parameter), join, states)
                if math.isfinite(value):
                    candidates.append((abs(value), float(parameter)))
            dense_minimum, dense_parameter = min(candidates)
        rows.append(
            {
                "benchmark": benchmark,
                "case_index": case_index,
                "cells_per_side": resolution,
                "cell_a": str(join.cells[0]),
                "cell_b": str(join.cells[1]),
                "current_mismatch": mismatch,
                "admissible_root_count": len(roots),
                "dense_minimum_mismatch": dense_minimum,
                "dense_minimum_parameter": dense_parameter,
            }
        )
    return {
        "benchmark": benchmark,
        "case_index": case_index,
        "cells_per_side": resolution,
        "sweeps_completed": result.sweeps_completed,
        "displacement_converged": result.converged,
        "final_max_update_displacement": (
            result.sweep_diagnostics[-1].max_update_displacement
        ),
        "final_max_c1_mismatch": result.sweep_diagnostics[-1].max_c1_mismatch_after,
        "joins": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--dense-samples", type=int, default=513)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    payloads = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _run_case,
                benchmark,
                case_index,
                args.resolution,
                args.max_sweeps,
                args.dense_samples,
            )
            for benchmark, case_index in DEFAULT_CASES
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            payloads.append(future.result())
            print(f"completed {completed}/{len(futures)}", flush=True)

    rows = [row for payload in payloads for row in payload["joins"]]
    write_csv(args.output / "join_results.csv", rows, JOIN_FIELDS)
    summary = []
    for payload in sorted(payloads, key=lambda item: item["benchmark"]):
        joins = payload.pop("joins")
        missing = [row for row in joins if row["admissible_root_count"] == 0]
        summary.append(
            {
                **payload,
                "edge_joins": len(joins),
                "joins_without_admissible_root": len(missing),
                "worst_current_mismatch_without_root": max(
                    (row["current_mismatch"] for row in missing), default=0.0
                ),
                "best_dense_mismatch_without_root": min(
                    (row["dense_minimum_mismatch"] for row in missing), default=0.0
                ),
                "worst_dense_mismatch_without_root": max(
                    (row["dense_minimum_mismatch"] for row in missing), default=0.0
                ),
            }
        )
    write_csv(args.output / "case_summary.csv", summary, tuple(summary[0]))
    write_json(
        args.output / "manifest.json",
        {
            "cases": [list(item) for item in DEFAULT_CASES],
            "resolution": args.resolution,
            "max_sweeps": args.max_sweeps,
            "dense_samples_per_missed_join": args.dense_samples,
            "interpretation": (
                "For every final edge join, enumerate verified algebraic C1 roots. "
                "For joins with none, scan the whole shared edge to distinguish a "
                "root-enumeration miss from an incompatible one-parameter update."
            ),
        },
    )


if __name__ == "__main__":
    main()
