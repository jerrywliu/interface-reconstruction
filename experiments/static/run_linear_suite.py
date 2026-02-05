#!/usr/bin/env python3
"""
Run all static experiments with linear / linear-corner algorithms only.
Continues on errors and reports a summary at the end.
"""

import argparse
import subprocess
import sys
from datetime import datetime


EXPERIMENTS = [
    {
        "name": "circles",
        "module": "experiments.static.circles",
        "config": "static/circle",
        "num_arg": "--num_circles",
        "algorithms": ["Youngs", "LVIRA", "safe_linear", "linear"],
    },
    {
        "name": "ellipses",
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "num_arg": "--num_ellipses",
        "algorithms": ["Youngs", "LVIRA", "safe_linear", "linear"],
    },
    {
        "name": "lines",
        "module": "experiments.static.lines",
        "config": "static/line",
        "num_arg": "--num_lines",
        "algorithms": ["Youngs", "LVIRA", "safe_linear", "linear"],
    },
    {
        "name": "squares",
        "module": "experiments.static.squares",
        "config": "static/square",
        "num_arg": "--num_squares",
        "algorithms": [
            "Youngs",
            "LVIRA",
            "safe_linear",
            "linear",
            "safe_linear_corner",
            "linear+corner",
        ],
    },
    {
        "name": "zalesak",
        "module": "experiments.static.zalesak",
        "config": "static/zalesak",
        "num_arg": "--num_cases",
        "algorithms": [
            "Youngs",
            "LVIRA",
            "safe_linear",
            "linear",
            "safe_linear_corner",
            "linear+corner",
        ],
    },
]


def algo_tag(algo: str) -> str:
    return algo.lower().replace("+", "plus")


def build_command(exp, algo, args):
    cmd = [
        sys.executable,
        "-m",
        exp["module"],
        "--config",
        exp["config"],
        "--facet_algo",
        algo,
        "--save_name",
        f"linear_suite_{exp['name']}_{algo_tag(algo)}",
    ]

    if args.resolution is not None:
        cmd += ["--resolution", str(args.resolution)]

    num_value = getattr(args, exp["name"], None)
    if num_value is not None:
        cmd += [exp["num_arg"], str(num_value)]

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run all static experiments with linear-only methods."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="override resolution for all experiments",
        required=False,
    )
    parser.add_argument(
        "--circles",
        type=int,
        help="override num_circles for circles",
        required=False,
    )
    parser.add_argument(
        "--ellipses",
        type=int,
        help="override num_ellipses for ellipses",
        required=False,
    )
    parser.add_argument(
        "--lines",
        type=int,
        help="override num_lines for lines",
        required=False,
    )
    parser.add_argument(
        "--squares",
        type=int,
        help="override num_squares for squares",
        required=False,
    )
    parser.add_argument(
        "--zalesak",
        type=int,
        help="override num_cases for zalesak",
        required=False,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print commands without running",
        default=False,
    )

    args = parser.parse_args()

    failures = []
    total = 0

    print(f"Linear static suite started at {datetime.now().isoformat()}")
    for exp in EXPERIMENTS:
        for algo in exp["algorithms"]:
            total += 1
            cmd = build_command(exp, algo, args)
            cmd_str = " ".join(cmd)
            print(f"\n[{exp['name']}] {algo}: {cmd_str}")
            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append(
                    {
                        "experiment": exp["name"],
                        "algo": algo,
                        "returncode": result.returncode,
                    }
                )
                print(
                    f"[ERROR] {exp['name']} {algo} failed with code {result.returncode}"
                )

    print("\n=== Linear static suite summary ===")
    print(f"Total runs: {total}")
    print(f"Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(
                f"- {failure['experiment']} / {failure['algo']} (code {failure['returncode']})"
            )


if __name__ == "__main__":
    main()
