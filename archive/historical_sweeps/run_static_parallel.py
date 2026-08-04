#!/usr/bin/env python3
"""
Run each static experiment for both non-perturbed and perturbed sweeps in parallel.

This script launches one subprocess per (experiment, grid_type) pair:
- grid_type=linear  -> experiments.static.run_linear_sweeps --only <exp>
- grid_type=perturbed -> experiments.static.run_perturbed_sweeps --only <exp>

Example:
  python -m experiments.static.run_static_parallel --experiments lines,ellipses --max_procs 4
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


EXPERIMENTS = ["circles", "ellipses", "lines", "squares", "zalesak"]


@dataclass
class Job:
    name: str
    cmd: list
    log_path: Path


def _parse_str_list(value):
    if value is None:
        return []
    return [p.strip().lower() for p in value.split(",") if p.strip()]


def _build_jobs(args, log_dir):
    jobs = []
    for exp in EXPERIMENTS:
        if args.experiments and exp not in args.experiments:
            continue

        linear_cmd = [
            sys.executable,
            "-m",
            "experiments.static.run_linear_sweeps",
            "--only",
            exp,
        ]
        if args.linear_subprocess:
            linear_cmd.append("--subprocess")
        if args.notify:
            linear_cmd.append("--notify")

        jobs.append(
            Job(
                name=f"{exp}-linear",
                cmd=linear_cmd,
                log_path=log_dir / f"{exp}_linear.log",
            )
        )

        perturbed_cmd = [
            sys.executable,
            "-m",
            "experiments.static.run_perturbed_sweeps",
            "--only",
            exp,
        ]
        if args.notify:
            perturbed_cmd.append("--notify")
        if args.resolutions:
            perturbed_cmd += ["--resolutions", args.resolutions]
        if args.wiggles:
            perturbed_cmd += ["--wiggles", args.wiggles]
        if args.seeds:
            perturbed_cmd += ["--seeds", args.seeds]
        if args.aggregate_samples is not None:
            perturbed_cmd += ["--aggregate_samples", str(args.aggregate_samples)]

        jobs.append(
            Job(
                name=f"{exp}-perturbed",
                cmd=perturbed_cmd,
                log_path=log_dir / f"{exp}_perturbed.log",
            )
        )

    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Run static linear + perturbed sweeps in parallel (one subprocess per experiment)."
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="comma-separated experiment names (default: all)",
    )
    parser.add_argument(
        "--max_procs",
        type=int,
        default=4,
        help="maximum concurrent subprocesses",
    )
    parser.add_argument(
        "--linear_subprocess",
        action="store_true",
        help="use --subprocess for linear sweeps",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="pass --notify to sweep scripts",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="override perturbed sweep resolutions (comma-separated)",
    )
    parser.add_argument(
        "--wiggles",
        type=str,
        default=None,
        help="override perturbed sweep wiggles (comma-separated)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="override perturbed sweep seeds (comma-separated)",
    )
    parser.add_argument(
        "--aggregate_samples",
        type=int,
        default=None,
        help="override perturbed aggregate sample count",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="log directory (default: logs/static_parallel/<timestamp>)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print commands without running",
    )
    args = parser.parse_args()

    selected = _parse_str_list(args.experiments)
    args.experiments = selected

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or os.path.join("logs", "static_parallel", stamp))
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args, log_dir)
    if args.dry_run:
        for job in jobs:
            print(job.name, ":", " ".join(job.cmd))
        return

    running = []
    completed = []

    def _start_job(job):
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(job.log_path, "w")
        proc = subprocess.Popen(job.cmd, stdout=log_file, stderr=subprocess.STDOUT)
        return {"job": job, "proc": proc, "log": log_file}

    queue = list(jobs)
    while queue or running:
        while queue and len(running) < args.max_procs:
            item = queue.pop(0)
            running.append(_start_job(item))
            print(f"[start] {item.name}")

        time.sleep(1.0)
        for item in list(running):
            ret = item["proc"].poll()
            if ret is None:
                continue
            item["log"].close()
            running.remove(item)
            completed.append({"job": item["job"], "code": ret})
            print(f"[done] {item['job'].name} (code {ret})")

    failures = [c for c in completed if c["code"] != 0]
    print("\n=== Static parallel summary ===")
    print(f"Total jobs: {len(completed)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"- {failure['job'].name} (code {failure['code']}) -> {failure['job'].log_path}")


if __name__ == "__main__":
    main()
