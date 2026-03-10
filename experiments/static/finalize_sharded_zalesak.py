#!/usr/bin/env python3
"""
Wait for sharded Zalesak workers to finish, then merge CSVs and regenerate summaries.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _merge_csvs(csv_paths, out_csv):
    fieldnames = None
    rows = []
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)
    if fieldnames is None:
        raise RuntimeError("No shard CSVs available to merge")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Finalize a sharded Zalesak sweep bundle."
    )
    parser.add_argument("--bundle", required=True, help="camera-ready bundle directory")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((bundle / "launch_manifest.json").read_text(encoding="utf-8"))
    pids = [info["pid"] for info in manifest["workers"].values()]

    while any(_pid_alive(pid) for pid in pids):
        time.sleep(args.poll_seconds)

    csv_dir = bundle / "csv" / "by_resolution"
    csv_paths = sorted(csv_dir.glob("*.csv"))
    merged_csv = bundle / "csv" / "zalesak_merged.csv"
    _merge_csvs(csv_paths, merged_csv)

    summary_dir = bundle / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.static.run_perturbed_sweeps",
        "--plot_from_csv",
        str(merged_csv),
        "--summary_dir",
        str(summary_dir),
    ]
    if args.notify:
        cmd.append("--notify")
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
