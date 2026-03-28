#!/usr/bin/env python3
"""
Wait for sharded ELVIRA/LVIRA workers to finish, then refresh the Section 6 bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a sharded ELVIRA/LVIRA static refresh bundle."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out-bundle", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--generate-maintext", action="store_true")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    out_bundle = args.out_bundle.resolve()
    manifest = json.loads((bundle / "launch_manifest.json").read_text(encoding="utf-8"))
    pids = [worker["pid"] for worker in manifest["workers"].values()]

    while any(_pid_alive(pid) for pid in pids):
        time.sleep(args.poll_seconds)

    cmd = [
        sys.executable,
        "-m",
        "experiments.static.merge_section6_with_lvira",
        "--bundle",
        str(bundle),
        "--out-bundle",
        str(out_bundle),
    ]
    if args.generate_maintext:
        cmd.append("--generate-maintext")
    if args.notify:
        cmd.append("--notify")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
