#!/usr/bin/env python3
"""
Run low-resolution Zalesak experiments with getArcFacet logging enabled.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from util.logging.analyze_log import DEFAULT_OUTPUT
from util.logging.get_arc_facet_logger import disable_logging, enable_logging, get_stats


DEFAULT_CONFIG = "static/zalesak"
DEFAULT_ALGOS = ["safe_circle", "circular"]
DEFAULT_RESOLUTIONS = [0.50, 0.64]
DEFAULT_WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]
DEFAULT_SEEDS = [0]
DEFAULT_NUM_CASES = 25


def _parse_list(raw_value, cast):
    if raw_value is None:
        return None
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    return values or None


def _make_save_name(exp_name, algo, resolution, wiggle, seed):
    res_tag = str(resolution).replace(".", "p")
    wiggle_tag = str(wiggle).replace(".", "p")
    algo_tag = algo.lower().replace("+", "plus")
    return f"perturb_sweep_{exp_name}_{algo_tag}_r{res_tag}_w{wiggle_tag}_s{seed}"


def _default_artifact_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results/static/debug/getarcfacet") / timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Run Zalesak harvest experiments with getArcFacet logging"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--algos", type=str, help="comma-separated algorithms")
    parser.add_argument("--resolutions", type=str, help="comma-separated resolutions")
    parser.add_argument("--wiggles", type=str, help="comma-separated wiggles")
    parser.add_argument("--seeds", type=str, help="comma-separated seeds")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_NUM_CASES)
    parser.add_argument(
        "--artifact-dir",
        type=str,
        help="artifact directory (default: results/static/debug/getarcfacet/<timestamp>)",
    )

    args = parser.parse_args()

    algos = _parse_list(args.algos, str) or DEFAULT_ALGOS
    resolutions = _parse_list(args.resolutions, float) or DEFAULT_RESOLUTIONS
    wiggles = _parse_list(args.wiggles, float) or DEFAULT_WIGGLES
    seeds = _parse_list(args.seeds, int) or DEFAULT_SEEDS

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else _default_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "get_arc_facet_calls.jsonl"
    manifest_path = artifact_dir / "harvest_manifest.json"

    manifest = {
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_path),
        "config": args.config,
        "algos": algos,
        "resolutions": resolutions,
        "wiggles": wiggles,
        "seeds": seeds,
        "num_cases": args.num_cases,
        "extraction_defaults": {
            "command": (
                "python -m util.logging.analyze_log "
                f"{log_path} --extract-problematic --experiment zalesak "
                "--algo safe_circle,circular --resolution 0.5,0.64 "
                "--source safe_circle,circular "
                f"--output {DEFAULT_OUTPUT}"
            ),
            "output_module": DEFAULT_OUTPUT,
            "threshold": 1.0,
            "experiment": "zalesak",
            "algos": ["safe_circle", "circular"],
            "resolutions": [0.50, 0.64],
            "sources": ["safe_circle", "circular"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 80)
    print("ZALESAK GETARCFACET HARVEST RUN")
    print("=" * 80)
    print(f"Artifact dir: {artifact_dir}")
    print(f"Log file: {log_path}")
    print(f"Algorithms: {algos}")
    print(f"Resolutions: {resolutions}")
    print(f"Wiggles: {wiggles}")
    print(f"Seeds: {seeds}")
    print(f"Num cases: {args.num_cases}")
    print("=" * 80)

    enable_logging(log_file=str(log_path))

    try:
        from experiments.static import zalesak

        for resolution in resolutions:
            for wiggle in wiggles:
                for seed in seeds:
                    for algo in algos:
                        save_name = _make_save_name("zalesak", algo, resolution, wiggle, seed)
                        print(
                            f"Running {algo} r={resolution} w={wiggle} s={seed} "
                            f"-> {save_name}"
                        )
                        zalesak.main(
                            config_setting=args.config,
                            resolution=resolution,
                            facet_algo=algo,
                            save_name=save_name,
                            num_cases=args.num_cases,
                            mesh_type="perturbed_quads",
                            perturb_wiggle=wiggle,
                            perturb_seed=seed,
                            perturb_fix_boundary=True,
                        )
    finally:
        disable_logging()
        stats = get_stats()
        print("=" * 80)
        print("LOGGING STATS")
        print("=" * 80)
        for key, value in stats.items():
            if key != "log_file":
                print(f"{key}: {value}")
        print(f"log_file: {stats.get('log_file', str(log_path))}")
        print()
        print("Suggested extraction command:")
        print(manifest["extraction_defaults"]["command"])


if __name__ == "__main__":
    main()
