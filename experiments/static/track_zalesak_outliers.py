#!/usr/bin/env python3
"""
Watch completed Zalesak subruns and record per-case circular+corner outliers.

This is intended to run alongside sharded `run_perturbed_sweeps` workers.
It only harvests completed subruns, so each outlier row corresponds to a stable
`save_name + case_index` pair that can be replayed later.
"""

import argparse
import csv
import json
import time
from pathlib import Path


DEFAULT_RESOLUTIONS = [0.50, 0.64, 1.00, 1.28, 1.50]
DEFAULT_WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]
DEFAULT_SEEDS = [0]
DEFAULT_ALGOS = ["circular+corner"]


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


def _read_numeric_values(path):
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def _expected_save_specs(resolutions, wiggles, seeds, algos):
    specs = []
    for resolution in resolutions:
        for wiggle in wiggles:
            for seed in seeds:
                for algo in algos:
                    save_name = _make_save_name("zalesak", algo, resolution, wiggle, seed)
                    specs.append(
                        {
                            "save_name": save_name,
                            "algo": algo,
                            "resolution": resolution,
                            "wiggle": wiggle,
                            "seed": seed,
                        }
                    )
    return specs


def _metrics_ready(metrics_dir, num_cases, run_start_epoch):
    required = [
        metrics_dir / "area_error.txt",
        metrics_dir / "facet_gap.txt",
        metrics_dir / "hausdorff.txt",
    ]
    for path in required:
        if not path.exists():
            return False
        if path.stat().st_mtime < run_start_epoch:
            return False
        if len(_read_numeric_values(path)) < num_cases:
            return False
    return True


def _load_tracker_state(state_path):
    if not state_path.exists():
        return {"processed_save_names": [], "recorded_outlier_keys": []}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _write_tracker_state(state_path, state):
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _write_status(
    status_path,
    expected_count,
    processed_save_names,
    outlier_rows,
    pending_save_names,
):
    payload = {
        "updated_at_epoch": time.time(),
        "expected_tracked_subruns": expected_count,
        "processed_tracked_subruns": len(processed_save_names),
        "remaining_tracked_subruns": len(pending_save_names),
        "recorded_outliers": len(outlier_rows),
        "pending_save_names": pending_save_names[:10],
    }
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Track per-case Zalesak circular+corner outliers from completed subruns."
    )
    parser.add_argument("--artifact-dir", required=True, help="output directory for tracker artifacts")
    parser.add_argument("--plots-root", default="plots", help="root plots directory")
    parser.add_argument("--resolutions", type=str, help="comma-separated resolutions")
    parser.add_argument("--wiggles", type=str, help="comma-separated wiggles")
    parser.add_argument("--seeds", type=str, help="comma-separated seeds")
    parser.add_argument("--algos", type=str, help="comma-separated algorithms")
    parser.add_argument("--num-cases", type=int, default=25, help="expected number of cases per subrun")
    parser.add_argument("--poll-seconds", type=float, default=20.0, help="poll interval")
    parser.add_argument("--run-start-epoch", type=float, required=True, help="epoch timestamp used to reject stale outputs")
    parser.add_argument("--hausdorff-threshold", type=float, default=1e-3)
    parser.add_argument("--facet-gap-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    resolutions = _parse_list(args.resolutions, float) or DEFAULT_RESOLUTIONS
    wiggles = _parse_list(args.wiggles, float) or DEFAULT_WIGGLES
    seeds = _parse_list(args.seeds, int) or DEFAULT_SEEDS
    algos = _parse_list(args.algos, str) or DEFAULT_ALGOS

    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outlier_csv = artifact_dir / "zalesak_circularpluscorner_outliers.csv"
    state_path = artifact_dir / "tracker_state.json"
    status_path = artifact_dir / "tracker_status.json"
    manifest_path = artifact_dir / "tracker_manifest.json"

    manifest = {
        "created_at_epoch": time.time(),
        "plots_root": str(Path(args.plots_root).resolve()),
        "resolutions": resolutions,
        "wiggles": wiggles,
        "seeds": seeds,
        "algorithms": algos,
        "num_cases": args.num_cases,
        "poll_seconds": args.poll_seconds,
        "run_start_epoch": args.run_start_epoch,
        "hausdorff_threshold": args.hausdorff_threshold,
        "facet_gap_threshold": args.facet_gap_threshold,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not outlier_csv.exists():
        with outlier_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "save_name",
                    "algo",
                    "resolution",
                    "wiggle",
                    "seed",
                    "case_index",
                    "area_error",
                    "facet_gap",
                    "hausdorff",
                    "plot_dir",
                ],
            )
            writer.writeheader()

    specs = _expected_save_specs(resolutions, wiggles, seeds, algos)
    expected_save_names = [spec["save_name"] for spec in specs]
    spec_by_name = {spec["save_name"]: spec for spec in specs}
    state = _load_tracker_state(state_path)
    processed_save_names = set(state.get("processed_save_names", []))
    recorded_outlier_keys = set(state.get("recorded_outlier_keys", []))
    recorded_outlier_rows = []

    plots_root = Path(args.plots_root).resolve()

    while True:
        made_progress = False

        for save_name in expected_save_names:
            if save_name in processed_save_names:
                continue

            metrics_dir = plots_root / save_name / "metrics"
            if not _metrics_ready(metrics_dir, args.num_cases, args.run_start_epoch):
                continue

            area = _read_numeric_values(metrics_dir / "area_error.txt")
            gaps = _read_numeric_values(metrics_dir / "facet_gap.txt")
            hausdorff = _read_numeric_values(metrics_dir / "hausdorff.txt")
            limit = min(len(area), len(gaps), len(hausdorff), args.num_cases)

            spec = spec_by_name[save_name]
            with outlier_csv.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "save_name",
                        "algo",
                        "resolution",
                        "wiggle",
                        "seed",
                        "case_index",
                        "area_error",
                        "facet_gap",
                        "hausdorff",
                        "plot_dir",
                    ],
                )
                for case_index in range(limit):
                    if (
                        hausdorff[case_index] <= args.hausdorff_threshold
                        and gaps[case_index] <= args.facet_gap_threshold
                    ):
                        continue
                    outlier_key = f"{save_name}:{case_index}"
                    if outlier_key in recorded_outlier_keys:
                        continue
                    row = {
                        "save_name": save_name,
                        "algo": spec["algo"],
                        "resolution": spec["resolution"],
                        "wiggle": spec["wiggle"],
                        "seed": spec["seed"],
                        "case_index": case_index,
                        "area_error": area[case_index],
                        "facet_gap": gaps[case_index],
                        "hausdorff": hausdorff[case_index],
                        "plot_dir": str((plots_root / save_name).resolve()),
                    }
                    writer.writerow(row)
                    recorded_outlier_rows.append(row)
                    recorded_outlier_keys.add(outlier_key)

            processed_save_names.add(save_name)
            made_progress = True

        pending_save_names = [
            save_name for save_name in expected_save_names if save_name not in processed_save_names
        ]
        state = {
            "processed_save_names": sorted(processed_save_names),
            "recorded_outlier_keys": sorted(recorded_outlier_keys),
        }
        _write_tracker_state(state_path, state)
        _write_status(
            status_path,
            len(expected_save_names),
            processed_save_names,
            recorded_outlier_rows,
            pending_save_names,
        )

        if not pending_save_names:
            break
        if not made_progress:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
