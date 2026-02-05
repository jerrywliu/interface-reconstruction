#!/usr/bin/env python3
"""
Run perturbed-quad sweeps for static experiments.
Sweeps resolution, perturbation amplitude (wiggle), and seed.
Writes a CSV with per-run metrics and (optionally) sends aggregate plots to Slack.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from util.io.slack import load_slack_env, send_results_to_slack


LINEAR_ALGOS = ["Youngs", "LVIRA", "safe_linear", "linear"]
CORNER_ALGOS = ["safe_linear_corner", "linear+corner"]

DEFAULT_RESOLUTIONS = [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
DEFAULT_RESOLUTIONS_SHORT = [0.50, 0.64, 1.00, 1.28, 1.50]
DEFAULT_WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]
DEFAULT_SEEDS = [0]


EXPERIMENTS = [
    {
        "name": "circles",
        "module": "experiments.static.circles",
        "config": "static/circle",
        "num_arg": "--num_circles",
        "algorithms": LINEAR_ALGOS,
    },
    {
        "name": "ellipses",
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "num_arg": "--num_ellipses",
        "algorithms": LINEAR_ALGOS,
    },
    {
        "name": "lines",
        "module": "experiments.static.lines",
        "config": "static/line",
        "num_arg": "--num_lines",
        "algorithms": LINEAR_ALGOS,
    },
    {
        "name": "squares",
        "module": "experiments.static.squares",
        "config": "static/square",
        "num_arg": "--num_squares",
        "algorithms": LINEAR_ALGOS + CORNER_ALGOS,
    },
    {
        "name": "zalesak",
        "module": "experiments.static.zalesak",
        "config": "static/zalesak",
        "num_arg": "--num_cases",
        "algorithms": LINEAR_ALGOS + CORNER_ALGOS,
    },
]


def _parse_list(value, cast=float):
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [cast(p) for p in parts]


def _parse_str_list(value):
    if value is None:
        return []
    return [p.strip().lower() for p in value.split(",") if p.strip()]


def _safe_mean(values):
    if not values:
        return float("nan")
    mean_val = float(np.mean(np.array(values)))
    return mean_val


def _safe_stats(values):
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _metric_stats(metric_name, values):
    stats = _safe_stats(values)
    return {f"{metric_name}_{key}": value for key, value in stats.items()}


def _make_save_name(exp_name, algo, resolution, wiggle, seed):
    res_tag = str(resolution).replace(".", "p")
    wiggle_tag = str(wiggle).replace(".", "p")
    algo_tag = algo.lower().replace("+", "plus")
    return f"perturb_sweep_{exp_name}_{algo_tag}_r{res_tag}_w{wiggle_tag}_s{seed}"


def _algo_tag(algo):
    return algo.lower().replace("+", "plus")


def _run_subprocess(cmd, log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def _read_metric_values(path):
    if not Path(path).exists():
        return []
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(line)
    return values


def _parse_numeric_values(path):
    values = []
    for line in _read_metric_values(path):
        try:
            values.append(float(line))
        except ValueError:
            pass
    return values


def _parse_labeled_values(path):
    values = []
    for line in _read_metric_values(path):
        try:
            values.append(float(line.split("_")[-1]))
        except ValueError:
            pass
    return values


def _collect_metrics(exp_name, save_name):
    metrics_dir = Path("plots") / save_name / "metrics"
    if exp_name in ["circles", "ellipses"]:
        curvature = _parse_numeric_values(metrics_dir / "curvature_error.txt")
        gaps = _parse_numeric_values(metrics_dir / "facet_gap.txt")
        hausdorff = _parse_numeric_values(metrics_dir / "hausdorff.txt")
        tangent = _parse_numeric_values(metrics_dir / "tangent_error.txt")
        curvature_proxy = _parse_numeric_values(
            metrics_dir / "curvature_proxy_error.txt"
        )
        metrics = {}
        metrics.update(_metric_stats("curvature_error", curvature))
        metrics.update(_metric_stats("facet_gap", gaps))
        metrics.update(_metric_stats("hausdorff", hausdorff))
        metrics.update(_metric_stats("tangent_error", tangent))
        metrics.update(_metric_stats("curvature_proxy_error", curvature_proxy))
        return metrics
    if exp_name == "lines":
        hausdorff = _parse_labeled_values(metrics_dir / "hausdorff.txt")
        gaps = _parse_labeled_values(metrics_dir / "facet_gap.txt")
        metrics = {}
        metrics.update(_metric_stats("hausdorff", hausdorff))
        metrics.update(_metric_stats("facet_gap", gaps))
        return metrics
    if exp_name == "squares":
        area = _parse_numeric_values(metrics_dir / "area_error.txt")
        edge = _parse_numeric_values(metrics_dir / "edge_alignment_error.txt")
        metrics = {}
        metrics.update(_metric_stats("area_error", area))
        metrics.update(_metric_stats("edge_alignment_error", edge))
        return metrics
    if exp_name == "zalesak":
        area = _parse_numeric_values(metrics_dir / "area_error.txt")
        gaps = _parse_numeric_values(metrics_dir / "facet_gap.txt")
        metrics = {}
        metrics.update(_metric_stats("area_error", area))
        metrics.update(_metric_stats("facet_gap", gaps))
        return metrics
    return {}


def _sample_indices(count, sample_count):
    if sample_count <= 0 or count <= 0:
        return []
    if count <= sample_count:
        return list(range(count))
    positions = np.linspace(0, count - 1, sample_count)
    indices = sorted({int(round(pos)) for pos in positions})
    if len(indices) < sample_count:
        for idx in range(count):
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= sample_count:
                break
        indices = sorted(indices)
    return indices


def _build_aggregate_plot(save_name, sample_count=5):
    if sample_count <= 0:
        return None, []
    base_dir = Path("plots") / save_name / "plt"
    areas_dir = base_dir / "areas"
    partial_dir = base_dir / "partial_areas"
    if not areas_dir.exists() or not partial_dir.exists():
        return None, []

    area_files = sorted(areas_dir.glob("*.png"))
    partial_files = sorted(partial_dir.glob("*.png"))
    count = min(len(area_files), len(partial_files))
    if count == 0:
        return None, []

    indices = _sample_indices(count, sample_count)
    if not indices:
        return None, []

    try:
        from PIL import Image
    except Exception:
        return None, indices

    first = Image.open(area_files[indices[0]])
    tile_w, tile_h = first.size
    first.close()

    pad = 10
    cols = len(indices)
    rows = 2
    width = cols * tile_w + (cols + 1) * pad
    height = rows * tile_h + (rows + 1) * pad

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    for col, idx in enumerate(indices):
        area_path = area_files[idx]
        partial_path = partial_files[idx]

        area_img = Image.open(area_path)
        partial_img = Image.open(partial_path)

        if area_img.size != (tile_w, tile_h):
            area_img = area_img.resize((tile_w, tile_h))
        if partial_img.size != (tile_w, tile_h):
            partial_img = partial_img.resize((tile_w, tile_h))

        x = pad + col * (tile_w + pad)
        y_top = pad
        y_bottom = pad * 2 + tile_h
        canvas.paste(area_img, (x, y_top))
        canvas.paste(partial_img, (x, y_bottom))

        area_img.close()
        partial_img.close()

    out_dir = Path("results") / "static" / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{save_name}_aggregate.png"
    canvas.save(out_path)
    return str(out_path), indices


def _load_sweep_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows


def _build_metric_index(rows):
    def _split_metric_key(metric_key):
        for suffix in ("_mean", "_median", "_p25", "_p75"):
            if metric_key.endswith(suffix):
                return metric_key[: -len(suffix)], suffix[1:]
        return metric_key, "value"

    data = {}
    for row in rows:
        try:
            exp = row["experiment"]
            algo = row["algo"]
            metric, stat = _split_metric_key(row["metric_key"])
            res = float(row["resolution"])
            wiggle = float(row["wiggle"])
            value = float(row["metric_value"])
        except (KeyError, ValueError):
            continue

        data.setdefault(exp, {}).setdefault(algo, {}).setdefault(metric, {}).setdefault(
            res, {}
        ).setdefault(wiggle, {}).setdefault(stat, []).append(value)

    return data


def _plot_metric_vs_wiggle(exp, algo, metric, res_map, out_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    min_error = 1e-14

    def _aggregate(values):
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=float)
        return float(np.median(arr))

    for res in sorted(res_map.keys()):
        wiggle_map = res_map[res]
        wiggles = sorted(wiggle_map.keys())
        medians = []
        p25 = []
        p75 = []
        for w in wiggles:
            stats = wiggle_map[w]
            if not stats:
                medians.append(float("nan"))
                p25.append(float("nan"))
                p75.append(float("nan"))
                continue

            med = (
                _aggregate(stats.get("median"))
                if "median" in stats
                else _aggregate(stats.get("mean"))
                if "mean" in stats
                else _aggregate(stats.get("value"))
            )
            q25 = _aggregate(stats.get("p25")) if "p25" in stats else med
            q75 = _aggregate(stats.get("p75")) if "p75" in stats else med

            medians.append(max(med, min_error))
            p25.append(max(q25, min_error))
            p75.append(max(q75, min_error))

        plt.plot(wiggles, medians, marker="o", linewidth=2.0, label=f"r={res}")
        plt.fill_between(wiggles, p25, p75, alpha=0.2)

    metric_label = metric.replace("_", " ").title()
    plt.xlabel("Wiggle", fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f"{exp} {algo} ({metric_label})", fontsize=14, fontweight="bold")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{exp}_{_algo_tag(algo)}_{metric}.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _generate_summary_plots(csv_path, out_dir):
    rows = _load_sweep_rows(csv_path)
    data = _build_metric_index(rows)
    plots_by_exp = {}

    for exp, exp_data in data.items():
        for algo, algo_data in exp_data.items():
            for metric, res_map in algo_data.items():
                plot_path = _plot_metric_vs_wiggle(
                    exp, algo, metric, res_map, out_dir
                )
                plots_by_exp.setdefault(exp, []).append(plot_path)

    return plots_by_exp


def main():
    parser = argparse.ArgumentParser(description="Run perturbed-quad sweeps for static experiments.")
    parser.add_argument("--circles", type=int, default=25, help="num circles")
    parser.add_argument("--ellipses", type=int, default=25, help="num ellipses")
    parser.add_argument("--lines", type=int, default=25, help="num lines")
    parser.add_argument("--squares", type=int, default=25, help="num squares")
    parser.add_argument("--zalesak", type=int, default=25, help="num zalesak cases")
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="comma-separated resolutions (default: match linear sweeps)",
    )
    parser.add_argument(
        "--wiggles",
        type=str,
        default=None,
        help="comma-separated wiggle amplitudes (default: 0,0.05,0.1,0.2,0.3)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds (default: 0)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated experiment names to run (e.g., lines,ellipses)",
    )
    parser.add_argument(
        "--aggregate_samples",
        type=int,
        default=0,
        help="number of sample cases in aggregate image (0 to disable)",
    )
    parser.add_argument("--notify", action="store_true", help="send aggregates to Slack")
    parser.add_argument("--dry_run", action="store_true", help="skip execution")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="directory to write subprocess logs (default: logs/perturbed_sweeps/<timestamp>)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="output CSV path (default: results/static/perturbed_sweep_<timestamp>.csv)",
    )

    args = parser.parse_args()

    load_slack_env()
    notify = args.notify or os.getenv("SLACK_NOTIFY", "").lower() in {"1", "true", "yes"}

    resolutions_override = _parse_list(args.resolutions, float)
    wiggles = _parse_list(args.wiggles, float) or DEFAULT_WIGGLES
    seeds = _parse_list(args.seeds, int) or DEFAULT_SEEDS
    only_experiments = set(_parse_str_list(args.only))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join("logs", "perturbed_sweeps", stamp)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    out_csv = args.out_csv or os.path.join(
        "results", "static", f"perturbed_sweep_{stamp}.csv"
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    failures = []

    with open(out_csv, "w", newline="") as csvfile:
        fieldnames = [
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "metric_key",
            "metric_value",
            "save_name",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Perturbed sweeps started at {datetime.now().isoformat()}")
        print(f"Logging to {log_dir}")
        print(f"Writing CSV to {out_csv}")

        if args.dry_run:
            return

        for exp in EXPERIMENTS:
            if only_experiments and exp["name"] not in only_experiments:
                continue
            num_value = getattr(args, exp["name"], None)
            if resolutions_override:
                exp_resolutions = resolutions_override
            elif exp["name"] in {"squares", "zalesak"}:
                exp_resolutions = DEFAULT_RESOLUTIONS_SHORT
            else:
                exp_resolutions = DEFAULT_RESOLUTIONS

            for resolution in exp_resolutions:
                for wiggle in wiggles:
                    for seed in seeds:
                        for algo in exp["algorithms"]:
                            save_name = _make_save_name(
                                exp["name"], algo, resolution, wiggle, seed
                            )
                            cmd = [
                                sys.executable,
                                "-m",
                                exp["module"],
                                "--config",
                                exp["config"],
                                "--resolution",
                                str(resolution),
                                "--facet_algo",
                                algo,
                                "--save_name",
                                save_name,
                                "--mesh_type",
                                "perturbed_quads",
                                "--perturb_wiggle",
                                str(wiggle),
                                "--perturb_seed",
                                str(seed),
                                "--perturb_fix_boundary",
                                "1",
                            ]
                            if num_value is not None:
                                cmd += [exp["num_arg"], str(num_value)]

                            log_path = Path(log_dir) / f"{save_name}.log"
                            code = _run_subprocess(cmd, log_path)
                            if code != 0:
                                failures.append(
                                    {
                                        "experiment": exp["name"],
                                        "algo": algo,
                                        "resolution": resolution,
                                        "wiggle": wiggle,
                                        "seed": seed,
                                        "code": code,
                                    }
                                )
                                print(
                                    f"[ERROR] {exp['name']} {algo} r={resolution} w={wiggle} s={seed} failed (code {code})"
                                )
                                continue

                            metrics = _collect_metrics(exp["name"], save_name)
                            for key, value in metrics.items():
                                writer.writerow(
                                    {
                                        "experiment": exp["name"],
                                        "algo": algo,
                                        "resolution": resolution,
                                        "wiggle": wiggle,
                                        "seed": seed,
                                        "metric_key": key,
                                        "metric_value": value,
                                        "save_name": save_name,
                                    }
                                )

                            if notify:
                                agg_path, indices = _build_aggregate_plot(
                                    save_name, sample_count=args.aggregate_samples
                                )
                                if agg_path:
                                    send_results_to_slack(
                                        f"{exp['name']} {algo} r={resolution} w={wiggle} s={seed}: aggregate samples {indices}",
                                        [agg_path],
                                    )

    print("\n=== Perturbed sweep summary ===")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(
            f"- {failure['experiment']} / {failure['algo']} / r={failure['resolution']} / w={failure['wiggle']} / s={failure['seed']} (code {failure['code']})"
        )

    summary_dir = (Path("results") / "static" / "perturbed_plots").resolve()
    plots_by_exp = _generate_summary_plots(out_csv, summary_dir)

    if notify:
        for exp, plot_paths in plots_by_exp.items():
            if plot_paths:
                resolved_paths = [str(Path(path).resolve()) for path in plot_paths]
                send_results_to_slack(
                    f"Perturbed sweep plots: {exp}",
                    resolved_paths,
                )
        send_results_to_slack(
            f"Perturbed sweep complete. CSV: {out_csv}",
            [str(Path(out_csv).resolve())],
        )


if __name__ == "__main__":
    main()
