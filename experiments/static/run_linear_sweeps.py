#!/usr/bin/env python3
"""
Run linear-only resolution sweeps for all static experiments and optionally
send plots to Slack.
"""

import argparse
import gc
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from util.io.slack import load_slack_env, send_results_to_slack

from experiments.static import circles, ellipses, lines, squares, zalesak


LINEAR_ALGOS = ["Youngs", "LVIRA", "safe_linear", "linear"]
CORNER_ALGOS = ["safe_linear_corner", "linear+corner"]

DEFAULT_RESOLUTIONS = [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
DEFAULT_RESOLUTIONS_SHORT = [0.50, 0.64, 1.00, 1.28, 1.50]


def _safe_mean(values, min_error):
    if not values:
        return float("nan")
    mean_val = float(np.mean(np.array(values)))
    if math.isnan(mean_val):
        return mean_val
    return max(mean_val, min_error)


def _record_failure(failures, exp_name, algo, resolution, err):
    failures.append(
        {
            "experiment": exp_name,
            "algo": algo,
            "resolution": resolution,
            "error": str(err),
        }
    )


def _make_save_name(exp_name, algo, resolution):
    res_tag = str(resolution).replace(".", "p")
    algo_tag = algo.lower().replace("+", "plus")
    return f"linear_sweep_{exp_name}_{algo_tag}_r{res_tag}"


def _log_tail(log_path, lines=20):
    try:
        content = Path(log_path).read_text().splitlines()
        tail = content[-lines:] if len(content) > lines else content
        print("\n".join(tail))
    except Exception:
        print(f"(Failed to read log tail: {log_path})")


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
        return curvature, gaps, hausdorff, tangent, curvature_proxy
    if exp_name == "lines":
        hausdorff = _parse_labeled_values(metrics_dir / "hausdorff.txt")
        gaps = _parse_labeled_values(metrics_dir / "facet_gap.txt")
        return hausdorff, gaps
    if exp_name == "squares":
        area = _parse_numeric_values(metrics_dir / "area_error.txt")
        edge = _parse_numeric_values(metrics_dir / "edge_alignment_error.txt")
        return area, edge
    if exp_name == "zalesak":
        area = _parse_numeric_values(metrics_dir / "area_error.txt")
        gaps = _parse_numeric_values(metrics_dir / "facet_gap.txt")
        return area, gaps
    return []


def _cleanup_memory():
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    gc.collect()


def _collect_result_plots(explicit_paths):
    results_dir = Path("results") / "static"
    found = []
    for path in explicit_paths:
        if path and Path(path).exists():
            found.append(path)
    if results_dir.exists():
        for path in sorted(results_dir.glob("linear_*.png")):
            found.append(str(path))
    unique = []
    seen = set()
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _parse_str_list(value):
    if value is None:
        return []
    return [p.strip().lower() for p in value.split(",") if p.strip()]


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


def run_circle_sweep(
    config,
    num_circles,
    radius,
    failures,
    use_subprocess,
    log_dir,
    notify=False,
    aggregate_samples=5,
):
    min_error = 1e-14
    resolutions = DEFAULT_RESOLUTIONS
    algos = LINEAR_ALGOS

    curvature_results = {algo: [] for algo in algos}
    gap_results = {algo: [] for algo in algos}
    hausdorff_results = {algo: [] for algo in algos}
    tangent_results = {algo: [] for algo in algos}
    curvature_proxy_results = {algo: [] for algo in algos}
    tangent_results = {algo: [] for algo in algos}
    curvature_proxy_results = {algo: [] for algo in algos}

    for resolution in resolutions:
        for algo in algos:
            save_name = _make_save_name("circles", algo, resolution) if use_subprocess else f"linear_sweep_circle_{algo}"
            if use_subprocess:
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.static.circles",
                    "--config",
                    config,
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    algo,
                    "--save_name",
                    save_name,
                    "--num_circles",
                    str(num_circles),
                    "--radius",
                    str(radius),
                ]
                log_path = Path(log_dir) / f"circles_{algo}_r{resolution}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    _record_failure(failures, "circles", algo, resolution, f"exit {code}")
                    print(f"[circles] {algo} r={resolution} failed. Log: {log_path}")
                    _log_tail(log_path)
                    curvature_results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    hausdorff_results[algo].append(float("nan"))
                    continue
                errors, gaps, hausdorffs, tangents, curv_proxies = _collect_metrics(
                    "circles", save_name
                )
            else:
                try:
                    errors, gaps, hausdorffs, tangents, curv_proxies = circles.main(
                        config_setting=config,
                        resolution=resolution,
                        facet_algo=algo,
                        save_name=save_name,
                        num_circles=num_circles,
                        radius=radius,
                    )
                except Exception as err:
                    _record_failure(failures, "circles", algo, resolution, err)
                    curvature_results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    hausdorff_results[algo].append(float("nan"))
                    tangent_results[algo].append(float("nan"))
                    curvature_proxy_results[algo].append(float("nan"))
                    _cleanup_memory()
                    continue

            curvature_results[algo].append(_safe_mean(errors, min_error))
            gap_results[algo].append(_safe_mean(gaps, min_error))
            hausdorff_results[algo].append(_safe_mean(hausdorffs, min_error))
            tangent_results[algo].append(_safe_mean(tangents, min_error))
            curvature_proxy_results[algo].append(_safe_mean(curv_proxies, min_error))

            agg_path, indices = _build_aggregate_plot(
                save_name, sample_count=aggregate_samples
            )
            if notify and agg_path:
                send_results_to_slack(
                    f"Circles {algo} r={resolution}: aggregate samples {indices}",
                    [agg_path],
                )
            _cleanup_memory()

    circles.create_combined_plot(
        resolutions,
        curvature_results,
        gap_results,
        radius=radius,
        save_path="results/static/linear_circle_reconstruction_combined.png",
    )
    if notify:
        send_results_to_slack(
            "Circles sweep: combined plot",
            ["results/static/linear_circle_reconstruction_combined.png"],
        )
    circles.create_hausdorff_plot(
        resolutions,
        hausdorff_results,
        radius=radius,
        save_path="results/static/linear_circle_reconstruction_hausdorff.png",
    )
    circles.create_tangent_plot(
        resolutions,
        tangent_results,
        radius=radius,
        save_path="results/static/linear_circle_reconstruction_tangent.png",
    )
    circles.create_curvature_proxy_plot(
        resolutions,
        curvature_proxy_results,
        radius=radius,
        save_path="results/static/linear_circle_reconstruction_curvature_proxy.png",
    )
    if notify:
        send_results_to_slack(
            "Circles sweep: Hausdorff plot",
            ["results/static/linear_circle_reconstruction_hausdorff.png"],
        )

    with open("results/static/linear_circle_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Curvature Results: {curvature_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
        f.write(f"Hausdorff Results: {hausdorff_results}\n")
        f.write(f"Tangent Results: {tangent_results}\n")
        f.write(f"Curvature Proxy Results: {curvature_proxy_results}\n")
    if notify:
        send_results_to_slack(
            "Circles sweep: results",
            ["results/static/linear_circle_reconstruction_results.txt"],
        )

    return [
        "results/static/linear_circle_reconstruction_combined.png",
        "results/static/linear_circle_reconstruction_hausdorff.png",
        "results/static/linear_circle_reconstruction_tangent.png",
        "results/static/linear_circle_reconstruction_curvature_proxy.png",
    ]


def run_ellipse_sweep(
    config,
    num_ellipses,
    failures,
    use_subprocess,
    log_dir,
    notify=False,
    aggregate_samples=5,
):
    min_error = 1e-14
    resolutions = DEFAULT_RESOLUTIONS
    algos = LINEAR_ALGOS

    curvature_results = {algo: [] for algo in algos}
    gap_results = {algo: [] for algo in algos}
    hausdorff_results = {algo: [] for algo in algos}

    for resolution in resolutions:
        for algo in algos:
            save_name = _make_save_name("ellipses", algo, resolution) if use_subprocess else f"linear_sweep_ellipse_{algo}"
            if use_subprocess:
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.static.ellipses",
                    "--config",
                    config,
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    algo,
                    "--save_name",
                    save_name,
                    "--num_ellipses",
                    str(num_ellipses),
                ]
                log_path = Path(log_dir) / f"ellipses_{algo}_r{resolution}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    _record_failure(failures, "ellipses", algo, resolution, f"exit {code}")
                    print(f"[ellipses] {algo} r={resolution} failed. Log: {log_path}")
                    _log_tail(log_path)
                    curvature_results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    hausdorff_results[algo].append(float("nan"))
                    continue
                errors, gaps, hausdorffs, tangents, curv_proxies = _collect_metrics(
                    "ellipses", save_name
                )
            else:
                try:
                    errors, gaps, hausdorffs, tangents, curv_proxies = ellipses.main(
                        config_setting=config,
                        resolution=resolution,
                        facet_algo=algo,
                        save_name=save_name,
                        num_ellipses=num_ellipses,
                    )
                except Exception as err:
                    _record_failure(failures, "ellipses", algo, resolution, err)
                    curvature_results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    hausdorff_results[algo].append(float("nan"))
                    tangent_results[algo].append(float("nan"))
                    curvature_proxy_results[algo].append(float("nan"))
                    _cleanup_memory()
                    continue

            curvature_results[algo].append(_safe_mean(errors, min_error))
            gap_results[algo].append(_safe_mean(gaps, min_error))
            hausdorff_results[algo].append(_safe_mean(hausdorffs, min_error))
            tangent_results[algo].append(_safe_mean(tangents, min_error))
            curvature_proxy_results[algo].append(_safe_mean(curv_proxies, min_error))

            agg_path, indices = _build_aggregate_plot(
                save_name, sample_count=aggregate_samples
            )
            if notify and agg_path:
                send_results_to_slack(
                    f"Ellipses {algo} r={resolution}: aggregate samples {indices}",
                    [agg_path],
                )
            _cleanup_memory()

    ellipses.create_combined_plot(
        resolutions,
        curvature_results,
        gap_results,
        save_path="results/static/linear_ellipse_reconstruction_combined.png",
    )
    if notify:
        send_results_to_slack(
            "Ellipses sweep: combined plot",
            ["results/static/linear_ellipse_reconstruction_combined.png"],
        )
    ellipses.create_hausdorff_plot(
        resolutions,
        hausdorff_results,
        save_path="results/static/linear_ellipse_reconstruction_hausdorff.png",
    )
    ellipses.create_tangent_plot(
        resolutions,
        tangent_results,
        save_path="results/static/linear_ellipse_reconstruction_tangent.png",
    )
    ellipses.create_curvature_proxy_plot(
        resolutions,
        curvature_proxy_results,
        save_path="results/static/linear_ellipse_reconstruction_curvature_proxy.png",
    )
    if notify:
        send_results_to_slack(
            "Ellipses sweep: Hausdorff plot",
            ["results/static/linear_ellipse_reconstruction_hausdorff.png"],
        )

    with open("results/static/linear_ellipse_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Curvature Results: {curvature_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
        f.write(f"Hausdorff Results: {hausdorff_results}\n")
        f.write(f"Tangent Results: {tangent_results}\n")
        f.write(f"Curvature Proxy Results: {curvature_proxy_results}\n")
    if notify:
        send_results_to_slack(
            "Ellipses sweep: results",
            ["results/static/linear_ellipse_reconstruction_results.txt"],
        )

    return [
        "results/static/linear_ellipse_reconstruction_combined.png",
        "results/static/linear_ellipse_reconstruction_hausdorff.png",
        "results/static/linear_ellipse_reconstruction_tangent.png",
        "results/static/linear_ellipse_reconstruction_curvature_proxy.png",
    ]


def run_line_sweep(
    config,
    num_lines,
    failures,
    use_subprocess,
    log_dir,
    notify=False,
    aggregate_samples=5,
):
    min_error = 1e-14
    resolutions = DEFAULT_RESOLUTIONS
    algos = LINEAR_ALGOS

    results = {algo: [] for algo in algos}
    gap_results = {algo: [] for algo in algos}

    for resolution in resolutions:
        for algo in algos:
            save_name = _make_save_name("lines", algo, resolution) if use_subprocess else f"linear_sweep_line_{algo}"
            if use_subprocess:
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.static.lines",
                    "--config",
                    config,
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    algo,
                    "--save_name",
                    save_name,
                    "--num_lines",
                    str(num_lines),
                ]
                log_path = Path(log_dir) / f"lines_{algo}_r{resolution}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    _record_failure(failures, "lines", algo, resolution, f"exit {code}")
                    print(f"[lines] {algo} r={resolution} failed. Log: {log_path}")
                    _log_tail(log_path)
                    results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    continue
                hausdorff, gaps = _collect_metrics("lines", save_name)
            else:
                try:
                    hausdorff, gaps = lines.main(
                        config_setting=config,
                        resolution=resolution,
                        facet_algo=algo,
                        save_name=save_name,
                        num_lines=num_lines,
                    )
                except Exception as err:
                    _record_failure(failures, "lines", algo, resolution, err)
                    results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    _cleanup_memory()
                    continue

            results[algo].append(_safe_mean(hausdorff, min_error))
            gap_results[algo].append(_safe_mean(gaps, min_error))

            agg_path, indices = _build_aggregate_plot(
                save_name, sample_count=aggregate_samples
            )
            if notify and agg_path:
                send_results_to_slack(
                    f"Lines {algo} r={resolution}: aggregate samples {indices}",
                    [agg_path],
                )
            _cleanup_memory()

    lines.create_performance_plot(
        resolutions,
        results,
        title="Line Static Reconstruction (Linear Only)",
        ylabel="Average Hausdorff Distance",
        save_path="results/static/linear_line_reconstruction_hausdorff.png",
    )
    if notify:
        send_results_to_slack(
            "Lines sweep: Hausdorff plot",
            ["results/static/linear_line_reconstruction_hausdorff.png"],
        )

    plt_gap_path = "results/static/linear_line_reconstruction_facet_gap.png"
    _plot_line_gap(resolutions, gap_results, plt_gap_path)
    if notify:
        send_results_to_slack("Lines sweep: facet gap plot", [plt_gap_path])

    with open("results/static/linear_line_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Results: {results}\n")
        f.write(f"Facet Gaps: {gap_results}\n")
    if notify:
        send_results_to_slack(
            "Lines sweep: results",
            ["results/static/linear_line_reconstruction_results.txt"],
        )

    return [
        "results/static/linear_line_reconstruction_hausdorff.png",
        plt_gap_path,
    ]


def _plot_line_gap(resolutions, gap_results, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in gap_results.items():
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Average Facet Gap", fontsize=14)
    plt.title("Facet Gap vs. Resolution", fontsize=16, fontweight="bold")
    plt.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_square_sweep(
    config,
    num_squares,
    failures,
    use_subprocess,
    log_dir,
    notify=False,
    aggregate_samples=5,
):
    min_error = 1e-14
    resolutions = DEFAULT_RESOLUTIONS_SHORT
    algos = LINEAR_ALGOS + CORNER_ALGOS

    area_results = {algo: [] for algo in algos}
    edge_results = {algo: [] for algo in algos}

    for resolution in resolutions:
        for algo in algos:
            save_name = _make_save_name("squares", algo, resolution) if use_subprocess else f"linear_sweep_square_{algo}"
            if use_subprocess:
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.static.squares",
                    "--config",
                    config,
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    algo,
                    "--save_name",
                    save_name,
                    "--num_squares",
                    str(num_squares),
                ]
                log_path = Path(log_dir) / f"squares_{algo}_r{resolution}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    _record_failure(failures, "squares", algo, resolution, f"exit {code}")
                    print(f"[squares] {algo} r={resolution} failed. Log: {log_path}")
                    _log_tail(log_path)
                    area_results[algo].append(float("nan"))
                    edge_results[algo].append(float("nan"))
                    continue
                areas, edges = _collect_metrics("squares", save_name)
            else:
                try:
                    areas, edges = squares.main(
                        config_setting=config,
                        resolution=resolution,
                        facet_algo=algo,
                        save_name=save_name,
                        num_squares=num_squares,
                    )
                except Exception as err:
                    _record_failure(failures, "squares", algo, resolution, err)
                    area_results[algo].append(float("nan"))
                    edge_results[algo].append(float("nan"))
                    _cleanup_memory()
                    continue

            area_results[algo].append(_safe_mean(areas, min_error))
            edge_results[algo].append(_safe_mean(edges, min_error))

            agg_path, indices = _build_aggregate_plot(
                save_name, sample_count=aggregate_samples
            )
            if notify and agg_path:
                send_results_to_slack(
                    f"Squares {algo} r={resolution}: aggregate samples {indices}",
                    [agg_path],
                )
            _cleanup_memory()

    squares.create_plots(
        resolutions,
        area_results,
        edge_results,
        area_save_path="results/static/linear_square_reconstruction_area.png",
        edge_save_path="results/static/linear_square_reconstruction_edge.png",
    )
    if notify:
        send_results_to_slack(
            "Squares sweep: area error plot",
            ["results/static/linear_square_reconstruction_area.png"],
        )
        send_results_to_slack(
            "Squares sweep: edge alignment plot",
            ["results/static/linear_square_reconstruction_edge.png"],
        )

    with open("results/static/linear_square_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Area Results: {area_results}\n")
        f.write(f"Edge Results: {edge_results}\n")
    if notify:
        send_results_to_slack(
            "Squares sweep: results",
            ["results/static/linear_square_reconstruction_results.txt"],
        )

    return [
        "results/static/linear_square_reconstruction_area.png",
        "results/static/linear_square_reconstruction_edge.png",
    ]


def run_zalesak_sweep(
    config,
    num_cases,
    failures,
    use_subprocess,
    log_dir,
    notify=False,
    aggregate_samples=5,
):
    min_error = 1e-14
    resolutions = DEFAULT_RESOLUTIONS_SHORT
    algos = LINEAR_ALGOS + CORNER_ALGOS

    area_results = {algo: [] for algo in algos}
    gap_results = {algo: [] for algo in algos}

    for resolution in resolutions:
        for algo in algos:
            save_name = _make_save_name("zalesak", algo, resolution) if use_subprocess else f"linear_sweep_zalesak_{algo}"
            if use_subprocess:
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.static.zalesak",
                    "--config",
                    config,
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    algo,
                    "--save_name",
                    save_name,
                    "--num_cases",
                    str(num_cases),
                ]
                log_path = Path(log_dir) / f"zalesak_{algo}_r{resolution}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    _record_failure(failures, "zalesak", algo, resolution, f"exit {code}")
                    print(f"[zalesak] {algo} r={resolution} failed. Log: {log_path}")
                    _log_tail(log_path)
                    area_results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    continue
                areas, gaps = _collect_metrics("zalesak", save_name)
            else:
                try:
                    areas, gaps = zalesak.main(
                        config_setting=config,
                        resolution=resolution,
                        facet_algo=algo,
                        save_name=save_name,
                        num_cases=num_cases,
                    )
                except Exception as err:
                    _record_failure(failures, "zalesak", algo, resolution, err)
                    area_results[algo].append(float("nan"))
                    gap_results[algo].append(float("nan"))
                    _cleanup_memory()
                    continue

            area_results[algo].append(_safe_mean(areas, min_error))
            gap_results[algo].append(_safe_mean(gaps, min_error))

            agg_path, indices = _build_aggregate_plot(
                save_name, sample_count=aggregate_samples
            )
            if notify and agg_path:
                send_results_to_slack(
                    f"Zalesak {algo} r={resolution}: aggregate samples {indices}",
                    [agg_path],
                )
            _cleanup_memory()

    zalesak.create_combined_plot(
        resolutions,
        area_results,
        gap_results,
        save_path="results/static/linear_zalesak_reconstruction_combined.png",
    )
    if notify:
        send_results_to_slack(
            "Zalesak sweep: combined plot",
            ["results/static/linear_zalesak_reconstruction_combined.png"],
        )

    with open("results/static/linear_zalesak_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Area Results: {area_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
    if notify:
        send_results_to_slack(
            "Zalesak sweep: results",
            ["results/static/linear_zalesak_reconstruction_results.txt"],
        )

    return ["results/static/linear_zalesak_reconstruction_combined.png"]


def main():
    parser = argparse.ArgumentParser(description="Run linear-only sweeps for static experiments.")
    parser.add_argument("--circles", type=int, default=25, help="num circles")
    parser.add_argument("--ellipses", type=int, default=25, help="num ellipses")
    parser.add_argument("--lines", type=int, default=25, help="num lines")
    parser.add_argument("--squares", type=int, default=25, help="num squares")
    parser.add_argument("--zalesak", type=int, default=25, help="num zalesak cases")
    parser.add_argument("--notify", action="store_true", help="send plots to Slack")
    parser.add_argument("--dry_run", action="store_true", help="skip execution")
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
        help="number of sample cases to include per-run aggregate plot (0 to disable)",
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="run each case in a subprocess (lower memory footprint)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="directory to write subprocess logs (default: logs/linear_sweeps/<timestamp>)",
    )
    args = parser.parse_args()

    failures = []
    all_plots = []

    print(f"Linear sweeps started at {datetime.now().isoformat()}")

    load_slack_env()
    auto_notify = os.getenv("SLACK_NOTIFY", "").lower() in {"1", "true", "yes"}
    notify = args.notify or auto_notify
    if notify and not args.notify and auto_notify:
        print("Slack auto-notify enabled via SLACK_NOTIFY.")

    only_experiments = set(_parse_str_list(args.only))

    log_dir = args.log_dir
    if args.subprocess and log_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", "linear_sweeps", stamp)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        print(f"Logging subprocess output to {log_dir}")

    if not args.dry_run:
        if not only_experiments or "circles" in only_experiments:
            all_plots += run_circle_sweep(
                "static/circle",
                args.circles,
                10.0,
                failures,
                args.subprocess,
                log_dir,
                notify,
                args.aggregate_samples,
            )
        if not only_experiments or "ellipses" in only_experiments:
            all_plots += run_ellipse_sweep(
                "static/ellipse",
                args.ellipses,
                failures,
                args.subprocess,
                log_dir,
                notify,
                args.aggregate_samples,
            )
        if not only_experiments or "lines" in only_experiments:
            all_plots += run_line_sweep(
                "static/line",
                args.lines,
                failures,
                args.subprocess,
                log_dir,
                notify,
                args.aggregate_samples,
            )
        if not only_experiments or "squares" in only_experiments:
            all_plots += run_square_sweep(
                "static/square",
                args.squares,
                failures,
                args.subprocess,
                log_dir,
                notify,
                args.aggregate_samples,
            )
        if not only_experiments or "zalesak" in only_experiments:
            all_plots += run_zalesak_sweep(
                "static/zalesak",
                args.zalesak,
                failures,
                args.subprocess,
                log_dir,
                notify,
                args.aggregate_samples,
            )

    print("\n=== Linear sweep summary ===")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(
            f"- {failure['experiment']} / {failure['algo']} / {failure['resolution']}: {failure['error']}"
        )

    if notify:
        message = "Linear sweep plots generated."
        if failures:
            message += f" Failures: {len(failures)} (see console)."
        send_results_to_slack(message, [])


if __name__ == "__main__":
    main()
