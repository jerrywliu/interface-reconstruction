#!/usr/bin/env python3
"""
Run deterministic single-case resolution-study reconstructions for the appendix.

This script runs one representative case per experiment, method, resolution, and
mesh perturbation setting, then assembles qualitative comparison figures with:
  - Cartesian grid on the left
  - Perturbed grid on the right
  - rows for N = 32, 50, 64

The best-method comparison used for each geometry is:
  - lines: linear
  - squares: linear+corner
  - circles: circular
  - ellipses: circular
  - zalesak: circular+corner
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import generate_section6_maintext_figures as maintext_figs
from experiments.static import run_perturbed_sweeps as sweeps


PLOTS_ROOT = REPO_ROOT / "plots"

APPENDIX_RESOLUTION_EXPERIMENTS = [
    {
        "name": "lines",
        "module": "experiments.static.lines",
        "config": "static/line",
        "num_arg": "--num_lines",
        "num_default": 25,
        "case_index": 12,
        "algo": "linear",
        "display": "Ours (linear)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 100.0,
        "margin_frac": 0.00,
        "inset": None,
    },
    {
        "name": "squares",
        "module": "experiments.static.squares",
        "config": "static/square",
        "num_arg": "--num_squares",
        "num_default": 25,
        "case_index": 12,
        "algo": "linear+corner",
        "display": "Ours (linear+corner)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 42.0,
        "margin_frac": 0.10,
        "inset": {"kind": "square_corner", "zoom": 2.8},
    },
    {
        "name": "circles",
        "module": "experiments.static.circles",
        "config": "static/circle",
        "num_arg": "--num_circles",
        "num_default": 25,
        "case_index": 12,
        "algo": "circular",
        "display": "Ours (circular)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 26.0,
        "margin_frac": 0.14,
        "inset": None,
    },
    {
        "name": "ellipses",
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "num_arg": "--num_ellipses",
        "num_default": 25,
        "case_index": 12,
        "algo": "circular",
        "display": "Ours (circular)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 66.0,
        "margin_frac": 0.12,
        "inset": None,
    },
    {
        "name": "zalesak",
        "module": "experiments.static.zalesak",
        "config": "static/zalesak",
        "num_arg": "--num_cases",
        "num_default": 25,
        "case_index": 12,
        "algo": "circular+corner",
        "display": "Ours (circular+corner)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 42.0,
        "margin_frac": 0.12,
        "inset": {"kind": "zalesak_corner", "zoom": 3.0},
    },
]


def _parse_str_list(raw):
    if raw is None:
        return []
    return [part.strip().lower() for part in str(raw).split(",") if part.strip()]


def _selected_experiments(raw_only):
    only = set(_parse_str_list(raw_only))
    if not only:
        return APPENDIX_RESOLUTION_EXPERIMENTS
    return [exp for exp in APPENDIX_RESOLUTION_EXPERIMENTS if exp["name"] in only]


def _save_name(exp_name: str, algo: str, resolution: float, wiggle: float, seed: int) -> str:
    base = sweeps._make_save_name(exp_name, algo, resolution, wiggle, seed)
    return f"appendix_resolution_{base}"


def _required_paths(save_name: str, case_index: int) -> tuple[Path, Path]:
    mesh_path = PLOTS_ROOT / save_name / "vtk" / "mesh.vtk"
    facet_path = (
        PLOTS_ROOT / save_name / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    )
    return mesh_path, facet_path


def _run_exists(save_name: str, case_index: int) -> bool:
    mesh_path, facet_path = _required_paths(save_name, case_index)
    return mesh_path.exists() and facet_path.exists()


def _run_subprocess(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def _true_vtp_segments(exp_name: str, save_name: str, case_index: int) -> np.ndarray:
    true_path = maintext_figs._true_vtp_path(exp_name, save_name, case_index)
    return maintext_figs._segments_from_polydata(maintext_figs._read_polydata(true_path))


def _dedupe_points(points: list[np.ndarray], tol: float = 1e-8) -> list[np.ndarray]:
    deduped = []
    for point in points:
        if any(np.linalg.norm(point - existing) <= tol for existing in deduped):
            continue
        deduped.append(point)
    return deduped


def _clip_infinite_line_to_bounds(
    p1: np.ndarray,
    p2: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    intersections = []

    if abs(dx) > 1e-14:
        for x in (x0, x1):
            t = (x - p1[0]) / dx
            y = p1[1] + t * dy
            if y0 - 1e-8 <= y <= y1 + 1e-8:
                intersections.append(np.asarray([x, y], dtype=float))
    if abs(dy) > 1e-14:
        for y in (y0, y1):
            t = (y - p1[1]) / dy
            x = p1[0] + t * dx
            if x0 - 1e-8 <= x <= x1 + 1e-8:
                intersections.append(np.asarray([x, y], dtype=float))

    intersections = _dedupe_points(intersections)
    if len(intersections) < 2:
        return np.asarray([[p1, p2]], dtype=float)
    if len(intersections) > 2:
        # Keep the farthest pair if the line passes exactly through a corner.
        best_pair = None
        best_dist = -1.0
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                dist = float(np.linalg.norm(intersections[j] - intersections[i]))
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (intersections[i], intersections[j])
        intersections = list(best_pair)
    return np.asarray([[intersections[0], intersections[1]]], dtype=float)


def _line_fill_polygon_from_points(
    p1: np.ndarray,
    p2: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    rect = np.asarray(
        [
            [bounds[0], bounds[2]],
            [bounds[1], bounds[2]],
            [bounds[1], bounds[3]],
            [bounds[0], bounds[3]],
        ],
        dtype=float,
    )

    def _cross(point):
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

    def _intersect(start, end):
        s_val = _cross(start)
        e_val = _cross(end)
        denom = s_val - e_val
        if abs(denom) < 1e-14:
            return end
        t = s_val / denom
        return start + t * (end - start)

    clipped = []
    for start, end in zip(rect, np.roll(rect, -1, axis=0)):
        start_inside = _cross(start) >= 0
        end_inside = _cross(end) >= 0
        if start_inside and end_inside:
            clipped.append(end)
        elif start_inside and not end_inside:
            clipped.append(_intersect(start, end))
        elif (not start_inside) and end_inside:
            clipped.append(_intersect(start, end))
            clipped.append(end)
    if not clipped:
        return np.empty((0, 2), dtype=float)
    return np.asarray(clipped, dtype=float)


def _ordered_loop_vertices(segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.empty((0, 2), dtype=float)
    remaining = [np.asarray(seg, dtype=float) for seg in segments]
    current = remaining.pop(0)
    points = [current[0], current[1]]

    while remaining:
        last = points[-1]
        next_index = None
        next_point = None
        for idx, seg in enumerate(remaining):
            if np.linalg.norm(seg[0] - last) <= 1e-6:
                next_index = idx
                next_point = seg[1]
                break
            if np.linalg.norm(seg[1] - last) <= 1e-6:
                next_index = idx
                next_point = seg[0]
                break
        if next_index is None:
            break
        points.append(next_point)
        remaining.pop(next_index)

    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-6:
        points = points[:-1]
    return np.asarray(points, dtype=float)


def _single_case_ellipse_segments(case_index: int, sample_count: int = 720) -> np.ndarray:
    rng = np.random.default_rng(maintext_figs.ELLIPSE_RANDOM_SEED)
    aspect_ratio = np.linspace(1.5, 3.0, 25)[case_index]
    center = np.asarray([rng.uniform(50, 51), rng.uniform(50, 51)], dtype=float)
    theta = float(rng.uniform(0, np.pi / 2))
    major_axis = 30.0
    minor_axis = major_axis / aspect_ratio
    ts = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    pts = np.zeros((sample_count, 2), dtype=float)
    c = np.cos(theta)
    s = np.sin(theta)
    for i, t in enumerate(ts):
        x_local = major_axis * np.cos(t)
        y_local = minor_axis * np.sin(t)
        pts[i, 0] = center[0] + c * x_local - s * y_local
        pts[i, 1] = center[1] + s * x_local + c * y_local
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


def _single_case_zalesak_truth(case_index: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    rng = np.random.default_rng(maintext_figs.ZALESAK_RANDOM_SEED)
    center = [rng.uniform(50, 51), rng.uniform(50, 51)]
    theta = float(rng.uniform(0, np.pi / 2))
    radius = 15.0
    slot_width = 5.0
    slot_top_rel = 10.0
    cx, cy = center
    half_w = slot_width * 0.5
    y_bottom = cy - radius - 1.0e-6
    y_top = cy + slot_top_rel
    rect = [
        [cx - half_w, y_bottom],
        [cx + half_w, y_bottom],
        [cx + half_w, y_top],
        [cx - half_w, y_top],
    ]
    rect = [maintext_figs.rotate_point_around_center(point, center, theta) for point in rect]
    true_reference = maintext_figs.build_true_reference_zalesak(center, radius, rect, theta)
    true_facets = true_reference["facets"]
    segment_chunks = []
    for facet in true_facets:
        chunk = maintext_figs._facet_segments(facet)
        if len(chunk):
            segment_chunks.append(chunk)
    true_segments = np.concatenate(segment_chunks, axis=0)
    fill_vertices = maintext_figs._concat_facet_points(true_facets)
    slot_rect = np.asarray(rect, dtype=float)
    corner = slot_rect[np.argmax(slot_rect[:, 0] + slot_rect[:, 1])]
    inset_bounds = (
        float(corner[0] - 4.5),
        float(corner[0] + 4.5),
        float(corner[1] - 4.5),
        float(corner[1] + 4.5),
    )
    return true_segments, fill_vertices, inset_bounds


def _truth_payload(
    exp_spec: dict,
    base_save_name: str,
    mesh_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float, float, float] | None]:
    exp_name = exp_spec["name"]
    case_index = exp_spec["case_index"]

    if exp_name == "lines":
        true_segments = maintext_figs._line_true_segments(case_index, mesh_bounds)
        # Let the shared plotting helper derive the filled half-plane from the
        # final panel bounds, exactly as in the main-text line representative.
        return true_segments, None, None

    if exp_name == "squares":
        true_segments = _true_vtp_segments(exp_name, base_save_name, case_index)
        fill_vertices = _ordered_loop_vertices(true_segments)
        corner = fill_vertices[np.argmax(fill_vertices[:, 0] + fill_vertices[:, 1])]
        inset_bounds = (
            float(corner[0] - 4.0),
            float(corner[0] + 4.0),
            float(corner[1] - 4.0),
            float(corner[1] + 4.0),
        )
        return true_segments, fill_vertices, inset_bounds

    if exp_name == "circles":
        true_segments = _true_vtp_segments(exp_name, base_save_name, case_index)
        fill_vertices = _ordered_loop_vertices(true_segments)
        return true_segments, fill_vertices, None

    if exp_name == "ellipses":
        true_segments = _single_case_ellipse_segments(case_index)
        fill_vertices = true_segments[:, 0, :]
        return true_segments, fill_vertices, None

    if exp_name == "zalesak":
        return _single_case_zalesak_truth(case_index)

    return maintext_figs._load_true_segments(exp_name, base_save_name, case_index), None, None


def _figure_bounds(
    exp_spec,
    true_segments: np.ndarray,
    mesh_bounds: tuple[float, float, float, float],
):
    if exp_spec["name"] == "lines":
        # Match the main-text line representative: show the full mesh-domain
        # window rather than collapsing the panel to the line's y-span.
        return mesh_bounds
    return maintext_figs._compute_view_bounds(
        true_segments,
        min_span=exp_spec["min_span"],
        margin_frac=exp_spec["margin_frac"],
    )


def _generate_figure(exp_spec: dict, out_path: Path):
    nrows = len(exp_spec["resolutions"])
    ncols = len(exp_spec["wiggles"])
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 3.1 * nrows))
    axes = np.atleast_2d(axes)

    base_save_name = _save_name(
        exp_spec["name"],
        exp_spec["algo"],
        exp_spec["resolutions"][0],
        exp_spec["wiggles"][0],
        exp_spec["seed"],
    )
    base_mesh_segments = maintext_figs._mesh_segments(PLOTS_ROOT / base_save_name / "vtk" / "mesh.vtk")
    mesh_bounds = maintext_figs._segments_bounds(base_mesh_segments)
    true_segments, true_fill_vertices, inset_bounds = _truth_payload(
        exp_spec,
        base_save_name,
        mesh_bounds,
    )
    bounds = _figure_bounds(exp_spec, true_segments, mesh_bounds)

    for row, resolution in enumerate(exp_spec["resolutions"]):
        for col, wiggle in enumerate(exp_spec["wiggles"]):
            ax = axes[row, col]
            save_name = _save_name(
                exp_spec["name"],
                exp_spec["algo"],
                resolution,
                wiggle,
                exp_spec["seed"],
            )
            mesh_segments = maintext_figs._mesh_segments(PLOTS_ROOT / save_name / "vtk" / "mesh.vtk")
            recon_segments, endpoint_points = maintext_figs._load_reconstructed_segments_and_endpoints(
                save_name, exp_spec["case_index"]
            )
            condition = "Cartesian" if wiggle == 0.0 else "Perturbed"
            title = f"{condition}, N={int(round(resolution * 100))}"
            maintext_figs._plot_panel(
                ax,
                exp_name=exp_spec["name"],
                spec={
                    "case_index": exp_spec["case_index"],
                    "inset": exp_spec["inset"],
                    "true_fill_vertices": true_fill_vertices,
                    "inset_bounds": inset_bounds,
                },
                algo=exp_spec["algo"],
                mesh_segments=mesh_segments,
                true_segments=true_segments,
                recon_segments=recon_segments,
                endpoint_points=endpoint_points,
                title=title,
                bounds=bounds,
            )

    fig.suptitle(f"{exp_spec['display']} on {exp_spec['name'].title()}", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic appendix resolution-study reconstruction visuals."
    )
    parser.add_argument("--only", type=str, default=None, help="comma-separated experiments to run")
    parser.add_argument("--out_dir", type=Path, default=None, help="artifact output directory")
    parser.add_argument("--log_dir", type=Path, default=None, help="log directory override")
    parser.add_argument("--skip_existing", action="store_true", help="skip runs whose required artifacts already exist")
    parser.add_argument("--plot_only", action="store_true", help="generate figures only from existing run outputs")
    parser.add_argument("--dry_run", action="store_true", help="print commands without executing")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir
        or REPO_ROOT
        / "results"
        / "static"
        / "camera_ready"
        / f"static_appendix_resolution_visuals_{stamp}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir or out_dir / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "out_dir": str(out_dir),
        "summary_plots": {},
        "runs": [],
    }

    for exp_spec in _selected_experiments(args.only):
        for resolution in exp_spec["resolutions"]:
            for wiggle in exp_spec["wiggles"]:
                save_name = _save_name(
                    exp_spec["name"],
                    exp_spec["algo"],
                    resolution,
                    wiggle,
                    exp_spec["seed"],
                )
                run_record = {
                    "experiment": exp_spec["name"],
                    "algo": exp_spec["algo"],
                    "resolution": resolution,
                    "wiggle": wiggle,
                    "seed": exp_spec["seed"],
                    "case_index": exp_spec["case_index"],
                    "save_name": save_name,
                }
                manifest["runs"].append(run_record)

                if args.plot_only:
                    continue

                cmd = [
                    sys.executable,
                    "-m",
                    exp_spec["module"],
                    "--config",
                    exp_spec["config"],
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    exp_spec["algo"],
                    "--save_name",
                    save_name,
                    "--mesh_type",
                    "perturbed_quads",
                    "--perturb_wiggle",
                    str(wiggle),
                    "--perturb_seed",
                    str(exp_spec["seed"]),
                    "--perturb_fix_boundary",
                    "1",
                    "--case_indices",
                    str(exp_spec["case_index"]),
                    exp_spec["num_arg"],
                    str(exp_spec["num_default"]),
                ]

                if args.dry_run:
                    print(" ".join(cmd))
                    continue

                if args.skip_existing and _run_exists(save_name, exp_spec["case_index"]):
                    print(
                        f"[SKIP] {exp_spec['name']} {exp_spec['algo']} "
                        f"N={int(round(resolution * 100))} wiggle={wiggle}"
                    )
                    continue

                log_path = log_dir / f"{save_name}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    raise RuntimeError(
                        f"{exp_spec['name']} {exp_spec['algo']} N={resolution} w={wiggle} failed; see {log_path}"
                    )

        if args.dry_run:
            continue

        out_path = summary_dir / f"{exp_spec['name']}_resolution_cartesian_vs_perturbed.png"
        _generate_figure(exp_spec, out_path)
        manifest["summary_plots"][exp_spec["name"]] = str(out_path)
        print(f"[summary] {exp_spec['name']}: {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
