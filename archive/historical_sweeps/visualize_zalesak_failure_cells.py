#!/usr/bin/env python3
"""
Render local Zalesak diagnostics around specific failing merged cells.

The goal is to make shoulder / corner failure modes concrete by showing:
- local base-cell area fractions
- full / empty / mixed classification
- target merged cell and any oriented left/right support cells
- true interface vs. reconstructed local facet geometry
- direct raw getArcFacet probe outcome when applicable
"""

import argparse
import contextlib
import csv
import io
import json
import math
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import zalesak
from main.geoms.circular_facet import getArcFacet
from main.geoms.geoms import getCentroid
from main.structs.interface_geometry import composite_from_facet
from util.io.slack import load_slack_env, send_results_to_slack


DEFAULT_DEBUG_ROOT = "results/static/debug/zalesak_failure_cells"
DEFAULT_TARGETS = [
    "5:272",
    "5:260",
    "24:266",
    "24:248",
]

STATUS_COLORS = {
    "empty": "#f7f7f7",
    "mixed": "#f4a261",
    "full": "#4d4d4d",
}


@dataclass(frozen=True)
class TargetSpec:
    case_index: int
    merge_id: int


def _parse_target(raw_value: str) -> TargetSpec:
    case_part, merge_part = raw_value.split(":")
    return TargetSpec(case_index=int(case_part), merge_id=int(merge_part))


def _parse_targets(raw_values):
    return [_parse_target(value) for value in raw_values]


def _make_save_name(exp_name, algo, resolution, wiggle, seed):
    res_tag = str(resolution).replace(".", "p")
    wiggle_tag = str(wiggle).replace(".", "p")
    algo_tag = algo.lower().replace("+", "plus")
    return f"debug_{exp_name}_{algo_tag}_r{res_tag}_w{wiggle_tag}_s{seed}"


def _target_label(target_spec):
    return f"case{target_spec.case_index}_merge{target_spec.merge_id}"


def _status_name(poly):
    if poly.isEmpty():
        return "empty"
    if poly.isFull():
        return "full"
    return "mixed"


def _probe_arc_fit(target_poly, timeout_seconds=10.0):
    if not target_poly.fullyOriented():
        return {
            "status": "not_fully_oriented",
            "stdout": "",
            "result": None,
            "exception": None,
        }

    probe_stdout = io.StringIO()

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"probe timed out after {timeout_seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            with contextlib.redirect_stdout(probe_stdout):
                result = getArcFacet(
                    target_poly.getLeftNeighbor().points,
                    target_poly.points,
                    target_poly.getRightNeighbor().points,
                    target_poly.getLeftNeighbor().getFraction(),
                    target_poly.getFraction(),
                    target_poly.getRightNeighbor().getFraction(),
                    1e-10,
                )
            stdout_text = probe_stdout.getvalue()
            if result == (None, None, None):
                if "Max timesteps reached" in stdout_text:
                    status = "getArcFacet_none_max_timesteps"
                elif "Error in getArcFacet" in stdout_text:
                    status = "getArcFacet_none_error"
                else:
                    status = "getArcFacet_none"
                return {
                    "status": status,
                    "stdout": stdout_text,
                    "result": None,
                    "exception": None,
                }
            return {
                "status": "getArcFacet_success",
                "stdout": stdout_text,
                "result": result,
                "exception": None,
            }
        except Exception as error:  # noqa: BLE001
            return {
                "status": "getArcFacet_exception",
                "stdout": probe_stdout.getvalue(),
                "result": None,
                "exception": repr(error),
            }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _sample_points_for_facet(facet, sample_spacing, xlim=None, ylim=None):
    points = []
    if facet is None:
        return points
    for primitive in composite_from_facet(facet).primitives:
        for point in primitive.sample_by_max_spacing(sample_spacing):
            if xlim is not None and not (xlim[0] <= point[0] <= xlim[1]):
                continue
            if ylim is not None and not (ylim[0] <= point[1] <= ylim[1]):
                continue
            points.append([float(point[0]), float(point[1])])
    return points


def _collect_support_merge_ids(mesh, poly_to_id, target_poly):
    support_ids = []
    for support_poly in [target_poly.getLeftNeighbor(), target_poly.getRightNeighbor()]:
        if support_poly is None:
            support_ids.append(None)
        else:
            support_ids.append(poly_to_id.get(id(support_poly)))
    return support_ids


def _normalize_merge_coords(raw_coords):
    if not raw_coords:
        return []
    first = raw_coords[0]
    if isinstance(first, (int, np.integer, float)):
        return [list(raw_coords)]
    return [list(coord) for coord in raw_coords]


def _window_bounds(mesh, merge_ids, pad):
    xs = []
    ys = []
    for merge_id in merge_ids:
        if merge_id is None:
            continue
        for x, y in _normalize_merge_coords(mesh.merge_ids_to_coords[merge_id]):
            xs.append(x)
            ys.append(y)
    x0 = max(min(xs) - pad, 0)
    x1 = min(max(xs) + pad, len(mesh.polys) - 1)
    y0 = max(min(ys) - pad, 0)
    y1 = min(max(ys) + pad, len(mesh.polys[0]) - 1)
    return x0, x1, y0, y1


def _axis_limits(mesh, x0, x1, y0, y1, resolution):
    verts = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            verts.extend(mesh.polys[x][y].points)
    xs = [point[0] for point in verts]
    ys = [point[1] for point in verts]
    margin = 0.4 * resolution
    return (min(xs) - margin, max(xs) + margin), (min(ys) - margin, max(ys) + margin)


def _draw_cell_background(ax, mesh, x0, x1, y0, y1, font_size=7):
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            poly = mesh.polys[x][y]
            status = _status_name(poly)
            facecolor = STATUS_COLORS[status]
            patch = MplPolygon(
                poly.points,
                closed=True,
                facecolor=facecolor,
                edgecolor="#bdbdbd",
                linewidth=0.75,
                alpha=0.95,
            )
            ax.add_patch(patch)
            centroid = getCentroid(poly.points)
            text_color = "white" if status == "full" else "black"
            ax.text(
                centroid[0],
                centroid[1],
                f"{poly.getFraction():.3f}",
                ha="center",
                va="center",
                fontsize=font_size,
                color=text_color,
            )


def _draw_merge_outline(ax, poly, color, linewidth, linestyle="-", label=None):
    if poly is None:
        return
    points = np.asarray(poly.points, dtype=float)
    closed = np.vstack([points, points[0]])
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


def _draw_local_interface(ax, case_record, target_poly, sample_spacing, xlim, ylim):
    for facet in case_record["true_facets"]:
        pts = _sample_points_for_facet(facet, sample_spacing, xlim=xlim, ylim=ylim)
        if pts:
            arr = np.asarray(pts, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="black", linewidth=2.0, linestyle="--")

    for facet in case_record["reconstructed_facets"]:
        pts = _sample_points_for_facet(facet, sample_spacing, xlim=xlim, ylim=ylim)
        if pts:
            arr = np.asarray(pts, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="#6baed6", linewidth=1.0, alpha=0.5)

    if target_poly.getFacet() is not None:
        pts = _sample_points_for_facet(
            target_poly.getFacet(), sample_spacing, xlim=xlim, ylim=ylim
        )
        if pts:
            arr = np.asarray(pts, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="#d62728", linewidth=2.5)


def _summarize_target(case_record, target_spec, sample_spacing):
    mesh = case_record["mesh"]
    target_poly = mesh.merged_polys[target_spec.merge_id]
    poly_to_id = {id(poly): merge_id for merge_id, poly in mesh.merged_polys.items()}
    left_id, right_id = _collect_support_merge_ids(mesh, poly_to_id, target_poly)
    arc_probe = _probe_arc_fit(target_poly)
    coords = _normalize_merge_coords(mesh.merge_ids_to_coords[target_spec.merge_id])
    x0, x1, y0, y1 = _window_bounds(
        mesh,
        [target_spec.merge_id, left_id, right_id],
        pad=2,
    )
    return {
        "case_index": target_spec.case_index,
        "merge_id": target_spec.merge_id,
        "merge_coords": coords,
        "fraction": float(target_poly.getFraction()),
        "facet_name": target_poly.getFacet().name if target_poly.getFacet() is not None else None,
        "fully_oriented": bool(target_poly.fullyOriented()),
        "has_3x3_stencil": bool(target_poly.has3x3Stencil()),
        "left_merge_id": left_id,
        "right_merge_id": right_id,
        "left_fraction": (
            float(target_poly.getLeftNeighbor().getFraction())
            if target_poly.getLeftNeighbor() is not None
            else None
        ),
        "right_fraction": (
            float(target_poly.getRightNeighbor().getFraction())
            if target_poly.getRightNeighbor() is not None
            else None
        ),
        "window": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "probe": arc_probe,
        "sample_spacing": sample_spacing,
    }


def _plot_target_row(ax_left, ax_right, case_record, target_summary, resolution):
    mesh = case_record["mesh"]
    target_poly = mesh.merged_polys[target_summary["merge_id"]]
    left_poly = (
        mesh.merged_polys[target_summary["left_merge_id"]]
        if target_summary["left_merge_id"] is not None
        else None
    )
    right_poly = (
        mesh.merged_polys[target_summary["right_merge_id"]]
        if target_summary["right_merge_id"] is not None
        else None
    )
    x0 = target_summary["window"]["x0"]
    x1 = target_summary["window"]["x1"]
    y0 = target_summary["window"]["y0"]
    y1 = target_summary["window"]["y1"]
    xlim, ylim = _axis_limits(mesh, x0, x1, y0, y1, resolution)

    for ax in [ax_left, ax_right]:
        _draw_cell_background(ax, mesh, x0, x1, y0, y1)
        _draw_merge_outline(ax, left_poly, color="#1f77b4", linewidth=2.0)
        _draw_merge_outline(ax, right_poly, color="#2ca02c", linewidth=2.0)
        _draw_merge_outline(ax, target_poly, color="#d62728", linewidth=2.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    _draw_local_interface(
        ax_right,
        case_record=case_record,
        target_poly=target_poly,
        sample_spacing=target_summary["sample_spacing"],
        xlim=xlim,
        ylim=ylim,
    )

    probe = target_summary["probe"]
    probe_bits = [probe["status"]]
    if probe["exception"] is not None:
        probe_bits.append(probe["exception"])
    elif probe["stdout"]:
        stdout_line = probe["stdout"].strip().splitlines()[0]
        probe_bits.append(stdout_line)
    probe_text = " | ".join(probe_bits)
    coords_text = ",".join(f"({x},{y})" for x, y in target_summary["merge_coords"])

    ax_left.set_title(
        f"case {target_summary['case_index']} | merge {target_summary['merge_id']} | "
        f"facet={target_summary['facet_name']}\n"
        f"coords={coords_text}",
        fontsize=10,
        loc="left",
    )
    ax_right.set_title(
        f"fraction={target_summary['fraction']:.6f} | "
        f"fullyOriented={target_summary['fully_oriented']} | "
        f"left={target_summary['left_fraction']} | right={target_summary['right_fraction']}\n"
        f"{probe_text}",
        fontsize=10,
        loc="left",
    )


def _write_summary_csv(path, summaries):
    fields = [
        "case_index",
        "merge_id",
        "merge_coords",
        "fraction",
        "facet_name",
        "fully_oriented",
        "has_3x3_stencil",
        "left_merge_id",
        "right_merge_id",
        "left_fraction",
        "right_fraction",
        "probe_status",
        "probe_exception",
        "probe_stdout_first_line",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "case_index": summary["case_index"],
                    "merge_id": summary["merge_id"],
                    "merge_coords": json.dumps(summary["merge_coords"]),
                    "fraction": summary["fraction"],
                    "facet_name": summary["facet_name"],
                    "fully_oriented": summary["fully_oriented"],
                    "has_3x3_stencil": summary["has_3x3_stencil"],
                    "left_merge_id": summary["left_merge_id"],
                    "right_merge_id": summary["right_merge_id"],
                    "left_fraction": summary["left_fraction"],
                    "right_fraction": summary["right_fraction"],
                    "probe_status": summary["probe"]["status"],
                    "probe_exception": summary["probe"]["exception"],
                    "probe_stdout_first_line": (
                        summary["probe"]["stdout"].strip().splitlines()[0]
                        if summary["probe"]["stdout"].strip()
                        else ""
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Visualize local Zalesak failure cells.")
    parser.add_argument("--config", default="static/zalesak", help="config path without .yaml")
    parser.add_argument("--facet-algo", default="circular+corner")
    parser.add_argument("--resolution", type=float, default=1.5)
    parser.add_argument("--wiggle", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=25)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="target specs as case_index:merge_id")
    parser.add_argument("--sample-spacing", type=float, default=0.05)
    parser.add_argument("--mesh-type", default="perturbed_quads")
    parser.add_argument("--perturb-fix-boundary", type=int, choices=[0, 1], default=1)
    parser.add_argument("--debug-root", default=DEFAULT_DEBUG_ROOT)
    parser.add_argument("--debug-save-name", default=None)
    parser.add_argument("--notify", action="store_true", help="send the artifact bundle to Slack")
    args = parser.parse_args()

    targets = _parse_targets(args.targets)
    target_case_indices = sorted({target.case_index for target in targets})
    debug_root = Path(args.debug_root).resolve()
    debug_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_save_name = (
        args.debug_save_name
        or f"zalesak_failure_cells_r{str(args.resolution).replace('.', 'p')}_w{str(args.wiggle).replace('.', 'p')}_{stamp}"
    )
    artifact_dir = debug_root / debug_save_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    replay_save_name = _make_save_name(
        "zalesak_failure_cells",
        args.facet_algo,
        args.resolution,
        args.wiggle,
        args.seed,
    )
    _, _, _, case_records = zalesak.main(
        config_setting=args.config,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=replay_save_name,
        num_cases=args.num_cases,
        mesh_type=args.mesh_type,
        perturb_wiggle=args.wiggle,
        perturb_seed=args.seed,
        perturb_fix_boundary=bool(args.perturb_fix_boundary),
        case_indices=target_case_indices,
        return_case_records=True,
    )
    case_record_by_index = {record["case_index"]: record for record in case_records}

    summaries = []
    for target in targets:
        if target.case_index not in case_record_by_index:
            raise RuntimeError(f"Missing replay record for case {target.case_index}")
        summaries.append(
            _summarize_target(
                case_record=case_record_by_index[target.case_index],
                target_spec=target,
                sample_spacing=args.sample_spacing,
            )
        )

    fig, axes = plt.subplots(len(summaries), 2, figsize=(16, 4.6 * len(summaries)))
    if len(summaries) == 1:
        axes = np.asarray([axes])
    for row_index, summary in enumerate(summaries):
        case_record = case_record_by_index[summary["case_index"]]
        _plot_target_row(
            ax_left=axes[row_index, 0],
            ax_right=axes[row_index, 1],
            case_record=case_record,
            target_summary=summary,
            resolution=args.resolution,
        )

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["empty"], edgecolor="#bdbdbd", label="empty"),
        Patch(facecolor=STATUS_COLORS["mixed"], edgecolor="#bdbdbd", label="mixed"),
        Patch(facecolor=STATUS_COLORS["full"], edgecolor="#bdbdbd", label="full"),
        Line2D([0], [0], color="#d62728", lw=2.6, label="target merged cell"),
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="left support"),
        Line2D([0], [0], color="#2ca02c", lw=2.0, label="right support"),
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="true interface"),
        Line2D([0], [0], color="#6baed6", lw=1.0, label="local reconstructed interface"),
        Line2D([0], [0], color="#d62728", lw=2.5, label="target facet"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(
        "Zalesak local failure diagnostics\n"
        f"algo={args.facet_algo}, resolution={args.resolution}, wiggle={args.wiggle}, seed={args.seed}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = artifact_dir / "zalesak_failure_cells.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    summary_csv = artifact_dir / "zalesak_failure_cells_summary.csv"
    _write_summary_csv(summary_csv, summaries)
    summary_json = artifact_dir / "zalesak_failure_cells_summary.json"
    summary_json.write_text(json.dumps({"targets": summaries}, indent=2), encoding="utf-8")

    print(f"Artifact dir: {artifact_dir}")
    print(f"Plot: {plot_path}")
    print(f"Summary CSV: {summary_csv}")

    if args.notify:
        load_slack_env()
        message = (
            "Zalesak moderate failure diagnostics: local fraction/status views for "
            f"algo={args.facet_algo}, r={args.resolution}, w={args.wiggle}, seed={args.seed}"
        )
        ok = send_results_to_slack(message, [str(plot_path), str(summary_csv)])
        print(f"Slack send: {'ok' if ok else 'failed'}")


if __name__ == "__main__":
    main()
