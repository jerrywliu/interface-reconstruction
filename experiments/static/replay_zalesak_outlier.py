#!/usr/bin/env python3
"""
Replay a deterministic Zalesak outlier case and dump primitive-level diagnostics.

This is intended for targeted debugging of `circular+corner` outliers harvested by
`track_zalesak_outliers.py`.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import zalesak
from main.structs.interface import Interface
from main.structs.interface_geometry import ArcPrimitive, composite_from_facet
from util.metrics.metrics import hausdorff_points, point_to_primitives_distance


DEFAULT_OUTLIER_CSV = (
    "results/static/camera_ready/"
    "static_cameraready_zalesak_25case_sharded_20260310_094212/"
    "outliers/zalesak_circularpluscorner_outliers.csv"
)
DEFAULT_DEBUG_ROOT = "results/static/debug/zalesak_outlier_replays"


def _make_save_name(exp_name, algo, resolution, wiggle, seed):
    res_tag = str(resolution).replace(".", "p")
    wiggle_tag = str(wiggle).replace(".", "p")
    algo_tag = algo.lower().replace("+", "plus")
    return f"perturb_sweep_{exp_name}_{algo_tag}_r{res_tag}_w{wiggle_tag}_s{seed}"


def _parse_csv_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    value = value.strip()
    if value == "":
        return value
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_case_spec_from_csv(csv_path, sort_by, row_index, save_name, case_index):
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        rows = [{key: _parse_csv_value(value) for key, value in row.items()} for row in csv.DictReader(fh)]
    if save_name is not None and case_index is not None:
        for row in rows:
            if row.get("save_name") == save_name and int(row.get("case_index")) == int(case_index):
                return row
        raise ValueError(f"Could not find save_name={save_name!r}, case_index={case_index} in {csv_path}")

    filtered_rows = rows
    if save_name is not None:
        filtered_rows = [row for row in filtered_rows if row.get("save_name") == save_name]
    if case_index is not None:
        filtered_rows = [row for row in filtered_rows if int(row.get("case_index")) == int(case_index)]
    if not filtered_rows:
        raise ValueError("No matching outlier rows found in CSV")

    filtered_rows.sort(key=lambda row: float(row.get(sort_by, 0.0)), reverse=True)
    if row_index < 0 or row_index >= len(filtered_rows):
        raise ValueError(
            f"row_index={row_index} out of range for filtered outlier set of size {len(filtered_rows)}"
        )
    return filtered_rows[row_index]


def _resolve_case_spec(args):
    if args.outlier_csv is not None:
        csv_path = Path(args.outlier_csv).resolve()
        row = _load_case_spec_from_csv(
            csv_path=csv_path,
            sort_by=args.sort_by,
            row_index=args.row_index,
            save_name=args.save_name,
            case_index=args.case_index,
        )
        return {
            "source": "csv",
            "csv_path": str(csv_path),
            "save_name": row["save_name"],
            "algo": row["algo"],
            "resolution": float(row["resolution"]),
            "wiggle": float(row["wiggle"]),
            "seed": int(row["seed"]),
            "case_index": int(row["case_index"]),
            "area_error": float(row["area_error"]),
            "facet_gap": float(row["facet_gap"]),
            "hausdorff": float(row["hausdorff"]),
            "plot_dir": row.get("plot_dir"),
        }

    if args.resolution is None or args.wiggle is None or args.seed is None or args.case_index is None:
        raise ValueError(
            "Direct replay requires --resolution, --wiggle, --seed, and --case_index when --outlier-csv is not used"
        )
    algo = args.facet_algo or "circular+corner"
    save_name = args.save_name or _make_save_name("zalesak", algo, args.resolution, args.wiggle, args.seed)
    return {
        "source": "direct",
        "save_name": save_name,
        "algo": algo,
        "resolution": float(args.resolution),
        "wiggle": float(args.wiggle),
        "seed": int(args.seed),
        "case_index": int(args.case_index),
        "area_error": None,
        "facet_gap": None,
        "hausdorff": None,
        "plot_dir": None,
    }


def _primitive_summary(facet_index, primitive_index, primitive, target_primitives, sample_spacing):
    sample_points = primitive.sample_by_max_spacing(sample_spacing)
    distances = [point_to_primitives_distance(point, target_primitives) for point in sample_points]
    summary = {
        "facet_index": facet_index,
        "primitive_index": primitive_index,
        "source_name": primitive.source_name,
        "primitive_type": "arc" if isinstance(primitive, ArcPrimitive) else "line",
        "pLeft": list(primitive.pLeft),
        "pRight": list(primitive.pRight),
        "length": primitive.length(),
        "sample_count": len(sample_points),
        "max_distance_to_target": max(distances) if distances else 0.0,
        "mean_distance_to_target": float(np.mean(distances)) if distances else 0.0,
    }
    if isinstance(primitive, ArcPrimitive):
        summary.update(
            {
                "center": list(primitive.center),
                "radius": primitive.radius,
                "is_major_arc": bool(primitive.is_major_arc),
                "midpoint": list(primitive.midpoint),
            }
        )
    return summary


def _collect_primitive_diagnostics(facets, target_primitives, sample_spacing):
    summaries = []
    for facet_index, facet in enumerate(facets):
        composite = composite_from_facet(facet)
        for primitive_index, primitive in enumerate(composite.primitives):
            summaries.append(
                _primitive_summary(
                    facet_index=facet_index,
                    primitive_index=primitive_index,
                    primitive=primitive,
                    target_primitives=target_primitives,
                    sample_spacing=sample_spacing,
                )
            )
    summaries.sort(key=lambda entry: entry["max_distance_to_target"], reverse=True)
    return summaries


def _plot_primitives(ax, primitives, color, linewidth, sample_spacing, alpha=1.0):
    for primitive in primitives:
        samples = primitive.sample_by_max_spacing(sample_spacing)
        if not samples:
            continue
        pts = np.asarray(samples, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def _write_overlay(path, true_primitives, reconstructed_primitives, worst_summaries, sample_spacing):
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_primitives(ax, true_primitives, color="black", linewidth=2.5, sample_spacing=sample_spacing)
    _plot_primitives(ax, reconstructed_primitives, color="#1f77b4", linewidth=1.5, sample_spacing=sample_spacing, alpha=0.75)

    top_labels = worst_summaries[:3]
    for summary in top_labels:
        if summary["primitive_type"] == "arc":
            point = summary["midpoint"]
        else:
            point = [
                0.5 * (summary["pLeft"][0] + summary["pRight"][0]),
                0.5 * (summary["pLeft"][1] + summary["pRight"][1]),
            ]
        ax.text(
            point[0],
            point[1],
            f"f{summary['facet_index']}:p{summary['primitive_index']}",
            color="red",
            fontsize=9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Canonical Primitive Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _point_distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _sample_points_in_window(primitives, center, radius, sample_spacing):
    points = []
    for primitive in primitives:
        samples = primitive.sample_by_max_spacing(sample_spacing)
        for point in samples:
            if _point_distance(point, center) <= radius:
                points.append([float(point[0]), float(point[1])])
    return points


def _primitive_type_counts_in_window(facets, center, radius, sample_spacing):
    counts = {}
    primitive_summaries = []
    for facet_index, facet in enumerate(facets):
        composite = composite_from_facet(facet)
        for primitive_index, primitive in enumerate(composite.primitives):
            samples = [
                point
                for point in primitive.sample_by_max_spacing(sample_spacing)
                if _point_distance(point, center) <= radius
            ]
            if not samples:
                continue
            key = f"{primitive.source_name}:{'arc' if isinstance(primitive, ArcPrimitive) else 'line'}"
            counts[key] = counts.get(key, 0) + 1
            primitive_summaries.append(
                {
                    "facet_index": facet_index,
                    "primitive_index": primitive_index,
                    "source_name": primitive.source_name,
                    "primitive_type": "arc" if isinstance(primitive, ArcPrimitive) else "line",
                    "sample_count_in_window": len(samples),
                    "pLeft": list(primitive.pLeft),
                    "pRight": list(primitive.pRight),
                }
            )
    return counts, primitive_summaries


def _windowed_gap_stats(mesh, reconstructed_facets, center, radius):
    interface = Interface.from_merge_mesh(
        mesh,
        reconstructed_facets=reconstructed_facets,
        infer_missing_neighbors=False,
    )
    gap_values = []
    for component in interface.components:
        records = component.records
        if len(records) < 2:
            continue
        for i in range(len(records)):
            j = (i + 1) % len(records) if component.is_closed else i + 1
            if j >= len(records):
                continue
            record_i = records[i]
            record_j = records[j]
            endpoint_pairs = [
                (record_i.facet.pLeft, record_j.facet.pLeft),
                (record_i.facet.pLeft, record_j.facet.pRight),
                (record_i.facet.pRight, record_j.facet.pLeft),
                (record_i.facet.pRight, record_j.facet.pRight),
            ]
            best_pair = min(endpoint_pairs, key=lambda pair: _point_distance(pair[0], pair[1]))
            midpoint = [
                0.5 * (best_pair[0][0] + best_pair[1][0]),
                0.5 * (best_pair[0][1] + best_pair[1][1]),
            ]
            if _point_distance(midpoint, center) > radius:
                continue
            gap_values.append(_point_distance(best_pair[0], best_pair[1]))

    if not gap_values:
        return {"count": 0, "mean": 0.0, "max": 0.0, "values": []}
    return {
        "count": len(gap_values),
        "mean": float(np.mean(gap_values)),
        "max": float(np.max(gap_values)),
        "values": [float(value) for value in gap_values],
    }


def _windowed_worst_primitives(facets, target_primitives, center, radius, sample_spacing, limit=5):
    summaries = []
    for facet_index, facet in enumerate(facets):
        composite = composite_from_facet(facet)
        for primitive_index, primitive in enumerate(composite.primitives):
            window_points = [
                point
                for point in primitive.sample_by_max_spacing(sample_spacing)
                if _point_distance(point, center) <= radius
            ]
            if not window_points:
                continue
            distances = [point_to_primitives_distance(point, target_primitives) for point in window_points]
            summaries.append(
                {
                    "facet_index": facet_index,
                    "primitive_index": primitive_index,
                    "source_name": primitive.source_name,
                    "primitive_type": "arc" if isinstance(primitive, ArcPrimitive) else "line",
                    "max_distance_to_target": float(max(distances)),
                    "mean_distance_to_target": float(np.mean(distances)),
                    "pLeft": list(primitive.pLeft),
                    "pRight": list(primitive.pRight),
                }
            )
    summaries.sort(key=lambda entry: entry["max_distance_to_target"], reverse=True)
    return summaries[:limit]


def _window_diagnostic(name, center, radius, true_primitives, reconstructed_facets, reconstructed_primitives, mesh, sample_spacing):
    true_points = _sample_points_in_window(true_primitives, center, radius, sample_spacing)
    reconstructed_points = _sample_points_in_window(
        reconstructed_primitives, center, radius, sample_spacing
    )
    if true_points and reconstructed_points:
        local_hausdorff = float(hausdorff_points(true_points, reconstructed_points))
    else:
        local_hausdorff = float("inf")
    gap_stats = _windowed_gap_stats(mesh, reconstructed_facets, center, radius)
    type_counts, primitive_summaries = _primitive_type_counts_in_window(
        reconstructed_facets, center, radius, sample_spacing
    )
    return {
        "name": name,
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "true_sample_count": len(true_points),
        "reconstructed_sample_count": len(reconstructed_points),
        "local_hausdorff": local_hausdorff,
        "local_gap": gap_stats,
        "reconstructed_primitive_type_counts": type_counts,
        "worst_reconstructed_primitives": _windowed_worst_primitives(
            reconstructed_facets,
            target_primitives=true_primitives,
            center=center,
            radius=radius,
            sample_spacing=sample_spacing,
        ),
        "reconstructed_primitives_in_window": primitive_summaries[:20],
    }


def _aggregate_window_group(name, diagnostics):
    hausdorff_values = [
        entry["local_hausdorff"]
        for entry in diagnostics
        if math.isfinite(entry["local_hausdorff"])
    ]
    gap_values = []
    for entry in diagnostics:
        gap_values.extend(entry["local_gap"]["values"])
    return {
        "name": name,
        "window_count": len(diagnostics),
        "hausdorff_mean": float(np.mean(hausdorff_values)) if hausdorff_values else float("inf"),
        "hausdorff_max": float(max(hausdorff_values)) if hausdorff_values else float("inf"),
        "gap_mean": float(np.mean(gap_values)) if gap_values else 0.0,
        "gap_max": float(max(gap_values)) if gap_values else 0.0,
        "gap_count": len(gap_values),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay a deterministic Zalesak outlier case.")
    parser.add_argument("--config", default="static/zalesak", help="config path without .yaml")
    parser.add_argument("--outlier-csv", default=DEFAULT_OUTLIER_CSV, help="outlier CSV to select from")
    parser.add_argument("--sort-by", default="hausdorff", choices=["hausdorff", "facet_gap", "area_error"])
    parser.add_argument("--row-index", type=int, default=0, help="rank within the sorted outlier CSV subset")
    parser.add_argument("--save_name", default=None, help="exact save_name to replay")
    parser.add_argument("--facet_algo", default=None, help="algorithm for direct replay")
    parser.add_argument("--resolution", type=float, default=None, help="resolution for direct replay")
    parser.add_argument("--wiggle", type=float, default=None, help="wiggle for direct replay")
    parser.add_argument("--seed", type=int, default=None, help="perturb seed for direct replay")
    parser.add_argument("--case_index", type=int, default=None, help="case index to replay")
    parser.add_argument("--num_cases", type=int, default=None, help="optional explicit num_cases upper bound")
    parser.add_argument("--sample_spacing", type=float, default=0.05, help="sampling spacing for diagnostics/overlay")
    parser.add_argument("--debug-root", default=DEFAULT_DEBUG_ROOT, help="root directory for replay artifacts")
    parser.add_argument("--debug-save-name", default=None, help="override replay save_name")
    parser.add_argument("--mesh-type", default="perturbed_quads", help="mesh type override for replay")
    parser.add_argument(
        "--window-radius",
        type=float,
        default=None,
        help="absolute diagnostic window radius around each true Zalesak corner",
    )
    parser.add_argument(
        "--window-radius-multiplier",
        type=float,
        default=3.0,
        help="window radius multiplier applied to resolution when --window-radius is omitted",
    )
    parser.add_argument(
        "--perturb-fix-boundary",
        type=int,
        choices=[0, 1],
        default=1,
        help="fix boundary nodes for perturbed mesh replay",
    )
    args = parser.parse_args()

    case_spec = _resolve_case_spec(args)
    debug_root = Path(args.debug_root).resolve()
    replay_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_tag = args.debug_save_name or f"replay_{case_spec['save_name']}_case{case_spec['case_index']}_{replay_stamp}"
    artifact_dir = debug_root / replay_tag
    artifact_dir.mkdir(parents=True, exist_ok=True)

    num_cases = args.num_cases if args.num_cases is not None else case_spec["case_index"] + 1
    num_cases = max(num_cases, case_spec["case_index"] + 1)

    area_errors, facet_gaps, hausdorff_values, case_records = zalesak.main(
        config_setting=args.config,
        resolution=case_spec["resolution"],
        facet_algo=case_spec["algo"],
        save_name=replay_tag,
        num_cases=num_cases,
        mesh_type=args.mesh_type,
        perturb_wiggle=case_spec["wiggle"],
        perturb_seed=case_spec["seed"],
        perturb_fix_boundary=bool(args.perturb_fix_boundary),
        case_indices=[case_spec["case_index"]],
        return_case_records=True,
    )
    if not case_records:
        raise RuntimeError("Replay completed without returning case records")
    if len(case_records) != 1:
        raise RuntimeError(f"Expected exactly one replayed case record, got {len(case_records)}")

    case_record = case_records[0]
    true_primitives = []
    for facet in case_record["true_facets"]:
        true_primitives.extend(composite_from_facet(facet).primitives)
    reconstructed_primitives = []
    for facet in case_record["reconstructed_facets"]:
        reconstructed_primitives.extend(composite_from_facet(facet).primitives)
    window_radius = (
        float(args.window_radius)
        if args.window_radius is not None
        else float(args.window_radius_multiplier * case_spec["resolution"])
    )

    recon_summaries = _collect_primitive_diagnostics(
        facets=case_record["reconstructed_facets"],
        target_primitives=true_primitives,
        sample_spacing=args.sample_spacing,
    )
    true_summaries = _collect_primitive_diagnostics(
        facets=case_record["true_facets"],
        target_primitives=reconstructed_primitives,
        sample_spacing=args.sample_spacing,
    )

    overlay_path = artifact_dir / "canonical_overlay.png"
    _write_overlay(
        path=overlay_path,
        true_primitives=true_primitives,
        reconstructed_primitives=reconstructed_primitives,
        worst_summaries=recon_summaries,
        sample_spacing=args.sample_spacing,
    )
    landmark_windows = {
        name: _window_diagnostic(
            name=name,
            center=center,
            radius=window_radius,
            true_primitives=true_primitives,
            reconstructed_facets=case_record["reconstructed_facets"],
            reconstructed_primitives=reconstructed_primitives,
            mesh=case_record["mesh"],
            sample_spacing=args.sample_spacing,
        )
        for name, center in case_record["true_landmarks"].items()
    }
    grouped_windows = {
        "linear_linear": _aggregate_window_group(
            "linear_linear",
            [
                landmark_windows["linear_linear_left"],
                landmark_windows["linear_linear_right"],
            ],
        ),
        "line_arc": _aggregate_window_group(
            "line_arc",
            [
                landmark_windows["line_arc_left"],
                landmark_windows["line_arc_right"],
            ],
        ),
    }

    summary = {
        "case_spec": case_spec,
        "replay": {
            "replay_save_name": replay_tag,
            "artifact_dir": str(artifact_dir),
            "plots_dir": str((Path("plots") / replay_tag).resolve()),
            "output_dirs": case_record["output_dirs"],
            "sample_spacing": args.sample_spacing,
            "num_cases": num_cases,
        },
        "metrics": {
            "area_error": float(area_errors[0]),
            "facet_gap": float(facet_gaps[0]),
            "hausdorff": float(hausdorff_values[0]),
        },
        "geometry": {
            "center": case_record["center"],
            "theta": case_record["theta"],
            "slot_rect": case_record["slot_rect"],
            "true_landmarks": case_record["true_landmarks"],
            "true_facet_count": len(case_record["true_facets"]),
            "true_primitive_count": len(true_primitives),
            "reconstructed_facet_count": len(case_record["reconstructed_facets"]),
            "reconstructed_primitive_count": len(reconstructed_primitives),
        },
        "local_windows": landmark_windows,
        "local_groups": grouped_windows,
        "top_reconstructed_primitives": recon_summaries[:10],
        "top_true_primitives": true_summaries[:10],
        "artifacts": {
            "canonical_overlay": str(overlay_path),
        },
    }

    summary_path = artifact_dir / "replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Replayed {case_spec['save_name']} case {case_spec['case_index']}")
    print(f"Metrics: area={summary['metrics']['area_error']:.6e}, gap={summary['metrics']['facet_gap']:.6e}, hausdorff={summary['metrics']['hausdorff']:.6e}")
    if recon_summaries:
        worst = recon_summaries[0]
        print(
            "Worst reconstructed primitive: "
            f"facet {worst['facet_index']} primitive {worst['primitive_index']} "
            f"{worst['primitive_type']} max_dist={worst['max_distance_to_target']:.6e}"
        )
    print(
        "Local groups: "
        f"linear-linear hausdorff={grouped_windows['linear_linear']['hausdorff_mean']:.6e}, "
        f"gap={grouped_windows['linear_linear']['gap_mean']:.6e}; "
        f"line-arc hausdorff={grouped_windows['line_arc']['hausdorff_mean']:.6e}, "
        f"gap={grouped_windows['line_arc']['gap_mean']:.6e}"
    )
    print(f"Summary: {summary_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
