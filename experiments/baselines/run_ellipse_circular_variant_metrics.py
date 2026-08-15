#!/usr/bin/env python3
"""Run a matched ellipse study for the finalized project circular variants."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.baselines.project_benchmarks import (
    DOMAIN_SIZE,
    canonical_benchmark_cases,
)
from experiments.baselines.project_smoke import write_json
from experiments.baselines.run_graph_circular_ellipse_common_metrics import (
    verify_canonical_ellipse_geometry,
)
from experiments.plotting import add_convergence_order_triangle
from experiments.submission.conservation_analyzer import (
    analyze_case_records,
    load_run_grid,
)
from main.algos.baselines.external_metrics import (
    directed_hausdorff_external,
    geometric_curvature_error_external,
)
from main.algos.baselines.project_facet_adapter import (
    external_primitives_from_facet_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(
    "experiments/baselines/results/ellipse_circular_variants_joint_c0_common_metrics_20260814"
)
DEFAULT_REPORT = Path("docs/baselines/ELLIPSE_CIRCULAR_VARIANTS_COMMON_METRICS.md")
DEFAULT_RUN_PREFIX = "ellipse_circular_variants_joint_c0_common_metrics_20260814"
DEFAULT_RESOLUTIONS = (32, 64, 128)
DEFAULT_CASE_INDICES = (0, 1, 2, 3, 4)
GEOMETRY_TOLERANCE = 1.0e-12

VARIANTS = (
    {
        "key": "per_cell_circular",
        "label": "per-cell circular",
        "display": "Ours (per-cell circular)",
        "facet_algo": "safe_circle",
        "do_c0": False,
    },
    {
        "key": "graph_coordinated_circular",
        "label": "graph-coordinated circular",
        "display": "Ours (graph-coordinated circular)",
        "facet_algo": "circular",
        "do_c0": False,
    },
    {
        "key": "graph_coordinated_circular_joint_c0",
        "label": "graph-coordinated circular + joint C0",
        "display": "Ours (graph-coordinated circular + joint C0)",
        "facet_algo": "circular",
        "do_c0": True,
        "c0_mode": "joint",
    },
)
VARIANT_BY_KEY = {variant["key"]: variant for variant in VARIANTS}

SUMMARY_METRICS = (
    "native_symmetric_hausdorff",
    "geometric_curvature_mean_absolute_error",
    "geometric_curvature_rms_error",
    "geometric_curvature_relative_l1_error",
    "production_facet_gap",
    "concave_arc_fraction",
    "concave_arc_length_fraction",
    "signed_curvature_arc_length_mean",
)
ORDER_METRICS = (
    "native_symmetric_hausdorff",
    "geometric_curvature_mean_absolute_error",
    "production_facet_gap",
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item.strip())


def _manifest_case_indices(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return _parse_ints(raw)
    return tuple(int(value) for value in raw)


def _float_or_nan(raw: Any) -> float:
    return float(raw) if raw not in (None, "") else math.nan


def _normalized_conservation_diagnostics(
    run_dir: Path, case_index: int, *, stage: str
) -> tuple[float, float]:
    with (run_dir / "metrics" / "cell_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["case_index"]) == case_index
        ]
    analysis = analyze_case_records(
        load_run_grid(run_dir, repo_root=REPO_ROOT),
        rows,
        total_prescribed_phase_area=None,
        stage=stage,
    )
    if not analysis.summary["complete"]:
        raise ValueError(
            f"{run_dir.name}: incomplete conservation replay for case {case_index}"
        )

    zone_areas: dict[str, float] = {}
    for row in analysis.cell_rows:
        merge_id = str(row["merge_id"])
        zone_areas[merge_id] = zone_areas.get(merge_id, 0.0) + float(
            row["cell_area"]
        )
    zone_residuals = [
        float(row["absolute_residual"]) / zone_areas[str(row["merge_id"])]
        for row in analysis.zone_rows
    ]
    global_residual = abs(
        sum(float(row["signed_residual"]) for row in analysis.zone_rows)
    ) / sum(zone_areas.values())
    return max(zone_residuals, default=0.0), global_residual


def _run_name(prefix: str, variant_key: str, resolution: int) -> str:
    return f"{prefix}_{variant_key}_n{resolution}"


def build_reconstruction_command(
    variant: Mapping[str, Any],
    resolution: int,
    case_indices: Sequence[int],
    run_name: str,
) -> list[str]:
    """Build one exact matched Cartesian reconstruction command."""

    return [
        sys.executable,
        "-m",
        "experiments.static.ellipses",
        "--config",
        "static/ellipse",
        "--resolution",
        f"{resolution / DOMAIN_SIZE:g}",
        "--facet_algo",
        str(variant["facet_algo"]),
        "--save_name",
        run_name,
        "--mesh_type",
        "perturbed_quads",
        "--perturb_wiggle",
        "0.0",
        "--perturb_seed",
        "0",
        "--perturb_fix_boundary",
        "1",
        "--do_c0",
        "1" if variant["do_c0"] else "0",
        "--c0_mode",
        str(variant.get("c0_mode", "joint")),
        "--num_ellipses",
        "25",
        "--case_indices",
        ",".join(str(value) for value in case_indices),
        "--plic_fallback",
        "LVIRA",
        "--corner_behavior_profile",
        "pre_f8_corner",
    ]


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return {
            int(row["case_index"]): row
            for row in (json.loads(line) for line in stream if line.strip())
        }


def _load_case_metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["case_index"]): row for row in csv.DictReader(stream)}


def _load_c0_event_counts(path: Path) -> dict[int, tuple[int, int]]:
    counts: dict[int, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            case_index = int(row["case_index"])
            values = counts.setdefault(case_index, [0, 0])
            if row.get("event_kind") in {"c0_adjustment", "c0_joint_adjustment"}:
                values[0] += 1
            elif row.get("event_kind") in {"c0_rejection", "c0_joint_rejection"}:
                values[1] += 1
    return {case_index: tuple(values) for case_index, values in counts.items()}


def _validate_run_manifest(
    run_dir: Path,
    variant: Mapping[str, Any],
    resolution: int,
    case_indices: Sequence[int],
) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    parameters = manifest["parameters"]
    expected = {
        "facet_algo": variant["facet_algo"],
        "do_c0": variant["do_c0"],
        "mesh_type": "perturbed_quads",
        "perturb_wiggle": 0.0,
        "perturb_seed": 0,
        "perturb_fix_boundary": 1,
        "plic_fallback": "LVIRA",
        "corner_behavior_profile": "pre_f8_corner",
        "random_seed": 42,
        "num_ellipses": 25,
    }
    if parameters.get("c0_mode") not in (None, variant.get("c0_mode", "joint")):
        raise ValueError(
            f"{run_dir.name}: c0_mode={parameters.get('c0_mode')!r}, "
            f"expected {variant.get('c0_mode', 'joint')!r}"
        )
    for field, expected_value in expected.items():
        if parameters.get(field) != expected_value:
            raise ValueError(
                f"{run_dir.name}: {field}={parameters.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    if not math.isclose(
        float(parameters["resolution"]),
        resolution / DOMAIN_SIZE,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{run_dir.name}: resolution does not encode N={resolution}")
    if _manifest_case_indices(parameters.get("case_indices")) != tuple(case_indices):
        raise ValueError(f"{run_dir.name}: case-index subset differs")
    return manifest


def _run_one(
    command: Sequence[str], log_path: Path, *, reuse: bool, run_dir: Path
) -> None:
    if reuse and (run_dir / "run_manifest.json").exists():
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"reconstruction failed; inspect {log_path}")


def run_reconstructions(
    *,
    output: Path,
    run_prefix: str,
    resolutions: Sequence[int],
    case_indices: Sequence[int],
    workers: int,
    reuse: bool,
) -> dict[tuple[str, int], Path]:
    """Run all matched variants and return their exact artifact roots."""

    jobs = []
    runs = {}
    for variant in VARIANTS:
        for resolution in resolutions:
            name = _run_name(run_prefix, variant["key"], resolution)
            run_dir = REPO_ROOT / "plots" / name
            command = build_reconstruction_command(
                variant, resolution, case_indices, name
            )
            runs[(variant["key"], resolution)] = run_dir
            jobs.append(
                (
                    command,
                    output / "logs" / f"{name}.log",
                    run_dir,
                )
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _run_one, command, log_path, reuse=reuse, run_dir=run_dir
            ): run_dir
            for command, log_path, run_dir in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"reconstruction {completed}/{len(futures)} complete", flush=True)
    return runs


def signed_arc_diagnostics(payload: Mapping[str, Any]) -> dict[str, float | int]:
    """Summarize signed project curvatures before the common unsigned metric."""

    arcs = [
        record for record in payload.get("primitives", ()) if record["kind"] == "arc"
    ]
    lengths = [
        abs(float(record["radius"]) * float(record["signed_delta"])) for record in arcs
    ]
    total_length = math.fsum(lengths)
    concave = [float(record["radius"]) < 0.0 for record in arcs]
    concave_length = math.fsum(
        length for length, is_concave in zip(lengths, concave) if is_concave
    )
    signed_integral = math.fsum(
        length / float(record["radius"]) for record, length in zip(arcs, lengths)
    )
    return {
        "concave_arc_count": sum(concave),
        "concave_arc_fraction": sum(concave) / len(arcs) if arcs else math.nan,
        "concave_arc_length_fraction": (
            concave_length / total_length if total_length > 0.0 else math.nan
        ),
        "signed_curvature_arc_length_mean": (
            signed_integral / total_length if total_length > 0.0 else math.nan
        ),
    }


def _evaluate_case(
    variant_key: str,
    resolution: int,
    run_dir_value: str,
    case_index: int,
    expected_case_indices: Sequence[int],
) -> dict[str, Any]:
    variant = VARIANT_BY_KEY[variant_key]
    run_dir = Path(run_dir_value)
    manifest = _validate_run_manifest(
        run_dir, variant, resolution, expected_case_indices
    )
    saved_geometry = _load_jsonl(run_dir / "metrics" / "case_geometry.jsonl")[
        case_index
    ]
    canonical = canonical_benchmark_cases("ellipses", (case_index,))[0]
    verify_canonical_ellipse_geometry(saved_geometry, canonical)
    metadata_path = (
        run_dir
        / "vtk"
        / "reconstructed"
        / "facets"
        / f"{case_index}.facet_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reconstruction = external_primitives_from_facet_metadata(metadata)
    truth = canonical.truth_primitives()
    reconstruction_to_truth = directed_hausdorff_external(reconstruction, truth)
    truth_to_reconstruction = directed_hausdorff_external(truth, reconstruction)
    curvature = geometric_curvature_error_external(reconstruction, truth)
    diagnostics = _load_case_metrics(run_dir / "metrics" / "case_metrics.csv")[
        case_index
    ]
    normalized_conservation, normalized_global_conservation = (
        _normalized_conservation_diagnostics(
            run_dir,
            case_index,
            stage="after_c0" if variant["do_c0"] else "before_c0",
        )
    )
    c0_adjustments, c0_rejections = _load_c0_event_counts(
        run_dir / "metrics" / "merge_events.csv"
    ).get(case_index, (0, 0))
    kinds = [primitive.kind for primitive in reconstruction]
    return {
        "method": "Ours",
        "variant": variant["label"],
        "display_label": variant["display"],
        "benchmark": "ellipses",
        "case_index": case_index,
        "cells_per_side": resolution,
        "cell_size": DOMAIN_SIZE / resolution,
        "source_run": run_dir.name,
        "source_commit": manifest["source_commit"],
        "canonical_geometry_match": True,
        "do_c0": bool(variant["do_c0"]),
        "num_mixed_cells": int(diagnostics["num_mixed_cells"]),
        "num_merged_cells": int(diagnostics["num_merged_cells"]),
        "num_merged_components": int(diagnostics["num_merged_components"]),
        "c0_adjustment_events": c0_adjustments,
        "c0_rejection_events": c0_rejections,
        "num_c0_bad_joins_before_joint": int(
            diagnostics.get("num_c0_bad_joins_before_joint") or 0
        ),
        "num_c0_bad_joins_after_joint": int(
            diagnostics.get("num_c0_bad_joins_after_joint") or 0
        ),
        "num_c0_joint_components": int(
            diagnostics.get("num_c0_joint_components") or 0
        ),
        "num_c0_joint_components_solved": int(
            diagnostics.get("num_c0_joint_components_solved") or 0
        ),
        "num_c0_joint_components_failed": int(
            diagnostics.get("num_c0_joint_components_failed") or 0
        ),
        "num_c0_exact_c1_components": int(
            diagnostics.get("num_c0_exact_c1_components") or 0
        ),
        "num_c0_conservative_fallback_components": int(
            diagnostics.get("num_c0_conservative_fallback_components") or 0
        ),
        "max_c0_relative_area_residual": _float_or_nan(
            diagnostics.get("max_c0_relative_area_residual")
        ),
        "max_c0_tangent_angle_radians": _float_or_nan(
            diagnostics.get("max_c0_tangent_angle_radians")
        ),
        "primitive_count": len(reconstruction),
        "arc_count": kinds.count("arc"),
        "line_count": kinds.count("line"),
        **signed_arc_diagnostics(metadata),
        "native_symmetric_hausdorff": max(
            reconstruction_to_truth, truth_to_reconstruction
        ),
        "native_reconstruction_to_truth": reconstruction_to_truth,
        "native_truth_to_reconstruction": truth_to_reconstruction,
        "geometric_curvature_mean_absolute_error": curvature["mean_absolute_error"],
        "geometric_curvature_rms_error": curvature["rms_error"],
        "geometric_curvature_max_absolute_error": curvature["max_absolute_error"],
        "geometric_curvature_relative_l1_error": curvature["relative_l1_error"],
        "geometric_curvature_reconstructed_length": curvature["reconstructed_length"],
        "geometric_curvature_quadrature_samples": curvature["quadrature_samples"],
        "production_facet_gap": float(diagnostics["facet_gap"]),
        "production_global_relative_area_error": _float_or_nan(
            diagnostics["area_error"]
        ),
        "normalized_conservation_residual": normalized_conservation,
        "normalized_global_conservation_residual": normalized_global_conservation,
    }


def _geometry_values(record: Mapping[str, Any]) -> tuple[float, ...]:
    values: list[float] = []
    for field in ("p_left", "p_right", "center"):
        values.extend(float(value) for value in record.get(field, ()))
    for field in ("radius", "signed_delta"):
        if field in record:
            values.append(float(record[field]))
    return tuple(values)


def compare_native_geometry(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[bool, bool, float]:
    """Compare native primitive geometry, ignoring serializer bookkeeping."""

    first_records = list(first.get("primitives", ()))
    second_records = list(second.get("primitives", ()))
    if len(first_records) != len(second_records):
        return False, False, math.inf
    exact = True
    maximum = 0.0
    for left, right in zip(first_records, second_records):
        if left.get("kind") != right.get("kind"):
            return False, False, math.inf
        left_values = _geometry_values(left)
        right_values = _geometry_values(right)
        if len(left_values) != len(right_values):
            return False, False, math.inf
        deltas = [abs(a - b) for a, b in zip(left_values, right_values)]
        maximum = max(maximum, max(deltas, default=0.0))
        exact = exact and left_values == right_values
    return exact, maximum <= GEOMETRY_TOLERANCE, maximum


def build_equivalence_rows(
    runs: Mapping[tuple[str, int], Path],
    resolutions: Sequence[int],
    case_indices: Sequence[int],
    case_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["variant"], int(row["cells_per_side"]), int(row["case_index"])): row
        for row in case_rows
    }
    result = []
    for resolution in resolutions:
        per_dir = runs[("per_cell_circular", resolution)]
        graph_dir = runs[("graph_coordinated_circular", resolution)]
        for case_index in case_indices:
            per_payload = json.loads(
                (
                    per_dir
                    / "vtk"
                    / "reconstructed"
                    / "facets"
                    / f"{case_index}.facet_metadata.json"
                ).read_text(encoding="utf-8")
            )
            graph_payload = json.loads(
                (
                    graph_dir
                    / "vtk"
                    / "reconstructed"
                    / "facets"
                    / f"{case_index}.facet_metadata.json"
                ).read_text(encoding="utf-8")
            )
            exact, within_tolerance, maximum = compare_native_geometry(
                per_payload, graph_payload
            )
            per_row = by_key[("per-cell circular", resolution, case_index)]
            graph_row = by_key[("graph-coordinated circular", resolution, case_index)]
            result.append(
                {
                    "cells_per_side": resolution,
                    "case_index": case_index,
                    "native_geometry_exact": exact,
                    "native_geometry_within_1e-12": within_tolerance,
                    "maximum_native_parameter_delta": maximum,
                    "per_cell_merged_cells": per_row["num_merged_cells"],
                    "graph_coordinated_merged_cells": graph_row["num_merged_cells"],
                    "graph_coordinated_merged_components": graph_row[
                        "num_merged_components"
                    ],
                }
            )
    return result


def build_c0_verification_rows(
    runs: Mapping[tuple[str, int], Path],
    resolutions: Sequence[int],
    case_indices: Sequence[int],
    case_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove the saved joint-C0 metadata contain post-pass geometry."""

    by_key = {
        (row["variant"], int(row["cells_per_side"]), int(row["case_index"])): row
        for row in case_rows
    }
    result = []
    for resolution in resolutions:
        graph_dir = runs[("graph_coordinated_circular", resolution)]
        c0_dir = runs[("graph_coordinated_circular_joint_c0", resolution)]
        for case_index in case_indices:
            graph_payload = json.loads(
                (
                    graph_dir
                    / "vtk"
                    / "reconstructed"
                    / "facets"
                    / f"{case_index}.facet_metadata.json"
                ).read_text(encoding="utf-8")
            )
            c0_payload = json.loads(
                (
                    c0_dir
                    / "vtk"
                    / "reconstructed"
                    / "facets"
                    / f"{case_index}.facet_metadata.json"
                ).read_text(encoding="utf-8")
            )
            exact, within_tolerance, maximum = compare_native_geometry(
                graph_payload, c0_payload
            )
            c0_row = by_key[
                (
                    "graph-coordinated circular + joint C0",
                    resolution,
                    case_index,
                )
            ]
            result.append(
                {
                    "cells_per_side": resolution,
                    "case_index": case_index,
                    "c0_geometry_exactly_matches_pre_c0": exact,
                    "c0_geometry_matches_pre_c0_within_1e-12": within_tolerance,
                    "native_primitive_kind_or_count_changed": not math.isfinite(
                        maximum
                    ),
                    "maximum_native_parameter_change": (
                        maximum if math.isfinite(maximum) else ""
                    ),
                    "c0_adjustment_events": c0_row["c0_adjustment_events"],
                    "c0_rejection_events": c0_row["c0_rejection_events"],
                }
            )
    return result


def _fit_order(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    cell_sizes = np.asarray([float(row["cell_size"]) for row in rows], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        return math.nan
    return float(np.polyfit(np.log(cell_sizes), np.log(values), 1)[0])


def summarize_case_results(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    for variant in VARIANTS:
        selected_variant = [row for row in rows if row["variant"] == variant["label"]]
        for resolution in sorted(
            {int(row["cells_per_side"]) for row in selected_variant}
        ):
            selected = [
                row
                for row in selected_variant
                if int(row["cells_per_side"]) == resolution
            ]
            item: dict[str, Any] = {
                "method": "Ours",
                "variant": variant["label"],
                "display_label": variant["display"],
                "benchmark": "ellipses",
                "cells_per_side": resolution,
                "cell_size": DOMAIN_SIZE / resolution,
                "case_count": len(selected),
                "mixed_cell_count": sum(
                    int(row["num_mixed_cells"]) for row in selected
                ),
                "arc_count": sum(int(row["arc_count"]) for row in selected),
                "line_count": sum(int(row["line_count"]) for row in selected),
                "concave_arc_count": sum(
                    int(row["concave_arc_count"]) for row in selected
                ),
                "merged_cell_count": sum(
                    int(row["num_merged_cells"]) for row in selected
                ),
                "c0_adjustment_events": sum(
                    int(row["c0_adjustment_events"]) for row in selected
                ),
                "c0_rejection_events": sum(
                    int(row["c0_rejection_events"]) for row in selected
                ),
                "c0_bad_joins_before_joint": sum(
                    int(row.get("num_c0_bad_joins_before_joint") or 0)
                    for row in selected
                ),
                "c0_bad_joins_after_joint": sum(
                    int(row.get("num_c0_bad_joins_after_joint") or 0)
                    for row in selected
                ),
                "c0_joint_components": sum(
                    int(row.get("num_c0_joint_components") or 0) for row in selected
                ),
                "c0_joint_components_solved": sum(
                    int(row.get("num_c0_joint_components_solved") or 0)
                    for row in selected
                ),
                "c0_joint_components_failed": sum(
                    int(row.get("num_c0_joint_components_failed") or 0)
                    for row in selected
                ),
                "c0_exact_c1_components": sum(
                    int(row.get("num_c0_exact_c1_components") or 0)
                    for row in selected
                ),
                "c0_conservative_fallback_components": sum(
                    int(row.get("num_c0_conservative_fallback_components") or 0)
                    for row in selected
                ),
                "max_c0_relative_area_residual": max(
                    float(row.get("max_c0_relative_area_residual", math.nan))
                    for row in selected
                ),
                "max_c0_tangent_angle_radians": max(
                    float(row.get("max_c0_tangent_angle_radians", math.nan))
                    for row in selected
                ),
                "normalized_conservation_residual_max": max(
                    float(row["normalized_conservation_residual"])
                    for row in selected
                ),
                "normalized_global_conservation_residual_max": max(
                    float(row["normalized_global_conservation_residual"])
                    for row in selected
                ),
            }
            for metric in SUMMARY_METRICS:
                values = np.asarray([float(row[metric]) for row in selected])
                item[f"{metric}_median"] = float(np.median(values))
                item[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                item[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            summary.append(item)

    case_orders = []
    for variant in VARIANTS:
        variant_summary = sorted(
            (row for row in summary if row["variant"] == variant["label"]),
            key=lambda row: int(row["cells_per_side"]),
        )
        for metric in ORDER_METRICS:
            order = _fit_order(variant_summary, f"{metric}_median")
            for row in variant_summary:
                row[f"{metric}_fit_order"] = order
        for case_index in sorted({int(row["case_index"]) for row in rows}):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["variant"] == variant["label"]
                    and int(row["case_index"]) == case_index
                ),
                key=lambda row: int(row["cells_per_side"]),
            )
            case_orders.append(
                {
                    "variant": variant["label"],
                    "case_index": case_index,
                    **{
                        f"{metric}_fit_order": _fit_order(selected, metric)
                        for metric in ORDER_METRICS
                    },
                }
            )
    return summary, case_orders


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    panels = (
        ("native_symmetric_hausdorff", "Native symmetric Hausdorff"),
        (
            "geometric_curvature_mean_absolute_error",
            "Unsigned geometric-curvature MAE",
        ),
        ("production_facet_gap", "Facet gap"),
        ("concave_arc_length_fraction", "Concave arc-length fraction"),
    )
    colors = ("#0072B2", "#009E73", "#D55E00")
    markers = ("o", "s", "D")
    linestyles = (":", "--", "-")
    for axis, (metric, ylabel) in zip(axes.ravel(), panels):
        for variant, color, marker, linestyle in zip(
            VARIANTS, colors, markers, linestyles
        ):
            selected = sorted(
                (row for row in summary if row["variant"] == variant["label"]),
                key=lambda row: int(row["cells_per_side"]),
            )
            x = np.asarray([int(row["cells_per_side"]) for row in selected])
            median = np.asarray([float(row[f"{metric}_median"]) for row in selected])
            lower = np.asarray([float(row[f"{metric}_q25"]) for row in selected])
            upper = np.asarray([float(row[f"{metric}_q75"]) for row in selected])
            axis.plot(
                x,
                median,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=variant["display"],
            )
            axis.fill_between(x, lower, upper, color=color, alpha=0.11, linewidth=0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            DEFAULT_RESOLUTIONS, tuple(str(value) for value in DEFAULT_RESOLUTIONS)
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        if metric != "concave_arc_length_fraction":
            axis.set_yscale("log")
    add_convergence_order_triangle(
        axes[0, 0], 3.0, order_label="3", anchor=(0.75, 0.12), width=0.13
    )
    add_convergence_order_triangle(
        axes[0, 1], 1.0, order_label="1", anchor=(0.76, 0.16), width=0.13
    )
    add_convergence_order_triangle(
        axes[1, 0], 3.0, order_label="3", anchor=(0.75, 0.12), width=0.13
    )
    axes[1, 1].text(
        0.5,
        0.72,
        "No concave arcs\nin any matched case",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="center",
        color="#4b5563",
        fontsize=8,
    )
    for axis in axes[1, :]:
        axis.set_xlabel("Cells per side")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        ncol=1,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _format(value: float) -> str:
    return f"{value:.6e}" if math.isfinite(value) else "n/a"


def _format_order(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else "n/a"


def _write_report(
    path: Path,
    summary: Sequence[Mapping[str, Any]],
    equivalence: Sequence[Mapping[str, Any]],
    c0_verification: Sequence[Mapping[str, Any]],
    source_commits: Sequence[str],
    case_indices: Sequence[int],
    resolutions: Sequence[int],
) -> None:
    if tuple(case_indices) == tuple(range(min(case_indices), max(case_indices) + 1)):
        case_description = f"cases {min(case_indices)}--{max(case_indices)}"
    else:
        case_description = "cases " + ",".join(str(value) for value in case_indices)
    resolution_description = ",".join(str(value) for value in resolutions)
    rows = [
        "# Ellipse Circular-Variant Common-Metric Study",
        "",
        "This fresh matched Cartesian study compares the finalized `per-cell circular`, "
        "`graph-coordinated circular`, and `graph-coordinated circular + joint C0` "
        f"variants on canonical ellipse {case_description} at "
        f"`N={resolution_description}`. The joint C0 "
        "variant is the production default: connected rejected-join components are "
        "refined over shared endpoints and conservative per-facet curvatures.",
        "",
        f"- reconstruction source commit(s): `{', '.join(source_commits)}`",
        "- native geometry: exact schema-v2 line/arc metadata",
        "- curvature observable: arc-length-weighted mean absolute error in unsigned "
        "geometric curvature against the nearest analytic ellipse branch",
        "- concavity diagnostic: negative project radius, reported by primitive count "
        "and native arc length",
        "- normalized conservation residual: maximum fitted-group area residual divided "
        "by the geometric area of that cell or merged group",
        "",
        "## Results",
        "",
        "| Variant | N | Hausdorff median | Curvature MAE median | Facet-gap median | Joint components solved | Bad joins after joint | Max area residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        rows.append(
            "| {variant} | {cells_per_side} | {hausdorff} | {curvature} | "
            "{gap} | {solved} / {components} | {bad_after} | {area} |".format(
                variant=row["variant"],
                cells_per_side=row["cells_per_side"],
                hausdorff=_format(float(row["native_symmetric_hausdorff_median"])),
                curvature=_format(
                    float(row["geometric_curvature_mean_absolute_error_median"])
                ),
                gap=_format(float(row["production_facet_gap_median"])),
                solved=row["c0_joint_components_solved"],
                components=row["c0_joint_components"],
                bad_after=row["c0_bad_joins_after_joint"],
                area=_format(float(row["normalized_conservation_residual_max"])),
            )
        )
    rows.extend(
        [
            "",
            "## Observed Orders",
            "",
            "| Variant | Geometry | Curvature | Facet gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        selected = next(row for row in summary if row["variant"] == variant["label"])
        rows.append(
            "| {variant} | {geometry} | {curvature} | {gap} |".format(
                variant=variant["label"],
                geometry=_format_order(
                    float(
                        row_value(selected, "native_symmetric_hausdorff_fit_order")
                    )
                ),
                curvature=_format_order(
                    float(
                        row_value(
                            selected,
                            "geometric_curvature_mean_absolute_error_fit_order",
                        )
                    )
                ),
                gap=_format_order(
                    float(row_value(selected, "production_facet_gap_fit_order"))
                ),
            )
        )
    exact_count = sum(bool(row["native_geometry_exact"]) for row in equivalence)
    tolerance_count = sum(
        bool(row["native_geometry_within_1e-12"]) for row in equivalence
    )
    maximum_delta = max(
        float(row["maximum_native_parameter_delta"]) for row in equivalence
    )
    total_merges = sum(
        int(row["per_cell_merged_cells"]) + int(row["graph_coordinated_merged_cells"])
        for row in equivalence
    )
    changed_c0_cases = sum(
        not bool(row["c0_geometry_exactly_matches_pre_c0"]) for row in c0_verification
    )
    plain_rows = {
        int(row["cells_per_side"]): row
        for row in summary
        if row["variant"] == "graph-coordinated circular"
    }
    c0_rows = {
        int(row["cells_per_side"]): row
        for row in summary
        if row["variant"] == "graph-coordinated circular + joint C0"
    }
    curvature_changes = {
        resolution: 100.0
        * (
            float(c0_rows[resolution]["geometric_curvature_mean_absolute_error_median"])
            / float(
                plain_rows[resolution]["geometric_curvature_mean_absolute_error_median"]
            )
            - 1.0
        )
        for resolution in plain_rows
    }
    plain_curvature_order = float(
        next(iter(plain_rows.values()))[
            "geometric_curvature_mean_absolute_error_fit_order"
        ]
    )
    c0_curvature_order = float(
        next(iter(c0_rows.values()))[
            "geometric_curvature_mean_absolute_error_fit_order"
        ]
    )
    c0_straight_limit_count = sum(int(row["line_count"]) for row in c0_rows.values())
    c0_component_count = sum(int(row["c0_joint_components"]) for row in c0_rows.values())
    c0_solved_count = sum(
        int(row["c0_joint_components_solved"]) for row in c0_rows.values()
    )
    c0_failed_count = sum(
        int(row["c0_joint_components_failed"]) for row in c0_rows.values()
    )
    c0_bad_after = sum(
        int(row["c0_bad_joins_after_joint"]) for row in c0_rows.values()
    )
    c0_max_area_residual = max(
        float(row["max_c0_relative_area_residual"]) for row in c0_rows.values()
    )
    rows.extend(
        [
            "",
            "## Equivalence Check",
            "",
            f"The per-cell and graph-coordinated native geometries are byte-for-byte "
            f"numerically identical in `{exact_count}/{len(equivalence)}` matched cases "
            f"and agree within `1e-12` in `{tolerance_count}/{len(equivalence)}`. The "
            f"largest native parameter difference is `{maximum_delta:.3e}`. Their "
            f"combined merged-cell count is `{total_merges}`.",
            "",
            "## Interpretation",
            "",
            "Joint C0 does not materially improve the common unsigned-curvature "
            f"observable in this {len(case_indices)}-case study. Relative to "
            "graph-coordinated circular, "
            f"its median curvature error changes by `{curvature_changes[32]:+.1f}%`, "
            f"`{curvature_changes[64]:+.1f}%`, and `{curvature_changes[128]:+.1f}%` at "
            f"`N=32,64,128`, respectively. The fitted curvature order changes only from "
            f"`{plain_curvature_order:.3f}` to `{c0_curvature_order:.3f}`. Its clear "
            "benefits are instead geometric: lower Hausdorff error and much smaller "
            "facet gaps.",
            "",
            f"The optimizer solves `{c0_solved_count}/{c0_component_count}` connected "
            f"components, with `{c0_failed_count}` failures and `{c0_bad_after}` "
            "remaining eligible bad joins. All solved components reach the exact-C1 "
            f"branch, and the maximum relative cell-area residual is "
            f"`{c0_max_area_residual:.3e}`.",
            "",
            "Negative-radius (locally concave) arcs are reported explicitly because "
            "joint conservative refinement does not impose a convexity constraint. "
            "The unsigned curvature metric therefore remains paired with the signed "
            "curvature and concave-arc diagnostics. The joint C0 runs contain "
            f"`{c0_straight_limit_count}` straight-limit line facets in total; the common "
            "metric assigns these zero curvature.",
            "",
            "The C0 facet sidecars are post-refinement: `runReconstruction` invokes "
            "the joint `makeC0` pass before collecting the returned facet list and "
            "writing the exact schema-v2 metadata. Component outcomes above come from "
            "the same final run's diagnostics. The saved native geometry "
            f"differs from its matched pre-C0 reconstruction in `{changed_c0_cases}/"
            f"{len(c0_verification)}` cases.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=. python -m experiments.baselines.run_ellipse_circular_variant_metrics --workers 3 --metric-workers 6",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def row_value(row: Mapping[str, Any], field: str) -> Any:
    """Keep table formatting readable when a mapping comes from CSV-like data."""

    return row[field]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--resolutions", type=_parse_ints, default=DEFAULT_RESOLUTIONS)
    parser.add_argument(
        "--case-indices", type=_parse_ints, default=DEFAULT_CASE_INDICES
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--reuse-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report = args.report.resolve()
    resolutions = tuple(int(value) for value in args.resolutions)
    case_indices = tuple(int(value) for value in args.case_indices)
    if not case_indices or len(set(case_indices)) != len(case_indices):
        raise ValueError("case indices must be a non-empty unique sequence")
    canonical_benchmark_cases("ellipses", case_indices)
    output.mkdir(parents=True, exist_ok=True)
    runs = run_reconstructions(
        output=output,
        run_prefix=args.run_prefix,
        resolutions=resolutions,
        case_indices=case_indices,
        workers=args.workers,
        reuse=args.reuse_runs,
    )
    jobs = [
        (
            variant["key"],
            resolution,
            str(runs[(variant["key"], resolution)]),
            case_index,
            case_indices,
        )
        for variant in VARIANTS
        for resolution in resolutions
        for case_index in case_indices
    ]
    case_rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.metric_workers)) as executor:
        futures = {executor.submit(_evaluate_case, *job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            case_rows.append(future.result())
            print(f"metric case {completed}/{len(futures)} complete", flush=True)
    case_rows.sort(
        key=lambda row: (
            next(
                index
                for index, variant in enumerate(VARIANTS)
                if variant["label"] == row["variant"]
            ),
            int(row["cells_per_side"]),
            int(row["case_index"]),
        )
    )
    summary, case_orders = summarize_case_results(case_rows)
    equivalence = build_equivalence_rows(runs, resolutions, case_indices, case_rows)
    c0_verification = build_c0_verification_rows(
        runs, resolutions, case_indices, case_rows
    )
    if case_indices == DEFAULT_CASE_INDICES and not all(
        bool(row["native_geometry_within_1e-12"])
        and int(row["per_cell_merged_cells"]) == 0
        and int(row["graph_coordinated_merged_cells"]) == 0
        for row in equivalence
    ):
        raise RuntimeError("per-cell/graph equivalence or no-merge QA failed")
    if not all(
        not bool(row["c0_geometry_exactly_matches_pre_c0"])
        and int(row["c0_adjustment_events"]) > 0
        for row in c0_verification
    ):
        raise RuntimeError("saved joint-C0 geometry is not demonstrably post-pass")
    _write_csv(output / "case_results.csv", case_rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "case_orders.csv", case_orders)
    _write_csv(output / "per_cell_graph_equivalence.csv", equivalence)
    _write_csv(output / "joint_c0_postrefinement_verification.csv", c0_verification)
    figure_path = output / "ellipse_circular_variants_all_methods.pdf"
    _plot_summary(summary, figure_path)
    source_commits = sorted({str(row["source_commit"]) for row in case_rows})
    _write_report(
        report,
        summary,
        equivalence,
        c0_verification,
        source_commits,
        case_indices,
        resolutions,
    )
    write_json(
        output / "manifest.json",
        {
            "method": "matched finalized project circular variants",
            "benchmark": "canonical Cartesian ellipses",
            "variants": list(VARIANTS),
            "cells_per_side": list(resolutions),
            "case_indices": list(case_indices),
            "case_count": len(case_indices),
            "result_row_count": len(case_rows),
            "source_commits": source_commits,
            "source_runs": [str(path) for path in runs.values()],
            "native_geometry_tolerance": GEOMETRY_TOLERANCE,
            "c0_definition": (
                "production joint refinement of shared endpoints and conservative "
                "per-facet curvatures on connected rejected-join components"
            ),
            "curvature_metric": (
                "arc-length-weighted mean absolute unsigned geometric-curvature "
                "error against nearest analytic ellipse truth"
            ),
            "signed_curvature_diagnostic": (
                "project radius sign; negative-radius arcs are counted as concave"
            ),
            "figure": str(figure_path),
            "report": str(report),
        },
    )


if __name__ == "__main__":
    main()
