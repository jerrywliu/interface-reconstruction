#!/usr/bin/env python3
"""Run the frozen project circular variants on canonical Cartesian circles."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from experiments.baselines.project_benchmarks import (
    CANONICAL_CASE_COUNT,
    DOMAIN_SIZE,
    ProjectBenchmarkCase,
    canonical_benchmark_cases,
)
from experiments.baselines.project_smoke import write_json
from experiments.baselines.run_ellipse_circular_variant_metrics import (
    VARIANTS,
    VARIANT_BY_KEY,
    _float_or_nan,
    _load_c0_event_counts,
    _load_case_metrics,
    _load_jsonl,
    _manifest_case_indices,
    _normalized_conservation_diagnostics,
    _parse_ints,
    _run_name,
    _run_one,
    _write_csv,
    build_c0_verification_rows,
    build_equivalence_rows,
    signed_arc_diagnostics,
    summarize_case_results,
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
    "experiments/baselines/results/" "circle_circular_variants_joint_c0_25case_20260814"
)
DEFAULT_RUN_PREFIX = "circle_circular_variants_joint_c0_25case_20260814"
DEFAULT_RESOLUTIONS = (32, 50, 64, 100, 128, 150, 256, 300)
DEFAULT_CASE_INDICES = tuple(range(CANONICAL_CASE_COUNT))
GEOMETRY_ABSOLUTE_TOLERANCE = 1.0e-12


def verify_canonical_circle_geometry(
    saved: Mapping[str, Any],
    canonical: ProjectBenchmarkCase,
    *,
    absolute_tolerance: float = GEOMETRY_ABSOLUTE_TOLERANCE,
) -> None:
    """Fail if a saved case is not the exact canonical seed-41 circle case."""

    if saved.get("geometry_type") != "circle":
        raise ValueError("saved geometry is not a circle")
    if int(saved.get("case_index", -1)) != canonical.case_index:
        raise ValueError("saved and canonical case indices differ")
    expected = canonical.parameters
    if not math.isclose(
        float(saved["radius"]),
        float(expected["radius"]),
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(
            f"case {canonical.case_index} radius differs from canonical geometry"
        )
    for axis, (actual, target) in enumerate(zip(saved["center"], expected["center"])):
        if not math.isclose(
            float(actual),
            float(target),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"case {canonical.case_index} center[{axis}] differs from "
                "canonical geometry"
            )


def build_reconstruction_command(
    variant: Mapping[str, Any],
    resolution: int,
    case_indices: Sequence[int],
    run_name: str,
) -> list[str]:
    """Build one exact matched Cartesian circle reconstruction command."""

    return [
        sys.executable,
        "-m",
        "experiments.static.circles",
        "--config",
        "static/circle",
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
        "--num_circles",
        str(CANONICAL_CASE_COUNT),
        "--case_indices",
        ",".join(str(value) for value in case_indices),
        "--radius",
        "10.0",
        "--plic_fallback",
        "LVIRA",
        "--corner_behavior_profile",
        "pre_f8_corner",
    ]


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
        "random_seed": 41,
        "num_circles": CANONICAL_CASE_COUNT,
        "radius": 10.0,
    }
    if manifest.get("experiment") != "circles":
        raise ValueError(f"{run_dir.name}: experiment is not circles")
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


def run_reconstructions(
    *,
    output: Path,
    run_prefix: str,
    resolutions: Sequence[int],
    case_indices: Sequence[int],
    workers: int,
    reuse: bool,
) -> dict[tuple[str, int], Path]:
    """Run all matched project variants and return their artifact roots."""

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
            jobs.append((command, output / "logs" / f"{name}.log", run_dir))

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
    canonical = canonical_benchmark_cases("circles", (case_index,))[0]
    verify_canonical_circle_geometry(saved_geometry, canonical)
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
        "benchmark": "circles",
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
        "num_c0_joint_components": int(diagnostics.get("num_c0_joint_components") or 0),
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


def _validate_case_rows(
    rows: Sequence[Mapping[str, Any]],
    resolutions: Sequence[int],
    case_indices: Sequence[int],
) -> None:
    expected = {
        (variant["label"], resolution, case_index)
        for variant in VARIANTS
        for resolution in resolutions
        for case_index in case_indices
    }
    observed = {
        (str(row["variant"]), int(row["cells_per_side"]), int(row["case_index"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            "project circle rows do not match the exact variant-resolution-case grid"
        )
    finite_fields = (
        "native_symmetric_hausdorff",
        "geometric_curvature_mean_absolute_error",
        "production_facet_gap",
        "normalized_conservation_residual",
        "normalized_global_conservation_residual",
    )
    for row in rows:
        for field in finite_fields:
            value = float(row[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"invalid {field} for {row['variant']} N={row['cells_per_side']} "
                    f"case={row['case_index']}: {value}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    resolutions = tuple(int(value) for value in args.resolutions)
    case_indices = tuple(int(value) for value in args.case_indices)
    if not case_indices or len(set(case_indices)) != len(case_indices):
        raise ValueError("case indices must be a non-empty unique sequence")
    canonical_benchmark_cases("circles", case_indices)
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
            if completed % 25 == 0 or completed == len(futures):
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
    _validate_case_rows(case_rows, resolutions, case_indices)
    summary, case_orders = summarize_case_results(case_rows)
    for row in summary:
        row["benchmark"] = "circles"
    equivalence = build_equivalence_rows(runs, resolutions, case_indices, case_rows)
    c0_verification = build_c0_verification_rows(
        runs, resolutions, case_indices, case_rows
    )
    _write_csv(output / "case_results.csv", case_rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "case_orders.csv", case_orders)
    _write_csv(output / "per_cell_graph_equivalence.csv", equivalence)
    _write_csv(output / "joint_c0_postrefinement_verification.csv", c0_verification)
    source_commits = sorted({str(row["source_commit"]) for row in case_rows})
    write_json(
        output / "manifest.json",
        {
            "method": "matched frozen project circular variants",
            "benchmark": "canonical Cartesian circles",
            "variants": list(VARIANTS),
            "cells_per_side": list(resolutions),
            "case_indices": list(case_indices),
            "case_count": len(case_indices),
            "result_row_count": len(case_rows),
            "source_commits": source_commits,
            "source_runs": [str(path) for path in runs.values()],
            "native_geometry_tolerance": GEOMETRY_ABSOLUTE_TOLERANCE,
            "mesh": "uniform axis-aligned Cartesian; w=0",
            "c0_definition": (
                "production joint refinement of shared endpoints and conservative "
                "per-facet curvatures on connected rejected-join components"
            ),
            "geometry_metric": (
                "partition-insensitive symmetric native point-to-curve supremum"
            ),
            "curvature_metric": (
                "arc-length-weighted mean absolute unsigned geometric-curvature "
                "error against analytic radius-10 circle truth"
            ),
            "artifacts": [
                "case_results.csv",
                "summary.csv",
                "case_orders.csv",
                "per_cell_graph_equivalence.csv",
                "joint_c0_postrefinement_verification.csv",
            ],
        },
    )


if __name__ == "__main__":
    main()
