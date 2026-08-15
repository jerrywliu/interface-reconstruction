#!/usr/bin/env python3
"""Build the frozen six-method Cartesian circle comparison."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.baselines.build_ellipse_all_method_comparison import (
    METHODS,
    _read_csv,
    _tracked_worktree_is_clean,
    _write_csv,
    _write_manifest,
    assemble_case_metrics,
    plot_summary,
    summarize_case_metrics,
    validate_study_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE_CASES = Path(
    "experiments/baselines/results/"
    "common_native_metric_circle_25case_20260814/case_results.csv"
)
DEFAULT_OURS_CASES = Path(
    "experiments/baselines/results/"
    "circle_circular_variants_joint_c0_25case_20260814/case_results.csv"
)
DEFAULT_BASELINE_CASES = (
    Path(
        "experiments/baselines/results/"
        "plvira_circle_25case_20260814/case_results.csv"
    ),
    Path(
        "experiments/baselines/results/"
        "pcic_center_circle_25case_20260814/case_results.csv"
    ),
    Path(
        "experiments/baselines/results/" "quasi_circle_25case_20260814/case_results.csv"
    ),
)
DEFAULT_OUTPUT = Path(
    "experiments/baselines/results/circle_all_method_25case_comparison_20260814"
)
DEFAULT_REPORT = Path("docs/baselines/CIRCLE_25CASE_HIGH_ORDER_COMPARISON.md")
DEFAULT_RESOLUTIONS = (32, 50, 64, 100, 128, 150, 256, 300)
DEFAULT_CASE_INDICES = tuple(range(25))
DEFAULT_METHOD_IDS = (
    "ours_per_cell",
    "ours_graph",
    "ours_c0",
    "plvira",
    "pcic_center",
    "quasi",
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _format(value: float) -> str:
    return f"{value:.6e}" if math.isfinite(value) else "n/a"


def _format_order(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else "n/a"


def _write_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    ours_cases: Path,
    baseline_cases: Sequence[Path],
) -> None:
    fine_resolution = max(int(row["cells_per_side"]) for row in summary)
    fine_rows = {
        str(row["method_id"]): row
        for row in summary
        if int(row["cells_per_side"]) == fine_resolution
    }
    order_rows = {
        str(row["method_id"]): row
        for row in summary
        if int(row["cells_per_side"])
        == min(int(item["cells_per_side"]) for item in summary)
    }
    c0_source = [
        row
        for row in _read_csv(ours_cases)
        if row["variant"] == "graph-coordinated circular + joint C0"
    ]
    c0_components = sum(int(row["num_c0_joint_components"]) for row in c0_source)
    c0_solved = sum(int(row["num_c0_joint_components_solved"]) for row in c0_source)
    c0_failed = sum(int(row["num_c0_joint_components_failed"]) for row in c0_source)
    c0_bad_after = sum(int(row["num_c0_bad_joins_after_joint"]) for row in c0_source)
    finite_c0_residuals = [
        float(row["max_c0_relative_area_residual"])
        for row in c0_source
        if row.get("max_c0_relative_area_residual") not in (None, "", "nan", "NaN")
    ]
    c0_residual = max(finite_c0_residuals, default=0.0)

    quasi_source = []
    for source in baseline_cases:
        quasi_source.extend(
            row
            for row in _read_csv(source)
            if row.get("method") == "QUASI" and row.get("benchmark") == "circles"
        )
    quasi_converged = sum(
        str(row.get("sweep_converged", "")).lower() in {"1", "true"}
        for row in quasi_source
    )
    maximum_conservation = max(
        float(row["normalized_conservation_residual"]) for row in rows
    )

    lines = [
        "# Frozen Six-Method Cartesian Circle Comparison",
        "",
        "This study compares six frozen methods on exactly 25 canonical radius-10 "
        "circle placements over uniform Cartesian meshes with `w=0` and "
        "`N={32,50,64,100,128,150,256,300}`. No external-method policy was changed "
        "after outcomes were inspected.",
        "",
        "## Protocol",
        "",
        "- Project methods: per-cell circular, graph-coordinated circular, and "
        "graph-coordinated circular plus the production joint C0 pass.",
        "- External methods: operational PLVIRA, bare PCIC with center translation, "
        "and the frozen Cartesian QUASI reproduction.",
        "- Common observables: partition-invariant native symmetric Hausdorff, "
        "arc-length-weighted unsigned geometric-curvature MAE, facet gap, "
        "mixed-cell coverage/status, and normalized conservation residual.",
        f"- Support QA: `{len(rows)}/1200` unique finite case rows and "
        f"`{len(summary)}/48` summary rows.",
        "",
        f"## Results At N={fine_resolution}",
        "",
        "| Method | Hausdorff median | Curvature MAE median | Facet-gap median | Coverage | Unresolved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = fine_rows[method["id"]]
        lines.append(
            "| {label} | {hausdorff} | {curvature} | {gap} | {coverage:.3f}% | "
            "{unresolved} |".format(
                label=method["label"],
                hausdorff=_format(float(row["native_symmetric_hausdorff_median"])),
                curvature=_format(
                    float(row["geometric_curvature_mean_absolute_error_median"])
                ),
                gap=_format(float(row["facet_gap_median"])),
                coverage=100.0 * float(row["reconstruction_coverage"]),
                unresolved=int(row["unresolved_cells"]),
            )
        )
    lines.extend(
        [
            "",
            "## Fitted Median Orders",
            "",
            "| Method | Hausdorff | Curvature | Facet gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in methods:
        row = order_rows[method["id"]]
        lines.append(
            "| {label} | {hausdorff} | {curvature} | {gap} |".format(
                label=method["label"],
                hausdorff=_format_order(
                    float(row["native_symmetric_hausdorff_fit_order"])
                ),
                curvature=_format_order(
                    float(row["geometric_curvature_mean_absolute_error_fit_order"])
                ),
                gap=_format_order(float(row["facet_gap_fit_order"])),
            )
        )
    lines.extend(
        [
            "",
            "## Status And Qualifications",
            "",
            f"The production joint C0 pass solves `{c0_solved}/{c0_components}` "
            f"recorded components, with `{c0_failed}` failed components and "
            f"`{c0_bad_after}` remaining eligible bad joins. Its maximum recorded "
            f"relative C0 area residual is `{c0_residual:.3e}`.",
            "",
            "PLVIRA retains its frozen Cartesian boundary-halo and GHF policies. "
            "PCIC uses only the preselected center-translation conservation correction; "
            "radius adjustment is excluded. QUASI retains the frozen ten-sweep "
            f"stopping and fallback rules; `{quasi_converged}/{len(quasi_source)}` "
            "circle cases report sweep convergence.",
            "",
            f"The maximum fitted-group normalized conservation residual over all "
            f"methods and cases is `{maximum_conservation:.3e}`. Exact-zero facet gaps "
            "are retained in CSV and shown at a labeled plotting floor only in the "
            "log-scale figure.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=. python -m experiments.baselines.build_circle_all_method_comparison",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-cases", type=Path, default=DEFAULT_NATIVE_CASES)
    parser.add_argument("--ours-cases", type=Path, default=DEFAULT_OURS_CASES)
    parser.add_argument(
        "--baseline-cases",
        nargs="+",
        type=Path,
        default=list(DEFAULT_BASELINE_CASES),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--resolutions", type=_parse_ints, default=DEFAULT_RESOLUTIONS)
    parser.add_argument(
        "--case-indices", type=_parse_ints, default=DEFAULT_CASE_INDICES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _tracked_worktree_is_clean():
        raise RuntimeError("commit tracked analysis changes before packaging results")
    methods_by_id = {method["id"]: method for method in METHODS}
    methods = tuple(methods_by_id[method_id] for method_id in DEFAULT_METHOD_IDS)
    rows = assemble_case_metrics(
        args.native_cases,
        tuple(args.baseline_cases),
        args.ours_cases,
        methods,
        benchmark="circles",
    )
    validate_study_grid(rows, methods, args.resolutions, args.case_indices)
    summary = summarize_case_metrics(rows, methods)
    expected_rows = len(methods) * len(args.resolutions) * len(args.case_indices)
    if len(rows) != expected_rows or expected_rows != 1200:
        raise ValueError(f"expected exactly 1200 case rows, found {len(rows)}")
    if len(summary) != len(methods) * len(args.resolutions):
        raise ValueError("summary does not contain the exact 48-row method grid")

    output = args.output.resolve()
    report = args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "case_metrics.csv", rows)
    _write_csv(output / "summary.csv", summary)
    plot_summary(
        summary,
        output / "circle_all_methods_metrics.pdf",
        methods,
        benchmark_title="Circle",
        convergence_orders=(3.0, 2.0, 2.0),
    )
    plot_summary(
        summary,
        output / "circle_all_methods_metrics.png",
        methods,
        benchmark_title="Circle",
        convergence_orders=(3.0, 2.0, 2.0),
    )
    _write_report(
        report,
        rows,
        summary,
        methods,
        args.ours_cases,
        tuple(args.baseline_cases),
    )
    _write_manifest(
        output / "manifest.json",
        args=args,
        methods=methods,
        rows=rows,
        summary=summary,
        benchmark="circles",
        artifact_stem="circle_all_methods_metrics",
        analysis_sources=(
            Path(__file__).resolve(),
            REPO_ROOT / "experiments/baselines/build_ellipse_all_method_comparison.py",
        ),
        report_path=report,
    )


if __name__ == "__main__":
    main()
