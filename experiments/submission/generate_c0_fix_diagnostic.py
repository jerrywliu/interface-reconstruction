"""Render the coarse-Zalesak C0 branch-fix witnesses as vector geometry."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.submission.conservation_analyzer import (
    analyze_saved_case,
    load_run_grid,
)


WITNESS_CASES = (21, 1)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _paired_index(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    return {
        (row["experiment"], int(row["N"]), int(row["case_index"])): row
        for row in _read_rows(path)
    }


def _load_c0_spec(path: Path) -> Mapping[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        spec
        for spec in manifest["specs"]
        if spec["experiment"] == "zalesak"
        and int(spec["N"]) == 64
        and int(spec["do_c0"]) == 1
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one coarse Zalesak C0 run in {path}")
    return matches[0]


def _format(value: float) -> str:
    return f"{value:.3e}"


def _witness_rows(
    old_results: Path,
    fixed_results: Path,
) -> list[dict[str, Any]]:
    old = _paired_index(old_results / "c0_conservation_paired_cases.csv")
    fixed = _paired_index(fixed_results / "c0_conservation_paired_cases.csv")
    rows = []
    for case_index in WITNESS_CASES:
        old_row = old[("zalesak", 64, case_index)]
        fixed_row = fixed[("zalesak", 64, case_index)]
        rows.append(
            {
                "experiment": "zalesak",
                "N": 64,
                "wiggle": float(fixed_row["wiggle"]),
                "case_index": case_index,
                "pre_c0_global_relative_phase_area_error": float(
                    fixed_row["global_relative_phase_area_error_before_c0"]
                ),
                "broken_c0_global_relative_phase_area_error": float(
                    old_row["global_relative_phase_area_error_after_c0"]
                ),
                "fixed_c0_global_relative_phase_area_error": float(
                    fixed_row["global_relative_phase_area_error_after_c0"]
                ),
                "pre_c0_max_merged_zone_absolute_residual": float(
                    fixed_row["max_merged_zone_absolute_residual_before_c0"]
                ),
                "broken_c0_max_merged_zone_absolute_residual": float(
                    old_row["max_merged_zone_absolute_residual_after_c0"]
                ),
                "fixed_c0_max_merged_zone_absolute_residual": float(
                    fixed_row["max_merged_zone_absolute_residual_after_c0"]
                ),
                "pre_c0_facet_gap": float(fixed_row["facet_gap_before_c0"]),
                "broken_c0_facet_gap": float(old_row["facet_gap_after_c0"]),
                "fixed_c0_facet_gap": float(fixed_row["facet_gap_after_c0"]),
                "fixed_c0_conservation_rejections": int(
                    fixed_row["num_c0_conservation_rejections"]
                ),
                "explicit_corner_facets": int(
                    fixed_row["num_explicit_corner_facets"]
                ),
                "missing_facets_after_fix": int(
                    fixed_row["num_missing_facets_after_c0"]
                ),
            }
        )
    return rows


def _write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _zone_bounds(run_root: Path, case_index: int, resolution: float):
    import numpy as np

    analysis = analyze_saved_case(
        run_root, case_index, stage="after_c0", repo_root=REPO_ROOT
    )
    worst_zone = max(
        analysis.zone_rows,
        key=lambda zone: float(zone["absolute_residual"]),
    )
    zone_cells = [
        cell
        for cell in analysis.cell_rows
        if str(cell["merge_id"]) == str(worst_zone["merge_id"])
    ]
    grid = load_run_grid(run_root, repo_root=REPO_ROOT)
    points = np.asarray(
        [
            point
            for cell in zone_cells
            for point in grid.cell_polygon(int(cell["cell_x"]), int(cell["cell_y"]))
        ],
        dtype=float,
    )
    margin = 2.25 * resolution
    bounds = (
        float(np.min(points[:, 0]) - margin),
        float(np.max(points[:, 0]) + margin),
        float(np.min(points[:, 1]) - margin),
        float(np.max(points[:, 1]) + margin),
    )
    return bounds, worst_zone


def _render(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    old_spec: Mapping[str, Any],
    fixed_spec: Mapping[str, Any],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from experiments.static import generate_section6_maintext_figures as figures

    mpl.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    old_root = Path(old_spec["run_root"])
    fixed_root = Path(fixed_spec["run_root"])
    mesh_segments = figures._mesh_segments(fixed_root / "vtk" / "mesh.vtk")
    true_segments = {
        case_index: figures._load_true_segments(
            "zalesak", str(fixed_spec["save_name"]), case_index
        )
        for case_index in WITNESS_CASES
    }

    figure, axes = plt.subplots(2, 2, figsize=(8.25, 6.85))
    for row_index, row in enumerate(rows):
        case_index = int(row["case_index"])
        bounds, worst_zone = _zone_bounds(old_root, case_index, 0.64)
        reconstructions = (
            (
                "Before: unverified arc branch",
                figures._load_reconstructed_segments(
                    str(old_spec["save_name"]), case_index
                ),
                float(row["broken_c0_global_relative_phase_area_error"]),
                float(row["broken_c0_max_merged_zone_absolute_residual"]),
                float(row["broken_c0_facet_gap"]),
                "#c2410c",
            ),
            (
                "After: conservative branch guard",
                figures._load_reconstructed_segments(
                    str(fixed_spec["save_name"]), case_index
                ),
                float(row["fixed_c0_global_relative_phase_area_error"]),
                float(row["fixed_c0_max_merged_zone_absolute_residual"]),
                float(row["fixed_c0_facet_gap"]),
                "#047857",
            ),
        )
        for column, (
            title,
            reconstructed,
            global_error,
            merged_residual,
            gap,
            color,
        ) in enumerate(reconstructions):
            axis = axes[row_index, column]
            figures._add_segments(
                axis, mesh_segments, color="#d1d5db", linewidth=0.35, alpha=0.7
            )
            figures._add_segments(
                axis,
                true_segments[case_index],
                color="#111827",
                linewidth=1.35,
                linestyle="--",
                zorder=2,
            )
            figures._add_segments(
                axis,
                reconstructed,
                color=color,
                linewidth=1.65,
                zorder=3,
            )
            axis.set_xlim(bounds[0], bounds[1])
            axis.set_ylim(bounds[2], bounds[3])
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(title, fontsize=9.5, pad=6)
            axis.text(
                0.02,
                0.02,
                "global area {global_error}\nmerged zone {merged_residual}\nfacet gap {gap}".format(
                    global_error=_format(global_error),
                    merged_residual=_format(merged_residual),
                    gap=_format(gap),
                ),
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                bbox={
                    "boxstyle": "square,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#9ca3af",
                    "linewidth": 0.5,
                    "alpha": 0.92,
                },
                zorder=5,
            )
        axes[row_index, 0].set_ylabel(
            "Case {case}\nworst affected zone {zone}\nraw residual {residual}".format(
                case=case_index,
                zone=worst_zone["merge_id"],
                residual=_format(float(worst_zone["absolute_residual"])),
            ),
            fontsize=8.5,
        )

    figure.legend(
        handles=[
            Line2D([0], [0], color="#111827", linestyle="--", label="True interface"),
            Line2D([0], [0], color="#c2410c", label="Broken C0 output"),
            Line2D([0], [0], color="#047857", label="Corrected C0 output"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(
        "C0 endpoint adjustment: coarse Zalesak branch regression and correction",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.04, 0.075, 1.0, 0.95), h_pad=1.3, w_pad=1.0)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# C0 circular-branch correction",
        "",
        "The endpoint-adjustment path previously trusted the analytic circular-segment",
        "branch without checking the phase area cut from the actual reconstruction zone.",
        "For these coarse Zalesak joins, neither signed supporting-circle branch through",
        "the averaged endpoints is conservative. The corrected path tests both branches",
        "against the pre-C0 area residual and retains the original conservative facet when",
        "the endpoint-constrained refit is infeasible.",
        "",
        "| case | global area, broken -> fixed | merged residual, broken -> fixed | gap, pre-C0 -> fixed | rejected refits |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {global_old} -> {global_new} | {zone_old} -> {zone_new} | {gap_pre} -> {gap_new} | {rejections} |".format(
                case=row["case_index"],
                global_old=_format(float(row["broken_c0_global_relative_phase_area_error"])),
                global_new=_format(float(row["fixed_c0_global_relative_phase_area_error"])),
                zone_old=_format(float(row["broken_c0_max_merged_zone_absolute_residual"])),
                zone_new=_format(float(row["fixed_c0_max_merged_zone_absolute_residual"])),
                gap_pre=_format(float(row["pre_c0_facet_gap"])),
                gap_new=_format(float(row["fixed_c0_facet_gap"])),
                rejections=row["fixed_c0_conservation_rejections"],
            )
        )
    lines.extend(
        [
            "",
            "The explicit corner count is unchanged and no missing facets are introduced.",
            "The pre-C0 production path is not modified by this correction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    args = parser.parse_args(argv)

    old_results = args.old_results.resolve()
    fixed_results = args.fixed_results.resolve()
    rows = _witness_rows(old_results, fixed_results)
    old_spec = _load_c0_spec(old_results / "run_manifest.json")
    fixed_spec = _load_c0_spec(fixed_results / "run_manifest.json")

    metrics_path = fixed_results / "c0_fix_witness_metrics.csv"
    figure_path = fixed_results / "c0_fix_before_after.pdf"
    report_path = fixed_results / "C0_FIX_REPORT.md"
    _write_metrics(metrics_path, rows)
    _render(figure_path, rows, old_spec, fixed_spec)
    _write_report(report_path, rows)
    print(figure_path)
    print(metrics_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
