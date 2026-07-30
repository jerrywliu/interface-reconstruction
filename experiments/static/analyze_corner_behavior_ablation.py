#!/usr/bin/env python3
"""Summarize the f8 corner-behavior ablation and render review artifacts."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.static.generate_section6_maintext_figures import (
    PLOTS_ROOT,
    _compute_view_bounds,
    _load_reconstructed_plot_geometry,
    _load_true_segments,
    _mesh_segments,
    _plot_panel,
    _save_figure,
)


PROFILES = [
    "current",
    "no_orientation_hint",
    "legacy_corner_acceptance",
    "no_corner_branch_propagation",
    "no_hint_legacy_acceptance",
    "no_hint_no_branch_propagation",
    "legacy_acceptance_no_branch_propagation",
    "pre_f8_corner",
]
PROFILE_LABELS = {
    "current": "Current",
    "no_orientation_hint": "No orientation hint",
    "legacy_corner_acceptance": "Legacy acceptance",
    "no_corner_branch_propagation": "No propagation",
    "no_hint_legacy_acceptance": "No hint + legacy acceptance",
    "no_hint_no_branch_propagation": "No hint + no propagation",
    "legacy_acceptance_no_branch_propagation": "Legacy acceptance + no propagation",
    "pre_f8_corner": "Pre-f8 corner logic",
}
SPLIT_PROFILES = [
    "no_locality_guard",
    "legacy_branch_intersection",
    "pre_f8_with_locality_guard",
    "pre_f8_with_both_branch_requirement",
    "pre_f8_corner",
]
PROFILE_LABELS.update(
    {
        "no_locality_guard": "Current without locality guard",
        "legacy_branch_intersection": "Current with legacy branch intersection",
        "pre_f8_with_locality_guard": "Pre-f8 plus locality guard",
        "pre_f8_with_both_branch_requirement": "Pre-f8 plus two-branch requirement",
    }
)
EXPERIMENT_ALGOS = {
    "squares": "linear+corner",
    "zalesak": "circular+corner",
}
RESOLUTIONS = [0.50, 0.64, 1.00, 1.28, 1.50]
WIGGLES = [0.0, 0.05, 0.10, 0.20, 0.30]
FLOOR = 1.0e-6


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _profile_from_row(row: dict[str, str]) -> str:
    profile = row.get("corner_behavior_profile", "")
    if profile:
        return profile
    save_name = row.get("save_name", "")
    for candidate in sorted(PROFILES, key=len, reverse=True):
        if f"_corner_{candidate}" in save_name:
            return candidate
    return ""


def _summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    hausdorff = [float(row["hausdorff"]) for row in rows]
    gaps = [float(row["facet_gap"]) for row in rows]
    return {
        "case_count": len(rows),
        "hausdorff_median": statistics.median(hausdorff),
        "hausdorff_mean": statistics.mean(hausdorff),
        "hausdorff_max": max(hausdorff),
        "facet_gap_median": statistics.median(gaps),
        "hausdorff_floor_cases": sum(value < FLOOR for value in hausdorff),
        "joint_floor_cases": sum(
            h_value < FLOOR and gap_value < FLOOR
            for h_value, gap_value in zip(hausdorff, gaps)
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hard_summary(hard_cases: Path) -> list[dict]:
    rows = _read_csv(hard_cases)
    summary_rows = []
    for experiment in EXPERIMENT_ALGOS:
        for profile in PROFILES:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment
                and _profile_from_row(row) == profile
            ]
            summary_rows.append(
                {
                    "experiment": experiment,
                    "profile": profile,
                    **_summarize(selected),
                }
            )
    return summary_rows


def _split_summary(split_cases: Path) -> list[dict]:
    rows = _read_csv(split_cases)
    summary_rows = []
    for experiment in EXPERIMENT_ALGOS:
        for profile in SPLIT_PROFILES:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment
                and _profile_from_row(row) == profile
            ]
            summary_rows.append(
                {
                    "experiment": experiment,
                    "profile": profile,
                    **_summarize(selected),
                }
            )
    return summary_rows


def _candidate_cases(paths: list[Path]) -> list[dict[str, str]]:
    selected = {}
    for path in paths:
        for row in _read_csv(path):
            if row["experiment"] not in EXPERIMENT_ALGOS:
                continue
            if _profile_from_row(row) != "pre_f8_corner":
                continue
            key = (
                row["experiment"],
                float(row["resolution"]),
                float(row["wiggle"]),
                int(row["case_index"]),
            )
            selected[key] = row
    return list(selected.values())


def _current_cases(path: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(path)
        if row["experiment"] in EXPERIMENT_ALGOS
        and row["algo"] == EXPERIMENT_ALGOS[row["experiment"]]
    ]


def _full_grid_summary(
    current_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]
) -> list[dict]:
    summary_rows = []
    for experiment in EXPERIMENT_ALGOS:
        for resolution in RESOLUTIONS:
            for wiggle in WIGGLES:
                for profile, rows in [
                    ("current", current_rows),
                    ("pre_f8_corner", candidate_rows),
                ]:
                    selected = [
                        row
                        for row in rows
                        if row["experiment"] == experiment
                        and float(row["resolution"]) == resolution
                        and float(row["wiggle"]) == wiggle
                    ]
                    if len(selected) != 25:
                        raise ValueError(
                            f"Expected 25 {experiment} cases for r={resolution}, "
                            f"w={wiggle}, profile={profile}; found {len(selected)}"
                        )
                    summary_rows.append(
                        {
                            "experiment": experiment,
                            "resolution": resolution,
                            "N": int(round(100 * resolution)),
                            "wiggle": wiggle,
                            "profile": profile,
                            **_summarize(selected),
                        }
                    )
    return summary_rows


def _plot_hard_ablation(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True)
    colors = ["#64748b"] * len(PROFILES)
    colors[-1] = "#0f766e"
    for ax, experiment in zip(axes, EXPERIMENT_ALGOS):
        selected = [row for row in rows if row["experiment"] == experiment]
        medians = [float(row["hausdorff_median"]) for row in selected]
        means = [float(row["hausdorff_mean"]) for row in selected]
        positions = np.arange(len(PROFILES))
        ax.bar(positions - 0.18, medians, 0.36, color=colors, label="Median")
        ax.bar(
            positions + 0.18,
            means,
            0.36,
            facecolor="none",
            edgecolor=colors,
            linewidth=1.2,
            label="Mean",
        )
        ax.axhline(FLOOR, color="#b91c1c", linestyle=(0, (3, 2)), linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylim(1.0e-10, 2.0)
        ax.set_xticks(positions, [PROFILE_LABELS[p] for p in PROFILES], rotation=50, ha="right")
        ax.set_title(experiment.title(), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Hausdorff distance")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Corner behavior ablation: N=150, w=0.3, 25 cases", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_figure(fig, out_path)
    plt.close(fig)


def _grid_values(rows: list[dict], experiment: str, profile: str, field: str) -> np.ndarray:
    lookup = {
        (float(row["resolution"]), float(row["wiggle"])): float(row[field])
        for row in rows
        if row["experiment"] == experiment and row["profile"] == profile
    }
    return np.asarray(
        [[lookup[(resolution, wiggle)] for wiggle in WIGGLES] for resolution in RESOLUTIONS]
    )


def _plot_full_grid(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), constrained_layout=True)
    image = None
    for col, experiment in enumerate(EXPERIMENT_ALGOS):
        for row_index, profile in enumerate(["current", "pre_f8_corner"]):
            values = _grid_values(rows, experiment, profile, "hausdorff_median")
            image = axes[row_index, col].imshow(
                np.log10(np.maximum(values, 1.0e-12)),
                origin="lower",
                aspect="auto",
                vmin=-10,
                vmax=0,
                cmap="viridis",
            )
            axes[row_index, col].set_xticks(range(len(WIGGLES)), [f"{w:g}" for w in WIGGLES])
            axes[row_index, col].set_yticks(range(len(RESOLUTIONS)), [str(int(100 * r)) for r in RESOLUTIONS])
            axes[row_index, col].set_xlabel("Perturbation w")
            axes[row_index, col].set_ylabel("Resolution N")
            title = "Current" if profile == "current" else "Pre-f8 corner logic"
            axes[row_index, col].set_title(f"{experiment.title()}: {title}", fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("log10 median Hausdorff distance")
    _save_figure(fig, out_path)
    plt.close(fig)


def _metric_lookup(hard_cases: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    lookup = {}
    for row in _read_csv(hard_cases):
        profile = _profile_from_row(row)
        if profile in {"current", "pre_f8_corner"}:
            lookup[(row["experiment"], profile, int(row["case_index"]))] = row
    return lookup


def _save_name(experiment: str, profile: str) -> str:
    if experiment == "squares":
        return f"perturb_sweep_squares_linearpluscorner_r1p5_w0p3_s0_corner_{profile}"
    return (
        f"perturb_sweep_zalesak_circularpluscorner_r1p5_w0p3_s0_corner_{profile}"
        "_no_curved_corner_rescues"
    )


def _plot_before_after(hard_cases: Path, out_path: Path) -> None:
    metric_rows = _metric_lookup(hard_cases)
    specs = [("squares", 0, 42.0, 0.10), ("zalesak", 14, 42.0, 0.12)]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.6))
    for row_index, (experiment, case_index, min_span, margin) in enumerate(specs):
        current_save = _save_name(experiment, "current")
        mesh_segments = _mesh_segments(PLOTS_ROOT / current_save / "vtk" / "mesh.vtk")
        true_segments = _load_true_segments(experiment, current_save, case_index)
        bounds = _compute_view_bounds(true_segments, min_span=min_span, margin_frac=margin)
        for col, profile in enumerate(["current", "pre_f8_corner"]):
            save_name = _save_name(experiment, profile)
            recon, endpoints, tips, crossings = _load_reconstructed_plot_geometry(
                save_name,
                case_index,
                exp_name=experiment,
                mesh_segments=mesh_segments,
            )
            metrics = metric_rows[(experiment, profile, case_index)]
            title_label = "Current" if profile == "current" else "Pre-f8 corner logic"
            title = (
                f"{title_label}\n"
                f"H={float(metrics['hausdorff']):.3g}, gap={float(metrics['facet_gap']):.3g}"
            )
            _plot_panel(
                axes[row_index, col],
                exp_name=experiment,
                spec={"case_index": case_index, "inset": None},
                algo=EXPERIMENT_ALGOS[experiment],
                mesh_segments=mesh_segments,
                true_segments=true_segments,
                recon_segments=recon,
                endpoint_points=endpoints,
                corner_tip_points=tips,
                corner_boundary_points=crossings,
                title=title,
                bounds=bounds,
            )
        axes[row_index, 0].set_ylabel(f"{experiment.title()} case {case_index}", fontsize=11)
    fig.suptitle("Matched reconstruction diagnostics: N=150, w=0.3", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, out_path)
    plt.close(fig)


def _write_markdown(
    path: Path,
    hard_rows: list[dict],
    split_rows: list[dict],
    grid_rows: list[dict],
) -> None:
    lines = [
        "# Corner behavior ablation",
        "",
        "## Recommendation",
        "",
        "Use the pre-f8 corner logic: disable the three-neighbor orientation hint, restore legacy linear-corner acceptance, and disable corner-branch propagation. Keep the post-f8 arc/root and rescue machinery outside these three corner behaviors.",
        "",
        "## Hard slice (N=150, w=0.3)",
        "",
        "| Benchmark | Profile | Hausdorff median | Hausdorff mean | H < 1e-6 | H and gap < 1e-6 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENT_ALGOS:
        for profile in ["current", "pre_f8_corner"]:
            row = next(
                item
                for item in hard_rows
                if item["experiment"] == experiment and item["profile"] == profile
            )
            lines.append(
                f"| {experiment} | {PROFILE_LABELS[profile]} | "
                f"{row['hausdorff_median']:.4g} | {row['hausdorff_mean']:.4g} | "
                f"{row['hausdorff_floor_cases']}/25 | {row['joint_floor_cases']}/25 |"
            )
    lines.extend(
        [
            "",
            "## Acceptance split (N=150, w=0.3)",
            "",
            "| Benchmark | Profile | Hausdorff median | Hausdorff mean | H < 1e-6 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for experiment in EXPERIMENT_ALGOS:
        for profile in SPLIT_PROFILES:
            row = next(
                item
                for item in split_rows
                if item["experiment"] == experiment and item["profile"] == profile
            )
            lines.append(
                f"| {experiment} | {PROFILE_LABELS[profile]} | "
                f"{row['hausdorff_median']:.4g} | {row['hausdorff_mean']:.4g} | "
                f"{row['hausdorff_floor_cases']}/25 |"
            )
    lines.extend(
        [
            "",
            "## Full-grid interpretation",
            "",
            "- Squares: the pre-f8 median is below 1e-6 in all 25 resolution/perturbation settings.",
            "- Zalesak: the pre-f8 median is below 1e-6 for all N >= 100 settings; N=50 remains coarse, and N=64 is the transition regime.",
            "- These are median-perfect results, not all-case guarantees. Perturbed runs retain a small outlier tail, reported in `full_grid_summary.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hard_cases",
        type=Path,
        default=Path("results/static/corner_ablation_hard_20260715/diagnostics/case_metrics.csv"),
    )
    parser.add_argument(
        "--acceptance_split_cases",
        type=Path,
        default=Path("results/static/corner_acceptance_split_20260715/diagnostics/case_metrics.csv"),
    )
    parser.add_argument(
        "--current_cases",
        type=Path,
        default=Path("results/static/static_paper_affected_diagnostics_20260714_102206/diagnostics/case_metrics.csv"),
    )
    parser.add_argument(
        "--candidate_cases",
        type=Path,
        nargs="+",
        default=[
            Path("results/static/corner_ablation_resolution_guard_20260715/diagnostics/case_metrics.csv"),
            Path("results/static/corner_ablation_full_grid_missing_a_20260715/diagnostics/case_metrics.csv"),
            Path("results/static/corner_ablation_full_grid_missing_b_20260715/diagnostics/case_metrics.csv"),
        ],
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/static/corner_ablation_analysis_20260715"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hard_rows = _hard_summary(args.hard_cases)
    split_rows = _split_summary(args.acceptance_split_cases)
    candidate_rows = _candidate_cases(args.candidate_cases)
    current_rows = _current_cases(args.current_cases)
    grid_rows = _full_grid_summary(current_rows, candidate_rows)

    _write_csv(args.out_dir / "hard_ablation_summary.csv", hard_rows)
    _write_csv(args.out_dir / "acceptance_split_summary.csv", split_rows)
    _write_csv(args.out_dir / "full_grid_summary.csv", grid_rows)
    _plot_hard_ablation(hard_rows, args.out_dir / "hard_profile_ablation.png")
    _plot_full_grid(grid_rows, args.out_dir / "full_grid_medians.png")
    _plot_before_after(args.hard_cases, args.out_dir / "matched_before_after.png")
    _write_markdown(args.out_dir / "README.md", hard_rows, split_rows, grid_rows)
    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
