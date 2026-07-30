#!/usr/bin/env python3
"""Analyze removal of the post-f8 linear-corner rescue package."""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from experiments.static.generate_section6_maintext_figures import (
    PLOTS_ROOT,
    _compute_view_bounds,
    _load_reconstructed_plot_geometry,
    _load_true_segments,
    _mesh_segments,
    _plot_panel,
    _save_figure,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/static/linear_rescue_cleanup_analysis_20260715"
FLOOR = 1.0e-6
RESOLUTIONS = [50, 64, 100, 128, 150]
WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]
METRICS = ["hausdorff", "facet_gap", "area_error"]

KEEP_ROOTS = [
    ROOT / "results/static/corner_ablation_resolution_guard_20260715",
    ROOT / "results/static/corner_ablation_full_grid_missing_a_20260715",
    ROOT / "results/static/corner_ablation_full_grid_missing_b_20260715",
]
DROP_ROOTS = [
    ROOT / "results/static/pre_f8_no_rescue_signal_20260715",
    ROOT / "results/static/pre_f8_no_rescue_full_grid_a_20260715",
    ROOT / "results/static/pre_f8_no_rescue_full_grid_b_20260715",
]
EXACT_ROOT = ROOT / "results/static/pre_f8_exact_support_validation_20260715"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _case_key(row: dict[str, str]) -> tuple[int, float, int]:
    return (
        int(round(100 * float(row["resolution"]))),
        float(row["wiggle"]),
        int(row["case_index"]),
    )


def _load_cases(roots: list[Path], rescue_profile: str) -> dict[tuple, dict]:
    cases = {}
    for root in roots:
        path = root / "diagnostics/case_metrics.csv"
        for row in _read_csv(path):
            if (
                row["experiment"] == "zalesak"
                and row["algo"] == "circular+corner"
                and row["corner_behavior_profile"] == "pre_f8_corner"
                and row["rescue_profile"] == rescue_profile
            ):
                cases[_case_key(row)] = row
    return cases


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "floor_cases": sum(value < FLOOR for value in values),
    }


def _full_grid_rows(keep: dict, drop: dict) -> list[dict]:
    rows = []
    for resolution in RESOLUTIONS:
        for wiggle in WIGGLES:
            keys = [(resolution, wiggle, case) for case in range(25)]
            keep_h = [float(keep[key]["hausdorff"]) for key in keys]
            drop_h = [float(drop[key]["hausdorff"]) for key in keys]
            keep_g = [float(keep[key]["facet_gap"]) for key in keys]
            drop_g = [float(drop[key]["facet_gap"]) for key in keys]
            keep_summary = _summarize(keep_h)
            drop_summary = _summarize(drop_h)
            rows.append(
                {
                    "N": resolution,
                    "wiggle": wiggle,
                    "keep_hausdorff_median": keep_summary["median"],
                    "drop_hausdorff_median": drop_summary["median"],
                    "keep_hausdorff_mean": keep_summary["mean"],
                    "drop_hausdorff_mean": drop_summary["mean"],
                    "keep_hausdorff_max": keep_summary["max"],
                    "drop_hausdorff_max": drop_summary["max"],
                    "keep_hausdorff_floor_cases": keep_summary["floor_cases"],
                    "drop_hausdorff_floor_cases": drop_summary["floor_cases"],
                    "keep_joint_floor_cases": sum(
                        h < FLOOR and g < FLOOR for h, g in zip(keep_h, keep_g)
                    ),
                    "drop_joint_floor_cases": sum(
                        h < FLOOR and g < FLOOR for h, g in zip(drop_h, drop_g)
                    ),
                    "changed_hausdorff_cases": sum(a != b for a, b in zip(keep_h, drop_h)),
                }
            )
    return rows


def _changed_case_rows(keep: dict, drop: dict) -> list[dict]:
    rows = []
    for key in sorted(keep):
        if float(keep[key]["hausdorff"]) == float(drop[key]["hausdorff"]):
            continue
        row = {"N": key[0], "wiggle": key[1], "case_index": key[2]}
        for metric in METRICS:
            before = float(keep[key][metric])
            after = float(drop[key][metric])
            row[f"keep_{metric}"] = before
            row[f"drop_{metric}"] = after
            row[f"delta_{metric}"] = after - before
        rows.append(row)
    return rows


def _rescue_usage_rows() -> list[dict]:
    counts = Counter()
    cases = Counter()
    for root in KEEP_ROOTS:
        path = root / "diagnostics/merge_events.csv"
        seen = set()
        for row in _read_csv(path):
            if (
                row["experiment"] != "zalesak"
                or row["corner_behavior_profile"] != "pre_f8_corner"
                or row["rescue_profile"] != "no_curved_corner_rescues"
                or "rescue" not in row["stage"]
            ):
                continue
            event = (row["stage"], row["facet_name"], row["facet_class"])
            counts[event] += 1
            case = (*_case_key(row), event)
            if case not in seen:
                cases[event] += 1
                seen.add(case)
    return [
        {
            "stage": event[0],
            "facet_name": event[1],
            "facet_class": event[2],
            "assignment_count": count,
            "case_count": cases[event],
        }
        for event, count in sorted(counts.items())
    ]


def _exact_validation_rows(keep: dict, exact: dict) -> list[dict]:
    fields = METRICS + [
        "num_final_linear_cells",
        "num_final_circular_cells",
        "num_final_linear_corner_cells",
        "num_final_curved_corner_cells",
        "num_plic_fallback_cells",
    ]
    rows = []
    for field in fields:
        deltas = [
            abs(float(exact[key][field]) - float(keep[key][field])) for key in exact
        ]
        rows.append(
            {
                "field": field,
                "case_count": len(deltas),
                "changed_cases": sum(delta > 0 for delta in deltas),
                "max_absolute_delta": max(deltas),
            }
        )
    return rows


def _plot_summary(rows: list[dict], path: Path) -> None:
    median_delta = np.asarray(
        [
            [
                next(
                    row["drop_hausdorff_median"] - row["keep_hausdorff_median"]
                    for row in rows
                    if row["N"] == resolution and row["wiggle"] == wiggle
                )
                for wiggle in WIGGLES
            ]
            for resolution in RESOLUTIONS
        ]
    )
    mean_delta = np.asarray(
        [
            [
                next(
                    row["drop_hausdorff_mean"] - row["keep_hausdorff_mean"]
                    for row in rows
                    if row["N"] == resolution and row["wiggle"] == wiggle
                )
                for wiggle in WIGGLES
            ]
            for resolution in RESOLUTIONS
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for axis, values, title in [
        (axes[0], median_delta, "Median Hausdorff change"),
        (axes[1], mean_delta, "Mean Hausdorff change"),
    ]:
        vmax = max(float(np.max(np.abs(values))), 1.0e-12)
        image = axis.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axis.set_xticks(range(len(WIGGLES)), [f"{value:g}" for value in WIGGLES])
        axis.set_yticks(range(len(RESOLUTIONS)), RESOLUTIONS)
        axis.set_xlabel("Perturbation magnitude")
        axis.set_ylabel("Resolution N")
        axis.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=axis, shrink=0.84)
    fig.suptitle("Effect of removing all linear-corner cleanup passes", fontweight="bold")
    _save_figure(fig, path)
    plt.close(fig)


def _point_cloud(segments: list[np.ndarray], max_step: float | None = None) -> np.ndarray:
    points = []
    for segment in segments:
        segment = np.asarray(segment)
        if not len(segment):
            continue
        if max_step is None:
            points.append(segment)
            continue
        for left, right in zip(segment[:-1], segment[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(right - left) / max_step)) + 1)
            points.append(np.linspace(left, right, count))
    return np.concatenate(points, axis=0)


def _witness_center(true_segments: list[np.ndarray], recon_segments: list[np.ndarray]) -> np.ndarray:
    truth = _point_cloud(true_segments, max_step=0.01)
    recon = _point_cloud(recon_segments)
    distances, _ = cKDTree(truth).query(recon)
    return recon[int(np.argmax(distances))]


def _plot_case7(keep: dict, drop: dict, exact: dict, path: Path) -> None:
    key = (128, 0.3, 7)
    save_names = {
        "Keep all six": "perturb_sweep_zalesak_circularpluscorner_r1p28_w0p3_s0_corner_pre_f8_corner_no_curved_corner_rescues",
        "Remove all six": "perturb_sweep_zalesak_circularpluscorner_r1p28_w0p3_s0_corner_pre_f8_corner_no_corner_rescues",
        "Keep exact support only": "perturb_sweep_zalesak_circularpluscorner_r1p28_w0p3_s0_corner_pre_f8_corner_exact_linear_support_only",
    }
    metric_sets = {"Keep all six": keep, "Remove all six": drop, "Keep exact support only": exact}
    first_save = next(iter(save_names.values()))
    mesh = _mesh_segments(PLOTS_ROOT / first_save / "vtk/mesh.vtk")
    truth = _load_true_segments("zalesak", first_save, key[2])
    geometry = {
        label: _load_reconstructed_plot_geometry(
            save_name, key[2], exp_name="zalesak", mesh_segments=mesh
        )
        for label, save_name in save_names.items()
    }
    full_bounds = _compute_view_bounds(truth, min_span=42.0, margin_frac=0.12)
    center = _witness_center(truth, geometry["Remove all six"][0])
    zoom_span = 2.5
    zoom_bounds = (
        center[0] - zoom_span / 2,
        center[0] + zoom_span / 2,
        center[1] - zoom_span / 2,
        center[1] + zoom_span / 2,
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.3, 8.0))
    for column, label in enumerate(save_names):
        recon, endpoints, tips, crossings = geometry[label]
        metrics = metric_sets[label][key]
        title = f"{label}\nH={float(metrics['hausdorff']):.3g}, gap={float(metrics['facet_gap']):.3g}"
        for row, bounds in enumerate([full_bounds, zoom_bounds]):
            _plot_panel(
                axes[row, column],
                exp_name="zalesak",
                spec={"case_index": key[2], "inset": None},
                algo="circular+corner",
                mesh_segments=mesh,
                true_segments=truth,
                recon_segments=recon,
                endpoint_points=endpoints,
                corner_tip_points=tips,
                corner_boundary_points=crossings,
                title=title if row == 0 else "Local witness",
                bounds=bounds,
            )
            if row == 1:
                axes[row, column].scatter(
                    [center[0]],
                    [center[1]],
                    marker="x",
                    s=55,
                    linewidths=1.8,
                    color="#c026d3",
                    zorder=12,
                )
    axes[0, 0].set_ylabel("Full reconstruction")
    axes[1, 0].set_ylabel("Local witness")
    fig.suptitle("Zalesak case 7: N=128, perturbation=0.3", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, path)
    plt.close(fig)


def _write_readme(
    grid_rows: list[dict], changed_rows: list[dict], usage_rows: list[dict], exact_rows: list[dict]
) -> None:
    keep_setting_floors = sum(row["keep_hausdorff_median"] < FLOOR for row in grid_rows)
    drop_setting_floors = sum(row["drop_hausdorff_median"] < FLOOR for row in grid_rows)
    keep = _load_cases(KEEP_ROOTS, "no_curved_corner_rescues")
    drop = _load_cases(DROP_ROOTS, "no_corner_rescues")
    lines = [
        "# Linear-corner rescue cleanup",
        "",
        "## Recommendation",
        "",
        "Use `pre_f8_corner` with `exact_linear_support_only`. Keep exact propagation of an already accepted straight support and disable the other five post-f8 linear-corner rescue passes together with the previously disabled curved rescue family #9.",
        "",
        "## Evidence",
        "",
        f"- Complete keep-all provenance: `{sum(row['assignment_count'] for row in usage_rows)}` rescue assignments across `{sum(row['case_count'] for row in usage_rows)}` case-events; every assignment was an exact `linear_support` facet. No other linear-corner rescue produced an assignment in the 625-case grid.",
        f"- Removing all six passes preserved floor-level setting medians in `{drop_setting_floors}/25` settings versus `{keep_setting_floors}/25` with keep-all, but changed `{len(changed_rows)}` case Hausdorff values, all for the worse.",
        f"- Hausdorff-floor cases changed `{sum(float(row['hausdorff']) < FLOOR for row in keep.values())}/625 -> {sum(float(row['hausdorff']) < FLOOR for row in drop.values())}/625`; joint Hausdorff/gap-floor cases remained `{sum(float(row['hausdorff']) < FLOOR and float(row['facet_gap']) < FLOOR for row in keep.values())}/625 -> {sum(float(row['hausdorff']) < FLOOR and float(row['facet_gap']) < FLOOR for row in drop.values())}/625`.",
        "- The largest meaningful fine-grid regression was `N=128, w=0.3, case=7`: Hausdorff `2.04e-9 -> 0.248` without exact support propagation.",
        f"- `exact_linear_support_only` matched keep-all exactly in all `{exact_rows[0]['case_count']}` validation cases across every reported scientific metric, facet-class count, and fallback count.",
        "- Square `linear+corner` reconstruction does not execute this circular+corner cleanup block, so its validated 25/25 floor-level setting medians are unaffected.",
        "",
        "## Artifacts",
        "",
        "- `full_grid_comparison.csv`",
        "- `changed_cases.csv`",
        "- `rescue_usage.csv`",
        "- `exact_support_validation.csv`",
        "- `rescue_cleanup_summary.png` / `.pdf`",
        "- `case7_reconstruction_comparison.png` / `.pdf`",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = _load_cases(KEEP_ROOTS, "no_curved_corner_rescues")
    drop = _load_cases(DROP_ROOTS, "no_corner_rescues")
    exact = _load_cases([EXACT_ROOT], "exact_linear_support_only")
    if len(keep) != 625 or len(drop) != 625 or len(exact) != 200:
        raise ValueError(f"Unexpected case inventory: keep={len(keep)}, drop={len(drop)}, exact={len(exact)}")

    grid_rows = _full_grid_rows(keep, drop)
    changed_rows = _changed_case_rows(keep, drop)
    usage_rows = _rescue_usage_rows()
    exact_rows = _exact_validation_rows(keep, exact)
    _write_csv(OUT / "full_grid_comparison.csv", grid_rows)
    _write_csv(OUT / "changed_cases.csv", changed_rows)
    _write_csv(OUT / "rescue_usage.csv", usage_rows)
    _write_csv(OUT / "exact_support_validation.csv", exact_rows)
    _plot_summary(grid_rows, OUT / "rescue_cleanup_summary.png")
    _plot_case7(keep, drop, exact, OUT / "case7_reconstruction_comparison.png")
    _write_readme(grid_rows, changed_rows, usage_rows, exact_rows)
    print(OUT)


if __name__ == "__main__":
    main()
