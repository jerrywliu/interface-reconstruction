#!/usr/bin/env python3
"""Build a matched ellipse comparison across project and external baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import add_convergence_order_triangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE_CASES = Path(
    "experiments/baselines/results/common_native_metric_replay_20260813/"
    "case_results.csv"
)
DEFAULT_OURS_CASES = Path(
    "experiments/baselines/results/"
    "ellipse_circular_variants_joint_c0_common_metrics_20260814/case_results.csv"
)
DEFAULT_BASELINE_CASES = (
    Path(
        "experiments/baselines/results/"
        "plvira_project_smoke_20260813_final/case_results.csv"
    ),
    Path(
        "experiments/baselines/results/"
        "pcic_project_smoke_20260813_final/case_results.csv"
    ),
    Path(
        "experiments/baselines/results/"
        "quasi_project_smoke_20260813_qa_fixed/case_results.csv"
    ),
)
DEFAULT_OUTPUT = Path(
    "experiments/baselines/results/ellipse_all_method_comparison_20260814"
)
RESOLUTIONS = (32, 64, 128)
GAP_DISPLAY_FLOOR = 1.0e-7
METRICS = (
    "native_symmetric_hausdorff",
    "geometric_curvature_mean_absolute_error",
    "facet_gap",
)

METHODS = (
    {
        "id": "ours_per_cell",
        "method": "Ours",
        "variant": "per-cell circular",
        "label": "Ours: circular (per-cell)",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": ":",
    },
    {
        "id": "ours_graph",
        "method": "Ours",
        "variant": "graph-coordinated circular",
        "label": "Ours: circular (graph-coordinated)",
        "color": "#009E73",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "id": "ours_c0",
        "method": "Ours",
        "variant": "graph-coordinated circular + joint C0",
        "label": "Ours: circular (graph-coordinated + joint C0)",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "id": "plvira",
        "method": "PLVIRA",
        "variant": "PLVIRA",
        "label": "PLVIRA",
        "color": "#CC79A7",
        "marker": "^",
        "linestyle": "-.",
    },
    {
        "id": "pcic_center",
        "method": "PCIC",
        "variant": "bare PCIC (center translation)",
        "label": "PCIC (center translation)",
        "color": "#6B7280",
        "marker": "v",
        "linestyle": "--",
    },
    {
        "id": "pcic_radius",
        "method": "PCIC",
        "variant": "bare PCIC (radius adjustment)",
        "label": "PCIC (radius adjustment)",
        "color": "#8C6D31",
        "marker": "<",
        "linestyle": ":",
    },
    {
        "id": "quasi",
        "method": "QUASI",
        "variant": "QUASI (frozen Cartesian port)",
        "label": "QUASI (frozen port)",
        "color": "#7E57C2",
        "marker": "X",
        "linestyle": "-",
    },
)
METHOD_BY_KEY = {(item["method"], item["variant"]): item for item in METHODS}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, "", "nan", "NaN"):
        return math.nan
    return float(value)


def _int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    return int(float(value)) if value not in (None, "") else 0


def assemble_case_metrics(
    native_cases: Path,
    baseline_cases: Sequence[Path],
    ours_cases: Path,
    methods: Sequence[Mapping[str, Any]] = METHODS,
) -> list[dict[str, Any]]:
    """Join native observables to baseline diagnostics and append project variants."""

    method_by_key = {(item["method"], item["variant"]): item for item in methods}
    diagnostics: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    for path in baseline_cases:
        for row in _read_csv(path):
            if row["benchmark"] != "ellipses":
                continue
            key = (
                row["method"],
                row["variant"],
                _int(row, "cells_per_side"),
                _int(row, "case_index"),
            )
            if key in diagnostics:
                raise ValueError(f"duplicate baseline diagnostic key: {key}")
            diagnostics[key] = row

    result: list[dict[str, Any]] = []
    for native in _read_csv(native_cases):
        if native["benchmark"] != "ellipses":
            continue
        key = (
            native["method"],
            native["variant"],
            _int(native, "cells_per_side"),
            _int(native, "case_index"),
        )
        method = method_by_key.get(key[:2])
        if method is None:
            continue
        diagnostic = diagnostics.get(key)
        if diagnostic is None:
            raise ValueError(f"missing baseline diagnostics for {key}")
        mixed = _int(diagnostic, "mixed_cells")
        reconstructed = _int(diagnostic, "reconstructed_cells") + _int(
            diagnostic, "paper_fallback_cells"
        )
        result.append(
            {
                "method_id": method["id"],
                "display_label": method["label"],
                "method": key[0],
                "variant": key[1],
                "case_index": key[3],
                "cells_per_side": key[2],
                "cell_size": _float(native, "cell_size"),
                "native_symmetric_hausdorff": _float(
                    native, "native_symmetric_hausdorff"
                ),
                "geometric_curvature_mean_absolute_error": _float(
                    native, "geometric_curvature_mean_absolute_error"
                ),
                "facet_gap": _float(diagnostic, "shared_edge_gap_mean"),
                "normalized_conservation_residual": _float(
                    diagnostic, "conservation_max_absolute_residual"
                )
                / _float(native, "cell_size") ** 2,
                "mixed_cells": mixed,
                "reconstructed_cells": reconstructed,
                "unsupported_cells": _int(diagnostic, "unsupported_cells"),
                "unresolved_cells": _int(diagnostic, "unresolved_cells"),
            }
        )

    for row in _read_csv(ours_cases):
        if row["benchmark"] != "ellipses":
            continue
        key = row["method"], row["variant"]
        method = method_by_key.get(key)
        if method is None:
            continue
        mixed = _int(row, "num_mixed_cells")
        result.append(
            {
                "method_id": method["id"],
                "display_label": method["label"],
                "method": key[0],
                "variant": key[1],
                "case_index": _int(row, "case_index"),
                "cells_per_side": _int(row, "cells_per_side"),
                "cell_size": _float(row, "cell_size"),
                "native_symmetric_hausdorff": _float(row, "native_symmetric_hausdorff"),
                "geometric_curvature_mean_absolute_error": _float(
                    row, "geometric_curvature_mean_absolute_error"
                ),
                "facet_gap": _float(row, "production_facet_gap"),
                "normalized_conservation_residual": _float(
                    row, "normalized_conservation_residual"
                ),
                "mixed_cells": mixed,
                "reconstructed_cells": mixed,
                "unsupported_cells": 0,
                "unresolved_cells": 0,
            }
        )

    case_keys_by_method = {
        method["id"]: {
            (int(row["cells_per_side"]), int(row["case_index"]))
            for row in result
            if row["method_id"] == method["id"]
        }
        for method in methods
    }
    reference_case_keys = case_keys_by_method[methods[0]["id"]]
    if not reference_case_keys or any(
        keys != reference_case_keys for keys in case_keys_by_method.values()
    ):
        raise ValueError("selected methods do not contain the same case-resolution keys")
    expected = len(methods) * len(reference_case_keys)
    if len(result) != expected:
        raise ValueError(f"expected {expected} matched case rows, found {len(result)}")
    result.sort(
        key=lambda row: (
            next(
                i
                for i, method in enumerate(methods)
                if method["id"] == row["method_id"]
            ),
            row["cells_per_side"],
            row["case_index"],
        )
    )
    return result


def _observed_order(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    resolutions = np.asarray([float(row["cells_per_side"]) for row in rows])
    if len(values) < 3 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        return math.nan
    return float(-np.polyfit(np.log(resolutions), np.log(values), 1)[0])


def summarize_case_metrics(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]] = METHODS,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method_id"] == method["id"]]
        method_summary = []
        for resolution in RESOLUTIONS:
            selected = [
                row for row in method_rows if int(row["cells_per_side"]) == resolution
            ]
            item: dict[str, Any] = {
                "method_id": method["id"],
                "display_label": method["label"],
                "method": method["method"],
                "variant": method["variant"],
                "cells_per_side": resolution,
                "case_count": len(selected),
            }
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in selected])
                item[f"{metric}_median"] = float(np.median(values))
                item[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                item[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            mixed = sum(int(row["mixed_cells"]) for row in selected)
            reconstructed = sum(int(row["reconstructed_cells"]) for row in selected)
            item["mixed_cells"] = mixed
            item["reconstructed_cells"] = reconstructed
            item["unsupported_cells"] = sum(
                int(row["unsupported_cells"]) for row in selected
            )
            item["unresolved_cells"] = sum(
                int(row["unresolved_cells"]) for row in selected
            )
            item["reconstruction_coverage"] = reconstructed / mixed
            item["normalized_conservation_residual_max"] = max(
                float(row["normalized_conservation_residual"])
                for row in selected
            )
            method_summary.append(item)
        for metric in METRICS:
            order = _observed_order(method_summary, f"{metric}_median")
            for item in method_summary:
                item[f"{metric}_fit_order"] = order
        summary.extend(method_summary)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _file_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    methods: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    analysis_git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    inputs = [args.native_cases, args.ours_cases, *args.baseline_cases]
    artifacts = [
        path.parent / "case_metrics.csv",
        path.parent / "summary.csv",
        path.parent / "ellipse_all_methods_metrics.pdf",
        path.parent / "ellipse_all_methods_metrics.png",
    ]
    payload = {
        "schema_version": 1,
        "analysis_git_head": analysis_git_head,
        "selected_methods": [
            {
                key: method[key]
                for key in ("id", "method", "variant", "label")
            }
            for method in methods
        ],
        "case_row_count": len(rows),
        "summary_row_count": len(summary),
        "case_indices": sorted({int(row["case_index"]) for row in rows}),
        "cells_per_side": sorted({int(row["cells_per_side"]) for row in rows}),
        "metric_definitions": {
            "native_symmetric_hausdorff": (
                "partition-insensitive symmetric native point-to-curve supremum"
            ),
            "geometric_curvature_mean_absolute_error": (
                "arc-length-weighted native geometric-curvature MAE against the "
                "nearest analytic ellipse branch"
            ),
            "facet_gap": "mean shared-edge endpoint gap from method diagnostics",
            "reconstruction_coverage": (
                "reconstructed mixed cells, including published method fallbacks, "
                "divided by mixed cells"
            ),
            "normalized_conservation_residual": (
                "maximum absolute fitted-group area residual divided by the "
                "geometric area of that cell or merged group"
            ),
        },
        "inputs": [_file_record(input_path) for input_path in inputs],
        "artifacts": [_file_record(artifact) for artifact in artifacts],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def plot_summary(
    summary: Sequence[Mapping[str, Any]],
    path: Path,
    methods: Sequence[Mapping[str, Any]] = METHODS,
) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), sharex=True)
    panels = (
        ("native_symmetric_hausdorff", "Native symmetric Hausdorff"),
        (
            "geometric_curvature_mean_absolute_error",
            "Unsigned geometric-curvature MAE",
        ),
        ("facet_gap", "Facet gap"),
    )
    for axis, (metric, ylabel) in zip(axes.ravel()[:3], panels):
        for method in methods:
            selected = [row for row in summary if row["method_id"] == method["id"]]
            x = np.asarray([int(row["cells_per_side"]) for row in selected])
            y = np.asarray([float(row[f"{metric}_median"]) for row in selected])
            if metric == "facet_gap" and np.all(y == 0.0):
                y = np.full_like(y, GAP_DISPLAY_FLOOR)
            axis.plot(
                x,
                y,
                color=method["color"],
                marker=method["marker"],
                linestyle=method["linestyle"],
                linewidth=1.35,
                markersize=5.0,
                label=method["label"],
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(RESOLUTIONS, tuple(str(value) for value in RESOLUTIONS))
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)

    for method in methods:
        selected = [row for row in summary if row["method_id"] == method["id"]]
        axes[1, 1].plot(
            [int(row["cells_per_side"]) for row in selected],
            [100.0 * float(row["reconstruction_coverage"]) for row in selected],
            color=method["color"],
            marker=method["marker"],
            linestyle=method["linestyle"],
            linewidth=1.35,
            markersize=5.0,
        )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(RESOLUTIONS, tuple(str(value) for value in RESOLUTIONS))
    axes[1, 1].set_ylabel("Reconstructed mixed cells (%)")
    coverage_values = [
        100.0 * float(row["reconstruction_coverage"]) for row in summary
    ]
    axes[1, 1].set_ylim(max(0.0, min(coverage_values) - 0.5), 100.15)
    axes[1, 1].grid(True, alpha=0.25)

    add_convergence_order_triangle(
        axes[0, 0], 3.0, order_label="3", anchor=(0.75, 0.10), width=0.13
    )
    add_convergence_order_triangle(
        axes[0, 1], 1.0, order_label="1", anchor=(0.76, 0.13), width=0.13
    )
    add_convergence_order_triangle(
        axes[1, 0], 3.0, order_label="3", anchor=(0.74, 0.24), width=0.13
    )
    if any(method["id"] == "quasi" for method in methods):
        axes[1, 0].annotate(
            "Joint C0 and QUASI exact zero\n(shown at plotting floor)",
            (64, GAP_DISPLAY_FLOOR),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#7E57C2",
            fontsize=7,
        )
    for axis in axes[1, :]:
        axis.set_xlabel("Cells per side")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
        ncol=min(3, len(methods)),
        fontsize=7,
    )
    figure.suptitle(
        "Ellipse benchmark: circular variants and higher-order baselines",
        fontsize=11,
        y=0.985,
    )
    figure.text(
        0.5,
        0.935,
        f"Medians over {max(int(row['case_count']) for row in summary)} matched "
        "Cartesian cases; native geometry and curvature",
        ha="center",
        va="center",
        fontsize=8,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    figure.savefig(
        path,
        bbox_inches="tight",
        dpi=360 if path.suffix.lower() == ".png" else None,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-cases", type=Path, default=DEFAULT_NATIVE_CASES)
    parser.add_argument("--ours-cases", type=Path, default=DEFAULT_OURS_CASES)
    parser.add_argument(
        "--baseline-cases",
        nargs="+",
        type=Path,
        default=list(DEFAULT_BASELINE_CASES),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method-ids",
        nargs="+",
        default=[method["id"] for method in METHODS],
        help="method IDs to include in the comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_by_id = {method["id"]: method for method in METHODS}
    unknown = [method_id for method_id in args.method_ids if method_id not in selected_by_id]
    if unknown:
        raise ValueError(f"unknown method IDs: {unknown}")
    methods = tuple(selected_by_id[method_id] for method_id in args.method_ids)
    rows = assemble_case_metrics(
        args.native_cases, tuple(args.baseline_cases), args.ours_cases, methods
    )
    summary = summarize_case_metrics(rows, methods)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "case_metrics.csv", rows)
    _write_csv(args.output / "summary.csv", summary)
    plot_summary(summary, args.output / "ellipse_all_methods_metrics.pdf", methods)
    plot_summary(summary, args.output / "ellipse_all_methods_metrics.png", methods)
    _write_manifest(
        args.output / "manifest.json",
        args=args,
        methods=methods,
        rows=rows,
        summary=summary,
    )


if __name__ == "__main__":
    main()
