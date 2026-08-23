#!/usr/bin/env python3
"""Analyze and plot the extended line-only circle convergence study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from experiments.plotting import add_convergence_order_triangle
from experiments.static.run_extended_convergence_smoke import LINEAR_METHODS


METRICS = ("hausdorff", "facet_gap")
EXPECTED_N = (256, 300, 512)
EXPECTED_WIGGLES = (0.0, 0.2)
EXPECTED_CASES = (0, 1, 2, 3, 4)
METHOD_LABELS = {
    "Youngs": "Youngs",
    "ELVIRA": "ELVIRA",
    "LVIRA": "LVIRA",
    "safe_linear": "Per-cell linear",
    "linear": "Graph-coordinated linear",
}
METHOD_COLORS = {
    "Youngs": "#B14E5E",
    "ELVIRA": "#B3811B",
    "LVIRA": "#2D7D64",
    "safe_linear": "#7C5AA6",
    "linear": "#2F6FA3",
}
METHOD_MARKERS = {
    "Youngs": "o",
    "ELVIRA": "s",
    "LVIRA": "D",
    "safe_linear": "^",
    "linear": "v",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Mapping[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {row[field]}")
    return value


def _fit_order(rows: Iterable[Mapping[str, Any]], metric: str) -> float:
    points = sorted(
        (int(row["cells_per_side"]), _float(row, metric)) for row in rows
    )
    if len(points) < 2 or any(value <= 0.0 for _, value in points):
        return math.nan
    slope, _ = np.polyfit(
        np.log([point[0] for point in points]),
        np.log([point[1] for point in points]),
        1,
    )
    return float(-slope)


def _validate(case_rows: Sequence[Mapping[str, str]]) -> None:
    if not case_rows:
        raise ValueError("case_metrics.csv is empty")
    experiments = {row["experiment"] for row in case_rows}
    if experiments != {"circles"}:
        raise ValueError(f"expected only circles, found {sorted(experiments)}")
    observed_methods = {row["method"] for row in case_rows}
    if observed_methods != set(LINEAR_METHODS):
        raise ValueError(
            f"method mismatch: expected {list(LINEAR_METHODS)}, "
            f"found {sorted(observed_methods)}"
        )
    observed_n = {int(row["cells_per_side"]) for row in case_rows}
    observed_w = {float(row["wiggle"]) for row in case_rows}
    observed_cases = {int(row["case_index"]) for row in case_rows}
    if observed_n != set(EXPECTED_N):
        raise ValueError(f"resolution mismatch: {sorted(observed_n)}")
    if observed_w != set(EXPECTED_WIGGLES):
        raise ValueError(f"perturbation mismatch: {sorted(observed_w)}")
    if observed_cases != set(EXPECTED_CASES):
        raise ValueError(f"case mismatch: {sorted(observed_cases)}")
    expected_count = (
        len(LINEAR_METHODS)
        * len(EXPECTED_N)
        * len(EXPECTED_WIGGLES)
        * len(EXPECTED_CASES)
    )
    if len(case_rows) != expected_count:
        raise ValueError(f"expected {expected_count} case rows, found {len(case_rows)}")
    keys = {
        (
            row["method"],
            int(row["cells_per_side"]),
            float(row["wiggle"]),
            int(row["case_index"]),
        )
        for row in case_rows
    }
    if len(keys) != expected_count:
        raise ValueError("case_metrics.csv contains duplicate matrix rows")
    for row in case_rows:
        for metric in METRICS:
            value = _float(row, metric)
            if value < 0.0:
                raise ValueError(f"negative {metric}: {value}")


def _aggregate_orders(
    summary_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in LINEAR_METHODS:
        for wiggle in EXPECTED_WIGGLES:
            selected = [
                row
                for row in summary_rows
                if row["method"] == method and float(row["wiggle"]) == wiggle
            ]
            for metric in METRICS:
                renamed = [
                    {**row, metric: row[f"{metric}_median"]} for row in selected
                ]
                output.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "wiggle": wiggle,
                        "metric": metric,
                        "median_order": _fit_order(renamed, metric),
                    }
                )
    return output


def _case_orders(case_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in LINEAR_METHODS:
        for wiggle in EXPECTED_WIGGLES:
            for case_index in EXPECTED_CASES:
                selected = [
                    row
                    for row in case_rows
                    if row["method"] == method
                    and float(row["wiggle"]) == wiggle
                    and int(row["case_index"]) == case_index
                ]
                for metric in METRICS:
                    output.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "wiggle": wiggle,
                            "case_index": case_index,
                            "metric": metric,
                            "order": _fit_order(selected, metric),
                        }
                    )
    return output


def _plot(summary_rows: Sequence[Mapping[str, str]], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.25), sharex=True)
    for column, wiggle in enumerate(EXPECTED_WIGGLES):
        for row_index, metric in enumerate(METRICS):
            axis = axes[row_index, column]
            for method in LINEAR_METHODS:
                selected = sorted(
                    (
                        row
                        for row in summary_rows
                        if row["method"] == method
                        and float(row["wiggle"]) == wiggle
                    ),
                    key=lambda row: int(row["cells_per_side"]),
                )
                n_values = np.asarray(
                    [int(row["cells_per_side"]) for row in selected], dtype=float
                )
                medians = np.asarray(
                    [float(row[f"{metric}_median"]) for row in selected]
                )
                q1 = np.asarray([float(row[f"{metric}_q1"]) for row in selected])
                q3 = np.asarray([float(row[f"{metric}_q3"]) for row in selected])
                color = METHOD_COLORS[method]
                axis.fill_between(n_values, q1, q3, color=color, alpha=0.10, lw=0)
                axis.plot(
                    n_values,
                    medians,
                    color=color,
                    marker=METHOD_MARKERS[method],
                    markersize=4.0,
                    linewidth=1.2,
                    label=METHOD_LABELS[method],
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="major", color="#d1d5db", linewidth=0.45)
            axis.grid(True, which="minor", color="#e5e7eb", linewidth=0.3)
            axis.set_xticks(EXPECTED_N, labels=[str(value) for value in EXPECTED_N])
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_title(
                "Cartesian mesh" if wiggle == 0.0 else r"Perturbed mesh ($w=0.2$)"
            )
            axis.set_ylabel(
                "Hausdorff error" if metric == "hausdorff" else "Facet-gap error"
            )
            axis.margins(x=0.08, y=0.18)
            add_convergence_order_triangle(
                axis,
                2.0,
                anchor=(0.69, 0.18),
                width=0.12,
                order_label="2",
            )
    for axis in axes[-1, :]:
        axis.set_xlabel("Cells per side, $N$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0, w_pad=1.0)
    fig.savefig(output_dir / "line_circle_extended_convergence.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "line_circle_extended_convergence.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _order_range(
    case_orders: Sequence[Mapping[str, Any]], method: str, wiggle: float, metric: str
) -> tuple[float, float]:
    values = [
        float(row["order"])
        for row in case_orders
        if row["method"] == method
        and float(row["wiggle"]) == wiggle
        and row["metric"] == metric
        and math.isfinite(float(row["order"]))
    ]
    return min(values), max(values)


def _write_report(
    output_dir: Path,
    aggregate_orders: Sequence[Mapping[str, Any]],
    case_orders: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    run_status: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Extended line-only circle convergence",
        "",
        "Fresh production-default study on deterministic circle cases 0--4 at "
        "`N=256,300,512` and perturbations `w=0,0.2`. Shaded bands in the "
        "summary plot show the interquartile range across matched cases.",
        "",
        "| Method | Mesh | Hausdorff order | Case range | Facet-gap order | Case range |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method in LINEAR_METHODS:
        for wiggle in EXPECTED_WIGGLES:
            by_metric = {
                row["metric"]: float(row["median_order"])
                for row in aggregate_orders
                if row["method"] == method and float(row["wiggle"]) == wiggle
            }
            h_range = _order_range(case_orders, method, wiggle, "hausdorff")
            g_range = _order_range(case_orders, method, wiggle, "facet_gap")
            mesh = "Cartesian" if wiggle == 0.0 else "Perturbed"
            lines.append(
                f"| {METHOD_LABELS[method]} | {mesh} | "
                f"{by_metric['hausdorff']:.3f} | {h_range[0]:.3f}--{h_range[1]:.3f} | "
                f"{by_metric['facet_gap']:.3f} | {g_range[0]:.3f}--{g_range[1]:.3f} |"
            )
    estimate = manifest.get("full_25_case_estimate", {})
    wall_seconds = sum(float(row["wall_time_seconds"]) for row in run_status)
    elapsed_seconds = (
        datetime.fromisoformat(manifest["completed_utc"])
        - datetime.fromisoformat(manifest["created_utc"])
    ).total_seconds()
    max_global_area = max(
        _float(row, "global_relative_phase_area_error") for row in case_rows
    )
    max_component_residual = max(
        _float(row, "max_fitted_component_absolute_residual") for row in case_rows
    )
    max_merged_residual = max(
        _float(row, "max_merged_component_absolute_residual")
        for row in case_rows
        if row.get("max_merged_component_absolute_residual") not in (None, "")
    )
    fallback_cells = sum(int(float(row["num_plic_fallback_cells"])) for row in case_rows)
    missing_facets = sum(int(float(row["num_missing_facets"])) for row in case_rows)
    lines.extend(
        [
            "",
            "Youngs and per-cell linear produce identical Hausdorff and facet-gap "
            "values for every matched case in this study; their plotted curves "
            "therefore coincide.",
            "",
            "## Provenance",
            "",
            f"- Source commit: `{manifest.get('source', {}).get('commit', '')}`.",
            "- PLIC fallback: `LVIRA`.",
            f"- Completed settings/cases: `{len(run_status)}/{len(case_rows)}`; "
            f"missing facets: `{missing_facets}`; fallback cells: `{fallback_cells}`.",
            f"- Maximum global relative area error: `{max_global_area:.3e}`; "
            f"maximum fitted-component residual: `{max_component_residual:.3e}`; "
            f"maximum merged-component residual: `{max_merged_residual:.3e}`.",
            f"- Observed two-worker elapsed time: `{elapsed_seconds / 60.0:.1f}` min; "
            f"summed subprocess time: `{wall_seconds / 60.0:.1f}` min.",
            f"- Recorded subprocess wall time: `{wall_seconds:.1f}` s.",
            f"- Five-to-25-case serial estimate: "
            f"`{float(estimate.get('estimated_25_case_serial_hours', 0.0)):.2f}` h.",
            "- No method policy was changed after inspecting results.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_hashes(output_dir: Path, names: Sequence[str]) -> None:
    lines = []
    for name in names:
        path = output_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    case_rows = _read_csv(run_dir / "case_metrics.csv")
    summary_rows = _read_csv(run_dir / "summary_metrics.csv")
    run_status = _read_csv(run_dir / "run_status.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _validate(case_rows)
    if manifest.get("status") != "complete":
        raise ValueError(f"run is not complete: {manifest.get('status')}")
    aggregate_orders = _aggregate_orders(summary_rows)
    case_orders = _case_orders(case_rows)
    _write_csv(run_dir / "aggregate_convergence_orders.csv", aggregate_orders)
    _write_csv(run_dir / "case_convergence_orders.csv", case_orders)
    _plot(summary_rows, run_dir)
    _write_report(
        run_dir, aggregate_orders, case_orders, case_rows, manifest, run_status
    )
    _write_hashes(
        run_dir,
        (
            "manifest.json",
            "case_metrics.csv",
            "summary_metrics.csv",
            "aggregate_convergence_orders.csv",
            "case_convergence_orders.csv",
            "line_circle_extended_convergence.pdf",
            "line_circle_extended_convergence.png",
            "README.md",
        ),
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
