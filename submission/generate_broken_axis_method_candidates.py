#!/usr/bin/env python3
"""Render review-only square and Zalesak panels with broken logarithmic y-axes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import (
    PAPER_METRIC_LABELS,
    apply_paper_metric_axis_style,
    apply_paper_serif_style,
    paper_markers_by_label,
)
from experiments.static.generate_pooled_perturbed_panels import (
    DEFAULT_CURVATURE,
    DEFAULT_SOURCE,
    _pooled_curves,
    load_case_index,
)
from experiments.static.run_perturbed_sweeps import (
    DISPLAY_LABELS,
    _draw_method_curves,
    _merge_legend_entries,
)
from submission.pdf_vector_qa import inspect_pdf


OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "submission"
    / "figure_consistency_20260903"
    / "broken_axis_candidates"
)

LIMITS = {
    "squares": {
        "hausdorff": {"upper": (1.5e-1, 1.1), "lower": (2.0e-10, 2.0e-8)},
        "facet_gap": {"upper": (4.0e-3, 3.5e-1), "lower": (2.0e-11, 5.0e-10)},
    },
    "zalesak": {
        "hausdorff": {"upper": (1.5e-1, 1.2), "lower": (2.0e-9, 4.0e-8)},
        "facet_gap": {"upper": (3.0e-3, 3.0e-1), "lower": (1.0e-10, 1.0e-9)},
    },
}

MARKERS_BY_LABEL = paper_markers_by_label(DISPLAY_LABELS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_break_marks(upper, lower) -> None:
    """Draw conventional diagonal marks where the logarithmic axis is omitted."""

    size = 0.014
    style = {"color": "#111827", "clip_on": False, "linewidth": 0.8}
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    lower.tick_params(axis="x", which="both", top=False)

    for x_coordinate in (0.0, 1.0):
        upper.plot(
            (x_coordinate - size, x_coordinate + size),
            (-size, size),
            transform=upper.transAxes,
            **style,
        )
        lower.plot(
            (x_coordinate - size, x_coordinate + size),
            (1.0 - size, 1.0 + size),
            transform=lower.transAxes,
            **style,
        )


def _style_axis(axis, metric: str, axis_name: str) -> None:
    x_mode = "perturbation" if axis_name == "wiggle" else "resolution"
    apply_paper_metric_axis_style(
        axis,
        metric,
        x_mode,
        markers_by_label=MARKERS_BY_LABEL,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")


def _draw_curve_copy(axis, curves, metric: str, axis_name: str, experiment: str) -> None:
    _draw_method_curves(
        axis,
        curves,
        metric,
        x_label="",
        x_mode="perturbation" if axis_name == "wiggle" else "resolution",
        exp_name=experiment,
    )
    _style_axis(axis, metric, axis_name)


def render_benchmark(data: dict, experiment: str, output: Path) -> None:
    apply_paper_serif_style()
    fig = plt.figure(figsize=(7.05, 6.45))
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=(2.0, 0.82, 0.30, 2.0, 0.82),
        hspace=0.08,
        wspace=0.28,
    )
    axes = {}
    legend_entries = {}

    for metric_index, metric in enumerate(("hausdorff", "facet_gap")):
        upper_row = 0 if metric_index == 0 else 3
        lower_row = upper_row + 1
        for column, axis_name in enumerate(("wiggle", "resolution")):
            upper = fig.add_subplot(grid[upper_row, column])
            lower = fig.add_subplot(grid[lower_row, column], sharex=upper)
            curves = _pooled_curves(data[experiment], metric, axis_name)
            _draw_curve_copy(upper, curves, metric, axis_name, experiment)
            _draw_curve_copy(lower, curves, metric, axis_name, experiment)
            upper.set_ylim(*LIMITS[experiment][metric]["upper"])
            lower.set_ylim(*LIMITS[experiment][metric]["lower"])
            _draw_break_marks(upper, lower)
            _merge_legend_entries(legend_entries, upper)

            if metric_index == 0:
                upper.set_title(
                    "Perturbation sweep"
                    if axis_name == "wiggle"
                    else "Resolution study"
                )
            if metric_index == 1:
                lower.set_xlabel(
                    r"Perturbation magnitude, $w$"
                    if axis_name == "wiggle"
                    else r"Cells per side, $N$"
                )
            axes[(metric, axis_name)] = (upper, lower)

    fig.text(
        0.012,
        0.676,
        PAPER_METRIC_LABELS["hausdorff"],
        rotation=90,
        va="center",
        fontsize=8.5,
    )
    fig.text(
        0.012,
        0.285,
        PAPER_METRIC_LABELS["facet_gap"],
        rotation=90,
        va="center",
        fontsize=8.5,
    )
    fig.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        loc="upper center",
        ncol=min(3, len(legend_entries)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.998),
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.875)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_case_index(
        DEFAULT_SOURCE / "diagnostics" / "case_metrics.csv",
        DEFAULT_CURVATURE,
    )
    outputs = []
    for experiment in ("squares", "zalesak"):
        output = OUTPUT_ROOT / f"{experiment}_broken_y_axis.pdf"
        render_benchmark(data, experiment, output)
        report = inspect_pdf(output, require_fonts=True)
        if not report.passed:
            raise RuntimeError(f"vector QA failed for {output}: {report.issues}")
        outputs.append(
            {
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "vector_qa": True,
                "limits": LIMITS[experiment],
            }
        )

    manifest = {
        "purpose": "review-only broken-log-axis candidates",
        "scientific_data_changed": False,
        "aggregation": "pooled cases at each abscissa",
        "outputs": outputs,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
