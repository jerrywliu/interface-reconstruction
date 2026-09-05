#!/usr/bin/env python3
"""Generate Appendix C0 panels from pooled case-level observations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from experiments.submission.generate_revision_layout_prototypes import _trim_for_box
from experiments.plotting import (
    add_shared_vertical_metric_label,
    apply_paper_metric_axis_style,
    apply_paper_serif_style,
    draw_log_axis_break_marks,
    paper_markers_by_label,
    paper_metric_panel_title,
)
from experiments.static.generate_pooled_perturbed_panels import _pooled_curves
from experiments.static.run_perturbed_sweeps import (
    DISPLAY_LABELS,
    RESOLUTION_AXIS_LABEL,
    _draw_method_curves,
    _merge_legend_entries,
    _save_figure,
)


apply_paper_serif_style()


DEFAULT_SEALED = (
    REPO_ROOT
    / "results/static/submission_static_20260731_012430_505aefa45432.sealed"
    / "diagnostics/case_metrics.csv"
)
DEFAULT_JOINT_GLOB = "appendix_b5_joint_c0_20260814_perturb_sweep_*"
DEFAULT_OUTPUT = REPO_ROOT / "results/submission/c0_pooled_panels_20260828"
JOINT_METHOD = {"ellipses": "linear+C0", "zalesak": "circular+C0"}
BASELINE_METHODS = {
    "ellipses": ("linear", "circular"),
    "zalesak": ("circular", "circular+corner"),
}
MARKERS_BY_LABEL = paper_markers_by_label(DISPLAY_LABELS)


def _add_value(data: dict, row: dict, method: str) -> None:
    experiment = row["experiment"]
    resolution = float(row["resolution"])
    wiggle = float(row["wiggle"])
    for metric in ("hausdorff", "facet_gap"):
        raw = row.get(metric)
        if raw in (None, ""):
            continue
        data.setdefault(experiment, {}).setdefault(method, {}).setdefault(
            metric, {}
        ).setdefault(resolution, {}).setdefault(wiggle, {}).setdefault(
            "value", []
        ).append(
            float(raw)
        )


def load_c0_case_index(sealed_csv: Path, plots_root: Path) -> dict:
    data = {}
    with sealed_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["experiment"] not in BASELINE_METHODS:
                continue
            if row["algo"] in BASELINE_METHODS[row["experiment"]]:
                _add_value(data, row, row["algo"])

    for run_dir in sorted(plots_root.glob(DEFAULT_JOINT_GLOB)):
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "metrics/case_metrics.csv"
        if not manifest_path.is_file() or not metrics_path.is_file():
            continue
        parameters = json.loads(manifest_path.read_text())["parameters"]
        experiment = "ellipses" if "_ellipses_" in run_dir.name else "zalesak"
        method = JOINT_METHOD[experiment]
        with metrics_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                _add_value(
                    data,
                    {
                        **row,
                        "experiment": experiment,
                        "resolution": parameters["resolution"],
                        "wiggle": parameters["perturb_wiggle"],
                    },
                    method,
                )
    return data


BREAK_SPECS = {
    "ellipses": {
        "facet_gap": {"lower": (5.0e-15, 8.0e-11), "upper": (5.0e-5, 2.0e-1)},
    },
    "zalesak": {
        "hausdorff": {"lower": (2.0e-11, 4.0e-8), "upper": (3.0e-3, 1.0)},
        "facet_gap": {"lower": (2.0e-11, 4.0e-8), "upper": (3.0e-3, 1.0)},
    },
}


def plot_resolution_panel(data: dict, experiment: str, output: Path) -> None:
    fig = plt.figure(figsize=(7.05, 3.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.55, 0.55),
        hspace=0.06,
        wspace=0.34,
    )
    legend_entries = {}
    axes_by_metric = {}
    for column, metric in enumerate(("hausdorff", "facet_gap")):
        break_spec = BREAK_SPECS[experiment].get(metric)
        if break_spec is None:
            panel_axes = (fig.add_subplot(grid[:, column]),)
        else:
            upper = fig.add_subplot(grid[0, column])
            lower = fig.add_subplot(grid[1, column], sharex=upper)
            panel_axes = (upper, lower)
        axes_by_metric[metric] = panel_axes
        curves = _pooled_curves(data[experiment], metric, "resolution")
        for band_index, axis in enumerate(panel_axes):
            y_window = None
            if break_spec is not None:
                y_window = break_spec["upper" if band_index == 0 else "lower"]
            _draw_method_curves(
                axis,
                curves,
                metric,
                x_label="",
                x_mode="resolution",
                exp_name=experiment,
                y_window=y_window,
            )
            apply_paper_metric_axis_style(
                axis,
                metric,
                "resolution",
                markers_by_label=MARKERS_BY_LABEL,
            )
            axis.set_ylabel("")
            if y_window is not None:
                axis.set_ylim(*y_window)
            _merge_legend_entries(legend_entries, axis)
        panel_axes[0].set_title(
            paper_metric_panel_title(metric),
            fontsize=10.2,
            fontweight="normal",
        )
        panel_axes[-1].set_xlabel(RESOLUTION_AXIS_LABEL)
        if break_spec is not None:
            draw_log_axis_break_marks(panel_axes[0], panel_axes[1])
    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=3,
            fontsize=8.5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.24, top=0.90)
    for metric in ("hausdorff", "facet_gap"):
        panel_axes = axes_by_metric[metric]
        add_shared_vertical_metric_label(
            fig,
            panel_axes,
            metric,
            fontsize=9.8,
        )
    _save_figure(fig, output)
    plt.close(fig)


def build_benchmark_pages(output_dir: Path) -> tuple[Path, Path]:
    ellipse_metrics = output_dir / "ellipses_appendix_c0_resolution.pdf"
    zalesak_metrics = output_dir / "zalesak_appendix_c0_resolution.pdf"
    representative_dir = (
        REPO_ROOT
        / "results/static/camera_ready/appendix_b5_joint_c0_20260814"
        / "representative_cases"
    )
    pages = []
    page_specs = (
        (
            "ellipse_joint_c0_comparison",
            ellipse_metrics,
            representative_dir / "ellipses_appendix_c0_representative_clean.pdf",
            representative_dir
            / "ellipses_appendix_c0_representative_with_endpoints.pdf",
            (
                r"\shortstack{Ours (linear,\\graph-coordinated)}",
                r"\shortstack{Ours (linear,\\graph-coordinated + joint $C^0$)}",
                r"\shortstack{Ours (circular,\\graph-coordinated)}",
            ),
        ),
        (
            "zalesak_joint_c0_comparison",
            zalesak_metrics,
            representative_dir / "zalesak_appendix_c0_representative_clean.pdf",
            None,
            (
                r"\shortstack{Ours (circular,\\graph-coordinated)}",
                r"\shortstack{Ours (circular,\\graph-coordinated + joint $C^0$)}",
                r"\shortstack{Ours (circular + corners,\\graph-coordinated)}",
            ),
        ),
    )
    panel_boxes = (
        (0.0, 0.04, 0.5, 0.5),
        (0.5, 0.04, 1.0, 0.5),
        (0.0, 0.54, 0.5, 1.0),
    )
    ellipse_zoom_boxes = (
        (0.31, 0.11, 0.44, 0.24),
        (0.81, 0.11, 0.94, 0.24),
        (0.31, 0.61, 0.44, 0.74),
    )
    for stem, metrics, representatives, zoom_source, labels in page_specs:
        if not representatives.is_file():
            raise FileNotFoundError(
                f"missing representative panel source: {representatives}"
            )
        representative_cells = []
        for index, (label, box) in enumerate(zip(labels, panel_boxes)):
            if zoom_source is None and index == 2:
                box = (box[0], 0.56, box[2], box[3])
            main_trim = _trim_for_box(representatives, box)
            if zoom_source is not None:
                zoom_trim = _trim_for_box(zoom_source, ellipse_zoom_boxes[index])
                representative_cells.append(
                    rf"\begin{{minipage}}[t]{{2.28in}}\centering"
                    rf"\fontsize{{8.5}}{{9.2}}\selectfont {label}\\[7pt]"
                    rf"\zoomcell{{{representatives}}}{{{main_trim}}}"
                    rf"{{{zoom_source}}}{{{zoom_trim}}}\end{{minipage}}"
                )
            else:
                representative_cells.append(
                    rf"\begin{{minipage}}[t]{{2.28in}}\centering"
                    rf"\fontsize{{8.5}}{{9.2}}\selectfont {label}\\[7pt]"
                    rf"\includegraphics[width=2.28in,trim={{{main_trim}}},clip]"
                    rf"{{{representatives}}}\end{{minipage}}"
                )
        tex_path = output_dir / f"{stem}.tex"
        tex_path.write_text(
            rf"""\documentclass[border=3pt]{{standalone}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{tikz}}
\usetikzlibrary{{calc}}
\definecolor{{spyglass}}{{RGB}}{{128,28,238}}
\newcommand{{\zoomcell}}[4]{{%
  \begin{{tikzpicture}}[baseline=0pt]
    \path[use as bounding box] (0,-0.79in) rectangle (2.28in,0.79in);
    \node[inner sep=0pt] (main) at (1.39in,0)
      {{\includegraphics[height=1.55in,trim={{#2}},clip]{{#1}}}};
    \begin{{scope}}[
      shift={{(main.south west)}},
      x={{($(main.south east)-(main.south west)$)}},
      y={{($(main.north west)-(main.south west)$)}}]
      \draw[spyglass,dashed,line width=0.5pt] (0.62,0.565) rectangle (0.88,0.848);
    \end{{scope}}
    \node[inner sep=0pt] (zoom) at (0.29in,0)
      {{\includegraphics[width=0.56in,trim={{#4}},clip]{{#3}}}};
    \draw[spyglass,line width=0.75pt] (zoom.south west) rectangle (zoom.north east);
  \end{{tikzpicture}}%
}}
\begin{{document}}
\begin{{minipage}}{{7.15in}}
\centering
\includegraphics[width=6.95in]{{{metrics}}}\\[-4pt]
\setlength{{\tabcolsep}}{{1pt}}
\begin{{tabular}}{{ccc}}
{" & ".join(representative_cells)}
\end{{tabular}}
\end{{minipage}}
\end{{document}}
""",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=output_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        pages.append(tex_path.with_suffix(".pdf"))
    return tuple(pages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--plots-root", type=Path, default=REPO_ROOT / "plots")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_c0_case_index(args.sealed, args.plots_root)
    for experiment in ("ellipses", "zalesak"):
        plot_resolution_panel(
            data,
            experiment,
            args.output / f"{experiment}_appendix_c0_resolution.png",
        )
    for page in build_benchmark_pages(args.output):
        print(page)


if __name__ == "__main__":
    main()
