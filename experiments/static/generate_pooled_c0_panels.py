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

from experiments.plotting import (
    apply_paper_metric_axis_style,
    apply_paper_serif_style,
    paper_markers_by_label,
)
from experiments.static.generate_pooled_perturbed_panels import _pooled_curves
from experiments.static.run_perturbed_sweeps import (
    DISPLAY_LABELS,
    RESOLUTION_AXIS_LABEL,
    _draw_method_curves,
    _merge_legend_entries,
    _metric_label,
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
PAPER_ASSETS = REPO_ROOT.parent / "overleaf/interface-reconstruction-paper/figs/cameraready"
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
        ).append(float(raw))


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


def plot_resolution_panel(data: dict, experiment: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85))
    legend_entries = {}
    for axis, metric in zip(axes, ("hausdorff", "facet_gap")):
        curves = _pooled_curves(data[experiment], metric, "resolution")
        _draw_method_curves(
            axis,
            curves,
            metric,
            x_label=RESOLUTION_AXIS_LABEL,
            x_mode="resolution",
            exp_name=experiment,
        )
        apply_paper_metric_axis_style(
            axis,
            metric,
            "resolution",
            markers_by_label=MARKERS_BY_LABEL,
        )
        axis.set_title(
            _metric_label(metric),
            fontsize=9.0,
            fontweight="normal",
        )
        _merge_legend_entries(legend_entries, axis)
    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=3,
            fontsize=7.2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=[0, 0.16, 1, 1], w_pad=1.2)
    _save_figure(fig, output)
    plt.close(fig)


def build_compact_page(output_dir: Path) -> Path:
    ellipse_metrics = output_dir / "ellipses_appendix_c0_resolution.pdf"
    zalesak_metrics = output_dir / "zalesak_appendix_c0_resolution.pdf"
    reviewed_compact = PAPER_ASSETS / "compact_joint_c0_one_page.pdf"
    tex_path = output_dir / "compact_joint_c0_one_page.tex"
    tex_path.write_text(
        rf"""\documentclass[border=3pt]{{standalone}}
\usepackage{{graphicx}}
\begin{{document}}
\begin{{minipage}}{{7.15in}}
\textbf{{Ellipse: resolution metrics}}\\[-2pt]
\includegraphics[width=6.8in]{{{ellipse_metrics}}}\\[-2pt]
\includegraphics[width=7.0in,trim={{0bp 340bp 0bp 212bp}},clip]{{{reviewed_compact}}}\\[2pt]
\textbf{{Zalesak: resolution metrics}}\\[-2pt]
\includegraphics[width=6.8in]{{{zalesak_metrics}}}\\[-2pt]
\includegraphics[width=7.0in,trim={{0bp 0bp 0bp 529bp}},clip]{{{reviewed_compact}}}
\end{{minipage}}
\end{{document}}
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output_dir,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--plots-root", type=Path, default=REPO_ROOT / "plots")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_c0_case_index(args.sealed, args.plots_root)
    for experiment in ("ellipses", "zalesak"):
        plot_resolution_panel(
            data,
            experiment,
            args.output / f"{experiment}_appendix_c0_resolution.png",
        )
    print(build_compact_page(args.output))


if __name__ == "__main__":
    main()
