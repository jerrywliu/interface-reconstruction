#!/usr/bin/env python3
"""Render representative paper plots with the proposed method color mapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.baselines.plot_ellipse_all_method_paper_comparison import (
    DEFAULT_INPUT_ROOT as HIGH_ORDER_INPUT,
    PAPER_METHOD_SPECS,
    compute_summary,
    load_case_metrics,
    plot_paper_figure,
)
import experiments.static.generate_pooled_perturbed_panels as pooled
import experiments.static.run_perturbed_sweeps as perturbed
from submission.pdf_vector_qa import inspect_pdf


OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "submission"
    / "figure_consistency_20260903"
    / "method_color_plot_candidates_v2"
)

METHOD_COLORS = {
    "Youngs": "#D94F9D",
    "ELVIRA": "#00A6C8",
    "LVIRA": "#84B547",
    "safe_linear": "#74A9CF",
    "linear": "#2F6FA3",
    "linear+C0": "#2F6FA3",
    "linear+corner": "#173F73",
    "safe_circle": "#E6AB02",
    "circular": "#D55E00",
    "circular+C0": "#D55E00",
    "circular+corner": "#B91C1C",
    "circular+corner+C0": "#B91C1C",
}

METHOD_LINESTYLES = {
    "Youngs": "-",
    "ELVIRA": "--",
    "LVIRA": "-.",
    "safe_linear": "--",
    "linear": "-",
    "linear+C0": ":",
    "linear+corner": "-",
    "safe_circle": "--",
    "circular": "-",
    "circular+C0": ":",
    "circular+corner": "-",
    "circular+corner+C0": ":",
}

METHOD_MARKERS = {
    "Youngs": "o",
    "ELVIRA": "s",
    "LVIRA": "D",
    "safe_linear": "^",
    "linear": "v",
    "linear+C0": "P",
    "linear+corner": "X",
    "safe_circle": "^",
    "circular": "v",
    "circular+C0": "P",
    "circular+corner": "X",
    "circular+corner+C0": "P",
}

HIGH_ORDER_COLORS = {
    "plvira": "#2D7D64",
    "pcic_center": "#7C5AA6",
    "quasi": "#4B5563",
    "ours_per_cell": "#E6AB02",
    "ours_graph": "#D55E00",
    "ours_c0": "#D55E00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proposed_high_order_methods() -> tuple[dict, ...]:
    methods = []
    for method in PAPER_METHOD_SPECS:
        candidate = {**method, "color": HIGH_ORDER_COLORS[method["id"]]}
        if method["id"] == "ours_c0":
            candidate["linestyle"] = ":"
        methods.append(candidate)
    return tuple(methods)


def render_perturbed_candidates() -> list[Path]:
    data = pooled.load_case_index(
        pooled.DEFAULT_SOURCE / "diagnostics" / "case_metrics.csv",
        pooled.DEFAULT_CURVATURE,
    )
    original_styles = {key: dict(value) for key, value in perturbed.METHOD_STYLES.items()}
    original_markers = dict(pooled.MARKERS_BY_LABEL)
    try:
        for method, color in METHOD_COLORS.items():
            perturbed.METHOD_STYLES[method]["color"] = color
            perturbed.METHOD_STYLES[method]["linestyle"] = METHOD_LINESTYLES[method]
        pooled.MARKERS_BY_LABEL = {
            perturbed.DISPLAY_LABELS.get(method, method): marker
            for method, marker in METHOD_MARKERS.items()
        }

        outputs = []
        for benchmark in ("squares", "zalesak"):
            output = OUTPUT_ROOT / f"{benchmark}_method_colors_v2.png"
            pooled._plot_grid(
                data,
                benchmark,
                ("hausdorff", "facet_gap"),
                output,
            )
            outputs.append(output.with_suffix(".pdf"))
        return outputs
    finally:
        perturbed.METHOD_STYLES.clear()
        perturbed.METHOD_STYLES.update(original_styles)
        pooled.MARKERS_BY_LABEL = original_markers


def render_higher_order_candidate() -> Path:
    summary = compute_summary(load_case_metrics(HIGH_ORDER_INPUT / "case_metrics.csv"))
    pdf = OUTPUT_ROOT / "ellipse_higher_order_method_colors_v2.pdf"
    png = pdf.with_suffix(".png")
    plot_paper_figure(
        summary,
        pdf,
        png,
        methods=proposed_high_order_methods(),
        figure_size=(9.2, 6.2),
        large_text=True,
    )
    return pdf


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = render_perturbed_candidates()
    outputs.append(render_higher_order_candidate())

    records = []
    for output in outputs:
        report = inspect_pdf(output, require_fonts=True)
        if not report.passed:
            raise RuntimeError(f"vector QA failed for {output}: {report.issues}")
        records.append(
            {
                "path": str(output.resolve()),
                "sha256": sha256(output),
                "vector_qa": True,
            }
        )

    manifest = {
        "purpose": "review-only method color and marker candidates",
        "scientific_data_changed": False,
        "method_colors": METHOD_COLORS,
        "higher_order_colors": HIGH_ORDER_COLORS,
        "outputs": records,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
