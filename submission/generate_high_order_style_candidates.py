#!/usr/bin/env python3
"""Generate review-only color and typography candidates for high-order panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.baselines.plot_circle_all_method_paper_comparison import (
    CIRCLE_TRIANGLE_ANCHORS,
)
from experiments.baselines.plot_ellipse_all_method_paper_comparison import (
    REPO_ROOT,
    compute_summary,
    load_case_metrics,
    paper_methods,
    plot_paper_figure,
)


INPUTS = {
    "circles": REPO_ROOT
    / "experiments/baselines/results/circle_all_method_25case_comparison_20260814",
    "ellipses": REPO_ROOT
    / (
        "experiments/baselines/results/"
        "ellipse_all_method_25case_extended_comparison_20260814"
    ),
}

PALETTES = (
    "b19_categorical",
    "semantic_hybrid",
    "colorblind_categorical",
    "grouped",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "results/submission/figure_consistency_20260903/color_candidates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for benchmark, input_root in INPUTS.items():
        summary = compute_summary(load_case_metrics(input_root / "case_metrics.csv"))
        for palette in PALETTES:
            output_dir = output_root / palette
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{benchmark}_higher_order_{palette}"
            plot_paper_figure(
                summary,
                output_dir / f"{stem}.pdf",
                output_dir / f"{stem}.png",
                triangle_anchors=(
                    CIRCLE_TRIANGLE_ANCHORS if benchmark == "circles" else None
                ),
                methods=paper_methods(palette),
                figure_size=(9.2, 6.2),
                large_text=True,
            )

    print(output_root)


if __name__ == "__main__":
    main()
