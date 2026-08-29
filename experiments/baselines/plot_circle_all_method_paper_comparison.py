#!/usr/bin/env python3
"""Render the frozen Cartesian circle comparison in the ellipse paper style."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.baselines.plot_ellipse_all_method_paper_comparison import (
    REPO_ROOT,
    compute_paired_win_counts,
    compute_summary,
    inspect_pdf_document,
    load_case_metrics,
    plot_paper_figure,
    validate_frozen_manifest,
    validate_frozen_summary,
    verify_hash_ledger,
    write_csv,
    write_hash_ledger,
    write_paper_manifest,
)


DEFAULT_INPUT_ROOT = REPO_ROOT / (
    "experiments/baselines/results/" "circle_all_method_25case_comparison_20260814"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT
CIRCLE_TRIANGLE_ANCHORS = {
    "native_symmetric_hausdorff": (0.52, 0.36),
    "geometric_curvature_mean_absolute_error": (0.52, 0.42),
}


def validate_circle_manifest(input_root: Path) -> dict:
    """Verify the frozen common package and require the circle benchmark."""

    payload = validate_frozen_manifest(input_root)
    if payload.get("benchmark") != "circles":
        raise ValueError(
            f"frozen manifest benchmark is {payload.get('benchmark')!r}, "
            "expected 'circles'"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    frozen_manifest = validate_circle_manifest(input_root)
    case_rows = load_case_metrics(input_root / "case_metrics.csv")
    summary = compute_summary(case_rows)
    validate_frozen_summary(summary, input_root / "summary.csv")
    win_counts = compute_paired_win_counts(case_rows)

    pdf_path = output_root / "circle_all_methods_metrics_paper.pdf"
    png_path = output_root / "circle_all_methods_metrics_paper.png"
    win_path = output_root / "circle_all_methods_paired_win_counts.csv"
    hash_path = output_root / "circle_all_methods_metrics_paper.sha256"
    manifest_path = output_root / "circle_all_methods_metrics_paper.manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    plot_paper_figure(
        summary,
        pdf_path,
        png_path,
        triangle_anchors=CIRCLE_TRIANGLE_ANCHORS,
    )
    write_csv(win_path, win_counts)
    write_hash_ledger(hash_path, (pdf_path, png_path, win_path))
    hash_issues = verify_hash_ledger(hash_path)
    if hash_issues:
        raise ValueError(
            "generated artifact hash verification failed:\n" + "\n".join(hash_issues)
        )
    pdf_audit = inspect_pdf_document(pdf_path)
    write_paper_manifest(
        manifest_path,
        input_root=input_root,
        frozen_manifest=frozen_manifest,
        case_rows=case_rows,
        summary=summary,
        win_counts=win_counts,
        artifacts=(pdf_path, png_path, win_path, hash_path),
        pdf_audit=pdf_audit,
        artifact_kind="paper-ready Cartesian circle comparison",
        benchmark="circles",
        generator_sources=(
            Path(__file__).resolve(),
            REPO_ROOT
            / "experiments/baselines/plot_ellipse_all_method_paper_comparison.py",
        ),
    )

    for artifact in (pdf_path, png_path, win_path, hash_path, manifest_path):
        print(artifact)
    print(
        "QA: frozen hashes verified; 1-page vector PDF; no raster objects; "
        "all fonts embedded; ellipse paper style shared exactly"
    )


if __name__ == "__main__":
    main()
