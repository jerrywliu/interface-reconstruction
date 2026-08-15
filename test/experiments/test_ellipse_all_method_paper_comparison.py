import json
from pathlib import Path
import shutil

import pytest

from experiments.baselines.plot_ellipse_all_method_paper_comparison import (
    ERROR_PANELS,
    EXPECTED_CASE_INDICES,
    EXPECTED_RESOLUTIONS,
    PAPER_METHODS,
    compute_paired_win_counts,
    compute_summary,
    file_record,
    inspect_pdf_document,
    plot_paper_figure,
    validate_frozen_manifest,
    validate_frozen_summary,
    verify_hash_ledger,
    write_csv,
    write_hash_ledger,
)


def _synthetic_rows(methods=PAPER_METHODS, case_indices=(0, 1, 2)):
    rows = []
    for method_index, method in enumerate(methods, 1):
        for resolution in EXPECTED_RESOLUTIONS:
            for case_index in case_indices:
                factor = method_index * (1.0 + 0.1 * (case_index - 1))
                rows.append(
                    {
                        "method_id": method["id"],
                        "case_index": case_index,
                        "cells_per_side": resolution,
                        "native_symmetric_hausdorff": factor / resolution**3,
                        "geometric_curvature_mean_absolute_error": (
                            factor / resolution
                        ),
                        "facet_gap": (
                            0.0
                            if method["id"] in {"quasi", "ours_c0"}
                            else factor / resolution**3
                        ),
                        "normalized_conservation_residual": 1.0e-12,
                        "mixed_cells": 100,
                        "reconstructed_cells": 100 - (method_index - 1),
                        "reconstruction_coverage": 1.0 - (method_index - 1) / 100.0,
                    }
                )
    return rows


def test_compute_summary_matches_frozen_csv_contract(tmp_path):
    summary = compute_summary(_synthetic_rows())
    frozen_rows = []
    for row in summary:
        frozen = dict(row)
        for metric, *_ in ERROR_PANELS:
            frozen[f"{metric}_fit_order"] = 1.0
        frozen_rows.append(frozen)
    frozen_path = tmp_path / "summary.csv"
    write_csv(frozen_path, frozen_rows)

    validate_frozen_summary(summary, frozen_path)

    frozen_rows[0]["native_symmetric_hausdorff_median"] *= 2.0
    write_csv(frozen_path, frozen_rows)
    with pytest.raises(ValueError, match="frozen summary mismatch"):
        validate_frozen_summary(summary, frozen_path)


def test_paired_win_counts_cover_each_resolution_and_all_cases():
    methods = (
        {"id": "a", "label": "A"},
        {"id": "b", "label": "B"},
    )
    rows = []
    for resolution in EXPECTED_RESOLUTIONS:
        for case_index in (0, 1):
            rows.extend(
                [
                    {
                        "method_id": "a",
                        "case_index": case_index,
                        "cells_per_side": resolution,
                        "native_symmetric_hausdorff": 1.0,
                        "geometric_curvature_mean_absolute_error": 1.0,
                        "facet_gap": 1.0,
                        "reconstruction_coverage": 1.0,
                    },
                    {
                        "method_id": "b",
                        "case_index": case_index,
                        "cells_per_side": resolution,
                        "native_symmetric_hausdorff": 2.0,
                        "geometric_curvature_mean_absolute_error": 2.0,
                        "facet_gap": 2.0,
                        "reconstruction_coverage": 0.5,
                    },
                ]
            )

    counts = compute_paired_win_counts(rows, methods)

    assert len(counts) == (len(EXPECTED_RESOLUTIONS) + 1) * 4
    all_scope = [row for row in counts if row["cells_per_side"] == "all"]
    assert {row["metric"] for row in all_scope} == {
        "native_symmetric_hausdorff",
        "geometric_curvature_mean_absolute_error",
        "facet_gap",
        "reconstruction_coverage",
    }
    assert all(row["paired_count"] == 16 for row in all_scope)
    assert all(row["method_a_wins"] == 16 for row in all_scope)
    assert all(row["ties"] == 0 and row["method_b_wins"] == 0 for row in all_scope)


def test_frozen_manifest_hash_validation_detects_tampering(tmp_path):
    case_path = tmp_path / "case_metrics.csv"
    summary_path = tmp_path / "summary.csv"
    case_path.write_text("case\n", encoding="utf-8")
    summary_path.write_text("summary\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "analysis_git_head": "abc123",
        "case_row_count": len(PAPER_METHODS)
        * len(EXPECTED_RESOLUTIONS)
        * len(EXPECTED_CASE_INDICES),
        "cells_per_side": list(EXPECTED_RESOLUTIONS),
        "case_indices": list(EXPECTED_CASE_INDICES),
        "selected_methods": [{"id": method["id"]} for method in PAPER_METHODS],
        "analysis_sources": [],
        "inputs": [],
        "artifacts": [file_record(case_path), file_record(summary_path)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    assert validate_frozen_manifest(tmp_path) == payload

    case_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_frozen_manifest(tmp_path)


def test_hash_ledger_detects_artifact_changes(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    ledger = tmp_path / "SHA256SUMS"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    write_hash_ledger(ledger, (first, second))
    assert verify_hash_ledger(ledger) == []

    second.write_text("changed\n", encoding="utf-8")
    assert "SHA-256 mismatch" in verify_hash_ledger(ledger)[0]


@pytest.mark.skipif(
    any(
        shutil.which(tool) is None
        for tool in ("pdfimages", "pdffonts", "pdfinfo", "pdftotext")
    ),
    reason="Poppler command-line tools are required for vector PDF QA",
)
def test_paper_plot_is_one_page_vector_pdf_with_approved_labels(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    png_path = tmp_path / "paper.png"

    plot_paper_figure(compute_summary(_synthetic_rows()), pdf_path, png_path)
    report = inspect_pdf_document(pdf_path)

    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
    assert report["pages"] == 1
    assert report["vector_qa"]["passed"]
    assert report["vector_qa"]["image_objects"] == 0
    assert all(font["embedded"] for font in report["vector_qa"]["fonts"])
