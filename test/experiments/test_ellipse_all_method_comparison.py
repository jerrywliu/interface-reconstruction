import math

import pytest

from experiments.baselines.build_ellipse_all_method_comparison import (
    METHODS,
    RESOLUTIONS,
    plot_summary,
    summarize_case_metrics,
)


def _synthetic_cases():
    rows = []
    for method in METHODS:
        for resolution in RESOLUTIONS:
            for case_index, factor in enumerate((0.9, 1.0, 1.1)):
                unresolved = (
                    1 if method["id"] == "pcic_center" and resolution == 128 else 0
                )
                rows.append(
                    {
                        "method_id": method["id"],
                        "display_label": method["label"],
                        "method": method["method"],
                        "variant": method["variant"],
                        "case_index": case_index,
                        "cells_per_side": resolution,
                        "cell_size": 100.0 / resolution,
                        "native_symmetric_hausdorff": factor / resolution**2,
                        "geometric_curvature_mean_absolute_error": (
                            factor / resolution
                        ),
                        "facet_gap": (
                            0.0 if method["id"] == "quasi" else factor / resolution**3
                        ),
                        "mixed_cells": 10,
                        "reconstructed_cells": 10 - unresolved,
                        "unsupported_cells": 0,
                        "unresolved_cells": unresolved,
                    }
                )
    return rows


def test_summary_recovers_orders_and_weighted_coverage():
    summary = summarize_case_metrics(_synthetic_cases())

    per_cell = [row for row in summary if row["method_id"] == "ours_per_cell"]
    assert per_cell[0]["native_symmetric_hausdorff_fit_order"] == pytest.approx(2.0)
    assert per_cell[0][
        "geometric_curvature_mean_absolute_error_fit_order"
    ] == pytest.approx(1.0)
    assert per_cell[0]["facet_gap_fit_order"] == pytest.approx(3.0)

    pcic_fine = next(
        row
        for row in summary
        if row["method_id"] == "pcic_center" and row["cells_per_side"] == 128
    )
    assert pcic_fine["reconstruction_coverage"] == pytest.approx(0.9)

    quasi = next(row for row in summary if row["method_id"] == "quasi")
    assert math.isnan(quasi["facet_gap_fit_order"])


def test_plot_handles_exact_zero_quasi_gap(tmp_path):
    path = tmp_path / "comparison.pdf"

    plot_summary(summarize_case_metrics(_synthetic_cases()), path)

    assert path.exists()
    assert path.stat().st_size > 0
