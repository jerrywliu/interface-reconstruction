import csv

from experiments.baselines.build_circle_all_method_comparison import (
    DEFAULT_CASE_INDICES,
    DEFAULT_METHOD_IDS,
    DEFAULT_RESOLUTIONS,
)
from experiments.baselines.build_ellipse_all_method_comparison import (
    METHODS,
    assemble_case_metrics,
)


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_circle_builder_defaults_freeze_six_method_full_grid():
    assert DEFAULT_METHOD_IDS == (
        "ours_per_cell",
        "ours_graph",
        "ours_c0",
        "plvira",
        "pcic_center",
        "quasi",
    )
    assert DEFAULT_RESOLUTIONS == (32, 50, 64, 100, 128, 150, 256, 300)
    assert DEFAULT_CASE_INDICES == tuple(range(25))


def test_common_builder_assembles_circle_rows_without_ellipse_leakage(tmp_path):
    native = tmp_path / "native.csv"
    diagnostics = tmp_path / "diagnostics.csv"
    ours = tmp_path / "ours.csv"
    _write_rows(
        native,
        [
            {
                "method": "PLVIRA",
                "variant": "PLVIRA",
                "benchmark": benchmark,
                "cells_per_side": 32,
                "case_index": 0,
                "cell_size": 3.125,
                "native_symmetric_hausdorff": 0.1,
                "geometric_curvature_mean_absolute_error": 0.01,
            }
            for benchmark in ("circles", "ellipses")
        ],
    )
    _write_rows(
        diagnostics,
        [
            {
                "method": "PLVIRA",
                "variant": "PLVIRA",
                "benchmark": benchmark,
                "cells_per_side": 32,
                "case_index": 0,
                "mixed_cells": 10,
                "reconstructed_cells": 9,
                "paper_fallback_cells": 0,
                "unsupported_cells": 1,
                "unresolved_cells": 0,
                "shared_edge_gap_mean": 0.001,
                "conservation_max_absolute_residual": 1.0e-12,
            }
            for benchmark in ("circles", "ellipses")
        ],
    )
    _write_rows(
        ours,
        [
            {
                "method": "Ours",
                "variant": "per-cell circular",
                "benchmark": benchmark,
                "cells_per_side": 32,
                "case_index": 0,
                "cell_size": 3.125,
                "num_mixed_cells": 10,
                "native_symmetric_hausdorff": 1.0e-6,
                "geometric_curvature_mean_absolute_error": 1.0e-7,
                "production_facet_gap": 1.0e-8,
                "normalized_conservation_residual": 1.0e-13,
            }
            for benchmark in ("circles", "ellipses")
        ],
    )
    methods_by_id = {method["id"]: method for method in METHODS}
    methods = (methods_by_id["ours_per_cell"], methods_by_id["plvira"])

    rows = assemble_case_metrics(
        native, (diagnostics,), ours, methods, benchmark="circles"
    )

    assert len(rows) == 2
    assert {row["method_id"] for row in rows} == {"ours_per_cell", "plvira"}
    assert all(row["cells_per_side"] == 32 for row in rows)
