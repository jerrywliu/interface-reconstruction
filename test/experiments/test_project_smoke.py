import numpy as np
import pytest

from experiments.baselines.project_smoke import (
    aggregate_case_rows,
    sampled_directed_hausdorff,
    sampled_symmetric_hausdorff,
)
from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.baselines.run_plvira_project_smoke import _plvira_method
from experiments.baselines.external_runner import run_external_static_baseline
from main.algos.baselines.external_geometry import (
    ExternalBaselineResult,
    ExternalCellReconstruction,
    ExternalInterfaceComponent,
    ExternalLinePrimitive,
    ExternalReconstructionStatus,
)


def _line_result(offset=0.0):
    primitive = ExternalLinePrimitive((0.0, offset), (1.0, offset))
    record = ExternalCellReconstruction(
        (0, 0),
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        (ExternalInterfaceComponent((primitive,)),),
        "test",
        "test",
        ExternalReconstructionStatus.RECONSTRUCTED,
        target_phase_area=0.5,
        stored_exact_phase_area=0.5,
    )
    return ExternalBaselineResult("test", "test", {(0, 0): record})


def test_sampled_hausdorff_has_explicit_geometric_meaning():
    truth = (ExternalLinePrimitive((0.0, 0.25), (1.0, 0.25)),)
    assert sampled_symmetric_hausdorff(_line_result(0.0), truth, spacing=0.1) == pytest.approx(0.25)
    assert sampled_directed_hausdorff(_line_result(0.0), truth, spacing=0.1) == pytest.approx(0.25)


def test_aggregate_rows_reports_case_medians_and_unresolved_fraction():
    template = {
        "method": "PLVIRA",
        "variant": "PLVIRA",
        "benchmark": "lines",
        "cells_per_side": 32,
        "sampled_hausdorff": 1.0,
        "sampled_reconstruction_to_truth": 0.5,
        "shared_edge_gap_mean": 2.0,
        "shared_edge_gap_max": 3.0,
        "conservation_max_absolute_residual": 4.0,
        "curvature_estimator_median_absolute_error": 5.0,
        "runtime_seconds": 6.0,
        "mixed_cells": 10,
        "reconstructed_cells": 8,
        "paper_fallback_cells": 0,
        "unsupported_cells": 1,
        "unresolved_cells": 1,
        "optimizer_failures": 0,
        "unmatched_crossings": 2,
    }
    rows = [template, {**template, "sampled_hausdorff": 3.0}]
    summary = aggregate_case_rows(rows)[0]
    assert summary["method"] == "PLVIRA"
    assert summary["variant"] == "PLVIRA"
    assert summary["sampled_hausdorff_median"] == pytest.approx(2.0)
    assert summary["unresolved_fraction"] == pytest.approx(0.2)


def test_aggregate_rows_never_pools_methods_or_variants():
    template = {
        "method": "PLVIRA",
        "variant": "PLVIRA",
        "benchmark": "circles",
        "cells_per_side": 32,
        "sampled_hausdorff": 1.0,
        "sampled_reconstruction_to_truth": 1.0,
        "shared_edge_gap_mean": 1.0,
        "shared_edge_gap_max": 1.0,
        "conservation_max_absolute_residual": 1.0,
        "curvature_estimator_median_absolute_error": 1.0,
        "runtime_seconds": 1.0,
        "mixed_cells": 1,
        "reconstructed_cells": 1,
        "paper_fallback_cells": 0,
        "unsupported_cells": 0,
        "unresolved_cells": 0,
        "optimizer_failures": 0,
        "unmatched_crossings": 0,
    }
    rows = [
        template,
        {**template, "method": "PCIC", "variant": "center translation"},
        {**template, "method": "PCIC", "variant": "radius adjustment"},
    ]
    summary = aggregate_case_rows(rows)
    assert {(row["method"], row["variant"]) for row in summary} == {
        ("PLVIRA", "PLVIRA"),
        ("PCIC", "center translation"),
        ("PCIC", "radius adjustment"),
    }
    assert all(row["cases"] == 1 for row in summary)


def test_aggregate_rows_counts_nonfinite_metrics_instead_of_hiding_them():
    template = {
        "method": "QUASI",
        "variant": "frozen",
        "benchmark": "ellipses",
        "cells_per_side": 64,
        "sampled_hausdorff": 1.0,
        "sampled_reconstruction_to_truth": 1.0,
        "shared_edge_gap_mean": 1.0,
        "shared_edge_gap_max": 1.0,
        "conservation_max_absolute_residual": 1.0,
        "curvature_estimator_median_absolute_error": 1.0,
        "runtime_seconds": 1.0,
        "mixed_cells": 1,
        "reconstructed_cells": 1,
        "paper_fallback_cells": 0,
        "unsupported_cells": 0,
        "unresolved_cells": 0,
        "optimizer_failures": 0,
        "unmatched_crossings": 0,
    }
    summary = aggregate_case_rows(
        [template, {**template, "sampled_hausdorff": float("inf")}]
    )[0]
    assert summary["sampled_hausdorff_median"] == pytest.approx(1.0)
    assert summary["sampled_hausdorff_nonfinite_cases"] == 1


def test_point_cloud_smoke_metric_has_a_partition_dependent_sampling_floor():
    truth = (ExternalLinePrimitive((0.0, 0.0), (1.0, 0.0)),)
    split = ExternalBaselineResult(
        "test",
        "test",
        {
            (0, 0): ExternalCellReconstruction(
                (0, 0),
                ((0.0, -1.0), (0.3, -1.0), (0.3, 1.0), (0.0, 1.0)),
                (
                    ExternalInterfaceComponent(
                        (ExternalLinePrimitive((0.0, 0.0), (0.3, 0.0)),)
                    ),
                ),
                "test",
                "test",
                ExternalReconstructionStatus.RECONSTRUCTED,
                target_phase_area=0.3,
                stored_exact_phase_area=0.3,
            ),
            (1, 0): ExternalCellReconstruction(
                (1, 0),
                ((0.3, -1.0), (1.0, -1.0), (1.0, 1.0), (0.3, 1.0)),
                (
                    ExternalInterfaceComponent(
                        (ExternalLinePrimitive((0.3, 0.0), (1.0, 0.0)),)
                    ),
                ),
                "test",
                "test",
                ExternalReconstructionStatus.RECONSTRUCTED,
                target_phase_area=0.7,
                stored_exact_phase_area=0.7,
            ),
        },
    )
    # The represented curves coincide exactly. The nonzero value is solely a
    # consequence of comparing independently sampled point clouds.
    assert sampled_symmetric_hausdorff(split, truth, spacing=0.2) > 0.0


def test_plvira_project_adapter_transposes_mesh_coordinates_into_row_column_order():
    case = canonical_benchmark_cases("circles", (0,))[0]
    mesh = case.build_mesh(32)
    case.initialize_fractions(mesh)
    result = run_external_static_baseline(
        mesh,
        _plvira_method(mesh, 100.0 / 32.0),
        source_method="PLVIRA",
        source_variant="PLVIRA",
    )
    assert result.metadata["status_counts"]["reconstructed"] == 24
    assert result.metadata["status_counts"]["unresolved"] == 0
    assert all(
        record.diagnostics["curvature_source"] == "cartesian-ghf"
        for record in result.cells.values()
    )
