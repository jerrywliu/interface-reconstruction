import json

import pytest

from experiments.baselines.external_runner import run_external_static_baseline
from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.baselines.run_pcic_project_smoke import (
    _complete_7x7_block,
    _pcic_method,
    aggregate_pcic_rows,
    run_case,
)


def test_complete_7x7_block_is_centered_and_rejects_boundary():
    case = canonical_benchmark_cases("circles", (0,))[0]
    mesh = case.build_mesh(32)
    case.initialize_fractions(mesh)
    block = _complete_7x7_block(mesh, 16, 16)
    assert len(block) == 7
    assert all(len(column) == 7 for column in block)
    assert block[3][3] is mesh.polys[16][16]
    assert _complete_7x7_block(mesh, 2, 16) is None


@pytest.mark.parametrize("correction", ("translate_center", "adjust_radius"))
def test_pcic_project_adapter_runs_frozen_variant(correction):
    case = canonical_benchmark_cases("circles", (0,))[0]
    mesh = case.build_mesh(32)
    case.initialize_fractions(mesh)
    result = run_external_static_baseline(
        mesh,
        _pcic_method(mesh, correction),
        source_method="PCIC",
        source_variant=(
            "bare PCIC (center translation)"
            if correction == "translate_center"
            else "bare PCIC (radius adjustment)"
        ),
    )
    assert result.metadata["status_counts"]["reconstructed"] > 0
    assert result.metadata["status_counts"]["unresolved"] == 0
    assert all(
        record.diagnostics["required_halo"] == "7x7" for record in result.cells.values()
    )


def test_case_output_contains_parameters_counts_and_unsigned_curvature(tmp_path):
    case = canonical_benchmark_cases("circles", (0,))[0]
    row = run_case(case, 32, "adjust_radius", tmp_path)
    assert json.loads(row["parameters_json"])["parameters"]["radius"] == 10.0
    assert row["curvature_estimator_samples"] > 0
    assert row["curvature_estimator_mean_absolute_error"] >= 0.0
    assert row["reconstructed_components"] >= row["reconstructed_cells"]
    assert (tmp_path / "geometry/adjust_radius_circles_N32_case00.json").exists()
    assert (tmp_path / "cells/adjust_radius_circles_N32_case00.csv").exists()


def test_variant_aggregation_does_not_pool_corrections():
    template = {
        "method": "PCIC",
        "variant": "variant",
        "benchmark": "circles",
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
        "paper_fallback_cells": 1,
        "unsupported_cells": 1,
        "unresolved_cells": 0,
        "optimizer_failures": 0,
        "unmatched_crossings": 2,
        "reconstructed_components": 8,
        "reconstructed_primitives": 8,
        "multi_component_cells": 0,
        "boundary_7x7_unsupported_cells": 1,
        "straight_limit_fallback_cells": 1,
    }
    rows = [
        {**template, "correction": "translate_center", "sampled_hausdorff": 1.0},
        {**template, "correction": "adjust_radius", "sampled_hausdorff": 3.0},
    ]
    summary = aggregate_pcic_rows(rows)
    assert len(summary) == 2
    assert sorted(item["sampled_hausdorff_median"] for item in summary) == [1.0, 3.0]
