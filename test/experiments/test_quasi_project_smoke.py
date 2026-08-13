import csv
import json

import pytest

from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.baselines.run_quasi_project_smoke import (
    QUASI_CASE_FIELDS,
    _aggregate,
    _external_result,
    _fallback_reasons_by_cell,
    run_case,
)
from main.algos.baselines.quasi import reconstruct_quasi


def test_fallback_messages_map_to_every_referenced_cell():
    message = (
        "Section 2.5 correction retained the area-preserving local quadratic "
        "in (3, 4) for pair (4, 5)"
    )

    reasons = _fallback_reasons_by_cell([message])

    assert set(reasons) == {(3, 4), (4, 5)}
    assert reasons[(3, 4)] == [message]


def test_aggregate_parses_serialized_sweep_booleans():
    common = {
        "method": "QUASI",
        "variant": "frozen",
        "benchmark": "lines",
        "cells_per_side": 8,
        "sampled_hausdorff": 1.0,
        "sampled_reconstruction_to_truth": 1.0,
        "shared_edge_gap_mean": 1.0,
        "shared_edge_gap_max": 1.0,
        "conservation_max_absolute_residual": 1.0e-12,
        "curvature_estimator_median_absolute_error": 1.0,
        "runtime_seconds": 0.1,
        "mixed_cells": 2,
        "reconstructed_cells": 2,
        "paper_fallback_cells": 0,
        "unsupported_cells": 0,
        "unresolved_cells": 0,
        "optimizer_failures": 0,
        "unmatched_crossings": 0,
        "conservative_fallback_cells": 0,
        "unresolved_events": 0,
        "joins": 1,
        "c1_updates": 1,
        "c1_misses": 0,
        "curvature_updates": 0,
        "vertex_jumps": 0,
        "sweeps_completed": 1,
    }

    summary = _aggregate(
        [{**common, "sweep_converged": "False"}, {**common, "sweep_converged": "True"}]
    )

    assert summary[0]["sweep_converged_cases"] == 1
    assert summary[0]["sweep_converged_fraction"] == pytest.approx(0.5)


def test_single_case_writes_native_quadratics_and_conservative_metrics(tmp_path):
    case = canonical_benchmark_cases("lines", (0,))[0]

    row = run_case(case, 8, tmp_path)

    assert set(QUASI_CASE_FIELDS).issuperset(row)
    assert row["mixed_cells"] > 0
    assert row["curvature_estimator_samples"] == row["mixed_cells"]
    assert row["conservation_max_absolute_residual"] < 1.0e-8
    geometry_path = tmp_path / "geometry" / "lines_N8_case00.json"
    payload = json.loads(geometry_path.read_text())
    assert payload["cells"]
    assert {
        primitive["kind"]
        for cell in payload["cells"]
        for component in cell["components"]
        for primitive in component["primitives"]
    } == {"quadratic"}
    with (tmp_path / "cells" / "lines_N8_case00.csv").open(newline="") as stream:
        cell_rows = list(csv.DictReader(stream))
    assert len(cell_rows) == row["mixed_cells"]
    assert all("curvature_absolute_error" in item for item in cell_rows)


def test_external_result_accounts_for_every_mixed_cell_when_quasi_omits_a_facet():
    case = canonical_benchmark_cases("lines", (2,))[0]
    mesh = case.build_mesh(64)
    case.initialize_fractions(mesh)
    quasi_result = reconstruct_quasi(mesh)

    result = _external_result(mesh, quasi_result, case, 64)
    expected = {
        (x, y)
        for x, column in enumerate(mesh.polys)
        for y, polygon in enumerate(column)
        if polygon.isMixed(tolerance=1.0e-10)
    }
    missing = expected - set(quasi_result.facets)

    assert set(result.cells) == expected
    assert result.metadata["status_counts"]["unresolved"] == len(missing)
    assert all(result.cells[index].diagnostics["missing_facet"] for index in missing)
