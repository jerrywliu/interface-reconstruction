import math

import pytest

from experiments.baselines.project_benchmarks import canonical_benchmark_cases
from experiments.baselines.run_circle_circular_variant_metrics import (
    DEFAULT_CASE_INDICES,
    DEFAULT_RESOLUTIONS,
    VARIANTS,
    _validate_case_rows,
    build_reconstruction_command,
    verify_canonical_circle_geometry,
)


def test_circle_commands_preserve_exact_cartesian_variant_contract():
    commands = {
        variant["key"]: build_reconstruction_command(
            variant, 300, DEFAULT_CASE_INDICES, "matched_circle"
        )
        for variant in VARIANTS
    }
    per_cell = commands["per_cell_circular"]
    graph = commands["graph_coordinated_circular"]
    joint = commands["graph_coordinated_circular_joint_c0"]

    assert per_cell[2] == "experiments.static.circles"
    assert per_cell[per_cell.index("--resolution") + 1] == "3"
    assert per_cell[per_cell.index("--facet_algo") + 1] == "safe_circle"
    assert graph[graph.index("--facet_algo") + 1] == "circular"
    assert graph[graph.index("--do_c0") + 1] == "0"
    assert joint[joint.index("--do_c0") + 1] == "1"
    assert joint[joint.index("--c0_mode") + 1] == "joint"
    assert joint[joint.index("--perturb_wiggle") + 1] == "0.0"
    assert joint[joint.index("--num_circles") + 1] == "25"
    assert joint[joint.index("--case_indices") + 1] == ",".join(
        str(value) for value in range(25)
    )


def test_canonical_circle_geometry_requires_seed_41_case():
    case = canonical_benchmark_cases("circles", (24,))[0]
    saved = {
        "geometry_type": "circle",
        "case_index": 24,
        "center": list(case.parameters["center"]),
        "radius": case.parameters["radius"],
    }
    verify_canonical_circle_geometry(saved, case)

    saved["center"][0] += 1.0e-6
    with pytest.raises(ValueError, match=r"center\[0\]"):
        verify_canonical_circle_geometry(saved, case)


def _case_rows(resolutions=(32, 64), case_indices=(0, 1)):
    rows = []
    for variant in VARIANTS:
        for resolution in resolutions:
            for case_index in case_indices:
                rows.append(
                    {
                        "variant": variant["label"],
                        "cells_per_side": resolution,
                        "case_index": case_index,
                        "native_symmetric_hausdorff": 1.0e-5,
                        "geometric_curvature_mean_absolute_error": 1.0e-6,
                        "production_facet_gap": 0.0,
                        "normalized_conservation_residual": 1.0e-12,
                        "normalized_global_conservation_residual": 1.0e-13,
                    }
                )
    return rows


def test_circle_case_grid_gate_requires_unique_finite_support():
    rows = _case_rows()
    _validate_case_rows(rows, (32, 64), (0, 1))

    missing = rows[:-1]
    with pytest.raises(ValueError, match="exact variant-resolution-case grid"):
        _validate_case_rows(missing, (32, 64), (0, 1))

    nonfinite = _case_rows()
    nonfinite[0]["native_symmetric_hausdorff"] = math.nan
    with pytest.raises(ValueError, match="invalid native_symmetric_hausdorff"):
        _validate_case_rows(nonfinite, (32, 64), (0, 1))


def test_circle_defaults_are_exact_requested_matrix():
    assert DEFAULT_RESOLUTIONS == (32, 50, 64, 100, 128, 150, 256, 300)
    assert DEFAULT_CASE_INDICES == tuple(range(25))
