import copy

import pytest

from experiments.baselines.run_ellipse_circular_variant_metrics import (
    VARIANTS,
    build_reconstruction_command,
    compare_native_geometry,
    signed_arc_diagnostics,
    summarize_case_results,
)


def _arc(radius, signed_delta=0.25):
    return {
        "kind": "arc",
        "p_left": [0.0, 0.0],
        "p_right": [1.0, 0.0],
        "center": [0.5, 1.0],
        "radius": radius,
        "signed_delta": signed_delta,
    }


def test_matched_commands_keep_exact_variant_contract():
    commands = {
        variant["key"]: build_reconstruction_command(
            variant, 64, (0, 1, 2, 3, 4), "matched"
        )
        for variant in VARIANTS
    }
    assert (
        commands["per_cell_circular"][
            commands["per_cell_circular"].index("--facet_algo") + 1
        ]
        == "safe_circle"
    )
    assert (
        commands["graph_coordinated_circular"][
            commands["graph_coordinated_circular"].index("--facet_algo") + 1
        ]
        == "circular"
    )
    assert (
        commands["graph_coordinated_circular"][
            commands["graph_coordinated_circular"].index("--do_c0") + 1
        ]
        == "0"
    )
    assert (
        commands["graph_coordinated_circular_guarded_c0"][
            commands["graph_coordinated_circular_guarded_c0"].index("--do_c0") + 1
        ]
        == "1"
    )
    assert (
        commands["per_cell_circular"][
            commands["per_cell_circular"].index("--resolution") + 1
        ]
        == "0.64"
    )


def test_signed_arc_diagnostics_preserve_concavity_information():
    diagnostics = signed_arc_diagnostics(
        {"primitives": [_arc(2.0), _arc(-4.0), {"kind": "line"}]}
    )
    assert diagnostics["concave_arc_count"] == 1
    assert diagnostics["concave_arc_fraction"] == pytest.approx(0.5)
    assert diagnostics["concave_arc_length_fraction"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["signed_curvature_arc_length_mean"] == pytest.approx(0.0)


def test_native_geometry_comparison_ignores_bookkeeping_only():
    first = {"primitives": [_arc(2.0)]}
    second = copy.deepcopy(first)
    first["primitives"][0]["facet_index"] = 10
    second["primitives"][0]["facet_index"] = 11
    assert compare_native_geometry(first, second) == (True, True, 0.0)

    second["primitives"][0]["center"][0] += 5.0e-13
    exact, within_tolerance, maximum = compare_native_geometry(first, second)
    assert not exact
    assert within_tolerance
    assert maximum == pytest.approx(5.0e-13)


def test_summary_recovers_variant_specific_orders():
    rows = []
    for variant_index, variant in enumerate(VARIANTS, start=1):
        for resolution in (32, 64, 128):
            h = 100.0 / resolution
            for case_index, factor in enumerate((0.8, 0.9, 1.0, 1.1, 1.2)):
                row = {
                    "variant": variant["label"],
                    "case_index": case_index,
                    "cells_per_side": resolution,
                    "cell_size": h,
                    "num_mixed_cells": 20,
                    "num_merged_cells": 0,
                    "c0_adjustment_events": 2 if variant["do_c0"] else 0,
                    "c0_rejection_events": 0,
                }
                for metric in (
                    "native_symmetric_hausdorff",
                    "geometric_curvature_mean_absolute_error",
                    "geometric_curvature_rms_error",
                    "geometric_curvature_relative_l1_error",
                    "production_facet_gap",
                    "concave_arc_fraction",
                    "concave_arc_length_fraction",
                    "signed_curvature_arc_length_mean",
                ):
                    power = 3.0 if metric == "native_symmetric_hausdorff" else 1.0
                    row[metric] = factor * variant_index * h**power
                rows.append(row)

    summary, case_orders = summarize_case_results(rows)

    assert len(summary) == 9
    assert len(case_orders) == 15
    assert summary[0]["native_symmetric_hausdorff_fit_order"] == pytest.approx(3.0)
    assert summary[0][
        "geometric_curvature_mean_absolute_error_fit_order"
    ] == pytest.approx(1.0)
