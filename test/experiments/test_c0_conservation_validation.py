import pytest

from experiments.submission.c0_conservation_validation import (
    aggregate_paired_rows,
    build_paired_case_row,
    conservation_regression_rows,
)


def _stage(stage, *, geometry="same", gap=0.2, global_error=1.0e-12):
    return {
        "experiment": "ellipses",
        "facet_algo": "circular",
        "resolution": 0.64,
        "wiggle": 0.1,
        "seed": 0,
        "case_index": 3,
        "save_name": stage,
        "geometry_fingerprint": geometry if stage == "before" else "adjusted",
        "input_geometry_fingerprint": geometry,
        "case_geometry_fingerprint": "analytic-case",
        "global_relative_phase_area_error": global_error,
        "max_merged_zone_absolute_residual": 2.0e-10,
        "facet_gap": gap,
        "num_final_missing_cells": 0,
        "failure_count": 0,
    }


def _audit(eligible=4, changed=3):
    return {
        "num_eligible_joins": eligible,
        "num_changed_eligible_joins": changed,
        "fraction_eligible_joins_changed": changed / eligible,
        "num_explicit_corner_facets": 0,
        "num_unresolved_fallback_facets": 1,
        "num_missing_facets_before_c0": 0,
        "num_missing_facets_after_c0": 0,
    }


def test_pairing_keeps_stage_values_and_deltas():
    before = _stage("before", gap=0.2, global_error=1.0e-12)
    after = _stage("after", gap=1.0e-10, global_error=4.0e-12)
    pair = build_paired_case_row(before, after, _audit())

    assert pair["facet_gap_before_c0"] == pytest.approx(0.2)
    assert pair["facet_gap_after_c0"] == pytest.approx(1.0e-10)
    assert pair["global_relative_phase_area_error_delta_after_minus_before"] == pytest.approx(
        3.0e-12
    )
    assert pair["num_changed_eligible_joins"] == 3


def test_pairing_rejects_nonidentical_pre_c0_reconstruction():
    before = _stage("before", geometry="off")
    after = _stage("after", geometry="on")
    with pytest.raises(ValueError, match="pre-adjustment geometry"):
        build_paired_case_row(before, after, _audit())


def test_aggregation_weights_join_fraction_and_counts_failures():
    first = build_paired_case_row(
        _stage("before", gap=0.2), _stage("after", gap=0.0), _audit(4, 3)
    )
    second_before = {**_stage("before", gap=0.4), "case_index": 4}
    second_after = {
        **_stage("after", gap=0.1),
        "case_index": 4,
        "num_final_missing_cells": 1,
    }
    second = build_paired_case_row(second_before, second_after, _audit(6, 1))

    summary = aggregate_paired_rows([first, second])
    assert summary["median_facet_gap_before_c0"] == pytest.approx(0.3)
    assert summary["num_eligible_joins"] == 10
    assert summary["num_changed_eligible_joins"] == 4
    assert summary["fraction_eligible_joins_changed"] == pytest.approx(0.4)
    assert summary["num_missing_facets_after_c0"] == 1


def test_empty_merged_zone_metric_is_preserved_as_missing():
    before = {**_stage("before"), "max_merged_zone_absolute_residual": None}
    after = {**_stage("after"), "max_merged_zone_absolute_residual": None}
    pair = build_paired_case_row(before, after, _audit())
    summary = aggregate_paired_rows([pair])

    assert summary["median_max_merged_zone_absolute_residual_before_c0"] is None
    assert summary["max_max_merged_zone_absolute_residual_after_c0"] is None


def test_material_conservation_regressions_are_selected():
    before = _stage("before", global_error=1.0e-12)
    after = _stage("after", global_error=2.0e-4)
    pair = build_paired_case_row(before, after, _audit())

    regressions = conservation_regression_rows([pair])
    assert len(regressions) == 1
    assert regressions[0]["exceeds_global_error_threshold"] == 1
