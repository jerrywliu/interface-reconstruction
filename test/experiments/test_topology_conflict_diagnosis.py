import math

from experiments.submission.diagnose_topology_conflicts import (
    TAXONOMY,
    _build_readme,
    assign_taxonomy,
    direct_phase_test,
)


def test_direct_phase_test_respects_signed_circle_complement():
    positive = {
        "class": "circular",
        "center": [0.0, 0.0],
        "radius": 2.0,
        "p_left": [2.0, 0.0],
        "p_right": [0.0, 2.0],
    }
    negative = {**positive, "radius": -2.0}

    assert direct_phase_test(positive, [0.0, 0.0], 1.0e-12)[0] == "full"
    assert direct_phase_test(positive, [3.0, 0.0], 1.0e-12)[0] == "empty"
    assert direct_phase_test(negative, [0.0, 0.0], 1.0e-12)[0] == "empty"
    assert direct_phase_test(negative, [3.0, 0.0], 1.0e-12)[0] == "full"


def test_direct_phase_test_normalizes_linear_margin():
    line = {
        "class": "linear",
        "p_left": [0.0, 0.0],
        "p_right": [2.0, 0.0],
    }
    label, margin = direct_phase_test(line, [1.0, 0.25], 1.0e-12)
    assert label == "full"
    assert math.isclose(margin, 0.25)


def test_taxonomy_prioritizes_auditable_data_and_diagnostic_failures():
    common = {
        "vtk_conflict": True,
        "exact_conflict": True,
        "metadata_mismatch": False,
        "classification_mismatch": False,
        "min_abs_exact_margin": 1.0e-2,
        "ambiguity_tolerance": 1.0e-6,
    }
    assert assign_taxonomy(**common) == TAXONOMY["a"]
    assert assign_taxonomy(**{**common, "metadata_mismatch": True}) == TAXONOMY["d"]
    assert (
        assign_taxonomy(**{**common, "classification_mismatch": True}) == TAXONOMY["b"]
    )
    assert assign_taxonomy(**{**common, "exact_conflict": False}) == TAXONOMY["c"]
    assert (
        assign_taxonomy(**{**common, "min_abs_exact_margin": 1.0e-8}) == TAXONOMY["e"]
    )


def test_conflict_readme_derives_all_incidence_denominators():
    taxonomy_row = {
        "experiment": "ellipses",
        "taxonomy": TAXONOMY["a"],
        "exact_conflict": 1,
        "vtk_labels": "full;empty",
        "exact_labels": "full;empty",
        "metadata_mismatch": 0,
        "classification_mismatch": 0,
        "contains_fallback": 1,
        "min_abs_exact_phase_margin": 2.0e-4,
        "ambiguity_tolerance": 1.0e-6,
    }

    readme = _build_readme(
        [taxonomy_row],
        [{"experiment": "ellipses", "case_index": 3}],
        {"audited_case_count": 4, "complete_evaluated_shared_vertices": 9},
    )

    assert "All 1 conflicts from the source full audit" in readme
    assert "`1/9` evaluated vertices" in readme
    assert "`1/4` audited cases" in readme
    assert "`1` flagged vertices use a PLIC fallback" in readme
    assert "22" not in readme
    assert "500" not in readme
