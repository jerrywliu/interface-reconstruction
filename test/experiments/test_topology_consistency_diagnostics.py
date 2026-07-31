import json
import math

import pytest

from experiments.submission.topology_consistency_diagnostics import (
    FULL_METHODS,
    FULL_RESOLUTIONS,
    StructuredMesh,
    _aggregate_case_rows_by,
    _ensure_output_outside_input_release,
    _full_readme,
    _resolve_run_bundle,
    build_full_selectors,
    classify_vertex,
    evaluate_case,
)


def _two_cell_mesh():
    points = tuple(
        tuple((float(cell_x), float(cell_y)) for cell_y in range(2))
        for cell_x in range(3)
    )
    return StructuredMesh(points=points, nx=3, ny=2, domain_diagonal=math.sqrt(5.0))


def _record(cell_id, geometry, *, oriented=True, fallback=False):
    cell_x, cell_y = cell_id
    return {
        "cell_id": f"{cell_x},{cell_y}",
        "cell_x": str(cell_x),
        "cell_y": str(cell_y),
        "merge_id": str(cell_x),
        "orientation_status": "oriented" if oriented else "unresolved_or_deadend",
        "construction_path": "plic_fallback" if fallback else "direct_fit",
        "fallback_policy": "LVIRA" if fallback else "",
        "facet_geometry_json": json.dumps(geometry),
    }


def _line(start, end):
    return {"class": "linear", "p_left": list(start), "p_right": list(end)}


def _arc(radius):
    if radius > 0:
        p_left, p_right = [1.25, 0.5], [1.0, 0.75]
    else:
        p_left, p_right = [1.0, 0.75], [1.25, 0.5]
    return {
        "class": "circular",
        "center": [1.0, 0.5],
        "radius": radius,
        "p_left": p_left,
        "p_right": p_right,
    }


def _corner(reverse=False):
    points = ([0.0, 0.5], [1.0, 0.25], [2.0, 0.5])
    if reverse:
        points = tuple(reversed(points))
    p_left, corner, p_right = points
    return {
        "class": "linear_corner",
        "p_left": p_left,
        "corner": corner,
        "p_right": p_right,
        "left_branch": _line(p_left, corner),
        "right_branch": _line(corner, p_right),
    }


def _curved_corner():
    p_left = [0.0, 0.5]
    corner = [1.0, 0.25]
    p_right = [2.0, 0.5]
    left_branch = {
        "class": "circular",
        "center": [0.5, 0.375],
        "radius": math.hypot(0.5, 0.125),
        "p_left": p_left,
        "p_right": corner,
    }
    return {
        "class": "curved_corner",
        "p_left": p_left,
        "corner": corner,
        "p_right": p_right,
        "left_branch": left_branch,
        "right_branch": _line(corner, p_right),
    }


def _scope(summary, name):
    return {
        "candidates": summary[f"{name}_candidate_shared_vertices"],
        "evaluated": summary[f"{name}_evaluated_shared_vertices"],
        "excluded": summary[f"{name}_on_facet_excluded_vertices"],
        "conflicts": summary[f"{name}_conflict_vertices"],
    }


def test_consistent_and_conflicting_lines():
    mesh = _two_cell_mesh()
    left = _line((0.0, 0.5), (1.0, 0.5))
    right = _line((1.0, 0.5), (2.0, 0.5))
    consistent, _ = evaluate_case(
        [_record((0, 0), left), _record((1, 0), right)], mesh, 0.0, 1.0e-12
    )
    conflicting, _ = evaluate_case(
        [_record((0, 0), left), _record((1, 0), _line((2.0, 0.5), (1.0, 0.5)))],
        mesh,
        0.0,
        1.0e-12,
    )

    assert _scope(consistent, "resolved") == {
        "candidates": 2,
        "evaluated": 2,
        "excluded": 0,
        "conflicts": 0,
    }
    assert _scope(conflicting, "resolved")["conflicts"] == 2


def test_consistent_and_conflicting_arcs():
    mesh = _two_cell_mesh()
    consistent, _ = evaluate_case(
        [_record((0, 0), _arc(0.25)), _record((1, 0), _arc(0.25))],
        mesh,
        0.0,
        1.0e-12,
    )
    conflicting, _ = evaluate_case(
        [_record((0, 0), _arc(0.25)), _record((1, 0), _arc(-0.25))],
        mesh,
        0.0,
        1.0e-12,
    )

    assert _scope(consistent, "complete")["conflicts"] == 0
    assert _scope(conflicting, "complete")["conflicts"] == 2


def test_consistent_and_conflicting_corners():
    mesh = _two_cell_mesh()
    consistent, _ = evaluate_case(
        [_record((0, 0), _corner()), _record((1, 0), _corner())],
        mesh,
        0.0,
        1.0e-12,
    )
    conflicting, _ = evaluate_case(
        [_record((0, 0), _corner()), _record((1, 0), _corner(reverse=True))],
        mesh,
        0.0,
        1.0e-12,
    )

    assert classify_vertex(_corner(), (1.0, 1.0), 1.0e-12) == "full"
    assert classify_vertex(_corner(), (1.0, 0.0), 1.0e-12) == "empty"
    assert _scope(consistent, "complete")["conflicts"] == 0
    assert _scope(conflicting, "complete")["conflicts"] == 2


def test_curved_corner_with_arc_branch_is_supported():
    geometry = _curved_corner()

    assert classify_vertex(geometry, (1.0, 1.0), 1.0e-12) in {"full", "empty"}
    assert classify_vertex(geometry, geometry["corner"], 1.0e-12) == "on_facet"


def test_vertices_on_facet_are_flagged_and_excluded():
    mesh = _two_cell_mesh()
    records = [
        _record((0, 0), _line((0.0, 0.0), (1.0, 0.0))),
        _record((1, 0), _line((1.0, 0.0), (2.0, 0.0))),
    ]
    summary, details = evaluate_case(records, mesh, 0.0, 1.0e-10)

    assert _scope(summary, "resolved") == {
        "candidates": 2,
        "evaluated": 1,
        "excluded": 1,
        "conflicts": 0,
    }
    excluded = [
        row
        for row in details
        if row["scope"] == "resolved" and row["on_facet_excluded"]
    ]
    assert len(excluded) == 1
    assert excluded[0]["vertex_y"] == 0.0


def test_fallback_cells_only_enter_complete_scope():
    mesh = _two_cell_mesh()
    records = [
        _record((0, 0), _line((0.0, 0.5), (1.0, 0.5))),
        _record(
            (1, 0),
            _line((2.0, 0.5), (1.0, 0.5)),
            oriented=False,
            fallback=True,
        ),
    ]
    summary, details = evaluate_case(records, mesh, 0.0, 1.0e-12)

    assert summary["num_plic_fallback_cells"] == 1
    assert _scope(summary, "resolved") == {
        "candidates": 0,
        "evaluated": 0,
        "excluded": 0,
        "conflicts": 0,
    }
    assert _scope(summary, "complete")["conflicts"] == 2
    complete_conflicts = [
        row for row in details if row["scope"] == "complete" and row["conflict"]
    ]
    assert all(row["contains_fallback"] for row in complete_conflicts)


def test_full_selector_generation_is_complete_and_unique():
    selectors = build_full_selectors()

    assert len(selectors) == 4 * 5 * 25 == 500
    assert len(set(selectors)) == len(selectors)
    assert {(item.experiment, item.algo) for item in selectors} == set(FULL_METHODS)
    assert {item.resolution for item in selectors} == set(FULL_RESOLUTIONS)
    assert {item.wiggle for item in selectors} == {0.1}
    assert {item.seed for item in selectors} == {0}
    assert {item.case_index for item in selectors} == set(range(25))


def test_aggregate_uses_mixed_cell_and_vertex_weighting():
    def row(mixed, oriented, resolved, fallback, evaluated, conflicts, excluded):
        values = {
            "experiment": "circles",
            "num_mixed_cells": mixed,
            "num_oriented_cells": oriented,
            "num_resolved_cells": resolved,
            "num_plic_fallback_cells": fallback,
        }
        for scope in ("resolved", "complete"):
            values.update(
                {
                    f"{scope}_candidate_shared_vertices": evaluated + excluded,
                    f"{scope}_evaluated_shared_vertices": evaluated,
                    f"{scope}_on_facet_excluded_vertices": excluded,
                    f"{scope}_invalid_excluded_vertices": 0,
                    f"{scope}_invalid_incident_labels": 0,
                    f"{scope}_conflict_vertices": conflicts,
                }
            )
        return values

    aggregate = _aggregate_case_rows_by(
        [
            row(2, 2, 2, 0, 2, 1, 1),
            row(8, 1, 1, 4, 8, 1, 2),
        ],
        ("experiment",),
    )[0]

    assert aggregate["case_count"] == 2
    assert aggregate["mixed_cells"] == 10
    assert aggregate["oriented_cell_fraction"] == 0.3
    assert aggregate["resolved_cell_fraction"] == 0.3
    assert aggregate["plic_fallback_cell_fraction"] == 0.4
    assert aggregate["resolved_evaluated_shared_vertices"] == 10
    assert aggregate["resolved_on_facet_excluded_vertices"] == 3
    assert aggregate["resolved_conflict_rate"] == 0.2


def test_final_release_relative_run_bundle_resolves_from_inventory(tmp_path):
    inventory = tmp_path / "release" / "diagnostics" / "run_inventory.csv"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("unused\n")

    assert (
        _resolve_run_bundle(inventory, "raw_runs/example")
        == (tmp_path / "release/raw_runs/example").resolve()
    )


def test_validation_output_cannot_be_inside_input_release(tmp_path):
    diagnostics = tmp_path / "release" / "diagnostics"
    diagnostics.mkdir(parents=True)
    metrics = diagnostics / "cell_metrics.csv"
    metrics.write_text("unused\n")

    with pytest.raises(ValueError, match="outside immutable release"):
        _ensure_output_outside_input_release(
            tmp_path / "release/validation/topology", metrics
        )
    _ensure_output_outside_input_release(tmp_path / "validation/topology", metrics)


def test_full_readme_uses_data_derived_case_and_fallback_counts():
    def aggregate(experiment, conflicts):
        return {
            "experiment": experiment,
            "algo": "circular",
            "case_count": 7,
            "mixed_cells": 20,
            "oriented_cell_fraction": 0.9,
            "plic_fallback_cell_fraction": 0.1,
            "resolved_evaluated_shared_vertices": 11,
            "resolved_on_facet_excluded_vertices": 1,
            "resolved_invalid_excluded_vertices": 0,
            "resolved_conflict_vertices": conflicts,
            "resolved_conflict_rate": conflicts / 11,
            "complete_evaluated_shared_vertices": 12,
            "complete_on_facet_excluded_vertices": 1,
            "complete_invalid_excluded_vertices": 0,
            "complete_conflict_vertices": conflicts,
            "complete_conflict_rate": conflicts / 12,
        }

    readme = _full_readme(
        [aggregate("ellipses", 2), aggregate("circles", 0)],
        [
            {
                **aggregate("all", 2),
                "relative_tolerance": 1.0e-10,
            }
        ],
        {
            "conflict_vertices": 2,
            "case_count": 2,
            "audited_case_count": 7,
            "fallback_involved_vertices": 1,
            "by_experiment": {
                "ellipses": {
                    "conflict_vertices": 2,
                    "case_count": 2,
                    "resolutions": [50.0],
                }
            },
        },
        {
            "input_validation": {
                "selector_count": 7,
                "setting_count": 1,
                "source_commits": ["a" * 40],
            }
        },
    )

    assert "`2/7` cases" in readme
    assert "`1` conflicting vertices contain a PLIC fallback" in readme
    assert "Zero conflicts at the paper tolerance: `circles`" in readme
    assert "July" not in readme
