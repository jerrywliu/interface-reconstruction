from types import SimpleNamespace

import pytest

from main.algos import c0_refinement
from main.algos.c0_refinement import plan_joint_c0_refinement
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.polys.neighbored_polygon import NeighboredPolygon


def _two_cell_gap():
    first = NeighboredPolygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    second = NeighboredPolygon([[1, 0], [2, 0], [2, 1], [1, 1]])
    first_facet = LinearFacet([0, 0.25], [1, 0.45])
    second_facet = LinearFacet([1, 0.55], [2, 0.75])
    for poly, facet in ((first, first_facet), (second, second_facet)):
        poly.setFacet(facet)
        poly.setArea(poly._facet_phase_area(facet))
    first.setNeighbor(second, "right")
    second.setNeighbor(first, "left")
    return first, second


def _two_cell_kink():
    first = NeighboredPolygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    second = NeighboredPolygon([[1, 0], [2, 0], [2, 1], [1, 1]])
    first_facet = LinearFacet([0, 0.25], [1, 0.5])
    second_facet = LinearFacet([1, 0.5], [2, 0.85])
    for poly, facet in ((first, first_facet), (second, second_facet)):
        poly.setFacet(facet)
        poly.setArea(poly._facet_phase_area(facet))
    first.setNeighbor(second, "right")
    second.setNeighbor(first, "left")
    return first, second


def _minimal_mesh(polys):
    mesh = object.__new__(MergeMesh)
    mesh.merged_polys = dict(enumerate(polys))
    mesh.facet_provenance_events = []
    mesh._provenance_event_order = 0
    mesh._provenance_stage = "test"
    mesh._provenance_override = None
    mesh.c0_refinement_report = None
    for merge_id, poly in mesh.merged_polys.items():
        mesh._attach_facet_provenance(poly, merge_id)
    return mesh


def test_joint_c0_repairs_gap_with_conservative_shared_endpoint():
    first, second = _two_cell_gap()
    mesh = SimpleNamespace(merged_polys={0: first, 1: second})

    assignments, report = plan_joint_c0_refinement(mesh, [first, second])

    assert report.bad_joins_before == 1
    assert report.bad_joins_after == 0
    assert report.components_solved == 1
    assert report.components_failed == 0
    assert report.max_relative_area_residual_after < 1.0e-10
    assert {assignment.solution_kind for assignment in assignments} == {"exact_c1"}
    facets = {assignment.merge_id: assignment.facet for assignment in assignments}
    assert facets[0].pRight == pytest.approx(facets[1].pLeft, abs=1.0e-12)


def test_joint_c0_treats_corner_facets_as_fixed_boundaries():
    first, second = _two_cell_gap()
    second.setFacet(SimpleNamespace(name="corner"))
    mesh = SimpleNamespace(merged_polys={0: first, 1: second})

    assignments, report = plan_joint_c0_refinement(mesh, [first, second])

    assert assignments == []
    assert report.eligible_joins == 0
    assert report.components == ()


def test_smooth_chain_refines_c0_join_with_tangent_jump():
    first, second = _two_cell_kink()
    mesh = SimpleNamespace(merged_polys={0: first, 1: second})

    assignments, report = plan_joint_c0_refinement(
        mesh, [first, second], component_mode="smooth_chain"
    )

    assert report.bad_joins_before == 0
    assert report.bad_tangent_joins_before == 1
    assert report.components_solved == 1
    assert report.bad_joins_after == 0
    assert report.bad_tangent_joins_after == 0
    assert {assignment.solution_kind for assignment in assignments} == {"exact_c1"}


def test_joint_c0_skips_degenerate_initial_seed(monkeypatch):
    first, second = _two_cell_gap()
    mesh = SimpleNamespace(merged_polys={0: first, 1: second})
    original = c0_refinement._initial_values

    def one_degenerate_seed(alpha, *args):
        if alpha == 0.0:
            raise ValueError("Cannot parameterize curvature on a zero-length chord")
        return original(alpha, *args)

    monkeypatch.setattr(c0_refinement, "_initial_values", one_degenerate_seed)

    assignments, report = plan_joint_c0_refinement(mesh, [first, second])

    assert assignments
    assert report.components_solved == 1
    assert report.components_failed == 0


def test_joint_c0_skips_infeasible_least_squares_seed(monkeypatch):
    first, second = _two_cell_gap()
    mesh = SimpleNamespace(merged_polys={0: first, 1: second})
    original = c0_refinement.least_squares
    calls = 0

    def one_infeasible_seed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("`x0` is infeasible.")
        return original(*args, **kwargs)

    monkeypatch.setattr(c0_refinement, "least_squares", one_infeasible_seed)

    assignments, report = plan_joint_c0_refinement(mesh, [first, second])

    assert calls > 1
    assert assignments
    assert report.components_solved == 1
    assert report.components_failed == 0


def test_make_c0_defaults_to_joint_and_keeps_guarded_mode_selectable():
    polygon = NeighboredPolygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    polygon.setArea(0.5)
    polygon.setFacet(LinearFacet([0.5, 0], [0.5, 1]))
    mesh = _minimal_mesh([polygon])

    mesh.makeC0([polygon])
    assert mesh.c0_refinement_report["mode"] == "joint"

    mesh.makeC0([polygon], mode="guarded")
    assert mesh.c0_refinement_report["mode"] == "guarded"

    mesh.makeC0([polygon], mode="g1_chain")
    assert mesh.c0_refinement_report["mode"] == "g1_chain"


def test_make_c0_rejects_unknown_mode():
    mesh = _minimal_mesh([])

    with pytest.raises(ValueError, match="Unknown C0 mode"):
        mesh.makeC0([], mode="not-a-mode")
