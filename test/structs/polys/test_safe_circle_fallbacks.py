import pytest

import main.structs.polys.base_polygon as base_polygon_module
import main.structs.polys.neighbored_polygon as neighbored_polygon_module
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.polys.base_polygon import BasePolygon
from main.structs.polys.neighbored_polygon import NeighboredPolygon


def _poly(fraction=0.5):
    poly = BasePolygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    poly.setFraction(fraction)
    poly.set3x3Stencil([[poly for _ in range(3)] for _ in range(3)])
    return poly


def test_safe_circle_defaults_unresolved_orientation_to_lvira(monkeypatch):
    poly = _poly()
    lvira = LinearFacet([0.0, 0.5], [1.0, 0.5], name="LVIRA")
    monkeypatch.setattr(poly, "findSafeOrientation", lambda fit_1neighbor=False: None)
    monkeypatch.setattr(poly, "runLVIRA", lambda ret=False: lvira)

    facet, fallback = poly.runSafeCircle(ret=True, return_info=True)

    assert facet is lvira
    assert fallback == {
        "event_kind": "plic_fallback",
        "reason": "unresolved_orientation",
        "policy": "LVIRA",
    }


def test_safe_circle_preserves_explicit_elvira_override(monkeypatch):
    poly = _poly()
    elvira = LinearFacet([0.0, 0.4], [1.0, 0.4], name="ELVIRA")
    monkeypatch.setattr(poly, "findSafeOrientation", lambda fit_1neighbor=False: None)
    monkeypatch.setattr(poly, "runELVIRA", lambda ret=False: elvira)

    facet, fallback = poly.runSafeCircle(
        ret=True, plic_fallback="ELVIRA", return_info=True
    )

    assert facet is elvira
    assert fallback["policy"] == "ELVIRA"


def test_safe_circle_arc_failure_uses_mass_matching_support_line(monkeypatch):
    poly = _poly()
    left = _poly(0.25)
    right = _poly(0.75)
    monkeypatch.setattr(
        poly, "findSafeOrientation", lambda fit_1neighbor=False: [left, right]
    )
    monkeypatch.setattr(
        base_polygon_module,
        "getLinearFacet",
        lambda *args: ([0.0, 0.25], [1.0, 0.25]),
    )
    monkeypatch.setattr(base_polygon_module, "getPolyLineArea", lambda *args: 0.1)
    monkeypatch.setattr(
        base_polygon_module,
        "getLinearFacetFromNormal",
        lambda *args: ([0.0, 0.5], [1.0, 0.5]),
    )
    monkeypatch.setattr(
        poly, "_run_arc_fit_with_timeout", lambda *args: (None, None, None)
    )
    monkeypatch.setattr(
        poly, "_try_arc_fit_root_fallbacks", lambda *args: (None, None, None)
    )

    facet, fallback = poly.runSafeCircle(ret=True, return_info=True)

    assert facet.name == "default_linear"
    assert facet.pLeft == [0.0, 0.5]
    assert facet.pRight == [1.0, 0.5]
    assert fallback == {
        "event_kind": "local_linear_fallback",
        "reason": "arc_fit_failed",
        "policy": "local_linear",
    }


def test_safe_circle_uses_lvira_when_local_line_fallback_fails(monkeypatch):
    poly = _poly()
    left = _poly(0.25)
    right = _poly(0.75)
    lvira = LinearFacet([0.0, 0.5], [1.0, 0.5], name="LVIRA")
    monkeypatch.setattr(
        poly, "findSafeOrientation", lambda fit_1neighbor=False: [left, right]
    )
    monkeypatch.setattr(
        base_polygon_module,
        "getLinearFacet",
        lambda *args: ([0.0, 0.25], [1.0, 0.25]),
    )
    monkeypatch.setattr(base_polygon_module, "getPolyLineArea", lambda *args: 0.1)
    monkeypatch.setattr(
        base_polygon_module,
        "getLinearFacetFromNormal",
        lambda *args: (_ for _ in ()).throw(RuntimeError("line fit failed")),
    )
    monkeypatch.setattr(
        poly, "_run_arc_fit_with_timeout", lambda *args: (None, None, None)
    )
    monkeypatch.setattr(
        poly, "_try_arc_fit_root_fallbacks", lambda *args: (None, None, None)
    )
    monkeypatch.setattr(poly, "runLVIRA", lambda ret=False: lvira)

    facet, fallback = poly.runSafeCircle(ret=True, return_info=True)

    assert facet is lvira
    assert fallback == {
        "event_kind": "plic_fallback",
        "reason": "arc_fit_failed_local_linear_failed",
        "policy": "LVIRA",
    }


def test_merge_mesh_records_independent_circle_fallback_provenance(monkeypatch):
    source = _poly()
    merged = _poly()
    lvira = LinearFacet([0.0, 0.5], [1.0, 0.5], name="LVIRA")
    fallback = {
        "event_kind": "plic_fallback",
        "reason": "unresolved_orientation",
        "policy": "LVIRA",
    }
    monkeypatch.setattr(
        source,
        "runSafeCircle",
        lambda **kwargs: (lvira, fallback),
    )

    mesh = object.__new__(MergeMesh)
    mesh.polys = [[source]]
    mesh.merged_polys = {0: merged}
    mesh.coords_to_merge_id = [[0]]
    mesh.merge_ids_to_coords = [[(0, 0)]]
    mesh.plic_fallback_records = []
    mesh.safe_circle_fallback_records = []
    mesh.facet_provenance_events = []
    mesh._provenance_event_order = 0
    mesh._provenance_stage = "initial"
    mesh._provenance_override = None
    monkeypatch.setattr(mesh, "get3x3Stencil", lambda x, y: source.stencil)
    mesh._attach_facet_provenance(merged, 0)

    mesh.runSafeCircle()

    assert mesh.plic_fallback_records[0]["policy"] == "LVIRA"
    assert mesh.safe_circle_fallback_records[0]["reason"] == "unresolved_orientation"
    assert mesh.facet_provenance_events[0]["event_kind"] == "plic_fallback"
    assert mesh.facet_provenance_events[0]["fallback_reason"] == "unresolved_orientation"


def test_safe_circle_rejects_unknown_fallback_policy():
    poly = _poly()
    with pytest.raises(ValueError, match="Unknown plic_fallback"):
        poly.runSafeCircle(ret=True, plic_fallback="not-a-policy")


@pytest.mark.parametrize("policy", ["Youngs", "ELVIRA", "LVIRA"])
def test_coordinated_support_failure_honors_fallback_policy(monkeypatch, policy):
    poly = NeighboredPolygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    poly.setFraction(0.5)
    poly.set3x3Stencil([[poly for _ in range(3)] for _ in range(3)])
    poly.setNeighbor(poly, "left")
    poly.setNeighbor(poly, "right")
    expected = LinearFacet([0.0, 0.5], [1.0, 0.5], name=policy)
    poly.plic_fallback_policy = policy
    monkeypatch.setattr(
        neighbored_polygon_module,
        "getLinearFacet",
        lambda *args: (_ for _ in ()).throw(RuntimeError("support failed")),
    )
    monkeypatch.setattr(poly, f"run{policy}", lambda ret=False: expected)

    poly.fitLinearFacet()

    assert poly.getFacet() is expected


def test_fit_facets_propagates_requested_policy_to_oriented_cells(monkeypatch):
    poly = NeighboredPolygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    poly.setFraction(0.5)
    poly.setNeighbor(poly, "left")
    poly.setNeighbor(poly, "right")
    seen = []

    def fit_linear():
        seen.append(poly.plic_fallback_policy)
        poly.setFacet(LinearFacet([0.0, 0.5], [1.0, 0.5]))

    monkeypatch.setattr(poly, "fitLinearFacet", fit_linear)
    mesh = object.__new__(MergeMesh)
    mesh.merged_polys = {0: poly}

    mesh.fitFacets([0], setting="linear", plic_fallback="Youngs")

    assert seen == ["Youngs"]
