from types import SimpleNamespace

import pytest

import main.structs.polys.base_polygon as base_polygon_module
from main.geoms.circular_facet import getCircleIntersectArea
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.polys.base_polygon import BasePolygon
from main.structs.polys.neighbored_polygon import NeighboredPolygon


def _polygon(points, target_area, original_facet):
    polygon = BasePolygon(points)
    polygon.setArea(target_area)
    polygon.setFacet(original_facet)
    return polygon


def test_c0_rejects_infeasible_coarse_zalesak_case21_branch():
    polygon = _polygon(
        [
            [40.485533382941256, 56.34676449016073],
            [42.05138323815612, 56.280377297838086],
            [42.03487237822918, 57.68554553937971],
            [40.651737628620744, 57.94562440226498],
        ],
        2.2164566702263073,
        ArcFacet(
            [66.71110975282082, 37.9541334586078],
            -30.826408500961303,
            [41.9275704356127, 56.28562657784222],
            [42.04939954764805, 56.44920066108686],
        ),
    )

    candidate = polygon.fitCurvature(
        [41.91872197637478, 56.28600172515487],
        [42.04847539614366, 56.52785121896753],
        ret=True,
    )

    assert candidate is None
    assert polygon.last_c0_fit_diagnostic["selected_branch"] == "rejected"
    assert polygon.last_c0_fit_diagnostic["candidate_error"] == pytest.approx(
        1.9327187144487765
    )
    assert polygon.last_c0_fit_diagnostic["alternate_error"] == pytest.approx(
        0.011875894794231012
    )


def test_c0_rejects_infeasible_coarse_zalesak_case1_merged_zone():
    polygon = _polygon(
        [
            [46.99304775284867, 57.6785602293994],
            [46.97799663283049, 59.386291541044045],
            [45.27243888656112, 59.258671440095604],
            [43.80412353306444, 59.23802259773869],
            [43.77887297488225, 57.9378306422579],
            [45.226977004999775, 57.731218423796584],
        ],
        2.1220794595842563,
        ArcFacet(
            [46.762210403423374, 58.098093160380984],
            -1.5997503250425793,
            [45.20433891552492, 57.73444837536833],
            [45.69554732209824, 59.290330961078396],
        ),
    )

    candidate = polygon.fitCurvature(
        [45.3754924355744, 57.72806768174097],
        [45.81536271847449, 59.29929627052873],
        ret=True,
    )

    assert candidate is None
    assert polygon.last_c0_fit_diagnostic["selected_branch"] == "rejected"
    assert polygon.last_c0_fit_diagnostic["candidate_error"] == pytest.approx(
        0.10248363701433938
    )


def test_c0_accepts_a_conservative_linear_adjustment():
    original = LinearFacet([0.5, 0.0], [0.5, 1.0])
    polygon = _polygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        0.5,
        original,
    )

    candidate = polygon.fitCurvature([0.5, 0.0], [0.5, 1.0], ret=True)

    assert isinstance(candidate, LinearFacet)
    assert polygon.last_c0_fit_diagnostic["selected_branch"] == "analytic"


def test_make_c0_preserves_explicit_corner_facets():
    polygon = NeighboredPolygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    corner = SimpleNamespace(name="corner")
    polygon.setFacet(corner)

    adjusted = MergeMesh.makeC0(SimpleNamespace(), [polygon])

    assert adjusted == [polygon]
    assert polygon.getFacet() is corner


def test_c0_rejects_numeric_fit_exception_and_retains_original(monkeypatch):
    original = LinearFacet([0.5, 0.0], [0.5, 1.0])
    polygon = _polygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        0.25,
        original,
    )
    monkeypatch.setattr(
        base_polygon_module,
        "matchArcArea",
        lambda *args: (_ for _ in ()).throw(RuntimeError("numeric failure")),
    )

    polygon.fitCurvature([0.4, 0.0], [0.6, 1.0])

    assert polygon.getFacet() is original
    assert polygon.last_c0_fit_diagnostic["selected_branch"] == "rejected_exception"
    assert polygon.last_c0_fit_diagnostic["rejection_reason"] == "RuntimeError"


def _provenance_mesh(poly, merge_id=7):
    mesh = object.__new__(MergeMesh)
    mesh.facet_provenance_events = []
    mesh._provenance_event_order = 0
    mesh._provenance_stage = "fitting"
    mesh._provenance_override = None
    mesh._attach_facet_provenance(poly, merge_id)
    return mesh


def test_make_c0_records_adjustment_provenance():
    polygon = NeighboredPolygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    polygon.setArea(0.5)
    polygon.setFacet(LinearFacet([0.5, 0.0], [0.5, 1.0]))
    mesh = _provenance_mesh(polygon)

    mesh.makeC0([polygon])

    event = mesh.facet_provenance_events[-1]
    assert event["stage"] == "c0"
    assert event["event_kind"] == "c0_adjustment"
    assert event["fallback_reason"] == "conservative_refit_accepted"


def test_make_c0_records_rejection_provenance(monkeypatch):
    polygon = NeighboredPolygon(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    polygon.setArea(0.5)
    original = LinearFacet([0.5, 0.0], [0.5, 1.0])
    polygon.setFacet(original)
    mesh = _provenance_mesh(polygon)

    def reject(*args, **kwargs):
        polygon.last_c0_fit_diagnostic = {
            "selected_branch": "rejected_exception",
            "rejection_reason": "RuntimeError",
        }
        return None

    monkeypatch.setattr(polygon, "fitCurvature", reject)
    mesh.makeC0([polygon])

    assert polygon.getFacet() is original
    event = mesh.facet_provenance_events[-1]
    assert event["stage"] == "c0"
    assert event["event_kind"] == "c0_rejection"
    assert event["fallback_reason"] == "RuntimeError"
