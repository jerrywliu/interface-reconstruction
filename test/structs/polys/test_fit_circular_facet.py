#!/usr/bin/env python3
"""
Integration-style test for NeighboredPolygon.fitCircularFacet().
"""

from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, pointInPoly
from main.structs.facets.circular_facet import ArcFacet
from main.structs.polys.base_polygon import BasePolygon
from main.structs.polys.neighbored_polygon import NeighboredPolygon


def test_fit_circular_facet():
    poly1_pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    poly2_pts = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]
    poly3_pts = [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]

    left = NeighboredPolygon(poly1_pts)
    mid = NeighboredPolygon(poly2_pts)
    right = NeighboredPolygon(poly3_pts)

    center = [1.0, 1.0]
    radius = 1.2
    max_area = abs(getArea(poly1_pts))

    left.setFraction(getCircleIntersectArea(center, radius, poly1_pts)[0] / max_area)
    mid.setFraction(getCircleIntersectArea(center, radius, poly2_pts)[0] / max_area)
    right.setFraction(getCircleIntersectArea(center, radius, poly3_pts)[0] / max_area)

    mid.setNeighbor(left, "left")
    mid.setNeighbor(right, "right")

    mid.fitCircularFacet()
    assert mid.hasFacet()
    assert isinstance(mid.getFacet(), ArcFacet)

    fit_area, _ = getCircleIntersectArea(
        mid.getFacet().center, mid.getFacet().radius, poly2_pts
    )
    fit_fraction = fit_area / max_area
    assert abs(fit_fraction - mid.getFraction()) < 1e-6


def test_root_fallback_prefers_candidate_with_better_selection_key():
    original = BasePolygon._try_arc_fit_root_fallback
    original_error = BasePolygon._arc_fit_max_fraction_error
    original_normalize = BasePolygon._normalize_root_fallback_arc
    original_key = BasePolygon._root_fallback_selection_key
    calls = []

    def fake_try_arc_fit_root_fallback(*args):
        calls.append(args)
        if len(args) == 7:
            return [1.0, 2.0], 3.0, [[0.0, 0.0], [1.0, 1.0]]
        return [9.0, 9.0], 99.0, [[0.0, 0.0], [1.0, 1.0]]

    BasePolygon._try_arc_fit_root_fallback = staticmethod(fake_try_arc_fit_root_fallback)
    BasePolygon._arc_fit_max_fraction_error = staticmethod(lambda *args: 0.0)
    BasePolygon._normalize_root_fallback_arc = staticmethod(
        lambda polys, fractions, center, radius, arcintersects, root_guess=None: (
            center,
            radius,
            arcintersects,
        )
    )
    BasePolygon._root_fallback_selection_key = staticmethod(
        lambda mid_poly, center, radius, arcintersects, root_guess=None: abs(center[0] - 1.0)
    )
    try:
        result = BasePolygon._try_arc_fit_root_fallbacks(
            ([0.0], [1.0], [2.0], 0.1, 0.2, 0.3, 1e-10),
            root_guess=(5.0, 6.0, 7.0),
        )
    finally:
        BasePolygon._try_arc_fit_root_fallback = original
        BasePolygon._arc_fit_max_fraction_error = original_error
        BasePolygon._normalize_root_fallback_arc = original_normalize
        BasePolygon._root_fallback_selection_key = original_key

    assert result[1] == 3.0
    assert len(calls) == 2


def test_root_fallback_uses_seed_only_after_unseeded_failure():
    original = BasePolygon._try_arc_fit_root_fallback
    original_error = BasePolygon._arc_fit_max_fraction_error
    original_normalize = BasePolygon._normalize_root_fallback_arc
    original_key = BasePolygon._root_fallback_selection_key
    calls = []

    def fake_try_arc_fit_root_fallback(*args):
        calls.append(args)
        if len(args) == 7:
            return None, None, None
        return [9.0, 9.0], 99.0, [[0.0, 0.0], [1.0, 1.0]]

    BasePolygon._try_arc_fit_root_fallback = staticmethod(fake_try_arc_fit_root_fallback)
    BasePolygon._arc_fit_max_fraction_error = staticmethod(lambda *args: 0.0)
    BasePolygon._normalize_root_fallback_arc = staticmethod(
        lambda polys, fractions, center, radius, arcintersects, root_guess=None: (
            center,
            radius,
            arcintersects,
        )
    )
    BasePolygon._root_fallback_selection_key = staticmethod(
        lambda mid_poly, center, radius, arcintersects, root_guess=None: abs(center[0] - 9.0)
    )
    try:
        result = BasePolygon._try_arc_fit_root_fallbacks(
            ([0.0], [1.0], [2.0], 0.1, 0.2, 0.3, 1e-10),
            root_guess=(5.0, 6.0, 7.0),
        )
    finally:
        BasePolygon._try_arc_fit_root_fallback = original
        BasePolygon._arc_fit_max_fraction_error = original_error
        BasePolygon._normalize_root_fallback_arc = original_normalize
        BasePolygon._root_fallback_selection_key = original_key

    assert result[1] == 99.0
    assert len(calls) == 2


def test_root_fallback_rejects_large_residual_candidate():
    original = BasePolygon._try_arc_fit_root_fallback
    original_error = BasePolygon._arc_fit_max_fraction_error
    original_normalize = BasePolygon._normalize_root_fallback_arc
    original_key = BasePolygon._root_fallback_selection_key

    BasePolygon._try_arc_fit_root_fallback = staticmethod(
        lambda *args: ([1.0, 2.0], 3.0, [[0.0, 0.0], [1.0, 1.0]])
    )
    BasePolygon._arc_fit_max_fraction_error = staticmethod(lambda *args: 1e-2)
    BasePolygon._normalize_root_fallback_arc = staticmethod(
        lambda polys, fractions, center, radius, arcintersects, root_guess=None: (
            center,
            radius,
            arcintersects,
        )
    )
    BasePolygon._root_fallback_selection_key = staticmethod(
        lambda mid_poly, center, radius, arcintersects, root_guess=None: 0.0
    )
    try:
        result = BasePolygon._try_arc_fit_root_fallbacks(
            ([0.0], [1.0], [2.0], 0.1, 0.2, 0.3, 1e-10),
            root_guess=None,
        )
    finally:
        BasePolygon._try_arc_fit_root_fallback = original
        BasePolygon._arc_fit_max_fraction_error = original_error
        BasePolygon._normalize_root_fallback_arc = original_normalize
        BasePolygon._root_fallback_selection_key = original_key

    assert result == (None, None, None)


def test_normalize_root_fallback_arc_prefers_local_midpoint_variant():
    poly1 = [
        [50.674383010137035, 64.50443453033003],
        [51.13351236474538, 64.59461935860502],
        [51.197867293648684, 65.18841340239373],
        [50.742042742041605, 65.2923163899722],
    ]
    poly2 = [
        [50.742042742041605, 65.2923163899722],
        [51.197867293648684, 65.18841340239373],
        [51.41212393322769, 65.87562355646607],
        [50.58729110936458, 65.80843897704065],
    ]
    poly3 = [
        [51.197867293648684, 65.18841340239373],
        [51.807407252645135, 65.26411207062758],
        [51.82890981672091, 65.92426075530699],
        [51.41212393322769, 65.87562355646607],
    ]
    center = [50.00407772533838, 50.30892377176704]
    radius = 14.999999995963115
    arcintersects = [
        [50.7507688342898, 65.29032731972298],
        [51.22004807705798, 65.25955639994142],
    ]

    normalized = BasePolygon._normalize_root_fallback_arc(
        [poly1, poly2, poly3],
        (2.177555089200034e-05, 0.954350326765554, 0.9652658456245118),
        center,
        radius,
        arcintersects,
    )
    facet = ArcFacet(normalized[0], normalized[1], normalized[2][0], normalized[2][1])

    assert normalized != (None, None, None)
    assert facet.is_major_arc is False
    assert pointInPoly(facet.midpoint, poly2)


if __name__ == "__main__":
    test_fit_circular_facet()
    test_root_fallback_prefers_candidate_with_better_selection_key()
    test_root_fallback_uses_seed_only_after_unseeded_failure()
    test_root_fallback_rejects_large_residual_candidate()
    test_normalize_root_fallback_arc_prefers_local_midpoint_variant()
    print("Circular facet fitting tests completed.")
