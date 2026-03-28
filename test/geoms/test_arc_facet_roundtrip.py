#!/usr/bin/env python3
"""
Round-trip test for getArcFacet using circle-derived area fractions.
"""

from main.geoms.circular_facet import getArcFacet, getCircleIntersectArea
from main.geoms.geoms import getArea


def test_arc_facet_roundtrip():
    poly1 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    poly2 = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]
    poly3 = [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]

    center = [1.0, 1.0]
    radius = 1.2

    max_area = abs(getArea(poly1))
    area1, _ = getCircleIntersectArea(center, radius, poly1)
    area2, _ = getCircleIntersectArea(center, radius, poly2)
    area3, _ = getCircleIntersectArea(center, radius, poly3)

    a1 = area1 / max_area
    a2 = area2 / max_area
    a3 = area3 / max_area

    ret_center, ret_radius, _ = getArcFacet(poly1, poly2, poly3, a1, a2, a3, 1e-10)
    assert ret_center is not None
    assert ret_radius is not None

    rarea1, _ = getCircleIntersectArea(ret_center, ret_radius, poly1)
    rarea2, _ = getCircleIntersectArea(ret_center, ret_radius, poly2)
    rarea3, _ = getCircleIntersectArea(ret_center, ret_radius, poly3)

    r1 = rarea1 / max_area
    r2 = rarea2 / max_area
    r3 = rarea3 / max_area

    tol = 1e-6
    assert abs(r1 - a1) < tol
    assert abs(r2 - a2) < tol
    assert abs(r3 - a3) < tol


def test_arc_facet_zalesak_case12_stencil_fails_gracefully():
    poly1 = [
        [64.65244101148592, 50.817570794173314],
        [65.28125207454978, 50.66703350352129],
        [65.50556501530274, 51.333462738535744],
        [64.8349116186194, 51.227175494578375],
    ]
    poly2 = [
        [65.28125207454978, 50.66703350352129],
        [66.02288169432336, 50.77732648423029],
        [66.0066360275915, 51.45248092856235],
        [65.50556501530274, 51.333462738535744],
    ]
    poly3 = [
        [65.32919447682659, 49.89462683602225],
        [65.89379653721898, 49.875795399173974],
        [66.02288169432336, 50.77732648423029],
        [65.28125207454978, 50.66703350352129],
    ]
    a1 = 0.8780358453509324
    a2 = 0.011719131631887647
    a3 = 0.047090906929863285

    ret_center, ret_radius, ret_intersects = getArcFacet(
        poly1, poly2, poly3, a1, a2, a3, 1e-10
    )
    assert ret_center is None
    assert ret_radius is None
    assert ret_intersects is None


if __name__ == "__main__":
    test_arc_facet_roundtrip()
    test_arc_facet_zalesak_case12_stencil_fails_gracefully()
    print("Arc facet roundtrip tests completed.")
