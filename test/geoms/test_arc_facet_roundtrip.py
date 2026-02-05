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


if __name__ == "__main__":
    test_arc_facet_roundtrip()
    print("Arc facet roundtrip tests completed.")
