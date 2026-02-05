#!/usr/bin/env python3
"""
Integration-style test for NeighboredPolygon.fitCircularFacet().
"""

from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea
from main.structs.facets.circular_facet import ArcFacet
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


if __name__ == "__main__":
    test_fit_circular_facet()
    print("Circular facet fitting tests completed.")
