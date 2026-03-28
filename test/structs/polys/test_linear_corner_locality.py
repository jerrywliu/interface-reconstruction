#!/usr/bin/env python3
"""
Regression coverage for the linear-corner locality guard.
"""

import math

from main.geoms.corner_facet import getPolyCornerArea
from main.geoms.corner_facet import getPolyCurvedCornerArea
from main.geoms.linear_facet import getPolyLineArea
from main.geoms.geoms import getArea
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.polys.neighbored_polygon import NeighboredPolygon


def test_check_corner_facet_accepts_local_corner():
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    poly = NeighboredPolygon(points)

    left_support = [[-0.5, 0.5], [0.0, 0.5]]
    right_support = [[0.5, -0.5], [0.5, 0.0]]
    target_fraction = getPolyCornerArea(
        points,
        left_support[1],
        [0.5, 0.5],
        right_support[1],
    ) / abs(getArea(points))

    poly.setFraction(target_fraction)
    poly.checkCornerFacet(
        left_support[0],
        left_support[1],
        right_support[0],
        right_support[1],
    )

    assert poly.hasFacet()
    assert isinstance(poly.getFacet(), CornerFacet)


def test_check_corner_facet_rejects_large_extrapolation():
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    poly = NeighboredPolygon(points)

    left_support = [[-1.0, 0.5], [0.0, 0.5]]
    right_support = [[5.0, -1.0], [5.0, 0.0]]
    target_fraction = getPolyCornerArea(
        points,
        left_support[1],
        [5.0, 0.5],
        right_support[1],
    ) / abs(getArea(points))

    poly.setFraction(target_fraction)
    poly.checkCornerFacet(
        left_support[0],
        left_support[1],
        right_support[0],
        right_support[1],
    )

    assert not poly.hasFacet()


def test_corner_branch_propagation_seeds_unset_neighbors():
    host = NeighboredPolygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    branch_left = NeighboredPolygon(
        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]
    )
    branch_right = NeighboredPolygon(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
    )

    host.setNeighbor(branch_left, "left")
    host.setNeighbor(branch_right, "right")
    branch_left.setNeighbor(host, "right")
    branch_right.setNeighbor(host, "left")

    host.setFraction(
        getPolyCornerArea(
            host.points,
            [1.0, 0.8],
            [0.8, 0.8],
            [0.8, 1.0],
        )
        / abs(getArea(host.points))
    )
    branch_left.setFraction(
        getPolyLineArea(branch_left.points, [2.0, 0.8], [0.8, 0.8])
        / abs(getArea(branch_left.points))
    )
    branch_right.setFraction(
        getPolyLineArea(branch_right.points, [0.8, 0.8], [0.8, 2.0])
        / abs(getArea(branch_right.points))
    )

    host.setFacet(
        CornerFacet(
            None,
            None,
            None,
            None,
            [1.0, 0.8],
            [0.8, 0.8],
            [0.8, 1.0],
        )
    )
    host.propagateCornerBranchFacets()

    assert branch_left.hasFacet()
    assert branch_right.hasFacet()
    assert branch_left.getFacet().name == "corner_branch_linear"
    assert branch_right.getFacet().name == "corner_branch_linear"


def test_check_curved_corner_accepts_default_linear_support():
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    poly = NeighboredPolygon(points)

    line_support = LinearFacet([-0.5, 0.5], [0.0, 0.5], name="default_linear")
    arc_support = ArcFacet(
        [1.0, 1.0],
        math.sqrt(0.5),
        [0.5, 1.0],
        [1.0, 0.5],
    )

    poly.setFraction(
        getPolyCurvedCornerArea(
            points,
            line_support.pRight,
            [0.5, 0.5],
            arc_support.pLeft,
            None,
            arc_support.radius,
        )
        / abs(getArea(points))
    )

    facet, error = poly.checkCurvedCornerFacet(line_support, arc_support, ret=True)

    assert facet is not None
    assert isinstance(facet, CornerFacet)
    assert facet.centerLeft is None
    assert facet.centerRight is not None
    assert error is not None
    assert error < NeighboredPolygon.curved_corner_area_threshold


if __name__ == "__main__":
    test_check_corner_facet_accepts_local_corner()
    test_check_corner_facet_rejects_large_extrapolation()
    test_corner_branch_propagation_seeds_unset_neighbors()
    test_check_curved_corner_accepts_default_linear_support()
    print("Linear corner locality tests completed.")
