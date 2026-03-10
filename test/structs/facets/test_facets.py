#!/usr/bin/env python3
"""
Basic sanity tests for LinearFacet, ArcFacet, and CornerFacet behavior.

Run with:
python -m test.test_facets
"""

import math

from main.geoms.geoms import getDistance
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface_geometry import ArcPrimitive, LinePrimitive, composite_from_facet


def assert_close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b} (tol={tol})"


def assert_point_close(p1, p2, tol=1e-9):
    assert getDistance(p1, p2) <= tol, f"{p1} != {p2} (tol={tol})"


def test_linear_facet_sample_and_tangent():
    facet = LinearFacet([0.0, 0.0], [2.0, 0.0])
    pts = facet.sample(5)
    assert len(pts) == 5
    assert_point_close(pts[0], [0.0, 0.0])
    assert_point_close(pts[-1], [2.0, 0.0])

    tangent = facet.getTangent([0.0, 0.0])
    assert_close(tangent[0], 2.0)
    assert_close(tangent[1], 0.0)
    assert_point_close(facet.getLeftTangent(), tangent)
    assert_point_close(facet.getRightTangent(), tangent)


def test_linear_facet_advected_zero_velocity():
    facet = LinearFacet([0.0, 0.0], [1.0, 0.0])

    def zero_velocity(t, p):
        return [0.0, 0.0]

    adv = facet.advected(zero_velocity, t=0.0, h=1.0, mode="RK1")
    assert isinstance(adv, LinearFacet)
    assert_point_close(adv.pLeft, facet.pLeft)
    assert_point_close(adv.pRight, facet.pRight)


def test_arc_facet_sample_and_tangent():
    center = [0.0, 0.0]
    facet = ArcFacet(center, 1.0, [1.0, 0.0], [0.0, 1.0])
    pts = facet.sample(3)
    assert len(pts) == 3
    assert_point_close(pts[0], facet.pLeft)
    assert_point_close(pts[-1], facet.pRight)

    mid = pts[1]
    expected_mid = [math.sqrt(0.5), math.sqrt(0.5)]
    assert_point_close(mid, expected_mid, tol=1e-6)

    for p in pts:
        assert_close(getDistance(p, center), 1.0, tol=1e-6)

    tangent = facet.getTangent(facet.pLeft)
    radius_vec = [facet.pLeft[0] - center[0], facet.pLeft[1] - center[1]]
    dot = tangent[0] * radius_vec[0] + tangent[1] * radius_vec[1]
    assert_close(dot, 0.0, tol=1e-9)


def test_arc_facet_advected_zero_velocity():
    center = [0.0, 0.0]
    facet = ArcFacet(center, 1.0, [1.0, 0.0], [0.0, 1.0])

    def zero_velocity(t, p):
        return [0.0, 0.0]

    adv = facet.advected(zero_velocity, t=0.0, h=1.0, mode="RK1")
    assert isinstance(adv, ArcFacet)
    assert_point_close(adv.pLeft, facet.pLeft)
    assert_point_close(adv.pRight, facet.pRight)
    assert_close(abs(adv.radius), 1.0, tol=1e-6)


def test_arc_facet_poly_intersect_area_handles_endpoint_roundoff_case():
    poly = [
        [46.9930902846328, 57.98514685599057],
        [48.035304846342704, 57.80830608587295],
        [47.700198930048906, 58.9957116764918],
        [47.12791288380223, 58.75634041000361],
    ]
    facet = ArcFacet(
        [215.81150889003243, 63.016632740100164],
        -167.98597195255488,
        [47.90561606954499, 57.83031140417026],
        [47.89123912574505, 58.318784814348184],
    )

    area = facet.getPolyIntersectArea(poly)

    assert math.isfinite(area)
    assert area >= 0.0
    assert area <= abs(getDistance(poly[0], poly[1]) * getDistance(poly[1], poly[2])) * 2


def test_corner_facet_sample_and_advected_zero_velocity():
    facet = CornerFacet(
        centerLeft=None,
        centerRight=None,
        radiusLeft=None,
        radiusRight=None,
        pLeft=[0.0, 0.0],
        corner=[1.0, 0.0],
        pRight=[1.0, 1.0],
    )
    pts = facet.sample(5)
    assert len(pts) == 5
    assert_point_close(pts[0], facet.pLeft)
    assert_point_close(pts[-1], facet.pRight)

    corner_hits = sum(getDistance(p, facet.corner) < 1e-9 for p in pts)
    assert corner_hits == 1

    def zero_velocity(t, p):
        return [0.0, 0.0]

    adv = facet.advected(zero_velocity, t=0.0, h=1.0, mode="RK1")
    assert isinstance(adv, CornerFacet)
    assert_point_close(adv.pLeft, facet.pLeft)
    assert_point_close(adv.corner, facet.corner)
    assert_point_close(adv.pRight, facet.pRight)


def test_linear_facet_canonicalization():
    facet = LinearFacet([0.0, 0.0], [2.0, 0.0])
    composite = composite_from_facet(facet)

    assert len(composite.primitives) == 1
    assert isinstance(composite.primitives[0], LinePrimitive)
    assert composite.joints == []
    assert_point_close(composite.primitives[0].pLeft, facet.pLeft)
    assert_point_close(composite.primitives[0].pRight, facet.pRight)


def test_arc_facet_canonicalization():
    facet = ArcFacet([0.0, 0.0], 1.0, [1.0, 0.0], [0.0, 1.0])
    composite = composite_from_facet(facet)

    assert len(composite.primitives) == 1
    assert isinstance(composite.primitives[0], ArcPrimitive)
    assert composite.joints == []
    assert_point_close(composite.primitives[0].pLeft, facet.pLeft)
    assert_point_close(composite.primitives[0].pRight, facet.pRight)


def test_linear_linear_corner_canonicalization():
    facet = CornerFacet(
        centerLeft=None,
        centerRight=None,
        radiusLeft=None,
        radiusRight=None,
        pLeft=[0.0, 0.0],
        corner=[1.0, 0.0],
        pRight=[1.0, 1.0],
    )
    composite = composite_from_facet(facet)

    assert len(composite.primitives) == 2
    assert isinstance(composite.primitives[0], LinePrimitive)
    assert isinstance(composite.primitives[1], LinePrimitive)
    assert len(composite.joints) == 1
    assert composite.joints[0].kind == "corner"
    assert_point_close(composite.primitives[0].pRight, facet.corner)
    assert_point_close(composite.primitives[1].pLeft, facet.corner)


def test_line_arc_corner_canonicalization():
    facet = CornerFacet(
        centerLeft=None,
        centerRight=[0.0, 0.0],
        radiusLeft=None,
        radiusRight=1.0,
        pLeft=[0.0, 0.0],
        corner=[1.0, 0.0],
        pRight=[0.0, 1.0],
    )
    composite = composite_from_facet(facet)

    assert len(composite.primitives) == 2
    assert isinstance(composite.primitives[0], LinePrimitive)
    assert isinstance(composite.primitives[1], ArcPrimitive)
    assert composite.joints[0].kind == "corner"
    assert_point_close(composite.primitives[1].pLeft, facet.corner)
    assert_point_close(composite.primitives[1].pRight, facet.pRight)


def test_arc_arc_corner_canonicalization():
    facet = CornerFacet(
        centerLeft=[0.0, 0.0],
        centerRight=[0.0, 0.0],
        radiusLeft=1.0,
        radiusRight=1.0,
        pLeft=[0.0, 1.0],
        corner=[1.0, 0.0],
        pRight=[0.0, -1.0],
    )
    composite = composite_from_facet(facet)

    assert len(composite.primitives) == 2
    assert isinstance(composite.primitives[0], ArcPrimitive)
    assert isinstance(composite.primitives[1], ArcPrimitive)
    assert composite.joints[0].kind == "corner"
    assert_point_close(composite.primitives[0].pRight, facet.corner)
    assert_point_close(composite.primitives[1].pLeft, facet.corner)


if __name__ == "__main__":
    test_linear_facet_sample_and_tangent()
    test_linear_facet_advected_zero_velocity()
    test_arc_facet_sample_and_tangent()
    test_arc_facet_advected_zero_velocity()
    test_arc_facet_poly_intersect_area_handles_endpoint_roundoff_case()
    test_corner_facet_sample_and_advected_zero_velocity()
    print("Facet tests completed.")
