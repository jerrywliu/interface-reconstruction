import math

from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface_geometry import ArcPrimitive
from util.metrics.metrics import (
    hausdorffFacets,
    hausdorff_interface,
    point_to_primitive_distance,
)


def _square_facets(dx=0.0, dy=0.0):
    points = [
        [dx - 1.0, dy - 1.0],
        [dx + 1.0, dy - 1.0],
        [dx + 1.0, dy + 1.0],
        [dx - 1.0, dy + 1.0],
    ]
    return [
        LinearFacet(points[i], points[(i + 1) % len(points)])
        for i in range(len(points))
    ]


def _zalesak_like_facets(dx=0.0, dy=0.0):
    radius = 2.0
    y_intersect = math.sqrt(radius * radius - 0.5 * 0.5)
    return [
        ArcFacet([dx, dy], -radius, [dx + 0.5, dy + y_intersect], [dx - 0.5, dy + y_intersect]),
        LinearFacet([dx - 0.5, dy - 2.1], [dx - 0.5, dy + 1.0]),
        LinearFacet([dx - 0.5, dy + 1.0], [dx + 0.5, dy + 1.0]),
        LinearFacet([dx + 0.5, dy + 1.0], [dx + 0.5, dy - 2.1]),
    ]


def test_linear_facets_parallel_distance():
    facet1 = LinearFacet([0.0, 0.0], [1.0, 0.0])
    facet2 = LinearFacet([0.0, 1.0], [1.0, 1.0])
    assert abs(hausdorffFacets(facet1, facet2) - 1.0) < 1e-9


def test_arc_facets_radius_offset_distance():
    center = [0.0, 0.0]
    facet1 = ArcFacet(center, 1.0, [1.0, 0.0], [0.0, 1.0])
    facet2 = ArcFacet(center, 2.0, [2.0, 0.0], [0.0, 2.0])
    assert abs(hausdorffFacets(facet1, facet2) - 1.0) < 1e-9


def test_corner_facets_identical_zero_distance():
    corner = CornerFacet(
        centerLeft=None,
        centerRight=None,
        radiusLeft=None,
        radiusRight=None,
        pLeft=[0.0, 0.0],
        corner=[1.0, 0.0],
        pRight=[1.0, 1.0],
    )
    assert hausdorffFacets(corner, corner) < 1e-12


def test_square_interface_identical_zero_distance():
    square = _square_facets()
    assert hausdorff_interface(square, square) < 1e-12


def test_square_interface_translation_matches_shift():
    square = _square_facets()
    translated = _square_facets(dy=0.25)
    assert abs(hausdorff_interface(square, translated) - 0.25) < 1e-6


def test_zalesak_like_interface_identical_zero_distance():
    zalesak_facets = _zalesak_like_facets()
    assert hausdorff_interface(zalesak_facets, zalesak_facets) < 1e-12


def test_point_to_arc_distance_respects_major_minor_choice():
    minor_arc = ArcPrimitive([0.0, 0.0], 1.0, [1.0, 0.0], [0.0, 1.0])
    major_arc = ArcPrimitive([0.0, 0.0], -1.0, [1.0, 0.0], [0.0, 1.0])
    opposite_point = [-1.0, 0.0]

    minor_distance = point_to_primitive_distance(opposite_point, minor_arc)
    major_distance = point_to_primitive_distance(opposite_point, major_arc)

    assert abs(minor_distance - math.sqrt(2.0)) < 1e-9
    assert major_distance < 1e-12
