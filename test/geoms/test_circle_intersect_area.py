#!/usr/bin/env python3
"""
Basic checks for getCircleIntersectArea parity/termination.
"""

from main.geoms.geoms import getArea
from main.geoms.circular_facet import getCircleIntersectArea


def test_circle_intersect_area_parity():
    polys = [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[0.5, 0.0], [1.5, 0.0], [1.5, 1.0], [0.5, 1.0]],
    ]
    circles = [
        ([0.5, 0.5], 0.8),
        ([1.0, 1.0], 1.2),
        ([0.75, 0.25], 0.6),
    ]

    for poly in polys:
        for center, radius in circles:
            area, arcpoints = getCircleIntersectArea(center, radius, poly)
            assert area >= 0
            assert len(arcpoints) % 2 == 0


def test_circle_intersect_area_uses_minor_cap_for_case2_repro():
    center = [50.751792271818616, 50.26369219747838]
    radius = 15.0
    poly = [
        [62.0, 40.0],
        [62.666666666666664, 40.0],
        [62.666666666666664, 40.666666666666664],
        [62.0, 40.666666666666664],
    ]
    expected_fraction = 0.103825964349862

    area, arcpoints = getCircleIntersectArea(center, radius, poly)

    assert len(arcpoints) == 2
    assert abs(area / getArea(poly) - expected_fraction) < 1e-6


if __name__ == "__main__":
    test_circle_intersect_area_parity()
    test_circle_intersect_area_uses_minor_cap_for_case2_repro()
    print("Circle intersect area tests completed.")
