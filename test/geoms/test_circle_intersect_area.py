#!/usr/bin/env python3
"""
Basic checks for getCircleIntersectArea parity/termination.
"""

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


if __name__ == "__main__":
    test_circle_intersect_area_parity()
    print("Circle intersect area tests completed.")
