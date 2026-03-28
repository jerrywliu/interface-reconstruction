#!/usr/bin/env python3
"""
Regression coverage for safe-orientation edge cases.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.structs.polys.base_polygon import BasePolygon


def _poly(points, fraction):
    poly = BasePolygon(points)
    poly.setFraction(fraction)
    return poly


def test_find_safe_orientation_handles_extreme_three_neighbor_case():
    center = _poly(
        [
            [64.02962065843354, 51.808854272431],
            [64.50863986686956, 52.19774323281559],
            [64.80385170705338, 52.82908356924063],
            [63.8503831102771, 52.64455949792059],
        ],
        0.9997707357510807,
    )
    right = _poly(
        [
            [64.50863986686956, 52.19774323281559],
            [65.19712590618519, 52.04805115832125],
            [65.48265978541593, 52.56454475619615],
            [64.80385170705338, 52.82908356924063],
        ],
        0.2916742279807388,
    )
    up = _poly(
        [
            [63.8503831102771, 52.64455949792059],
            [64.80385170705338, 52.82908356924063],
            [64.6469549479832, 53.27398996374289],
            [64.18239418858148, 53.3114905476879],
        ],
        0.998752734331801,
    )
    left = _poly(
        [
            [63.200241336410045, 52.04899809648456],
            [64.02962065843354, 51.808854272431],
            [63.8503831102771, 52.64455949792059],
            [63.139800590189765, 52.7445488404255],
        ],
        1.0,
    )
    down = _poly(
        [
            [64.099527131244, 51.190136212559004],
            [64.8349116186194, 51.227175494578375],
            [64.50863986686956, 52.19774323281559],
            [64.02962065843354, 51.808854272431],
        ],
        0.32310701798319624,
    )

    # Corners are irrelevant for this branch; only the cardinal neighbors matter.
    empty = _poly([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]], 0.0)
    full = _poly([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]], 1.0)
    center.stencil = [
        [empty, left, full],
        [down, center, up],
        [empty, right, full],
    ]

    orientation = center.findSafeOrientation()

    assert orientation is not None
    assert orientation[0] is right
    assert orientation[1] is up


if __name__ == "__main__":
    test_find_safe_orientation_handles_extreme_three_neighbor_case()
    print("Safe orientation tests completed.")
