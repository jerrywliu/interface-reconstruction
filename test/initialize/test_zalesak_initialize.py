#!/usr/bin/env python3
"""
Regression coverage for perturbed Zalesak area initialization.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static.zalesak import compute_zalesak_cell_area
from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, getPolyIntersectArea


def test_zalesak_cell_area_preserves_circle_only_shoulder_overlap():
    # Reproduced from the r=1.5, w=0.3, seed=0, case=24 shoulder inconsistency.
    center = [50.919486089099685, 50.880855811079044]
    radius = 15.0
    slot_rect = [
        [62.77155160790928, 41.3530767971721],
        [65.22098979694127, 45.712004711018764],
        [43.42634935592234, 57.95919614606636],
        [40.976911166890346, 53.60026823221969],
    ]
    cell_poly = [
        [64.68934542143934, 45.43897058301251],
        [65.38413158888528, 45.244981471804536],
        [65.39359768690076, 46.12103902882231],
        [64.51368682180798, 46.04393506058253],
    ]

    corrected_area = compute_zalesak_cell_area(cell_poly, center, radius, slot_rect)

    circle_area, _ = getCircleIntersectArea(center, radius, cell_poly)
    full_slot_overlap = sum(
        abs(getArea(inter)) for inter in getPolyIntersectArea(slot_rect, cell_poly)
    )
    naive_area = max(0.0, circle_area - full_slot_overlap)

    cell_max_area = abs(getArea(cell_poly))
    corrected_fraction = corrected_area / cell_max_area

    assert naive_area < 1e-12
    assert corrected_fraction > 0.12
    assert abs(corrected_fraction - 0.12447848242570808) < 5e-6


if __name__ == "__main__":
    test_zalesak_cell_area_preserves_circle_only_shoulder_overlap()
    print("Zalesak initialization tests completed.")
