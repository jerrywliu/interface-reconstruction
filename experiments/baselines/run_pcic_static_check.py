"""Run deterministic source-method checks for the Cartesian bare-PCIC kernel.

This is intentionally not labeled as a reproduction of the randomized ellipse
table in Maity, Sundararajan, and Velusamy (2021).  See
``experiments/baselines/PCIC_STATIC_CHECK.md`` for the remaining blockers.
"""

from __future__ import annotations

import json
import math

from main.algos.baselines.pcic import (
    PCICCircle,
    build_lls_parker_young_plic_stencil,
    reconstruct_bare_pcic_cartesian_cell,
    reconstruct_parker_young_plic,
)
from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, getPolyLineArea
from main.structs.polys.base_polygon import BasePolygon


def _subblock(block, center_i, center_j):
    return [
        [block[i][j] for j in range(center_j - 1, center_j + 2)]
        for i in range(center_i - 1, center_i + 2)
    ]


def _cartesian_block(fraction_function):
    block = []
    for i in range(7):
        column = []
        for j in range(7):
            x = i - 3
            y = j - 3
            points = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]
            polygon = BasePolygon(points)
            polygon.setFraction(fraction_function(points))
            column.append(polygon)
        block.append(column)
    return block


def _phase_normal(facet):
    dx = facet.pRight[0] - facet.pLeft[0]
    dy = facet.pRight[1] - facet.pLeft[1]
    magnitude = math.hypot(dx, dy)
    return [-dy / magnitude, dx / magnitude]


def _normal_angle(left, right):
    dot = max(-1.0, min(1.0, left[0] * right[0] + left[1] * right[1]))
    return math.acos(dot)


def _line_check():
    slope = 0.63
    intercept = 0.42
    line_left = [-20.0, -20.0 * slope + intercept]
    line_right = [20.0, 20.0 * slope + intercept]
    block = _cartesian_block(
        lambda points: getPolyLineArea(points, line_left, line_right)
    )
    parker_young = reconstruct_parker_young_plic(_subblock(block, 3, 3))
    lls = build_lls_parker_young_plic_stencil(block)[1][1]
    if lls is None:
        raise RuntimeError("The central LLS fixture was not reconstructed")

    expected = [-slope, 1.0]
    magnitude = math.hypot(*expected)
    expected = [expected[0] / magnitude, expected[1] / magnitude]
    target = block[3][3]
    lls_fraction = getPolyLineArea(target.points, lls.pLeft, lls.pRight) / abs(
        getArea(target.points)
    )
    return {
        "fixture": "deterministic straight line y=0.63x+0.42",
        "parker_young_normal_angle_rad": _normal_angle(
            _phase_normal(parker_young), expected
        ),
        "lls_normal_angle_rad": _normal_angle(_phase_normal(lls), expected),
        "lls_volume_fraction_residual": abs(lls_fraction - target.getFraction()),
    }


def _circle_check():
    exact_center = [0.5, -1.3]
    exact_radius = 2.0
    block = _cartesian_block(
        lambda points: getCircleIntersectArea(exact_center, exact_radius, points)[0]
    )
    rows = {}
    for correction in ("translate_center", "adjust_radius"):
        result = reconstruct_bare_pcic_cartesian_cell(
            block,
            correction=correction,
            phase="disk",
            center_translation_root_policy=(
                "nearest_bracket" if correction == "translate_center" else None
            ),
        )
        if not isinstance(result, PCICCircle):
            raise RuntimeError(f"{correction} unexpectedly retained a PLIC line")
        rows[result.source_variant] = {
            "target_fraction": block[3][3].getFraction(),
            "reconstructed_fraction": result.fraction_in(block[3][3]),
            "volume_fraction_residual": abs(
                result.fraction_in(block[3][3]) - block[3][3].getFraction()
            ),
            "center": list(result.center),
            "signed_radius": result.radius,
            "boundary_crossings": len(result.intersections),
            "connected_components": len(result.components),
            "component_pairing_status": result.component_pairing_status,
        }
    return {
        "fixture": "deterministic circle center=(0.5,-1.3), radius=2",
        "variants": rows,
    }


def main():
    line = _line_check()
    circle = _circle_check()
    if line["lls_normal_angle_rad"] >= line["parker_young_normal_angle_rad"]:
        raise RuntimeError("The cited one-pass LLS check did not improve the fixture")
    if any(
        row["volume_fraction_residual"] > 1.0e-9 for row in circle["variants"].values()
    ):
        raise RuntimeError("A bare-PCIC correction failed the conservation check")
    print(
        json.dumps(
            {
                "classification": "deterministic source-method kernel check",
                "paper_table_reproduction": False,
                "lls_line": line,
                "bare_pcic_circle": circle,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
