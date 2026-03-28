#!/usr/bin/env python3
"""
Focused regression tests for the ELVIRA / LVIRA baseline split.
"""

import math
import random

from main.geoms.geoms import getPolyLineArea
from main.structs.polys.base_polygon import BasePolygon


def _make_perturbed_stencil(seed):
    rng = random.Random(seed)
    vertices = []
    for i in range(4):
        column = []
        for j in range(4):
            x = float(i)
            y = float(j)
            if 0 < i < 3 and 0 < j < 3:
                x += rng.uniform(-0.18, 0.18)
                y += rng.uniform(-0.18, 0.18)
            column.append([x, y])
        vertices.append(column)

    return [
        [
            BasePolygon(
                [
                    vertices[i][j],
                    vertices[i + 1][j],
                    vertices[i + 1][j + 1],
                    vertices[i][j + 1],
                ]
            )
            for j in range(3)
        ]
        for i in range(3)
    ]


def _line_points(theta, offset, ref=(1.5, 1.5), span=10.0):
    normal = [math.cos(theta), math.sin(theta)]
    tangent = [-normal[1], normal[0]]
    point = [ref[0] + offset * normal[0], ref[1] + offset * normal[1]]
    l1 = [point[0] - span * tangent[0], point[1] - span * tangent[1]]
    l2 = [point[0] + span * tangent[0], point[1] + span * tangent[1]]
    return l1, l2


def _stencil_sse(stencil, facet):
    total = 0.0
    for row in stencil:
        for poly in row:
            predicted = getPolyLineArea(poly.points, facet.pLeft, facet.pRight) / poly.getMaxArea()
            total += (predicted - poly.getFraction()) ** 2
    return total


def test_lvira_improves_stencil_fit_on_perturbed_mesh():
    stencil = _make_perturbed_stencil(seed=2)
    true_l1, true_l2 = _line_points(theta=2.8653478367048724, offset=0.4030447383534144)

    for row in stencil:
        for poly in row:
            fraction = getPolyLineArea(poly.points, true_l1, true_l2) / poly.getMaxArea()
            poly.setFraction(fraction)

    center = stencil[1][1]
    center.set3x3Stencil(stencil)

    elvira = center.runELVIRA(ret=True)
    lvira = center.runLVIRA(ret=True)

    elvira_sse = _stencil_sse(stencil, elvira)
    lvira_sse = _stencil_sse(stencil, lvira)

    assert elvira.name == "ELVIRA"
    assert lvira.name == "LVIRA"
    assert elvira_sse > 1e-4
    assert lvira_sse < 1e-10
    assert lvira_sse < elvira_sse


if __name__ == "__main__":
    test_lvira_improves_stencil_fit_on_perturbed_mesh()
    print("LVIRA baseline tests completed.")
