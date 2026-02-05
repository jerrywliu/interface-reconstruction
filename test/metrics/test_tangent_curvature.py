import math

import numpy as np

from main.structs.facets.linear_facet import LinearFacet
from util.metrics.metrics import (
    polyline_average_curvature,
    tangent_error_to_curve,
)


def _circle_points(radius, n=64):
    points = []
    tangents = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        points.append([x, y])
        tangents.append([-math.sin(t), math.cos(t)])
    return points, tangents


def _ellipse_points(a, b, n=128):
    points = []
    tangents = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = a * math.cos(t)
        y = b * math.sin(t)
        points.append([x, y])
        tangents.append([-a * math.sin(t), b * math.cos(t)])
    return points, tangents


def _polyline_facets(points):
    facets = []
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        facets.append(LinearFacet(p0, p1))
    return facets


def test_tangent_error_line_zero():
    true_points = [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]]
    true_tangents = [[1.0, 0.0] for _ in true_points]
    facets = [LinearFacet([0.0, 0.0], [1.0, 0.0])]

    stats = tangent_error_to_curve(facets, true_points, true_tangents, n_per_facet=10)
    assert stats["mean"] < 1e-6


def test_circle_curvature_proxy():
    radius = 2.0
    points, _ = _circle_points(radius, n=128)
    avg_curv = polyline_average_curvature(points, closed=True)
    assert abs(avg_curv - 1 / radius) < 0.05


def test_circle_tangent_error_linear_facets():
    radius = 1.0
    points, tangents = _circle_points(radius, n=96)
    facets = _polyline_facets(points)
    stats = tangent_error_to_curve(facets, points, tangents, n_per_facet=5)
    assert stats["mean"] < 0.15


def test_ellipse_tangent_error_linear_facets():
    a, b = 2.0, 1.0
    points, tangents = _ellipse_points(a, b, n=128)
    facets = _polyline_facets(points)
    stats = tangent_error_to_curve(facets, points, tangents, n_per_facet=5)
    assert stats["mean"] < 0.2
