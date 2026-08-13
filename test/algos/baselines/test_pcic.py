import math

import pytest

from main.algos.baselines.pcic import (
    PCICCircle,
    PCICError,
    collect_stencil_samples,
    fit_riemann_sphere,
    reconstruct_bare_pcic_cell,
    sample_plic_segment,
)
from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getDistance
from main.geoms.linear_facet import getLinearFacetFromNormal, getPolyLineIntersects
from main.structs.facets.linear_facet import LinearFacet
from main.structs.polys.base_polygon import BasePolygon


def _cell(x, y, fraction):
    polygon = BasePolygon([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
    polygon.setFraction(fraction)
    return polygon


def _circle_stencil(center=(1.5, -0.3), radius=2.0):
    polygons = [[None for _ in range(3)] for _ in range(3)]
    facets = [[None for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            points = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]
            area, _ = getCircleIntersectArea(list(center), radius, points)
            polygon = _cell(i, j, area)
            polygons[i][j] = polygon
            if 0.0 < area < 1.0:
                centroid = [i + 0.5, j + 0.5]
                normal = [center[0] - centroid[0], center[1] - centroid[1]]
                magnitude = math.hypot(*normal)
                if magnitude == 0.0:
                    normal = [1.0, 0.0]
                else:
                    normal = [normal[0] / magnitude, normal[1] / magnitude]
                line = getLinearFacetFromNormal(points, area, normal, 1.0e-12)
                intersects = getPolyLineIntersects(points, *line)
                facets[i][j] = LinearFacet(intersects[0], intersects[-1])
    return polygons, facets


def test_sample_plic_segment_uses_published_apportionment_vector():
    facet = LinearFacet([0.0, 0.0], [10.0, 20.0])
    expected = [
        [8.0, 16.0],
        [7.5, 15.0],
        [7.0, 14.0],
        [3.0, 6.0],
        [2.5, 5.0],
        [2.0, 4.0],
    ]
    for actual_point, expected_point in zip(sample_plic_segment(facet), expected):
        assert actual_point == pytest.approx(expected_point)


def test_riemann_sphere_fit_recovers_exact_circle():
    center = [2.25, -0.75]
    radius = 1.4
    points = [
        [center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)]
        for theta in (0.1, 0.5, 1.0, 1.8, 2.4, 3.1, 4.0, 5.2)
    ]
    fitted_center, fitted_radius = fit_riemann_sphere(points)
    assert fitted_center == pytest.approx(center, abs=1.0e-11)
    assert fitted_radius == pytest.approx(radius, abs=1.0e-11)


def test_collect_stencil_samples_excludes_paper_low_fraction_cutoff():
    polygons, facets = _circle_stencil()
    polygons[0][0].setFraction(0.007)
    facets[0][0] = LinearFacet([0.0, 0.0], [1.0, 1.0])
    eligible = sum(
        1
        for i in range(3)
        for j in range(3)
        if facets[i][j] is not None and 0.01 <= polygons[i][j].getFraction() <= 0.99
    )
    assert len(collect_stencil_samples(polygons, facets)) == 6 * eligible


@pytest.mark.parametrize("correction", ["translate_center", "adjust_radius"])
def test_bare_pcic_conserves_target_cell_volume(correction):
    polygons, facets = _circle_stencil()
    result = reconstruct_bare_pcic_cell(
        polygons[1][1], polygons, facets, correction=correction
    )
    assert isinstance(result, PCICCircle)
    assert result.fraction_in(polygons[1][1]) == pytest.approx(
        polygons[1][1].getFraction(), abs=1.0e-10
    )
    assert getDistance(result.center, [1.5, -0.3]) < 0.3


def test_four_crossing_result_is_not_silently_truncated():
    result = PCICCircle(
        center=(0.5, 0.5),
        radius=0.45,
        intersections=((0.1, 0.5), (0.5, 0.9), (0.9, 0.5), (0.5, 0.1)),
        source_center=(0.5, 0.5),
        source_radius=0.45,
        correction="translate_center",
    )
    with pytest.raises(PCICError, match="cannot represent"):
        result.to_arc_facet()
