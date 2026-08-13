import math

import pytest

import main.algos.baselines.pcic as pcic
from main.algos.baselines.pcic import (
    CENTER_TRANSLATION_VARIANT,
    RADIUS_ADJUSTMENT_VARIANT,
    PCICAmbiguousSourceChoice,
    PCICCircle,
    PCICConfig,
    PCICError,
    build_lls_parker_young_plic_stencil,
    collect_stencil_samples,
    fit_riemann_sphere,
    parker_young_normal,
    reconstruct_bare_pcic_cell,
    reconstruct_bare_pcic_cartesian_cell,
    reconstruct_lls_plic,
    reconstruct_parker_young_plic,
    sample_plic_segment,
)
from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, getDistance, getPolyLineArea, pointInPoly
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


def _subblock(block, center_i, center_j):
    return [
        [block[i][j] for j in range(center_j - 1, center_j + 2)]
        for i in range(center_i - 1, center_i + 2)
    ]


def _circle_block(center=(0.5, -1.3), radius=2.0):
    block = []
    for i in range(7):
        column = []
        for j in range(7):
            x = i - 3
            y = j - 3
            points = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]
            area, _ = getCircleIntersectArea(list(center), radius, points)
            polygon = BasePolygon(points)
            polygon.setFraction(area)
            column.append(polygon)
        block.append(column)
    return block


def _line_block(slope=0.63, intercept=0.42):
    block = []
    line_left = [-20.0, -20.0 * slope + intercept]
    line_right = [20.0, 20.0 * slope + intercept]
    for i in range(7):
        column = []
        for j in range(7):
            x = i - 3
            y = j - 3
            points = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]
            polygon = BasePolygon(points)
            polygon.setFraction(getPolyLineArea(points, line_left, line_right))
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


def test_parker_young_normal_matches_published_a2_stencil():
    fractions = [
        [0.05, 0.10, 0.20],
        [0.15, 0.45, 0.70],
        [0.35, 0.75, 0.95],
    ]
    polygons = [[_cell(i, j, fractions[i][j]) for j in range(3)] for i in range(3)]
    gradient = [
        (
            fractions[2][0]
            + 2.0 * fractions[2][1]
            + fractions[2][2]
            - fractions[0][0]
            - 2.0 * fractions[0][1]
            - fractions[0][2]
        )
        / 8.0,
        (
            fractions[0][2]
            + 2.0 * fractions[1][2]
            + fractions[2][2]
            - fractions[0][0]
            - 2.0 * fractions[1][0]
            - fractions[2][0]
        )
        / 8.0,
    ]
    magnitude = math.hypot(*gradient)
    expected = [gradient[0] / magnitude, gradient[1] / magnitude]
    assert parker_young_normal(polygons) == pytest.approx(expected, abs=1.0e-14)


def test_one_pass_lls_parker_young_predictor_improves_source_line_fixture():
    slope = 0.63
    block = _line_block(slope=slope)
    central_stencil = _subblock(block, 3, 3)
    parker_young = reconstruct_parker_young_plic(central_stencil)
    lls = build_lls_parker_young_plic_stencil(block)[1][1]
    assert lls is not None

    expected = [-slope, 1.0]
    magnitude = math.hypot(*expected)
    expected = [expected[0] / magnitude, expected[1] / magnitude]
    parker_young_error = _normal_angle(_phase_normal(parker_young), expected)
    lls_error = _normal_angle(_phase_normal(lls), expected)
    assert parker_young_error == pytest.approx(2.9494863488e-2, rel=1.0e-9)
    assert lls_error == pytest.approx(6.0824367739e-4, rel=1.0e-8)
    assert lls_error < parker_young_error

    target = block[3][3]
    reconstructed_fraction = getPolyLineArea(
        target.points, lls.pLeft, lls.pRight
    ) / abs(getArea(target.points))
    assert reconstructed_fraction == pytest.approx(target.getFraction(), abs=1.0e-10)


def test_lls_overcrowded_radius_reduction_is_an_explicit_source_choice():
    polygons = [[_cell(i, j, 0.5) for j in range(3)] for i in range(3)]
    facets = [
        [LinearFacet([i, j + 0.5], [i + 1.0, j + 0.5]) for j in range(3)]
        for i in range(3)
    ]
    with pytest.raises(PCICAmbiguousSourceChoice, match="does not specify"):
        reconstruct_lls_plic(polygons, facets)


@pytest.mark.parametrize("correction", ["translate_center", "adjust_radius"])
def test_bare_pcic_conserves_target_cell_volume(correction):
    polygons, facets = _circle_stencil()
    result = reconstruct_bare_pcic_cell(
        polygons[1][1],
        polygons,
        facets,
        correction=correction,
        phase="disk",
        center_translation_root_policy=(
            "nearest_bracket" if correction == "translate_center" else None
        ),
    )
    assert isinstance(result, PCICCircle)
    assert result.fraction_in(polygons[1][1]) == pytest.approx(
        polygons[1][1].getFraction(), abs=1.0e-10
    )
    assert getDistance(result.center, [1.5, -0.3]) < 0.3


@pytest.mark.parametrize(
    ("correction", "variant"),
    [
        ("translate_center", CENTER_TRANSLATION_VARIANT),
        ("adjust_radius", RADIUS_ADJUSTMENT_VARIANT),
    ],
)
def test_cartesian_bare_pcic_builds_the_cited_lls_predictor(correction, variant):
    block = _circle_block()
    plic_stencil = build_lls_parker_young_plic_stencil(block)
    assert plic_stencil[1][1].name == "PCIC LLS/Parker-Young"

    result = reconstruct_bare_pcic_cartesian_cell(
        block,
        correction=correction,
        phase="disk",
        center_translation_root_policy=(
            "nearest_bracket" if correction == "translate_center" else None
        ),
    )
    assert isinstance(result, PCICCircle)
    assert result.source_variant == variant
    assert result.fraction_in(block[3][3]) == pytest.approx(
        block[3][3].getFraction(), abs=1.0e-10
    )
    assert result.component_pairing_status == "paired"
    assert len(result.components) == 1


def test_center_translation_direction_is_the_fitted_chord_bisector():
    direction = pcic._fitted_chord_perpendicular(
        [2.0, 0.5], [[1.0, 0.0], [1.0, 1.0]], PCICConfig()
    )
    assert direction == pytest.approx([1.0, 0.0], abs=1.0e-14)


def test_center_translation_requires_an_explicit_root_policy():
    polygons, facets = _circle_stencil()
    with pytest.raises(PCICAmbiguousSourceChoice, match="root policy"):
        reconstruct_bare_pcic_cell(
            polygons[1][1],
            polygons,
            facets,
            correction="translate_center",
            phase="disk",
        )


def test_center_translation_checkpoints_a_multi_chord_fitted_cell():
    with pytest.raises(PCICAmbiguousSourceChoice, match="4 boundary crossings"):
        pcic._fitted_chord_perpendicular(
            [0.5, 0.5],
            [[0.2, 0.0], [1.0, 0.2], [0.8, 1.0], [0.0, 0.8]],
            PCICConfig(),
        )


def test_four_crossing_result_is_not_silently_truncated():
    result = PCICCircle(
        center=(0.5, 0.5),
        radius=0.45,
        intersections=((0.1, 0.5), (0.5, 0.9), (0.9, 0.5), (0.5, 0.1)),
        source_center=(0.5, 0.5),
        source_radius=0.45,
        correction="translate_center",
        phase="disk",
    )
    with pytest.raises(PCICError, match="cannot represent"):
        result.to_arc_facet()


def test_four_crossings_are_paired_as_two_preserved_arc_components():
    polygon = _cell(0.0, 0.0, 0.5)
    center = [-0.25, 0.5]
    radius = 1.3
    _, intersections = getCircleIntersectArea(center, radius, polygon.points)
    assert len(intersections) == 4
    components, status = pcic._pair_circle_components(
        center, radius, intersections, polygon.points, PCICConfig()
    )
    result = PCICCircle(
        center=tuple(center),
        radius=radius,
        intersections=tuple(tuple(point) for point in intersections),
        source_center=tuple(center),
        source_radius=radius,
        correction="adjust_radius",
        phase="disk",
        components=components,
        component_pairing_status=status,
    )
    assert status == "paired"
    assert len(result.components) == 2
    assert len(result.to_arc_facets()) == 2
    for component in result.components:
        assert pointInPoly(component.sample(3)[1], polygon.points)
    with pytest.raises(PCICError, match="2 connected component"):
        result.to_arc_facet()
