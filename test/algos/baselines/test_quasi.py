import math

import pytest

from main.algos.baselines.quasi import (
    DEFAULT_QUASI_POLICY,
    QuadraticFacet,
    QuasiPolicy,
    QuasiTopologyError,
    _verified_algebraic_roots,
    reconstruct_quasi,
)
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.base_mesh import BaseMesh


def _cartesian_points(nx, ny):
    return [[[float(x), float(y)] for y in range(ny + 1)] for x in range(nx + 1)]


def _horizontal_interface_mesh(nx=4, ny=3, height=1.3):
    fractions = []
    for x in range(nx):
        column = []
        for y in range(ny):
            column.append(min(1.0, max(0.0, y + 1.0 - height)))
        fractions.append(column)
    return BaseMesh(_cartesian_points(nx, ny), 1.0e-10, fractions=fractions)


def test_quadratic_facet_preserves_analytic_cell_area():
    polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    facet = QuadraticFacet([0.0, 0.2], [1.0, 0.7], bulge=0.15)

    chord_area = 0.55
    expected = chord_area - (2.0 / 3.0) * 0.15 * math.sqrt(1.25)
    assert facet.represented_area(polygon) == pytest.approx(expected, abs=1.0e-14)
    assert facet.point(0.0) == pytest.approx(facet.pLeft)
    assert facet.point(1.0) == pytest.approx(facet.pRight)


def test_quasi_recovers_a_connected_straight_interface():
    mesh = _horizontal_interface_mesh()
    result = reconstruct_quasi(mesh)

    assert len(result.facets) == 4
    assert len(result.joins) == 3
    assert result.unresolved == []
    for (x, y), facet in result.facets.items():
        assert y == 1
        assert facet.represented_area(mesh.polys[x][y].points) == pytest.approx(
            mesh.polys[x][y].getArea(), abs=1.0e-10
        )

    ordered = [result.facets[(x, 1)] for x in range(4)]
    for left, right in zip(ordered[:-1], ordered[1:]):
        shared_distance = min(
            math.dist(a, b)
            for a in (left.pLeft, left.pRight)
            for b in (right.pLeft, right.pRight)
        )
        assert shared_distance < 1.0e-10

    for join in result.joins:
        tangents = []
        for cell, slot in zip(join.cells, join.slots):
            facet = result.facets[cell]
            tangents.append(
                facet.getLeftTangent() if slot == 0 else facet.getRightTangent()
            )
        cross = tangents[0][0] * tangents[1][1] - tangents[0][1] * tangents[1][0]
        scale = math.hypot(*tangents[0]) * math.hypot(*tangents[1])
        assert abs(cross) / scale < 1.0e-7
    assert result.sweeps_completed == DEFAULT_QUASI_POLICY.max_sweeps
    assert result.c1_misses == 0
    assert result.policy["update_order"] == "lexicographic Gauss-Seidel"


def _diagonal_interface_mesh(fraction=0.5):
    class StubPolygon:
        def __init__(self, points, fraction, endpoints):
            self.points = points
            self._fraction = fraction
            self._endpoints = endpoints

        def isMixed(self, tolerance=None):
            return 0.0 < self._fraction < 1.0

        def set3x3Stencil(self, stencil):
            self.stencil = stencil

        def runYoungs(self, ret=False):
            return LinearFacet(*self._endpoints)

        def getArea(self):
            return self._fraction

    class StubMesh:
        def __init__(self):
            self.polys = [
                [
                    StubPolygon(
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        fraction,
                        ([0.0, 0.5], [1.0, 0.5]),
                    ),
                    StubPolygon(
                        [[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
                        0.0,
                        ([0.0, 1.5], [1.0, 1.5]),
                    ),
                ],
                [
                    StubPolygon(
                        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]],
                        0.0,
                        ([1.0, 0.5], [2.0, 0.5]),
                    ),
                    StubPolygon(
                        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
                        fraction,
                        ([1.0, 1.5], [2.0, 1.5]),
                    ),
                ],
            ]

        def get3x3Stencil(self, x, y):
            return [[None] * 3 for _ in range(3)]

    return StubMesh()


def test_section_25_vertex_jump_connects_diagonal_cells_conservatively():
    mesh = _diagonal_interface_mesh()
    result = reconstruct_quasi(mesh)

    assert result.unresolved == []
    assert result.vertex_jumps == 1
    assert result.sweeps_completed == 0
    assert result.converged
    assert any(join.kind == "vertex-jump" for join in result.joins)
    first = result.facets[(0, 0)]
    second = result.facets[(1, 1)]
    assert (
        min(
            math.dist(a, b)
            for a in (first.pLeft, first.pRight)
            for b in (second.pLeft, second.pRight)
        )
        < 1.0e-12
    )
    for index, facet in result.facets.items():
        assert facet.represented_area(
            mesh.polys[index[0]][index[1]].points
        ) == pytest.approx(mesh.polys[index[0]][index[1]].getArea(), abs=1.0e-10)


def test_fallback_is_reported_and_strict_mode_raises():
    mesh = _diagonal_interface_mesh(fraction=0.01)
    result = reconstruct_quasi(mesh)
    assert result.unresolved
    assert result.curvature_updates == 0

    with pytest.raises(QuasiTopologyError, match="No eligible Section 2.5 target"):
        reconstruct_quasi(mesh, strict=True)


def test_repeated_continuity_root_is_not_lost():
    root = 0.375
    coefficients = [root * root, -2.0 * root, 1.0]
    roots = _verified_algebraic_roots(
        coefficients,
        lambda alpha: (alpha - root) ** 2,
        current_parameter=0.7,
        tolerance=1.0e-12,
    )
    assert roots == pytest.approx([root], abs=1.0e-9)


def test_policy_validation_and_zero_sweep_override():
    with pytest.raises(ValueError, match="target fraction bounds"):
        QuasiPolicy(target_fraction_lower=0.7, target_fraction_upper=0.2)

    result = reconstruct_quasi(_horizontal_interface_mesh(), iterations=0)
    assert result.sweeps_completed == 0
    assert not result.converged
