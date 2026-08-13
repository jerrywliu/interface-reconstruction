import pytest

from main.algos.baselines.plvira import parabolic_polygon_area
from main.algos.baselines.plvira_ghf import (
    GHFStencilError,
    cartesian_ghf_curvature,
    cartesian_myc_normal,
)


def _parabolic_fractions(size, curvature=0.2, shift=0.07):
    center = size // 2
    fractions = []
    for row in range(size):
        fraction_row = []
        for column in range(size):
            x = column - center
            y = row - center
            polygon = [
                (x - 0.5, y - 0.5),
                (x + 0.5, y - 0.5),
                (x + 0.5, y + 0.5),
                (x - 0.5, y + 0.5),
            ]
            fraction_row.append(
                parabolic_polygon_area(polygon, (0.0, 0.0), 0.0, curvature, shift)
            )
        fractions.append(fraction_row)
    return fractions


def test_myc_normal_points_from_liquid_to_empty():
    fractions = [[1.0, 0.5, 0.0] for _ in range(3)]
    normal = cartesian_myc_normal(fractions, (1, 1))
    assert normal[0] == pytest.approx(1.0)
    assert normal[1] == pytest.approx(0.0, abs=1.0e-29)


def test_height_function_is_exact_for_axis_aligned_parabola_at_center():
    fractions = _parabolic_fractions(15)
    center = len(fractions) // 2
    result = cartesian_ghf_curvature(fractions, (center, center), 1.0)
    assert result.method == "height_function"
    assert result.direction == "x"
    assert result.curvature == pytest.approx(0.2, abs=3.0e-13)


def test_height_function_includes_slope_denominator():
    fractions = _parabolic_fractions(15)
    center = len(fractions) // 2
    result = cartesian_ghf_curvature(fractions, (center + 1, center), 1.0)
    expected = 0.2 / (1.0 + 0.2**2) ** 1.5
    assert result.curvature == pytest.approx(expected, abs=3.0e-13)


def test_mixed_height_fallback_uses_both_directions():
    # Exact circle/square fractions for R/h=4, center=(0.013, 0.027), N=16,
    # around source-grid target (row=5, column=5).  This is a compact frozen
    # topology check; generation and convergence live in the reproducer.
    fractions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.178118185394581, 0.493886388742747, 0.546801716801658],
        [0.0, 0.0, 0.0, 0.485861884828692, 0.992713073763142, 1.0, 1.0],
        [0.0, 0.0, 0.279528650816171, 0.999314552416122, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.670544119228323, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.780976169302368, 1.0, 1.0, 1.0, 1.0],
    ]
    result = cartesian_ghf_curvature(fractions, (3, 3), 1.0 / 16.0)
    assert result.method == "mixed_height_parabola"
    assert result.height_points == 4
    assert result.independent_points == 3
    assert result.curvature > 0.0


def test_plic_centroid_fallback_is_preserved():
    # Exact circle/square fractions for R/h=2.4, center=(0.17h, 0.37h),
    # N=24, around source-grid target (row=10, column=10).
    fractions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00281665800493833, 0.0123337824235272, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.334871838176005,
            0.911955067211269,
            0.976872665608131,
            0.596683295000369,
            0.0132709447787758,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0772035602468687,
            0.967817952961788,
            1.0,
            1.0,
            1.0,
            0.385021513208659,
        ],
        [0.0, 0.0, 0.0, 0.208921359392554, 1.0, 1.0, 1.0, 1.0, 0.548921359392556],
        [
            0.0,
            0.0,
            0.0,
            0.03142176557259,
            0.890084372197316,
            1.0,
            1.0,
            0.997395474456753,
            0.264110663313155,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.152605418940479,
            0.654771725216209,
            0.72920644803166,
            0.339287820543619,
            0.0,
        ],
    ]
    result = cartesian_ghf_curvature(fractions, (4, 4), 1.0 / 24.0)
    assert result.method == "plic_centroid_parabola"
    assert result.fit_points == 5
    assert result.independent_points == 3
    assert result.curvature == pytest.approx(10.6076619865, rel=2.0e-10)


def test_boundary_policy_is_not_invented():
    fractions = _parabolic_fractions(5)
    with pytest.raises(GHFStencilError, match="no boundary-stencil policy"):
        cartesian_ghf_curvature(fractions, (0, 2), 1.0)
