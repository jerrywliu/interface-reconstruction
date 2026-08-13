import math

import numpy as np
import pytest

from experiments.baselines.project_benchmarks import (
    DEFAULT_BENCHMARKS,
    canonical_benchmark_cases,
)


SEEDS = {"lines": 42, "squares": 42, "circles": 41, "ellipses": 42, "zalesak": 43}


def test_fixture_advances_the_full_canonical_rng_sequence():
    for benchmark in DEFAULT_BENCHMARKS:
        selected = canonical_benchmark_cases(benchmark, (0, 4, 24))
        assert [case.case_index for case in selected] == [0, 4, 24]
        assert all(case.random_seed == SEEDS[benchmark] for case in selected)


def test_first_canonical_case_parameters_match_static_driver_draw_order():
    line_rng = np.random.default_rng(42)
    line = canonical_benchmark_cases("lines", (0,))[0]
    assert line.parameters["p_left"] == pytest.approx(
        [line_rng.uniform(50, 51), line_rng.uniform(50, 51)]
    )

    square_rng = np.random.default_rng(42)
    square = canonical_benchmark_cases("squares", (0,))[0]
    assert square.parameters["center"] == pytest.approx(
        [square_rng.uniform(50, 51), square_rng.uniform(50, 51)]
    )
    assert square.parameters["theta"] == pytest.approx(square_rng.uniform(0, math.pi / 2))

    circle_rng = np.random.default_rng(41)
    circle = canonical_benchmark_cases("circles", (0,))[0]
    assert circle.parameters["center"] == pytest.approx(
        [circle_rng.uniform(50, 51), circle_rng.uniform(50, 51)]
    )

    ellipse_rng = np.random.default_rng(42)
    ellipse = canonical_benchmark_cases("ellipses", (0,))[0]
    assert ellipse.parameters["center"] == pytest.approx(
        [ellipse_rng.uniform(50, 51), ellipse_rng.uniform(50, 51)]
    )
    assert ellipse.parameters["theta"] == pytest.approx(ellipse_rng.uniform(0, math.pi / 2))

    zalesak_rng = np.random.default_rng(43)
    zalesak = canonical_benchmark_cases("zalesak", (0,))[0]
    assert zalesak.parameters["center"] == pytest.approx(
        [zalesak_rng.uniform(50, 51), zalesak_rng.uniform(50, 51)]
    )
    assert zalesak.parameters["theta"] == pytest.approx(zalesak_rng.uniform(0, math.pi / 2))


@pytest.mark.parametrize("benchmark", DEFAULT_BENCHMARKS)
def test_fixture_initializes_nonempty_canonical_mixed_cells(benchmark):
    case = canonical_benchmark_cases(benchmark, (0,))[0]
    mesh = case.build_mesh(32)
    fractions = case.initialize_fractions(mesh)
    mixed = [value for column in fractions for value in column if 1e-10 < value < 1 - 1e-10]
    assert mixed
    assert len(mesh.polys) == 32
    assert len(mesh.polys[0]) == 32


def test_ellipse_curvature_is_exact_on_principal_axes_after_rotation():
    case = canonical_benchmark_cases("ellipses", (0,))[0]
    p = case.parameters
    point = (
        p["center"][0] + math.cos(p["theta"]) * p["major_axis"],
        p["center"][1] + math.sin(p["theta"]) * p["major_axis"],
    )
    assert case.true_curvature_at(point) == pytest.approx(
        p["major_axis"] / p["minor_axis"] ** 2
    )
