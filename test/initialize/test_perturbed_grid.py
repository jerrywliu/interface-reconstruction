#!/usr/bin/env python3
"""
Checks for perturbed Cartesian grid generation.
"""

from main.geoms.geoms import getArea
from util.initialize.points import makePerturbedCartesianGrid


def test_perturbed_grid_noninverted():
    grid_size = 4
    resolution = 2
    points = makePerturbedCartesianGrid(
        grid_size, resolution, wiggle=0.2, seed=1, fix_boundary=True
    )

    for i in range(len(points) - 1):
        for j in range(len(points[0]) - 1):
            quad = [
                points[i][j],
                points[i + 1][j],
                points[i + 1][j + 1],
                points[i][j + 1],
            ]
            assert getArea(quad) > 0


def test_perturbed_grid_boundary_fixed():
    grid_size = 4
    resolution = 2
    points = makePerturbedCartesianGrid(
        grid_size, resolution, wiggle=0.3, seed=2, fix_boundary=True
    )
    n = len(points) - 1

    for i in range(n + 1):
        assert points[i][0][1] == 0
        assert points[i][n][1] == grid_size
    for j in range(n + 1):
        assert points[0][j][0] == 0
        assert points[n][j][0] == grid_size


if __name__ == "__main__":
    test_perturbed_grid_noninverted()
    test_perturbed_grid_boundary_fixed()
    print("Perturbed grid tests completed.")
