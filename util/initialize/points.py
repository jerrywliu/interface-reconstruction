import numpy as np

from main.geoms.geoms import getArea

#Generate cartesian mesh, gridSize*resolution x gridSize*resolution grid
def makeFineCartesianGrid(gridSize, resolution):
    print("Making quad mesh")
    points = [[0] * (int(gridSize*resolution)+1) for _ in range(int(gridSize*resolution)+1)]
    for x in range(len(points)):
        for y in range(len(points)):
            points[x][y] = [x/resolution, y/resolution]
    print("Done")
    return points

#Generate quad mesh
def makeQuadGrid(gridSize, resolution, wiggle=0.25):
    rng = np.random.RandomState(0)
    print("Making quad mesh")
    points = [[0] * (int(gridSize*resolution)+1) for _ in range(int(gridSize*resolution)+1)]
    for x in range(len(points)):
        for y in range(len(points)):
            points[x][y] = [(x + wiggle*rng.rand())/resolution, (y + wiggle*rng.rand())/resolution]
    print("Done")
    return points

#Generate concave mesh
def makeConcaveGrid(gridSize, wiggle):
    print("Making quad mesh")
    points = [[0] * (gridSize+1) for _ in range(gridSize+1)]
    for x in range(gridSize+1):
        for y in range(gridSize+1):
            if (x+y) % 2 == 1:
                points[x][y] = [x-wiggle, y-wiggle]
            else:
                points[x][y] = [x, y]
    print("Done")
    return points


def makePerturbedCartesianGrid(
    gridSize,
    resolution,
    wiggle,
    seed=0,
    fix_boundary=True,
    max_tries=20,
    perturb_type="random",
):
    n = int(gridSize * resolution)
    dx0 = 1.0 / resolution
    rng = np.random.default_rng(seed)

    base_points = [
        [[i * dx0, j * dx0] for j in range(n + 1)] for i in range(n + 1)
    ]

    for attempt in range(max_tries):
        points = [[p[:] for p in row] for row in base_points]

        for i in range(n + 1):
            for j in range(n + 1):
                if fix_boundary and (i in (0, n) or j in (0, n)):
                    continue
                if perturb_type == "smooth":
                    freq = 1.0
                    phase_x = rng.uniform(0, 2 * np.pi)
                    phase_y = rng.uniform(0, 2 * np.pi)
                    dx = (
                        wiggle
                        * dx0
                        * np.sin(2 * np.pi * freq * i / n + phase_x)
                        * np.sin(2 * np.pi * freq * j / n + phase_y)
                    )
                    dy = (
                        wiggle
                        * dx0
                        * np.sin(2 * np.pi * freq * j / n + phase_y)
                        * np.sin(2 * np.pi * freq * i / n + phase_x)
                    )
                else:
                    dx = rng.uniform(-1.0, 1.0) * wiggle * dx0
                    dy = rng.uniform(-1.0, 1.0) * wiggle * dx0

                points[i][j][0] = base_points[i][j][0] + dx
                points[i][j][1] = base_points[i][j][1] + dy

        if _all_quads_positive_area(points):
            return points

    raise RuntimeError(
        f"Failed to generate non-inverted perturbed grid after {max_tries} tries"
    )


def _all_quads_positive_area(points):
    for i in range(len(points) - 1):
        for j in range(len(points[0]) - 1):
            quad = [
                points[i][j],
                points[i + 1][j],
                points[i + 1][j + 1],
                points[i][j + 1],
            ]
            if getArea(quad) <= 0:
                return False
    return True
