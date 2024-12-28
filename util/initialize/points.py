import numpy as np

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