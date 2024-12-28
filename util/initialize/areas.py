import math
import numpy as np

from main.geoms.geoms import (
    getDistance,
    getArea,
    getPolyIntersectArea,
    getPolyLineArea,
)
from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.corner_facet import getPolyCurvedCornerArea
from main.structs.polys.base_polygon import BasePolygon
from main.structs.meshes.base_mesh import BaseMesh

from util.initialize.velocity import initializeVelocity


# return areas array
def initializeCircle(m: BaseMesh, center, radius):
    areas = [[0] * len(m.polys[0]) for _ in range(len(m.polys))]

    for x in range(len(areas)):
        for y in range(len(areas[0])):
            poly: BasePolygon = m.polys[x][y]
            intersectarea, _ = getCircleIntersectArea(center, radius, poly.points)

            areas[x][y] = intersectarea / poly.getMaxArea()

    return areas


def initializePoly(m: BaseMesh, poly):
    areas = [[0] * len(m.polys[0]) for _ in range(len(m.polys))]

    for x in range(len(areas)):
        for y in range(len(areas[0])):
            mesh_poly: BasePolygon = m.polys[x][y]
            polyintersects = getPolyIntersectArea(poly, mesh_poly.points)
            for polyintersect in polyintersects:
                areas[x][y] += getArea(polyintersect)
            areas[x][y] /= mesh_poly.getMaxArea()

    return areas


# theta is angle from positive x-axis to major axis
def initializeEllipse(m: BaseMesh, major_axis, minor_axis, theta, center):
    areas = [[0] * len(m.polys[0]) for _ in range(len(m.polys))]

    circle_to_ellipse = np.array(
        [
            [
                major_axis * math.cos(theta) ** 2 + minor_axis * math.sin(theta) ** 2,
                (major_axis - minor_axis) * math.cos(theta) * math.sin(theta),
            ],
            [
                (major_axis - minor_axis) * math.cos(theta) * math.sin(theta),
                major_axis * math.sin(theta) ** 2 + minor_axis * math.cos(theta) ** 2,
            ],
        ]
    )
    ellipse_to_circle = np.linalg.inv(circle_to_ellipse)

    for x in range(len(areas)):
        for y in range(len(areas[0])):
            poly: BasePolygon = m.polys[x][y]
            centered_opoly = list(
                map(lambda x: [x[0] - center[0], x[1] - center[1]], poly.points)
            )
            squished_opoly = list(
                map(
                    lambda x: [
                        ellipse_to_circle[0][0] * x[0] + ellipse_to_circle[0][1] * x[1],
                        ellipse_to_circle[1][0] * x[0] + ellipse_to_circle[1][1] * x[1],
                    ],
                    centered_opoly,
                )
            )
            intersectarea, _ = getCircleIntersectArea([0, 0], 1, squished_opoly)
            areas[x][y] = intersectarea * major_axis * minor_axis
            areas[x][y] /= poly.getMaxArea()
            if areas[x][y] > 1:
                print(f"Error in initializeEllipse: {areas[x][y]}")
                areas[x][y] = 1

    return areas


# Area to left of line from l1 to l2
def initializeLine(m: BaseMesh, l1, l2):
    areas = [[0] * len(m.polys[0]) for _ in range(len(m.polys))]

    for x in range(len(areas)):
        for y in range(len(areas[0])):
            mesh_poly: BasePolygon = m.polys[x][y]
            area = getPolyLineArea(mesh_poly.points, l1, l2)
            areas[x][y] = area / mesh_poly.getMaxArea()

    return areas


# Example from Zalesak: circle with rectangle removed, use with 100x100 grid
def zalesak(m: BaseMesh):
    areas = [[0] * len(m.polys[0]) for _ in range(len(m.polys))]

    for x in range(len(areas)):
        for y in range(len(areas[0])):
            poly = m.polys[x][y]

            radiussmall = 15
            center = [50.05, 75.05]

            area, intersect = getCircleIntersectArea(center, radiussmall, poly.points)
            areas[x][y] += area

            rectangle = [[47.55, 59.5], [52.55, 59.5], [52.55, 85.05], [47.55, 85.05]]
            intersects = getPolyIntersectArea(rectangle, poly.points)
            for intersect in intersects:
                areas[x][y] -= getArea(intersect)
            areas[x][y] = max(0, areas[x][y])

            if getDistance(poly.points[0], [47.55, 60.25980054160715]) < math.sqrt(2):
                areas[x][y] = getPolyCurvedCornerArea(
                    poly.points,
                    [35.05, 75.05],
                    [47.55, 60.25980054160715],
                    [47.55, 85.05],
                    15,
                    None,
                )
            elif getDistance(poly.points[0], [52.55, 60.25980054160715]) < math.sqrt(2):
                areas[x][y] = getPolyCurvedCornerArea(
                    poly.points,
                    [52.55, 85.05],
                    [52.55, 60.25980054160715],
                    [65.05, 75.05],
                    None,
                    15,
                )

            areas[x][y] /= poly.getMaxArea()

    return areas


# x+o example: use with 100x100 grid
def xpluso(m: BaseMesh, dx=0):
    areas = [[0] * len(m.polys[0]) for _ in range(len(m.polys))]

    for x in range(len(areas)):
        for y in range(len(areas[0])):
            poly = m.polys[x][y]

            # material 1 cross
            xpoints = [
                [4, 0],
                [10, 6],
                [16, 0],
                [20, 4],
                [14, 10],
                [20, 16],
                [16, 20],
                [10, 14],
                [4, 20],
                [0, 16],
                [6, 10],
                [0, 4],
            ]
            xpoints = list(
                map(lambda x: [x[0] + 3.005 + dx, x[1] + 3.025 + dx], xpoints)
            )
            xpolyintersects = getPolyIntersectArea(xpoints, poly.points)
            for xpolyintersect in xpolyintersects:
                areas[x][y] += abs(getArea(xpolyintersect))

            # #material 2 cross
            xpoints = [
                [6, 0],
                [14, 0],
                [14, 6],
                [20, 6],
                [20, 14],
                [14, 14],
                [14, 20],
                [6, 20],
                [6, 14],
                [0, 14],
                [0, 6],
                [6, 6],
            ]
            xpoints = list(
                map(lambda x: [x[0] + 3.005 + dx, x[1] + 28.025 + dx], xpoints)
            )
            xpolyintersects = getPolyIntersectArea(xpoints, poly.points)
            for xpolyintersect in xpolyintersects:
                areas[x][y] += abs(getArea(xpolyintersect))

            # #material 1 ring
            radius = 10
            radiussmall = 6
            center = [38.005 + dx, 13.005 + dx]
            area, _ = getCircleIntersectArea(center, radius, poly.points)
            areas[x][y] += area
            area, _ = getCircleIntersectArea(center, radiussmall, poly.points)
            areas[x][y] -= area

            # #material 2 ring
            radius = 10
            radiussmall = 6
            center = [38.005 + dx, 38.005 + dx]
            area, _ = getCircleIntersectArea(center, radius, poly.points)
            areas[x][y] += area
            area, _ = getCircleIntersectArea(center, radiussmall, poly.points)
            areas[x][y] -= area

            # #material 2 dot
            radius = 3
            center = [38.005 + dx, 13.005 + dx]
            area, _ = getCircleIntersectArea(center, radius, poly.points)
            areas[x][y] += area

            areas[x][y] /= poly.getMaxArea()

    return areas


# Initialize areas for advection tests
def initializeAreas(m: BaseMesh, test_setting="vortex"):
    if test_setting == "vortex":
        fractions = initializeCircle(m, [50.01, 75.01], 15)
    if test_setting == "deformation":
        fractions = initializeCircle(m, [50.01, 75.01], 15)
    elif test_setting == "zalesak":
        fractions = zalesak(m)
    elif test_setting == "xpluso":
        fractions = xpluso(m)
    elif test_setting == "triangle":
        fractions = initializePoly(m, [[1.5, 1.5], [9.5, 11.5], [5.5, 15.2]])
    elif test_setting == "rectangle":
        fractions = initializePoly(
            m, [[8.2, 8.2], [12.2, 5.2], [15.2, 9.2], [11.2, 12.2]]
        )
    elif test_setting == "ellipse_merge":
        fractions = initializeEllipse(m, math.pi, 1.2, math.pi / 2, [50.5, 50])
    elif test_setting == "line":
        fractions = initializeLine(m, [50.2, 50], [50, 50.35])

    return fractions


# Get final areas for vortex, deformation, zalesak, xpluso tests
def trueFinalAreas(m: BaseMesh, test_setting="vortex", t=2):
    if test_setting == "vortex":
        fractions = initializeCircle(m, [50.01, 75.01], 15)
    if test_setting == "deformation":
        fractions = initializeCircle(m, [50.01, 75.01], 15)
    elif test_setting == "zalesak":
        fractions = zalesak(m)
    elif test_setting == "xpluso":
        velocity = initializeVelocity(m, t, test_setting=test_setting)
        dx = (
            velocity(0, [0, 0])[0] * t
        )  # constant velocity for x+o test with same x and y components
        fractions = xpluso(m, dx)

    return fractions
