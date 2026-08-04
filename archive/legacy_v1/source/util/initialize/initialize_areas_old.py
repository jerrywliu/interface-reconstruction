import math
from geoms import getDistance, getArea, mergePolys, getPolyIntersectArea, getPolyLineArea, getPolyLineIntersects, lineIntersect, getCentroid
from linear_facet import getLinearFacet
from circular_facet import getCircleIntersectArea, getCircleCircleIntersects, getArcFacet, getArcFacetNewton, getCircleLineIntersects2, getCenter
from corner_facet import getPolyCornerArea, getPolyCurvedCornerArea, getCurvedCornerFacet
import numpy as np
from scipy.integrate import dblquad

def initializeCircle(opolys, center, radius, threshold):
    areas = [[0] * len(opolys[0]) for _ in range(len(opolys))]

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            opoly = opolys[x][y]
            intersectarea, _ = getCircleIntersectArea(center, radius, opoly)
            areas[x][y] = intersectarea
            areas[x][y] /= getArea(opoly)

            if areas[x][y] > 1:
                areas[x][y] = 1
            if abs(1-areas[x][y]) < threshold:
                areas[x][y] = 1
            elif abs(areas[x][y]) < threshold:
                areas[x][y] = 0
            elif areas[x][y] < 0:
                areas[x][y] = 0

    return areas

def initializePoly(opolys, poly, threshold):
    areas = [[0] * len(opolys[0]) for _ in range(len(opolys))]

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            opoly = opolys[x][y]
            polyintersects = getPolyIntersectArea(poly, opoly)
            for polyintersect in polyintersects:
                areas[x][y] += getArea(polyintersect)
            areas[x][y] /= getArea(opoly)

            if areas[x][y] > 1:
                areas[x][y] = 1
            if abs(1-areas[x][y]) < threshold:
                areas[x][y] = 1
            elif abs(areas[x][y]) < threshold:
                areas[x][y] = 0
            elif areas[x][y] < 0:
                areas[x][y] = 0

    return areas

#theta is angle from positive x-axis to major axis
def initializeEllipse(opolys, major_axis, minor_axis, theta, center, threshold):
    areas = [[0] * len(opolys[0]) for _ in range(len(opolys))]

    circle_to_ellipse = np.array([[major_axis*math.cos(theta)**2 + minor_axis*math.sin(theta)**2, (major_axis-minor_axis)*math.cos(theta)*math.sin(theta)], [(major_axis-minor_axis)*math.cos(theta)*math.sin(theta), major_axis*math.sin(theta)**2 + minor_axis*math.cos(theta)**2]])
    ellipse_to_circle = np.linalg.inv(circle_to_ellipse)

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            opoly = opolys[x][y]
            centered_opoly = list(map(lambda x : [x[0]-center[0], x[1]-center[1]], opoly))
            squished_opoly = list(map(lambda x : [ellipse_to_circle[0][0]*x[0] + ellipse_to_circle[0][1]*x[1], ellipse_to_circle[1][0]*x[0] + ellipse_to_circle[1][1]*x[1]], centered_opoly))
            intersectarea, _ = getCircleIntersectArea([0, 0], 1, squished_opoly)
            areas[x][y] = intersectarea*major_axis*minor_axis
            areas[x][y] /= getArea(opoly)
            #if areas[x][y] != 0:
            #    print(areas[x][y])

            if areas[x][y] > 1:
                areas[x][y] = 1
            if abs(1-areas[x][y]) < threshold:
                areas[x][y] = 1
            elif abs(areas[x][y]) < threshold:
                areas[x][y] = 0
            elif areas[x][y] < 0:
                areas[x][y] = 0

    return areas

#Example from Zalesak: circle with rectangle removed, use with 100x100 grid
def zalesak(opolys, threshold):
    areas = [[0] * len(opolys[0]) for _ in range(len(opolys))]

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            opoly = opolys[x][y]

            radiussmall = 15
            center = [50.05, 75.05]
                
            area, intersect = getCircleIntersectArea(center, radiussmall, opoly)
            areas[x][y] += area
            
            rectangle = [[47.55, 59.5], [52.55, 59.5], [52.55, 85.05], [47.55, 85.05]]
            intersects = getPolyIntersectArea(rectangle, opoly)
            for intersect in intersects:
                areas[x][y] -= getArea(intersect)
            areas[x][y] = max(0, areas[x][y])

            if getDistance(opoly[0], [47.55, 60.25980054160715]) < math.sqrt(2):
                areas[x][y] = getPolyCurvedCornerArea(opoly, [35.05, 75.05], [47.55, 60.25980054160715], [47.55, 85.05], 15, None)
            elif getDistance(opoly[0], [52.55, 60.25980054160715]) < math.sqrt(2):
                areas[x][y] = getPolyCurvedCornerArea(opoly, [52.55, 85.05], [52.55, 60.25980054160715], [65.05, 75.05], None, 15)

            areas[x][y] /= getArea(opoly)

            if areas[x][y] > 1:
                areas[x][y] = 1
            if abs(1-areas[x][y]) < threshold:
                areas[x][y] = 1
            elif abs(areas[x][y]) < threshold:
                areas[x][y] = 0
            elif areas[x][y] < 0:
                areas[x][y] = 0

    return areas

#x+o example: use with 100x100 grid
def xpluso(opolys, threshold):
    areas = [[0] * len(opolys[0]) for _ in range(len(opolys))]

    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            opoly = opolys[x][y]

            #material 1 cross
            xpoints = [[4, 0], [10, 6], [16, 0], [20, 4], [14, 10], [20, 16], [16, 20], [10, 14], [4, 20], [0, 16], [6, 10], [0, 4]]
            xpoints = list(map(lambda x : [x[0]+3.005, x[1]+3.025], xpoints))
            xpolyintersects = getPolyIntersectArea(xpoints, opoly)
            for xpolyintersect in xpolyintersects:
                areas[x][y] += abs(getArea(xpolyintersect))

            #material 2 cross
            xpoints = [[6, 0], [14, 0], [14, 6], [20, 6], [20, 14], [14, 14], [14, 20], [6, 20], [6, 14], [0, 14], [0, 6], [6, 6]]
            xpoints = list(map(lambda x : [x[0]+3.005, x[1]+28.025], xpoints))
            xpolyintersects = getPolyIntersectArea(xpoints, opoly)
            for xpolyintersect in xpolyintersects:
                areas[x][y] += abs(getArea(xpolyintersect))

            #material 1 ring
            radius = 10
            radiussmall = 6
            center = [38.005, 13.005]
            area, intersect = getCircleIntersectArea(center, radius, opoly)
            areas[x][y] += area
            area, intersect = getCircleIntersectArea(center, radiussmall, opoly)
            areas[x][y] -= area

            #material 2 ring
            radius = 10
            radiussmall = 6
            center = [38.005, 38.005]
            area, intersect = getCircleIntersectArea(center, radius, opoly)
            areas[x][y] += area
            area, intersect = getCircleIntersectArea(center, radiussmall, opoly)
            areas[x][y] -= area

            #material 2 dot
            radius = 3
            center = [38.005, 13.005]
            area, intersect = getCircleIntersectArea(center, radius, opoly)
            areas[x][y] += area

            areas[x][y] /= getArea(opoly)

            if areas[x][y] > 1:
                areas[x][y] = 1
            if abs(1-areas[x][y]) < threshold:
                areas[x][y] = 1
            elif abs(areas[x][y]) < threshold:
                areas[x][y] = 0
            elif areas[x][y] < 0:
                areas[x][y] = 0

    return areas

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

def plotAreas(opolys, areas):
    listpatches = []
    listareas = []
    listpartials = []
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            opoly = opolys[x][y]
            opolyarea = getArea(opoly)
            #area_fraction = areas[x][y]/opolyarea
            area_fraction = areas[x][y]/opolyarea
            patch = Polygon(np.array(opoly), True)
            listpatches.append(patch)
            listareas.append(area_fraction)
            listpartials.append(math.ceil(area_fraction) - math.floor(area_fraction))

    #Max x and y of opolys grid
    maxX = max(list(map(lambda x : x[0], opolys[len(opolys)-1][len(opolys[0])-1])))
    maxY = max(list(map(lambda x : x[1], opolys[len(opolys)-1][len(opolys[0])-1])))

    p = PatchCollection(listpatches, cmap='jet')
    patchareas = np.array(listareas)
    p.set_array(patchareas)
    fig, ax = plt.subplots()
    ax.set_xlim(0, maxX)
    ax.set_ylim(0, maxY)
    ax.add_collection(p)
    plt.savefig("advection_plt/original_areas.png", dpi=199)
    plt.clf()

    p = PatchCollection(listpatches, cmap='jet')
    patchareas = np.array(listpartials)
    p.set_array(patchareas)
    fig, ax = plt.subplots()
    ax.set_xlim(0, maxX)
    ax.set_ylim(0, maxY)
    ax.add_collection(p)
    plt.savefig("advection_plt/original_partials.png", dpi=199)
    plt.clf()