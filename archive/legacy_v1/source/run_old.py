import math
import numpy as np

from mesh import QuadMesh, makeFineCartesianGrid, makeQuadGrid
from initialize_areas import initializeCircle, initializePoly, zalesak, xpluso
from interface_reconstruction import merge, makeFacets

from facet import getNormal, LinearFacet, ArcFacet, advectPoint
from strand import StrandPoly, Strand
from write_facets import writeFacets

import matplotlib.pyplot as plt

from geoms import getDistance, getArea, getPolyIntersectArea, getPolyLineArea
from circular_facet import getArcFacet, getCircleIntersectArea, getArcFacetNewton2, getArcFacetNewtonSigmoid, getArcFacetSigmoidRoot
from linear_facet import getLinearFacet, getLinearFacetFromNormal
from corner_facet import getPolyCurvedCornerArea

from local_reconstruction import local_reconstruction

#Hyperparameters
totalt = 2
dt = 0.01

checkSize = 2

makeGapless = False

#Test settings
threshold = 1e-6

meshSize = 100
resolution = 32/100 #128/150
meshType = 'cartesian' #quads #cartesian

#testType = 'vortex' #zalesak, xpluso, deformation, triangle, rectangle
testType = 'vortex'

#interface reconstruction = 'Youngs', 'Youngs+ours', 'ELVIRA', 'ours'
interfaceReconstruction = 'ours'

save_name = 'vortex_32x32_ours'

#----------------
#Code

#Initialize test settings
timesteps = int(totalt/dt)

if meshType == 'quads':
    opoints = makeQuadGrid(meshSize, resolution)
elif meshType == 'cartesian':
    opoints = makeFineCartesianGrid(meshSize, resolution)
mesh = QuadMesh(opoints)
opolys = mesh.polys

if testType == 'vortex':
    areas = initializeCircle(opolys, [50.01, 75.01], 15, threshold)
    velocity = lambda t, p : [-200*math.cos(math.pi*t/totalt)*(math.sin(math.pi*p[0]/100))**2 * math.sin(math.pi*p[1]/100) * math.cos(math.pi*p[1]/100),
                              200*math.cos(math.pi*t/totalt)*math.sin(math.pi*p[0]/100)*math.cos(math.pi*p[0]/100) * (math.sin(math.pi*p[1]/100))**2]
if testType == 'deformation':
    areas = initializeCircle(opolys, [50.01, 75.01], 15, threshold)
    velocity = lambda t, p: [100*math.cos(math.pi*t/totalt)*math.sin(4*math.pi*(p[0]+50)/100)*math.sin(4*math.pi*(p[1]+50)/100),
                             100*math.cos(math.pi*t/totalt)*math.cos(4*math.pi*(p[0]+50)/100)*math.cos(4*math.pi*(p[1]+50)/100)]
elif testType == 'zalesak':
    areas = zalesak(opolys, threshold)
    velocity = lambda t, p: [(2*math.pi)/totalt*(50-p[1]), (2*math.pi)/totalt*(p[0]-50)]
elif testType == 'xpluso':
    areas = xpluso(opolys, threshold)
    velocity = lambda t, p: [6, 6]
elif testType == 'triangle':
    areas = initializePoly(opolys, [[1.5, 1.5], [9.5, 11.5], [5.5, 15.2]], threshold)
    velocity = lambda t, p: [6, 6]
elif testType == 'rectangle':
    areas = initializePoly(opolys, [[8.2, 8.2], [12.2, 5.2], [15.2, 9.2], [11.2, 12.2]], threshold)
    velocity = lambda t, p: [6, 6]

#Plot mesh
mesh.plotMesh('advection_vtk/quads_{}x{}.vtk'.format(int(meshSize*resolution), int(meshSize*resolution)))
mesh.initializeAreas(areas)
mesh.plotAreas('advection_plt/original_areas.png')
mesh.plotPartialAreas('advection_plt/original_partials.png')

#------------------
#Interface reconstruction: advection


print("Advection")
t = 0
for timestep in range(timesteps+5):
    print("Advection: timestep {}".format(timestep))

    if interfaceReconstruction == 'ours':
        #Merge
        mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
        #Interface reconstruction
        predfacets, _ = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless)
    else:
        predfacets = local_reconstruction(opolys, areas, threshold, interfaceReconstruction=interfaceReconstruction, makeGapless=makeGapless)

    plot_predfacets = []
    
    if interfaceReconstruction == 'ours':
        for x in range(len(predfacets)):
            for y in range(len(predfacets[0])):
                if predfacets[x][y] is not None:
                    if predfacets[x][y][0] == 'linear':
                        facet = LinearFacet(predfacets[x][y][1][0], predfacets[x][y][1][1])
                        plot_predfacets.append(facet)
                        predfacets[x][y] = [facet]
                    elif predfacets[x][y][0] == 'corner':
                        facet1 = LinearFacet(predfacets[x][y][1][0], predfacets[x][y][1][1])
                        facet2 = LinearFacet(predfacets[x][y][1][1], predfacets[x][y][1][2])
                        plot_predfacets.append(facet1)
                        plot_predfacets.append(facet2)
                        predfacets[x][y] = [facet1, facet2]
                    elif predfacets[x][y][0] == 'arc':
                        facet = ArcFacet(predfacets[x][y][1], predfacets[x][y][2], predfacets[x][y][3][0], predfacets[x][y][3][1])
                        plot_predfacets.append(facet)
                        predfacets[x][y] = [facet]
                    elif predfacets[x][y][0] == 'curvedcorner':
                        if predfacets[x][y][1] is None: #predfacets[x][y][3] should also be None
                            #First facet is linear
                            facet1 = LinearFacet(predfacets[x][y][5][0], predfacets[x][y][5][1])
                        else:
                            facet1 = ArcFacet(predfacets[x][y][1], predfacets[x][y][3], predfacets[x][y][5][0], predfacets[x][y][5][1])
                        plot_predfacets.append(facet1)
                        if predfacets[x][y][2] is None: #predfacets[x][y][4] should also be None
                            #First facet is linear
                            facet2 = LinearFacet(predfacets[x][y][5][1], predfacets[x][y][5][2])
                        else:
                            facet2 = ArcFacet(predfacets[x][y][2], predfacets[x][y][4], predfacets[x][y][5][1], predfacets[x][y][5][2])
                        plot_predfacets.append(facet2)
                        predfacets[x][y] = [facet1, facet2]
                else:
                    predfacets[x][y] = []
    
    else:
        for x in range(len(predfacets)):
            for y in range(len(predfacets[0])):
                for facet in predfacets[x][y]:
                    plot_predfacets.append(facet)

    #Plot reconstructed interface
    writeFacets(plot_predfacets, '{}_{}'.format(save_name, timestep))
    print(len(plot_predfacets))

    if timestep == 200:
        dists = []
        for facet in plot_predfacets:
            dist = max(list(map(lambda x : abs(getDistance([50.01, 75.01], x)-15), [facet.pLeft, facet.midpoint, facet.pRight])))
            dists.append(dist)
        print(dists)
        print(sum(dists)/len(dists))

    #Updated areas
    nareas = [[0 for _ in range(len(opolys[0]))] for _ in range(len(opolys))]

    plot_advectedfacets = []

    #Advect interface and intersect with neighbors
    for advectx in range(len(opolys)):
        for advecty in range(len(opolys[0])):
            if areas[advectx][advecty] > threshold:
                #shiftpoly
                shiftpoly = list(map(lambda x : advectPoint(x, velocity, t, dt), opolys[advectx][advecty]))
                shiftbounds = [min(list(map(lambda x : x[0], shiftpoly))), min(list(map(lambda x : x[1], shiftpoly))), max(list(map(lambda x : x[0], shiftpoly))), max(list(map(lambda x : x[1], shiftpoly)))]

                for testx in range(-checkSize, checkSize+1):
                    for testy in range(-checkSize, checkSize+1):
                        checkx = advectx-testx
                        checky = advecty-testy
                        if checkx >= 0 and checkx < len(opolys) and checky >= 0 and checky < len(opolys[0]):
                            testpoly = opolys[checkx][checky]
                            testbounds = [min(list(map(lambda x : x[0], testpoly))), min(list(map(lambda x : x[1], testpoly))), max(list(map(lambda x : x[0], testpoly))), max(list(map(lambda x : x[1], testpoly)))]
                            if not(testbounds[2] <= shiftbounds[0] or shiftbounds[2] <= testbounds[0] or testbounds[3] <= shiftbounds[1] or shiftbounds[3] <= testbounds[1]):
                                #bounding boxes intersect, could be nonzero intersection
                                try:
                                    polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                except:
                                    print("Failed polyintersect: getPolyIntersectArea({}, {})".format(shiftpoly, testpoly))
                                    testpoly = list(map(lambda x : [x[0]+1e-13, x[1]+1e-13], testpoly))
                                    polyintersections = getPolyIntersectArea(shiftpoly, testpoly)
                                if len(polyintersections) == 0:
                                    #No intersection
                                    continue
                                #For each overlap region
                                for polyintersection in polyintersections:
                                    if predfacets[advectx][advecty] is not None: #added 1/20/22 to fix compatibility with merging
                                        if len(predfacets[advectx][advecty]) == 0:
                                            #Full cell
                                            assert areas[advectx][advecty] > 1-threshold
                                            nareas[checkx][checky] += abs(getArea(polyintersection))
                                        else:
                                            #Mixed cell with facet

                                            #PLIC
                                            if interfaceReconstruction == 'Youngs' or interfaceReconstruction == 'Youngs+ours' or interfaceReconstruction == 'ELVIRA':
                                                for num_advectedpredfacets in range(len(predfacets[advectx][advecty])):
                                                    advectedfacet = predfacets[advectx][advecty][num_advectedpredfacets].advected(velocity, t, dt)
                                                    plot_advectedfacets.append(advectedfacet)
                                                    if advectedfacet.name == 'linear':
                                                        polyintersectionarea = getPolyLineArea(polyintersection, advectedfacet.pLeft, advectedfacet.pRight)
                                                    elif advectedfacet.name == 'arc':
                                                        polyintersectionarea, _ = getCircleIntersectArea(advectedfacet.center, advectedfacet.radius, polyintersection)
                                                    nareas[checkx][checky] += abs(polyintersectionarea)

                                            #Ours
                                            elif interfaceReconstruction == 'ours':
                                                if len(predfacets[advectx][advecty]) == 1:
                                                    advectedfacet = predfacets[advectx][advecty][0].advected(velocity, t, dt)
                                                    plot_advectedfacets.append(advectedfacet)
                                                    if advectedfacet.name == 'linear':
                                                        polyintersectionarea = getPolyLineArea(polyintersection, advectedfacet.pLeft, advectedfacet.pRight)
                                                    elif advectedfacet.name == 'arc':
                                                        polyintersectionarea, _ = getCircleIntersectArea(advectedfacet.center, advectedfacet.radius, polyintersection)
                                                    nareas[checkx][checky] += polyintersectionarea #TODO: abs here?
                                                    if polyintersectionarea < 0:
                                                        print("Negative polyintersectionarea")
                                                elif len(predfacets[advectx][advecty]) == 2:
                                                    advectedfacet1 = predfacets[advectx][advecty][0].advected(velocity, t, dt)
                                                    advectedfacet1r = advectedfacet1.radius if advectedfacet1.name == 'arc' else None
                                                    plot_advectedfacets.append(advectedfacet1)
                                                    advectedfacet2 = predfacets[advectx][advecty][1].advected(velocity, t, dt)
                                                    advectedfacet2r = advectedfacet2.radius if advectedfacet2.name == 'arc' else None
                                                    plot_advectedfacets.append(advectedfacet2)
                                                    nareas[checkx][checky] += getPolyCurvedCornerArea(polyintersection, advectedfacet1.pLeft, advectedfacet1.pRight, advectedfacet2.pRight, advectedfacet1r, advectedfacet2r)
                                                else:
                                                    print("More than two facets in this cell?")

    #Update areas
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            areas[x][y] = min(max(nareas[x][y]/getArea(opolys[x][y]), 0), 1)
            if areas[x][y] < threshold:
                areas[x][y] = 0
            elif areas[x][y] > 1-threshold:
                areas[x][y] = 1

    #Plot areas
    mesh.initializeAreas(areas)
    mesh.plotAreas('advection_plt/areas_{}.png'.format(timestep))
    mesh.plotPartialAreas('advection_plt/partials_{}.png'.format(timestep))

    t += dt

