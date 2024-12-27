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

"""
#------------------
#Our interface reconstruction: baseline
mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless)

newfacetsunique = []
for facet in facetsunique:
    newfacet = ArcFacet(facet[1], facet[2], facet[3][0], facet[3][1])
    newfacetsunique.append(newfacet)
facetsunique = newfacetsunique

#Plot initial setup
writeFacets(facetsunique, 'facets_initial')
"""

#------------------
#Interface reconstruction: advection

"""
areas = np.load('areas.npy', allow_pickle=True)
areas[51][51] = 1
areas[52][51] += 0.1
areas[35][22] -= 0.1
mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, 1e-4, makeGapless=True)
newfacetsunique = []
for facet in facetsunique:
    if facet[0] == 'arc':
        newfacet = ArcFacet(facet[1], facet[2], facet[3][0], facet[3][1])
    elif facet[0] == 'linear':
        newfacet = LinearFacet(facet[1], facet[2])
    newfacetsunique.append(newfacet)
writeFacets(newfacetsunique, 'vortex_100')
print(1/0)
"""

print("Advection")
t = 0
for timestep in range(timesteps+5):
    print("Advection: timestep {}".format(timestep))

    """
    if timestep == 100:
        np.save('areas.npy', areas)
        #Merge
        mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
        #Interface reconstruction
        predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, 1e-5, makeGapless=True)

        newfacetsunique = []
        for facet in facetsunique:
            if facet[0] == 'arc':
                newfacet = ArcFacet(facet[1], facet[2], facet[3][0], facet[3][1])
            elif facet[0] == 'linear':
                newfacet = LinearFacet(facet[1], facet[2])
            newfacetsunique.append(newfacet)
        writeFacets(newfacetsunique, 'vortex_100')
        print(1/0)
    """

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
    

    """
    predfacets = [[[] for _ in range(len(opolys[0]))] for _ in range(len(opolys))]
    plot_predfacets = []
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            #Interface reconstruction step

            if areas[x][y] < threshold or areas[x][y] > 1-threshold:
                #Full cell, no facet needed
                predfacets[x][y] = []
            else:
                #Mixed cell
                def helperArea(x, y):
                    if x < 0 or y < 0 or x >= len(opolys) or y >= len(opolys[0]):
                        return 0
                    return areas[x][y]

                #Normal calculation
                if interfaceReconstruction == 'Youngs':
                    def getYoungsNormal(x, y):
                        #Youngs normal
                        youngs_alpha = 2
                        f_e = 1/(2+youngs_alpha)*(helperArea(x+1, y-1) + youngs_alpha*helperArea(x+1, y) + helperArea(x+1, y+1))
                        f_w = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x-1, y) + helperArea(x-1, y+1))
                        f_n = 1/(2+youngs_alpha)*(helperArea(x-1, y+1) + youngs_alpha*helperArea(x, y+1) + helperArea(x+1, y+1))
                        f_s = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x, y-1) + helperArea(x+1, y-1))
                        normal = [(f_e - f_w)/2, (f_n - f_s)/2]
                        normal_magnitude = getDistance([0,0], normal)
                        normal = [normal[0]/normal_magnitude, normal[1]/normal_magnitude]

                        return normal

                    l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], getYoungsNormal(x, y), threshold)
                    linearfacet = LinearFacet(l1, l2)
                    predfacets[x][y].append(linearfacet)
                    plot_predfacets.append(linearfacet)

                elif interfaceReconstruction == 'Youngs+ours':
                    def getYoungsNormal(x, y):
                        #Youngs normal
                        youngs_alpha = 2
                        f_e = 1/(2+youngs_alpha)*(helperArea(x+1, y-1) + youngs_alpha*helperArea(x+1, y) + helperArea(x+1, y+1))
                        f_w = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x-1, y) + helperArea(x-1, y+1))
                        f_n = 1/(2+youngs_alpha)*(helperArea(x-1, y+1) + youngs_alpha*helperArea(x, y+1) + helperArea(x+1, y+1))
                        f_s = 1/(2+youngs_alpha)*(helperArea(x-1, y-1) + youngs_alpha*helperArea(x, y-1) + helperArea(x+1, y-1))
                        normal = [(f_e - f_w)/2, (f_n - f_s)/2]
                        normal_magnitude = getDistance([0,0], normal)
                        normal = [normal[0]/normal_magnitude, normal[1]/normal_magnitude]

                        return normal

                    #condition on number of mixed neighbors
                    def getMixedNeighbors(x, y):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        mixed_neighbors = []
                        for i_check_neighbor, check_neighbor in enumerate(check_neighbors):
                            if helperArea(x+check_neighbor[0], y+check_neighbor[1]) > threshold and helperArea(x+check_neighbor[0], y+check_neighbor[1]) < 1-threshold:
                                mixed_neighbors.append(i_check_neighbor)
                        return mixed_neighbors

                    #condition on number of mixed neighbors with exactly 2 mixed neighbors
                    def getMixedCertainNeighbors(x, y):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        mixed_neighbors = []
                        for i_check_neighbor, check_neighbor in enumerate(check_neighbors):
                            if helperArea(x+check_neighbor[0], y+check_neighbor[1]) > threshold and helperArea(x+check_neighbor[0], y+check_neighbor[1]) < 1-threshold and len(getMixedNeighbors(x+check_neighbor[0], y+check_neighbor[1])) == 2:
                                mixed_neighbors.append(i_check_neighbor)
                        return mixed_neighbors

                    #Get orientation from mixed neighbors: can use mixed neighbors or mixed certain neighbors
                    def getOrientation(x, y, mixed_neighbors):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        ambiguous_neighbors = True
                        if len(mixed_neighbors) == 2:
                            #2 neighbors, determine orientation based on full neighbors
                            if abs(mixed_neighbors[0] - mixed_neighbors[1]) == 2:
                                #Opposite mixed cells
                                left_test_neighbor = (mixed_neighbors[0]+1) % 4
                                right_test_neighbor = (left_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) > 1-threshold and helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold:
                                    #Left is full
                                    prev_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) < threshold and helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold:
                                    #Right is full
                                    prev_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    ambiguous_neighbors = False
                            else:
                                #Adjacent mixed cells
                                if (mixed_neighbors[0]+1) % 4 == mixed_neighbors[1]:
                                    mid_test_neighbor = mixed_neighbors[0] #mid_test_neighbor and neighbor clockwise from it are mixed
                                else:
                                    mid_test_neighbor = mixed_neighbors[1]
                                right_test_neighbor = (mid_test_neighbor-1) % 4
                                opp_test_neighbor = (mid_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold and helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) > 1-threshold:
                                    #Both full
                                    prev_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    prev_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    next_x = x+check_neighbors[mid_test_neighbor][0]
                                    next_y = y+check_neighbors[mid_test_neighbor][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold and helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) < threshold:
                                    #Both empty
                                    prev_x = x+check_neighbors[mid_test_neighbor][0]
                                    prev_y = y+check_neighbors[mid_test_neighbor][1]
                                    next_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    next_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    ambiguous_neighbors = False

                        if not(ambiguous_neighbors):
                            return prev_x, prev_y, next_x, next_y
                        else:
                            return None, None, None, None

                    #Get orientation from mixed neighbors: can use mixed neighbors or mixed certain neighbors
                    def getOrientationWeak(x, y, mixed_neighbors):
                        check_neighbors = [[1, 0], [0, -1], [-1, 0], [0, 1]] #clockwise
                        ambiguous_neighbors = True
                        if len(mixed_neighbors) == 2:
                            #2 neighbors, determine orientation based on full neighbors
                            if abs(mixed_neighbors[0] - mixed_neighbors[1]) == 2:
                                #Opposite mixed cells
                                left_test_neighbor = (mixed_neighbors[0]+1) % 4
                                right_test_neighbor = (left_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) > 1-threshold or helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold:
                                    #Left is full
                                    prev_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[left_test_neighbor][0], y+check_neighbors[left_test_neighbor][1]) < threshold or helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold:
                                    #Right is full
                                    prev_x = x+check_neighbors[mixed_neighbors[1]][0]
                                    prev_y = y+check_neighbors[mixed_neighbors[1]][1]
                                    next_x = x+check_neighbors[mixed_neighbors[0]][0]
                                    next_y = y+check_neighbors[mixed_neighbors[0]][1]
                                    ambiguous_neighbors = False
                            else:
                                #Adjacent mixed cells
                                if (mixed_neighbors[0]+1) % 4 == mixed_neighbors[1]:
                                    mid_test_neighbor = mixed_neighbors[0] #mid_test_neighbor and neighbor clockwise from it are mixed
                                else:
                                    mid_test_neighbor = mixed_neighbors[1]
                                right_test_neighbor = (mid_test_neighbor-1) % 4
                                opp_test_neighbor = (mid_test_neighbor+2) % 4
                                if helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) > 1-threshold or helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) > 1-threshold:
                                    #Both full
                                    prev_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    prev_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    next_x = x+check_neighbors[mid_test_neighbor][0]
                                    next_y = y+check_neighbors[mid_test_neighbor][1]
                                    ambiguous_neighbors = False
                                elif helperArea(x+check_neighbors[right_test_neighbor][0], y+check_neighbors[right_test_neighbor][1]) < threshold or helperArea(x+check_neighbors[opp_test_neighbor][0], y+check_neighbors[opp_test_neighbor][1]) < threshold:
                                    #Both empty
                                    prev_x = x+check_neighbors[mid_test_neighbor][0]
                                    prev_y = y+check_neighbors[mid_test_neighbor][1]
                                    next_x = x+check_neighbors[(mid_test_neighbor+1) % 4][0]
                                    next_y = y+check_neighbors[(mid_test_neighbor+1) % 4][1]
                                    ambiguous_neighbors = False

                        if not(ambiguous_neighbors):
                            return prev_x, prev_y, next_x, next_y
                        else:
                            return None, None, None, None

                    ambiguous_neighbors = True
                    mixed_neighbors = getMixedNeighbors(x, y)
                    prev_x, prev_y, next_x, next_y = getOrientationWeak(x, y, mixed_neighbors)
                    if prev_x is not None:
                        ambiguous_neighbors = False
                    else:
                        #attempt with only unambiguous neighbors
                        mixed_neighbors = getMixedCertainNeighbors(x, y)
                        prev_x, prev_y, next_x, next_y = getOrientationWeak(x, y, mixed_neighbors)
                        if prev_x is not None:
                            ambiguous_neighbors = False
                        else:
                            print(opolys[x][y])

                    if not(ambiguous_neighbors):
                        #Try to fit a linear facet
                        l1, l2 = getLinearFacet(opolys[prev_x][prev_y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[next_x][next_y], threshold)
                        if abs(areas[x][y] - getPolyLineArea(opolys[x][y], l1, l2)/getArea(opolys[x][y])) < threshold:
                            #Area in central poly matches the line, form linear facet
                            #print("Linear facet")
                            linearfacet = LinearFacet(l1, l2)
                            predfacets[x][y].append(linearfacet)
                            plot_predfacets.append(linearfacet)
                        else:
                            #Try to fit a circular facet
                            try:
                                #arccenter, arcradius, arcintersects = getArcFacet(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                arccenter, arcradius, arcintersects = getArcFacetSigmoidRoot(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                #arccenter, arcradius, arcintersects = getArcFacetNewton2(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                
                                if arccenter is not None and arcradius is not None and arcintersects is not None:
                                    #Successful circular facet
                                    #print("Circular facet")
                                    arcfacet = ArcFacet(arccenter, arcradius, arcintersects[0], arcintersects[-1])
                                    predfacets[x][y].append(arcfacet)
                                    plot_predfacets.append(arcfacet)
                                else:
                                    #Failed circular facet, default to Youngs normal
                                    ambiguous_neighbors = True
                            except:
                                #arccenter, arcradius, arcintersects = getArcFacetNewton2(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold)
                                
                                print("getArcFacetNewton2({}, {}, {}, {}, {}, {}, {})".format(opolys[prev_x][prev_y], opolys[x][y], opolys[next_x][next_y], areas[prev_x][prev_y], areas[x][y], areas[next_x][next_y], threshold))
                                ambiguous_neighbors = True

                    #Note that failed circular facet runs go here
                    if ambiguous_neighbors:
                        #Ambiguous neighbors, default to Youngs normal
                        #print("Ambiguity: {}, {}".format(x, y))
                        try:
                            l1, l2 = getLinearFacetFromNormal(opolys[x][y], areas[x][y], getYoungsNormal(x, y), threshold)
                        except:
                            print("getLinearFacetFromNormal({}, {}, {}, {})".format(opolys[x][y], areas[x][y], getYoungsNormal(x, y), threshold))
                        linearfacet = LinearFacet(l1, l2)
                        predfacets[x][y].append(linearfacet)
                        plot_predfacets.append(linearfacet)
    """                    

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
                                    #if checkx == 12 and checky == 25:
                                    #    print(polyintersection)
                                    if predfacets[advectx][advecty] is not None: #added 1/20/22 to fix compatibility with merging
                                        if len(predfacets[advectx][advecty]) == 0:
                                            #Full cell
                                            assert areas[advectx][advecty] > 1-threshold
                                            nareas[checkx][checky] += abs(getArea(polyintersection))
                                            #if checkx == 12 and checky == 25:
                                            #    print("Whole: {}".format(getArea(polyintersection)))
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
                                            
                                            #Ours
                                            # elif interfaceReconstruction == 'ours':
                                            #     if len(predfacets[advectx][advecty]) == 0:
                                            #         continue
                                            #     nareas[checkx][checky] += getPolyLineArea(polyintersection, predfacets[advectx][advecty][0].pLeft, predfacets[advectx][advecty][-1].pRight)
                                            #     strandPoly = [predfacets[advectx][advecty].pLeft]
                                            #     for num_advectedpredfacets in range(len(predfacets[advectx][advecty])):
                                            #         advectedfacet = predfacets[advectx][advecty][num_advectedpredfacets]
                                            #         strandPoly.append(advectedfacet.pRight)
                                            #         if advectedfacet.name == 'arc':
                                            #             #TODO: figure out a way to better calculate the areas of sectors intersected with arbitrary polygons

                                                    


    #Plot advected facets
    #writeFacets(plot_advectedfacets, 'advectedfacets_youngs_{}'.format(timestep))

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

