import math
import numpy as np

from mesh import QuadMesh, makeFineCartesianGrid, makeQuadGrid
from initialize_areas import initializeCircle, initializePoly, initializeEllipse, zalesak, xpluso
from interface_reconstruction import merge, makeFacets

from facet import getNormal, LinearFacet, ArcFacet, advectPoint
from strand import StrandPoly, Strand
from write_facets import writeFacets

import matplotlib.pyplot as plt

from geoms import getDistance, getArea, getPolyIntersectArea, getPolyLineArea
from circular_facet import getArcFacet, getCircleIntersectArea, getArcFacetNewton2, getArcFacetNewtonSigmoid, getArcFacetSigmoidRoot
from linear_facet import getLinearFacet, getLinearFacetFromNormal

from local_reconstruction import local_reconstruction

#Hyperparameters
totalt = 2
dt = 0.01

checkSize = 2

makeGapless = False

#Test settings
threshold = 1e-6

meshSize = 100
resolution = 50/100 #128/150
meshType = 'cartesian' #quads #cartesian

#testType = 'vortex' #zalesak, xpluso, deformation, triangle, rectangle
testType = 'xpluso'

#interface reconstruction = 'Youngs', 'Youngs+ours', 'ELVIRA', 'ours'
interfaceReconstruction = 'ours'
config = 'linear' #'linear', 'circular' #for Youngs+ours

save_name = 'zalesak_example_ours_64'

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

areas = zalesak(opolys, threshold)

#Plot mesh
if meshType == 'quads':
    mesh.plotMesh('advection_vtk/perturbed_quads_{}x{}.vtk'.format(int(meshSize*resolution), int(meshSize*resolution)))
elif meshType == 'cartesian':
    mesh.plotMesh('advection_vtk/quads_{}x{}.vtk'.format(int(meshSize*resolution), int(meshSize*resolution)))
mesh.initializeAreas(areas)
mesh.plotAreas('advection_plt/original_areas.png')
mesh.plotPartialAreas('advection_plt/original_partials.png')

if interfaceReconstruction == 'ours':
    #Merge
    mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
    #Interface reconstruction
    predfacets, _ = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless, useCorners=True, useCircles=True)
else:
    predfacets = local_reconstruction(opolys, areas, threshold, interfaceReconstruction=interfaceReconstruction, makeGapless=makeGapless, config=config)

plot_predfacets = []

if interfaceReconstruction == 'ours':
    for x in range(len(predfacets)):
        for y in range(len(predfacets[0])):
            if predfacets[x][y] is not None:
                if predfacets[x][y][0] == 'linear':
                    facet = LinearFacet(predfacets[x][y][1][0], predfacets[x][y][1][1])
                    plot_predfacets.append(facet)
                elif predfacets[x][y][0] == 'corner':
                    facet1 = LinearFacet(predfacets[x][y][1][0], predfacets[x][y][1][1])
                    facet2 = LinearFacet(predfacets[x][y][1][1], predfacets[x][y][1][2])
                    plot_predfacets.append(facet1)
                    plot_predfacets.append(facet2)
                elif predfacets[x][y][0] == 'arc':
                    facet = ArcFacet(predfacets[x][y][1], predfacets[x][y][2], predfacets[x][y][3][0], predfacets[x][y][3][1])
                    plot_predfacets.append(facet)
                elif predfacets[x][y][0] == 'curvedcorner':
                    if predfacets[x][y][3] is None:
                        facet1 = LinearFacet(predfacets[x][y][5][0], predfacets[x][y][5][1])
                    else:
                        facet1 = ArcFacet(predfacets[x][y][1], predfacets[x][y][3], predfacets[x][y][5][0], predfacets[x][y][5][1])
                    if predfacets[x][y][4] is None:
                        facet2 = LinearFacet(predfacets[x][y][5][1], predfacets[x][y][5][2])
                    else:
                        facet2 = ArcFacet(predfacets[x][y][2], predfacets[x][y][4], predfacets[x][y][5][1], predfacets[x][y][5][2])
                    plot_predfacets.append(facet1)
                    plot_predfacets.append(facet2)

else:
    for x in range(len(predfacets)):
        for y in range(len(predfacets[0])):
            for facet in predfacets[x][y]:
                plot_predfacets.append(facet)

writeFacets(plot_predfacets, '{}'.format(save_name))
print(len(plot_predfacets))