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

makeGapless = True

#Test settings
threshold = 1e-6

meshSize = 100
resolution = 100/100 #128/150
meshType = 'cartesian' #quads #cartesian

#testType = 'vortex' #zalesak, xpluso, deformation, triangle, rectangle
testType = 'xpluso'

#interface reconstruction = 'Youngs', 'Youngs+ours', 'ELVIRA', 'ours'
interfaceReconstruction = 'Youngs'
config = 'linear' #'linear', 'circular' #for Youngs+ours

save_name = 'ellipse_example_youngs_c0'

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

for major_axis_range in range(290, 311):
    print("Major axis: {}".format(major_axis_range/100))
    areas = initializeEllipse(opolys, major_axis_range/100, 1.7, math.pi/2, [50.5, 50], threshold)

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
        predfacets, _ = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless)
    else:
        predfacets = local_reconstruction(opolys, areas, threshold, interfaceReconstruction=interfaceReconstruction, makeGapless=makeGapless, config=configs)

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
    
    else:
        for x in range(len(predfacets)):
            for y in range(len(predfacets[0])):
                for facet in predfacets[x][y]:
                    plot_predfacets.append(facet)

    writeFacets(plot_predfacets, '{}_{}'.format(save_name, major_axis_range))
    print(len(plot_predfacets))