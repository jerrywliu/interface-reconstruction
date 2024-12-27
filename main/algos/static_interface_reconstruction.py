import math
import random
import numpy as np
import scipy.stats as st

from mesh import QuadMesh, makeFineCartesianGrid, makeQuadGrid
from initialize_areas import initializeCircle, initializePoly, initializeEllipse, zalesak, xpluso
from interface_reconstruction import merge, makeFacets

from facet import getNormal, LinearFacet, ArcFacet
from write_facets import writeFacets

from geoms import *

from local_reconstruction import local_reconstruction

#Test settings
makeGapless = False

threshold = 1e-6

meshSize = 100
resolution = 10/100 #128/150
meshType = 'cartesian' #quads #cartesian

testType = 'ellipse'

#interface reconstruction = 'Youngs', 'Youngs+ours', 'ELVIRA', 'ours'
interfaceReconstruction = 'ours'

save_name = 'rotated_ellipse_example_ours'

#----------------------------------------------------------------------

for resolution in [10/100, 20/100, 40/100, 80/100, 160/100]:

    print("Resolution: {}".format(resolution))

    #Initialize test settings
    if meshType == 'quads':
        opoints = makeQuadGrid(meshSize, resolution)
    elif meshType == 'cartesian':
        opoints = makeFineCartesianGrid(meshSize, resolution)
    mesh = QuadMesh(opoints)
    opolys = mesh.polys

    #Plot mesh
    if meshType == 'quads':
        mesh.plotMesh('advection_vtk/perturbed_quads_{}x{}.vtk'.format(int(meshSize*resolution), int(meshSize*resolution)))
    elif meshType == 'cartesian':
        mesh.plotMesh('advection_vtk/quads_{}x{}.vtk'.format(int(meshSize*resolution), int(meshSize*resolution)))

    if testType == 'ellipse':
        #Static ellipse test

        num_trials = 100

        facet_gaps = []
        curvature_errors = []

        for trial in range(num_trials):
            theta = random.random()*math.pi/2
            center = [50+random.random(), 50+random.random()]
            major_axis = 10*math.sqrt(12)
            minor_axis = 10*math.sqrt(2)
            areas = initializeEllipse(opolys, major_axis, minor_axis, theta, center, threshold)

            """
            mesh.setAreas(areas)
            mesh.plotAreas('advection_plt/areas_initial.png')
            mesh.plotPartialAreas('advection_plt/partial_areas_initial.png')
            """

            if interfaceReconstruction == 'ours':
                #Merge
                mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
                #Interface reconstruction
                predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless=False)

                newfacetsunique = []
                for facet in facetsunique:
                    if facet[0] == 'arc':
                        newfacet = ArcFacet(facet[1], facet[2], facet[3][0], facet[3][1])
                    elif facet[0] == 'linear':
                        newfacet = LinearFacet(facet[1], facet[2])
                    newfacetsunique.append(newfacet)
                facetsunique = newfacetsunique

                #Compute gaps
                for i in range(len(facetsunique)):
                    prevfacet = facetsunique[i]
                    nextfacet = facetsunique[(i+1) % len(facetsunique)]
                    p1 = prevfacet.pRight
                    p2 = nextfacet.pLeft
                    facet_gaps.append(getDistance(p1, p2))  
                
            else:
                predfacets = local_reconstruction(opolys, areas, threshold, interfaceReconstruction=interfaceReconstruction, makeGapless=False)
                predfacets_flatten = [a for c in predfacets for b in c for a in b]
                newfacetsunique = []
                #Compute gaps
                for predfacet1 in predfacets_flatten:
                    min_facet = None
                    min_distance = float('inf')
                    for predfacet2 in predfacets_flatten:
                        d = getDistance(predfacet1.pRight, predfacet2.pLeft)
                        if d < min_distance:
                            min_facet = predfacet2
                            min_distance = d
                    newfacetsunique.append(predfacet1)
                    facet_gaps.append(min_distance)
                facetsunique = newfacetsunique

            writeFacets(facetsunique, '{}_{}_{}'.format(save_name, resolution*100, trial))

            if interfaceReconstruction == 'ours':

                #Gapless
                predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless=True)

                newfacetsunique = []
                for facet in facetsunique:
                    if facet[0] == 'arc':
                        newfacet = ArcFacet(facet[1], facet[2], facet[3][0], facet[3][1])
                    elif facet[0] == 'linear':
                        newfacet = LinearFacet(facet[1], facet[2])
                    newfacetsunique.append(newfacet)
                facetsunique = newfacetsunique

                for i in range(len(facetsunique)):
                    circle_to_ellipse = np.array([[major_axis*math.cos(theta)**2 + minor_axis*math.sin(theta)**2, (major_axis-minor_axis)*math.cos(theta)*math.sin(theta)], [(major_axis-minor_axis)*math.cos(theta)*math.sin(theta), major_axis*math.sin(theta)**2 + minor_axis*math.cos(theta)**2]])
                    ellipse_to_circle = np.linalg.inv(circle_to_ellipse)
                    ellipse_to_circle_proj = lambda x : [ellipse_to_circle[0][0]*x[0] + ellipse_to_circle[0][1]*x[1], ellipse_to_circle[1][0]*x[0] + ellipse_to_circle[1][1]*x[1]]

                    prevfacet = facetsunique[i]
                    curvature = lambda x : major_axis*minor_axis/(math.sqrt(major_axis**2 * (math.sin(x - theta))**2 + minor_axis**2 * (math.cos(x - theta))**2))**3
                    p1 = ellipse_to_circle_proj([prevfacet.pLeft[0]-center[0], prevfacet.pLeft[1]-center[1]])
                    p2 = ellipse_to_circle_proj([prevfacet.pRight[0]-center[0], prevfacet.pRight[1]-center[1]])
                    theta1 = math.atan(p1[1]/p1[0])
                    theta2 = math.atan(p2[1]/p2[0])
                    curvature1 = curvature(theta1)
                    curvature2 = curvature(theta2)
                    curvature_avg = (curvature1+curvature2)/2
                    curvature_errors.append(abs(curvature_avg - prevfacet.curvature))

                writeFacets(facetsunique, '{}_gapless_{}_{}'.format(save_name, resolution*100, trial))

        print("Facet gaps")
        #print(facet_gaps)
        print(len(facet_gaps))
        print(st.t.interval(alpha=0.95, df=len(facet_gaps)-1, loc=np.mean(facet_gaps), scale=st.sem(facet_gaps)))


        if interfaceReconstruction == 'ours':
            print("Curvature errors")
            #print(curvature_errors)
            print(len(curvature_errors))
            print(st.t.interval(alpha=0.95, df=len(curvature_errors)-1, loc=np.mean(curvature_errors), scale=st.sem(curvature_errors)))