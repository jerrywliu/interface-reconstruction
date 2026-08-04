import numpy as np
import math
import vtk
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from geoms import getDistance, getArea, mergePolys, getPolyIntersectArea, getPolyLineArea, getPolyLineIntersects, lineIntersect, getCentroid
from linear_facet import getLinearFacet
from circular_facet import getCircleIntersectArea, getCircleCircleIntersects, getArcFacet, getArcFacetNewton, getCircleLineIntersects2, getCenter
from corner_facet import getPolyCornerArea, getPolyCurvedCornerArea, getCurvedCornerFacet
from mesh import QuadMesh, makeQuadGrid, makeConcaveGrid, makeFineCartesianGrid
import initialize_areas
import initialize_velocity
from vtkplots import plotQuadGrid, plotFacets
from interface_reconstruction import merge, makeFacets, advectFacets

import argparse

def advection(gridSize, timesteps, plotMat=True, plotVtk=True, makeGapless=False):
    
    #Starting mesh
    opoints = makeFineCartesianGrid(gridSize, resolution)
    mesh = QuadMesh(opoints)
    opolys = mesh.polys
    
    #Compute initial area setup
    areas = initialize_areas.initializeCircle(opolys, [50.01, 75.01], 15, threshold)
    mesh.initializeAreas(areas)

    #Plot in matplotlib
    if plotMat:
        try:
            os.mkdir('advection_plots')
        except:
            print("Saving plots in ./advection_plots/.")
        mesh.plotAreas('advection_plots/original.png')
        mesh.plotPartialAreas('advection_plots/original_partials.png')
        mesh.plotInitialAreaCompare('advection_plots/original_compare.png')
    
    #Plot in vtk
    if plotVtk:
        try:
            os.mkdir('advection_vtk')
            print("Saving vtk files in ./advection_vtk/.")
        except:
            pass
        if os.path.exists('advection_vtk/quads_{}x{}_{}.vtk'.format(gridSize, gridSize, resolution)):
            print("Using preexisting mesh at ./advection_vtk/quads_{}x{}_{}.vtk".format(gridSize, gridSize, resolution))
        else:
            mesh.plotMesh('advection_vtk/quads_{}x{}_{}.vtk'.format(gridSize, gridSize, resolution))
    
    #Main advection loop
    t = 0

    #List of areas per timestep
    areas_per_time = []
    times = []
    
    area_current_time = 0
    for x in range(len(opolys)):
        for y in range(len(opolys[0])):
            area_current_time += getArea(opolys[x][y]) * areas[x][y]
    areas_per_time.append(area_current_time)
    times.append(t)

    for timestep in range(timesteps):

        #Mesh velocity per timestep
        velocity = initialize_velocity.vortex_tr(timedelta, t, totalt)
        
        print("Timestep: {}".format(timestep))
        
        #Merge
        mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
        #Interface reconstruction
        predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless)

        #Plot facets in vtk
        if plotVtk and (timestep % savepertime == 0 or timestep == timesteps-1):
            plotFacets(facetsunique, 'timestep_{}'.format(timestep))
        
        print("Computing new areas")
        nareas = advectFacets(opolys, areas, predfacets, velocity, checksize, threshold)
        areas = nareas
        mesh.setAreas(areas)
        
        if plotMat:
            print("Plotting")
            #plot in matplotlib
            mesh.plotAreas('advection_plots/pred_timestep_{}.png'.format(timestep))
            mesh.plotPartialAreas('advection_plots/predpartials_timestep_{}.png'.format(timestep))
            mesh.plotInitialAreaCompare('advection_plots/predcompare_timestep_{}.png'.format(timestep))

        t += timedelta

        #Update areas_per_time
        area_current_time = 0
        for x in range(len(opolys)):
            for y in range(len(opolys)):
                area_current_time += getArea(opolys[x][y]) * areas[x][y]
        areas_per_time.append(area_current_time)
        times.append(t)

    
    plt.plot(times, areas_per_time, 'ro')
    plt.title('Volume vs. time steps, {}x{}, timestep = {}'.format(resolution, resolution, timestep))
    plt.xlabel('Time step')
    plt.ylabel('Volume')
    plt.savefig("advection_plots/areas_per_time.png", dpi=199)

def interface_reconstruction(plotMat=True, plotVtk=True):

    #Starting mesh
    opoints = makeFineCartesianGrid(gridSize, resolution)
    mesh = QuadMesh(opoints)
    opolys = mesh.polys
    
    #Compute initial area setup
    areas = initialize_areas.initializeCircle(opolys, [50.01, 75.01], 15, threshold)
    mesh.initializeAreas(areas)

    #Plot in matplotlib
    if plotMat:
        try:
            os.mkdir('advection_plots')
        except:
            print("Saving plots in ./advection_plots/.")
        mesh.plotAreas('advection_plots/original.png')
        mesh.plotPartialAreas('advection_plots/original_partials.png')
        mesh.plotInitialAreaCompare('advection_plots/original_compare.png')
    
    #Plot in vtk
    if plotVtk:
        try:
            os.mkdir('advection_vtk')
            print("Saving vtk files in ./advection_vtk/.")
        except:
            pass
        if os.path.exists('advection_vtk/quads_{}x{}_{}.vtk'.format(gridSize, gridSize, resolution)):
            print("Using preexisting mesh at ./advection_vtk/quads_{}x{}_{}.vtk".format(gridSize, gridSize, resolution))
        else:
            mesh.plotMesh('advection_vtk/quads_{}x{}_{}.vtk'.format(gridSize, gridSize, resolution))

    #Merge
    mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions = merge(opolys, areas)
    #Interface reconstruction
    predfacets, facetsunique = makeFacets(mergedpolyindices, mergedpolyinfos, mergedcoords, mergedareafractions, opolys, areas, threshold, makeGapless)


def __init__():
    parser = argparse.ArgumentParser(description='High-order advection scheme')

    parser.add_argument('--name', '-n', type=str, metavar='N', help='description')

    args = parser.parse_args()


np.random.seed(17)
savepertime = 1
wiggle = 0.5

gridSize = 100
resolution = 1
#gridSize = 1
#resolution = 128
checksize = 2

threshold = 1e-10*resolution
vsize = 0.0905
totalt = 8
timedelta = 0.005
timesteps = 1 #int(totalt/timedelta)+5
advection(gridSize, timesteps, plotVtk=True, plotMat=True, makeGapless=False)
#advection(gridSize, timesteps, plotVtk=False, plotMat=False, makeGapless=True)