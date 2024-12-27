import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from matplotlib.collections import PatchCollection

from main.structs.polys.base_polygon import BasePolygon

class BaseMesh:

    def __init__(self, points, threshold, fractions=None):
        #Points = [[x, y]]
        self._points = points
        self.threshold = threshold

        #Compute extremes x and y
        self._max_x = -float('inf')
        self._max_y = -float('inf')
        self._min_x = float('inf')
        self._min_y = float('inf')

        for x in range(len(self._points)):
            for y in range(len(self._points[0])):
                #Set extremes
                if points[x][y][0] > self._max_x:
                    self._max_x = points[x][y][0]
                if points[x][y][1] > self._max_y:
                    self._max_y = points[x][y][1]
                if points[x][y][0] < self._min_x:
                    self._min_x = points[x][y][0]
                if points[x][y][1] < self._min_y:
                    self._min_y = points[x][y][1]

        self.polys: list[list[BasePolygon]] = [[None] * (len(points[0])-1) for _ in range(len(points)-1)]
        self._plt_patches = []
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                #Make quads
                poly = BasePolygon([points[x][y], points[x+1][y], points[x+1][y+1], points[x][y+1]])
                self.polys[x][y] = poly
                patch = plt_polygon(np.array(poly.points), True)
                self._plt_patches.append(patch)
        
        if fractions is None:
            self._plt_patchareas = np.array([])
            self._plt_patchpartialareas = np.array([])
            self._plt_patchinitialareas = np.array([])
            self._vtk_mixed_polys = []
            self._vtk_mixed_polyareas = []
        else:
            self.initializeFractions(fractions)

    def initializeFractions(self, fractions):
        patchinitialareas = []

        #Flatten array
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):

                adjusted_fraction = fractions[x][y]
                if abs(1-adjusted_fraction) < self.threshold:
                    adjusted_fraction = 1
                elif abs(adjusted_fraction) < self.threshold:
                    adjusted_fraction = 0

                patchinitialareas.append(adjusted_fraction)

        self._plt_patchinitialareas = np.array(patchinitialareas)
        self.setFractions(fractions)

    def setFractions(self, fractions):
        patchareas = []
        patchpartialareas = []
        self._vtk_mixed_polys = []
        self._vtk_mixed_polyareas = []

        #Flatten array
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                
                adjusted_fraction = fractions[x][y]
                if abs(1-adjusted_fraction) < self.threshold:
                    adjusted_fraction = 1
                elif abs(adjusted_fraction) < self.threshold:
                    adjusted_fraction = 0
                else:
                    # mixed cell
                    self._vtk_mixed_polys.append(self.polys[x][y].points)
                    self._vtk_mixed_polyareas.append(adjusted_fraction)

                # self.polys[x][y].setFraction(adjusted_fraction)
                # Raw (unrounded) area value stored in mesh
                self.polys[x][y].setFraction(fractions[x][y])

                patchareas.append(adjusted_fraction)
                patchpartialareas.append(math.ceil(adjusted_fraction - math.floor(adjusted_fraction)))

        self._plt_patchareas = np.array(patchareas)
        self._plt_patchpartialareas = np.array(patchpartialareas)

    def getFractions(self):
        fractions = [[None for _ in range(len(self.polys[0]))] for _ in range(len(self.polys))]
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                fractions[x][y] = self.polys[x][y].getFraction()
        return fractions

    # Take cell and return 3x3 array of area fractions (for Young's)
    def get3x3Stencil(self, x, y):
        def _helper_in_bounds(i, j):
            if i < 0 or j < 0 or i >= len(self.polys) or j >= len(self.polys[0]):
                return None
            return self.polys[i][j]

        ret = [[None, None, None],[None, None, None],[None, None, None]]
        for x_delta in range(-1, 2):
            for y_delta in range(-1, 2):
                ret[1+x_delta][1+y_delta] = _helper_in_bounds(x+x_delta, y+y_delta)

        return ret

    # Get full/empty cell coords whose fraction is closest to being mixed
    # If all are mixed, returns [None, None]
    def getMostMixedAdjacentFullCell(self, x, y):
        def _helper_diffs(i, j):
            # Ignore mixed cells
            if i < 0 or j < 0 or i >= len(self.polys) or j >= len(self.polys[0]) or self.polys[i][j].isMixed():
                return float("inf")
            return self.polys[i][j].diffFromMixed()
        
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        ret_x = None
        ret_y = None
        min_diff = float("inf")
        for dir in dirs:
            diff = _helper_diffs(x+dir[0], y+dir[1])
            if diff < min_diff:
                ret_x = x+dir[0]
                ret_y = y+dir[1]

        diffs = list(map(lambda dir : _helper_diffs(x+dir[0], y+dir[1]), dirs))
        print(diffs)
        return [ret_x, ret_y]
    
    # Youngs
    def runYoungs(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly : BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_poly.runYoungs()

    # LVIRA
    def runLVIRA(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly : BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_poly.runLVIRA()

    # Whenever orientation is unambiguous, run linear fit. Default to Youngs
    def runSafeLinear(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly : BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_poly.runSafeLinear(check_threshold=False, default_to_youngs=True)

    # Whenever orientation is unambiguous, run circle fit. Default to Youngs
    def runSafeCircle(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly : BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_poly.runSafeCircle()
