from main.geoms.geoms import (
    getArea,
    getDistance,
    getPolyLineArea,
    lineIntersect,
    pointInPoly,
)
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet
from main.algos.plic_normals import getYoungsNormal, getLVIRANormal
from main.geoms.linear_facet import (
    getLinearFacet,
    getPolyLineIntersects,
    getLinearFacetFromNormal,
)
from main.geoms.circular_facet import getArcFacet, matchArcArea, getCenter


class BasePolygon:

    fraction_tolerance = 1e-10
    C0_linear_tolerance = 1e-5  # should be a couple orders of magnitude higher than linearity_threshold in NeighboredPolygon

    linearity_threshold = 1e-6  # if area fraction error in linear facet < this value, use linear facet at this cell
    optimization_threshold = 1e-10  # in optimizations

    # Invariant: self.fraction should always be between 0 and 1
    def __init__(self, points):
        self.points = points
        self.max_area = abs(getArea(self.points))
        self.fraction = None
        self.facet = None

        # TODO read from config
        # Error less than this is acceptable
        self.fraction_tolerance = BasePolygon.fraction_tolerance

        # 3x3 stencil of fractions for Young's
        self.stencil = None

        # Adjacent polygons
        self.adjacent_polys = []

    # TODO set fraction to 0 or 1 if within threshold?
    def setArea(self, area):
        self.fraction = area / self.max_area
        if self.fraction < 0 or self.fraction > 1:
            raise ValueError(
                f"Fraction {self.fraction} is invalid for BasePolygon after setArea"
            )

    def setFraction(self, fraction):
        self.fraction = fraction
        if self.fraction < 0 or self.fraction > 1:
            raise ValueError(
                f"Fraction {self.fraction} is invalid for BasePolygon after setFraction"
            )

    def clearFraction(self):
        self.fraction = None

    def setFractionTolerance(self, fraction_tolerance):
        self.fraction_tolerance = fraction_tolerance

    def getArea(self):
        return self.fraction * self.max_area

    def getFraction(self):
        return self.fraction

    def getFractionTolerance(self):
        return self.fraction_tolerance

    def getMaxArea(self):
        return self.max_area

    def isFull(self, tolerance=None):
        if tolerance is None:
            return self.fraction > 1 - self.fraction_tolerance
        else:
            return self.fraction > 1 - tolerance

    def isEmpty(self, tolerance=None):
        if tolerance is None:
            return self.fraction < self.fraction_tolerance
        else:
            return self.fraction < tolerance

    def isMixed(self, tolerance=None):
        return not (
            self.isFull(tolerance=tolerance) or self.isEmpty(tolerance=tolerance)
        )

    def diffFromMixed(self):
        if self.isMixed():
            return 0
        else:
            return max(
                self.fraction_tolerance - self.fraction,
                self.fraction - (1 - self.fraction_tolerance),
            )

    def setFacet(self, facet):
        self.facet = facet

    def clearFacet(self):
        self.facet = None

    def getFacet(self):
        return self.facet

    def hasFacet(self):
        return self.facet is not None

    def set3x3Stencil(self, stencil):
        self.stencil = stencil

    def has3x3Stencil(self):
        return self.stencil is not None

    # If orientation is "easy" (only 2 mixed neighbors with consistent orientation), return those neighbors
    def findSafeOrientation(self, fit_1neighbor=False):
        assert self.has3x3Stencil()
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # counterclockwise from right

        def _helper_getNeighborFromDirIndex(dir_i):
            dir = dirs[dir_i]
            return self.stencil[1 + dir[0]][1 + dir[1]]

        # Find mixed neighbors
        mixed_neighbors = []
        mixed_dirs = []
        for dir in range(len(dirs)):
            temp = _helper_getNeighborFromDirIndex(dir)
            if temp is not None and temp.isMixed():
                mixed_neighbors.append(temp)
                mixed_dirs.append(dir)

        # If two mixed neighbors, check to see if their orientations are consistent with the area fractions of their nonmixed neighbors
        if len(mixed_neighbors) == 2:
            # Check if mixed neighbors are across or adjacent
            if abs(mixed_dirs[0] - mixed_dirs[1]) == 2:  # Across
                # Check if nonmixed neighbors' fractions are consistent
                nonmixed1 = _helper_getNeighborFromDirIndex((mixed_dirs[0] + 1) % 4)
                nonmixed2 = _helper_getNeighborFromDirIndex((mixed_dirs[1] + 1) % 4)
                if nonmixed1 is None or nonmixed2 is None:
                    return None
                if nonmixed1.isFull() and nonmixed2.isEmpty():
                    return [mixed_neighbors[1], mixed_neighbors[0]]
                elif nonmixed1.isEmpty() and nonmixed2.isFull():
                    return mixed_neighbors
                else:
                    return None
            else:  # Adjacent
                # Figure out which mixed neighbor comes first in counterclockwise order
                if (
                    mixed_dirs[1] - mixed_dirs[0]
                ) % 4 == 1:  # TODO what's going on here? look carefully at orientations for linear vs. circular
                    # mixed1 = mixed_neighbors[0]
                    # mixed2 = mixed_neighbors[1]
                    mixed1 = mixed_neighbors[1]
                    mixed2 = mixed_neighbors[0]
                else:
                    # mixed1 = mixed_neighbors[1]
                    # mixed2 = mixed_neighbors[0]
                    mixed1 = mixed_neighbors[0]
                    mixed2 = mixed_neighbors[1]
                # Check if nonmixed neighbors' fractions are consistent
                # Edge case: if cell is at a boundary so that nonmixed neighbor is None, say it's consistent
                nonmixed1 = _helper_getNeighborFromDirIndex((mixed_dirs[0] + 2) % 4)
                nonmixed2 = _helper_getNeighborFromDirIndex((mixed_dirs[1] + 2) % 4)
                if (nonmixed1 is None or nonmixed1.isEmpty()) and (
                    nonmixed2 is None or nonmixed2.isEmpty()
                ):
                    return [mixed1, mixed2]
                elif (nonmixed1 is None or nonmixed1.isFull()) and (
                    nonmixed2 is None or nonmixed2.isFull()
                ):
                    return [mixed2, mixed1]
                else:
                    return None

        # If only one mixed neighbor, return that neighbor and the cell itself
        elif len(mixed_neighbors) == 1 and fit_1neighbor:
            # Find a nonmixed, non-None neighbor from among the neighbors adjacent to the mixed neighbor
            for dir in [(mixed_dirs[0] + 1) % 4, (mixed_dirs[0] - 1) % 4]:
                temp = _helper_getNeighborFromDirIndex(dir)
                if temp is not None and not temp.isMixed():
                    break
            # If dir - mixed_dirs[0] == 1, then temp is the neighbor counterclockwise from the mixed neighbor. Otherwise, temp is the neighbor clockwise from the mixed neighbor.
            # If temp is full and dir - mixed_dirs[0] == 1, or if temp is empty and dir - mixed_dirs[0] == -1, then return [self, mixed_neighbor]. Otherwise, return [mixed_neighbor, self].
            if (temp.isFull() and (dir - mixed_dirs[0]) % 4 == 1) or (
                temp.isEmpty() and (dir - mixed_dirs[0]) % 4 == 3
            ):
                return [self, mixed_neighbors[0]]
            else:
                return [mixed_neighbors[0], self]

        # Otherwise, we don't have a safe orientation. Return None
        else:
            return None

    def runYoungs(self, ret=False):
        assert self.has3x3Stencil()
        normal = getYoungsNormal(self.stencil)
        l1, l2 = getLinearFacetFromNormal(
            self.points, self.getFraction(), normal, BasePolygon.optimization_threshold
        )
        intersects = getPolyLineIntersects(self.points, l1, l2)
        youngsFacet = LinearFacet(intersects[0], intersects[-1], name="Youngs")
        if ret:
            return youngsFacet
        else:
            self.setFacet(youngsFacet)

    def runLVIRA(self, ret=False):
        assert self.has3x3Stencil()
        normal = getLVIRANormal(self.stencil)
        l1, l2 = getLinearFacetFromNormal(
            self.points, self.getFraction(), normal, BasePolygon.optimization_threshold
        )
        intersects = getPolyLineIntersects(self.points, l1, l2)
        youngsFacet = LinearFacet(intersects[0], intersects[-1], name="LVIRA")
        if ret:
            return youngsFacet
        else:
            self.setFacet(youngsFacet)

    def runSafeLinear(
        self,
        ret=False,
        check_threshold=False,
        default_to_youngs=False,
        default_to_lvira=True,
        fit_1neighbor=False,
    ):
        assert self.has3x3Stencil()
        orientation = self.findSafeOrientation(fit_1neighbor=fit_1neighbor)
        if orientation is None:
            # Default to PLIC
            if default_to_youngs:
                facet = self.runYoungs(ret=True)
            elif default_to_lvira:
                facet = self.runLVIRA(ret=True)
            else:
                facet = None
        else:
            left_neighbor: BasePolygon = orientation[0]
            right_neighbor: BasePolygon = orientation[1]
            l1, l2 = getLinearFacet(
                left_neighbor.points,
                right_neighbor.points,
                left_neighbor.getFraction(),
                right_neighbor.getFraction(),
                BasePolygon.optimization_threshold,
            )
            if check_threshold:
                # Check whether middle area fraction is close to target area fraction
                if (
                    abs(
                        self.getFraction()
                        - getPolyLineArea(self.points, l1, l2) / self.getMaxArea()
                    )
                    < BasePolygon.linearity_threshold
                    and (
                        getPolyLineArea(self.points, l1, l2) / self.getMaxArea()
                        > BasePolygon.fraction_tolerance
                    )
                    and (
                        getPolyLineArea(self.points, l1, l2) / self.getMaxArea()
                        < 1 - BasePolygon.fraction_tolerance
                    )
                ):
                    # Linear: set facet
                    normal = [
                        (-l2[1] + l1[1]) / getDistance(l1, l2),
                        (l2[0] - l1[0]) / getDistance(l1, l2),
                    ]
                    l1, l2 = getLinearFacetFromNormal(
                        self.points,
                        self.getFraction(),
                        normal,
                        BasePolygon.optimization_threshold,
                    )
                    intersects = getPolyLineIntersects(self.points, l1, l2)
                    facet = LinearFacet(intersects[0], intersects[-1], name="linear")
                else:
                    if default_to_youngs:
                        facet = self.runYoungs(ret=True)
                    elif default_to_lvira:
                        facet = self.runLVIRA(ret=True)
                    else:
                        facet = None
            else:
                # No need to check linearity threshold
                normal = [
                    (-l2[1] + l1[1]) / getDistance(l1, l2),
                    (l2[0] - l1[0]) / getDistance(l1, l2),
                ]
                l1, l2 = getLinearFacetFromNormal(
                    self.points,
                    self.getFraction(),
                    normal,
                    BasePolygon.optimization_threshold,
                )
                # Linear
                intersects = getPolyLineIntersects(self.points, l1, l2)
                facet = LinearFacet(
                    intersects[0], intersects[-1], name="default_linear"
                )  # TODO change name to linear?

        if ret:
            return facet
        else:
            if facet is not None:
                self.setFacet(facet)

    def runSafeCircle(
        self,
        ret=False,
        default_to_youngs=False,
        default_to_lvira=True,
    ):
        assert self.has3x3Stencil()
        orientation = self.findSafeOrientation(fit_1neighbor=False)
        if orientation is None:
            # Default to PLIC
            if default_to_youngs:
                facet = self.runYoungs(ret=True)
            elif default_to_lvira:
                facet = self.runLVIRA(ret=True)
            else:
                facet = None
        else:
            left_neighbor: BasePolygon = orientation[0]
            right_neighbor: BasePolygon = orientation[1]
            l1, l2 = getLinearFacet(
                left_neighbor.points,
                right_neighbor.points,
                left_neighbor.getFraction(),
                right_neighbor.getFraction(),
                BasePolygon.optimization_threshold,
            )
            if (
                abs(
                    self.getFraction()
                    - getPolyLineArea(self.points, l1, l2) / self.getArea()
                )
                < BasePolygon.linearity_threshold
                and (
                    getPolyLineArea(self.points, l1, l2) / self.getArea()
                    > BasePolygon.optimization_threshold
                )
                and (
                    getPolyLineArea(self.points, l1, l2) / self.getArea()
                    < 1 - BasePolygon.optimization_threshold
                )
            ):
                # Linear
                intersects = getPolyLineIntersects(self.points, l1, l2)
                facet = LinearFacet(intersects[0], intersects[-1])
            else:
                try:
                    arccenter, arcradius, arcintersects = getArcFacet(
                        left_neighbor.points,
                        self.points,
                        right_neighbor.points,
                        left_neighbor.getFraction(),
                        self.getFraction(),
                        right_neighbor.getFraction(),
                        BasePolygon.optimization_threshold,
                    )
                    if arccenter is None or arcradius is None or arcintersects is None:
                        if default_to_youngs:
                            facet = self.runYoungs(ret=True)
                        elif default_to_lvira:
                            facet = self.runLVIRA(ret=True)
                        else:
                            facet = None
                    else:
                        # Arc
                        facet = ArcFacet(
                            arccenter, arcradius, arcintersects[0], arcintersects[-1]
                        )
                except:
                    if default_to_youngs:
                        facet = self.runYoungs(ret=True)
                    elif default_to_lvira:
                        facet = self.runLVIRA(ret=True)
                    else:
                        facet = None

        if ret:
            return facet
        else:
            self.setFacet(facet)

    # Given two endpoints of facet and area fraction, find unique curvature satisfying those constraints
    def fitCurvature(
        self, pLeft, pRight, fraction_tolerance=fraction_tolerance, ret=False
    ):
        d = getDistance(pLeft, pRight)
        lineArea = getPolyLineArea(self.points, pLeft, pRight)
        # If line area is close to target area, return linear facet
        if (
            abs(self.getArea() - lineArea) / self.getMaxArea()
            < BasePolygon.C0_linear_tolerance
        ):
            facet = LinearFacet(pLeft, pRight)
        # Otherwise, need to fit curvature
        else:
            radius = matchArcArea(
                d, self.getArea() - lineArea, fraction_tolerance * self.getMaxArea()
            )
            if radius == float("inf") or radius == -float("inf"):
                facet = LinearFacet(pLeft, pRight)
            else:
                center = getCenter(pLeft, pRight, radius)
                facet = ArcFacet(center, radius, pLeft, pRight)
        if ret:
            return facet
        else:
            self.setFacet(facet)

    # Given two endpoints of facet and slopes (both pointing from endpoint toward corner), form unique linear corner and #TODO
    def checkCorner(self, pLeft, pRight, slopeLeft, slopeRight):
        p2Left = [pLeft[0] + slopeLeft[0], pLeft[1] + slopeLeft[1]]
        p2Right = [pRight[0] + slopeRight[0], pRight[1] + slopeRight[1]]
        corner, _, _ = lineIntersect(pLeft, p2Left, pRight, p2Right)

    def __str__(self):
        if self.hasFacet:
            return f"\nPoints: {self.points}\nFraction: {self.fraction}\nFacet: {self.facet}\n"
        else:
            return f"\nPoints: {self.points}\nFraction: {self.fraction}\n"
