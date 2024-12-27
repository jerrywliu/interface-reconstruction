from main.geoms.corner_facet import getPolyCornerArea, getPolyCurvedCornerArea
from main.structs.polys.base_polygon import BasePolygon
from main.geoms.geoms import getArea, getDistance, lineIntersect, pointInPoly, lineAngleSine
from main.structs.facets.base_facet import Facet, LinearFacet, ArcFacet, CornerFacet
from main.geoms.circular_facet import getArcFacet, getCircleLineIntersects, getCircleCircleIntersects
from main.geoms.linear_facet import getLinearFacet, getPolyLineArea, getPolyLineIntersects, getLinearFacetFromNormal

class NeighboredPolygon(BasePolygon):

    linearity_threshold = 1e-6 #if area fraction error in linear facet < this value, use linear facet at this cell
    optimization_threshold = 1e-10 # in optimizations
    linear_corner_area_threshold = 1e-4 # if area fraction error in corner facet < this value, use (straight edged) corner facet at this cell
    corner_sharpness_threshold = 1e-2 # if abs(sine) of angle between corner edges < this value, it's too sharp to use
    curved_corner_curvature_threshold = 1e-2 #if adjacent curvatures > curvaturethreshold, try to fit a curved corner facet
    curved_corner_area_threshold = 1e-2 #if area fraction error in curved corner < curvedcornerthreshold, use curved corner at this cell

    def __init__(self, points):
        super().__init__(points)

        self.facet_type = None
        self.left_neighbor = None
        self.right_neighbor = None

        #TODO List of unoriented neighbors, maybe useful later?
        self.unoriented_neighbors = []

    def setFacetType(self, facet_type):
        self.facet_type = facet_type

    #poly = NeighboredPolygon
    #orientation = "left", "right", other
    def setNeighbor(self, poly, orientation):
        if orientation == "left":
            self.left_neighbor = poly
        elif orientation == "right":
            self.right_neighbor = poly
        else:
            self.unoriented_neighbors.append(poly)

    #orientation = "left", "right", other
    def clearNeighbor(self, orientation):
        if orientation == "left":
            self.left_neighbor = None
        elif orientation == "right":
            self.right_neighbor = None
        else:
            self.unoriented_neighbors = []

    def hasLeftNeighbor(self):
        return (self.left_neighbor is not None)

    def hasRightNeighbor(self):
        return (self.right_neighbor is not None)

    def getLeftNeighbor(self):
        return self.left_neighbor

    def getRightNeighbor(self):
        return self.right_neighbor

    def fullyOriented(self):
        return self.hasLeftNeighbor() and self.hasRightNeighbor()
    
    # Finds orientation of neighbors based on 3x3 stencil, sets self.left_neighbor and self.right_neighbor for easy cases
    def findSafeOrientation(self):
        # Inherit from BasePolygon
        orientation = super().findSafeOrientation()
        if orientation is None:
            return None
        else:
            [self.left_neighbor, self.right_neighbor] = orientation
            return orientation

    def fitCircularFacet(self):
        # If both neighbors, try linear and circular TODO
        if self.hasLeftNeighbor() and self.hasRightNeighbor():
            facetline1, facetline2 = getLinearFacet(self.left_neighbor.points, self.right_neighbor.points, self.left_neighbor.getFraction(), self.right_neighbor.getFraction(), NeighboredPolygon.optimization_threshold)
            if abs(self.getFraction() - getPolyLineArea(self.points, facetline1, facetline2)/self.getArea()) < NeighboredPolygon.linearity_threshold and (getPolyLineArea(self.points, facetline1, facetline2)/self.getArea() > NeighboredPolygon.optimization_threshold) and (getPolyLineArea(self.points, facetline1, facetline2)/self.getArea() < 1-NeighboredPolygon.optimization_threshold):
                intersects = getPolyLineIntersects(self.points, facetline1, facetline2)
                self.setFacet(LinearFacet(intersects[0], intersects[-1]))
            else:
                try:
                    arccenter, arcradius, arcintersects = getArcFacet(self.left_neighbor.points, self.points, self.right_neighbor.points, self.left_neighbor.getFraction(), self.getFraction(), self.right_neighbor.getFraction(), NeighboredPolygon.optimization_threshold)
                    #TODO If failed: default to linear
                    if arccenter is None or arcradius is None or arcintersects is None:
                        pass
                        # l1 = facetline1
                        # l2 = facetline2
                        # normal = [(-l2[1]+l1[1])/getDistance(l1, l2), (l2[0]-l1[0])/getDistance(l1, l2)]
                        # facetline1, facetline2 = getLinearFacetFromNormal(self.points, self.getFraction(), normal, NeighboredPolygon.optimization_threshold)
                        # self.setFacet(LinearFacet(facetline1, facetline2))
                    else:
                        self.setFacet(ArcFacet(arccenter, arcradius, arcintersects[0], arcintersects[-1]))
                except:
                    pass
                    # l1 = facetline1
                    # l2 = facetline2
                    # normal = [(-l2[1]+l1[1])/getDistance(l1, l2), (l2[0]-l1[0])/getDistance(l1, l2)]
                    # facetline1, facetline2 = getLinearFacetFromNormal(self.points, self.getFraction(), normal, NeighboredPolygon.optimization_threshold)
                    # self.setFacet(LinearFacet(facetline1, facetline2))
        else:
            print("Not enough neighbors: failed to make circular facet")

    # if doCollinearityCheck, then only set linear facet if middle area fraction matches within threshold
    def fitLinearFacet(self, doCollinearityCheck=False):
        if self.hasLeftNeighbor() and self.hasRightNeighbor():
            l1, l2 = getLinearFacet(self.left_neighbor.points, self.right_neighbor.points, self.left_neighbor.getFraction(), self.right_neighbor.getFraction(), NeighboredPolygon.optimization_threshold)
            
            # Check if this facet matches middle area (and actually intersects middle poly in case of nearly empty or full cell)
            isValidLinearFacet = False
            if abs(self.getFraction() - (getPolyLineArea(self.points, l1, l2)/self.getMaxArea())) < NeighboredPolygon.linearity_threshold:
                intersects = getPolyLineIntersects(self.points, l1, l2)
                if len(intersects) > 0:
                    if doCollinearityCheck:
                        isValidLinearFacet = True
                        self.setFacet(LinearFacet(intersects[0], intersects[-1]))

            if not(isValidLinearFacet) and not(doCollinearityCheck):
                # Use this as normal and fit a linear facet
                normal = [(-l2[1]+l1[1])/getDistance(l1, l2), (l2[0]-l1[0])/getDistance(l1, l2)]
                #TODO getLinearFacetFromNormal failed on areafraction=1e-8, threshold=1e-12
                try:
                    l1, l2 = getLinearFacetFromNormal(self.points, self.getFraction(), normal, NeighboredPolygon.optimization_threshold)
                except:
                    #TODO hack to fix getLinearFacetFromNormal for small areas
                    print(f"getLinearFacetFromNormal({self.points}, {self.getFraction()}, {normal}, {NeighboredPolygon.optimization_threshold})")
                    l1, l2 = getLinearFacetFromNormal(self.points, self.getFraction()*1e3, normal, NeighboredPolygon.optimization_threshold)
                
                self.setFacet(LinearFacet(l1, l2, name="default_linear"))

        else:
            print("Not enough neighbors: failed to make linear facet")

    # Extends lines l1 to l2, p1 to p2, calculates intersection if it exists, checks whether this corner matches area fraction, updates self.facet if it does
    # (l1, l2, p1, p2) = (l.l, l.r, r.r, r.l)
    def checkCornerFacet(self, l1, l2, p1, p2):
        corner, _, _ = lineIntersect(l1, l2, p1, p2)
        if corner is not None:
            # Feasible intersection, continue #TODO: do a test to see if corner point lies within the two segments
            corner = [l2, corner, p2]
            cornerareafraction = getPolyCornerArea(self.points, corner[0], corner[1], corner[2])/self.getMaxArea()
            if abs(cornerareafraction - self.getFraction()) < NeighboredPolygon.linear_corner_area_threshold:
                # Check if this corner facet intersects middle poly
                intersects1 = getPolyLineIntersects(self.points, corner[0], corner[1])
                intersects2 = getPolyLineIntersects(self.points, corner[1], corner[2])
                # Check that corner is not too sharp
                if (len(intersects1) > 0 or len(intersects2)) and abs(lineAngleSine(l1, l2, p1, p2)) > NeighboredPolygon.corner_sharpness_threshold:
                    self.setFacet(CornerFacet(None, None, None, None, corner[0], corner[1], corner[2]))
                    # print("checkCornerFacet formed corner:")
                    # print(self)
                    # print("Real area fraction: {}".format(cornerareafraction))
                    # print("Target area fraction: {}".format(self.getFraction()))
            else:
                # print("Unmatching corner area:")
                # print(cornerareafraction)
                # print(self.getFraction())
                pass

    # Facet 1: l1, l2, center, radius
    # Facet 2: p1, p2, center, radius
    # Returns closest intersection point, if there is one
    #TODO uses self.points[0] to decide which line-arc intersection is the corner point
    def checkCurvedCornerFacet(self, facet1: Facet, facet2: Facet, ret=False):
        if facet1.name == "linear" and facet2.name == "arc":
            intersects = getCircleLineIntersects(facet1.pLeft, facet1.pRight, facet2.center, facet2.radius, checkWithinLine=False)
            if len(intersects) == 1:
                corner_point = intersects[0]
            elif len(intersects) > 1:
                if getDistance(self.points[0], intersects[0]) >= getDistance(self.points[0], intersects[1]):
                    corner_point = intersects[1]
                else:
                    corner_point = intersects[0]
            else: # no intersects
                print("checkCurvedCornerFacet: no intersects between line and arc")
                return None, None
            
        elif facet1.name == "arc" and facet2.name == "linear":
            intersects = getCircleLineIntersects(facet2.pLeft, facet2.pRight, facet1.center, facet1.radius, checkWithinLine=False)
            if len(intersects) == 1:
                corner_point = intersects[0]
            elif len(intersects) > 1:
                if getDistance(self.points[0], intersects[0]) >= getDistance(self.points[0], intersects[1]):
                    corner_point = intersects[1]
                else:
                    corner_point = intersects[0]
            else: # no intersects
                print("checkCurvedCornerFacet: no intersects between line and arc")
                return None, None

        elif facet1.name == "arc" and facet2.name == "arc":
            try:
                intersects = getCircleCircleIntersects(facet1.center, facet2.center, facet1.radius, facet2.radius)
                if len(intersects) == 0:
                    print("checkCurvedCornerFacet: no intersects between arc and arc")
                    return None, None
                else: # two intersects TODO what if more than 2?
                    assert len(intersects) == 2
                    if getDistance(self.points[0], intersects[0]) >= getDistance(self.points[0], intersects[1]):
                        corner_point = intersects[1]
                    else:
                        corner_point = intersects[0]
            except:
                # Failed two arc curved corner
                print("checkCurvedCornerFacet: failed intersect between arc and arc")
                return None, None
        else: # linear, linear
            #TODO: skip linear, linear case in curved corner
            # self.checkCornerFacet(facet1.pLeft, facet1.pRight, facet2.pRight, facet2.pLeft)
            return None, None

        cornerareafraction = getPolyCurvedCornerArea(self.points, facet1.pRight, corner_point, facet2.pLeft, facet1.radius if facet1.name == "arc" else None, facet2.radius if facet2.name == "arc" else None)/self.getMaxArea()
        facet1_tangent = facet1.getTangent(corner_point)
        facet2_tangent = facet2.getTangent(corner_point)
        # Check that corner is not too sharp
        # TODO Check if this corner facet intersects middle poly
        if abs(lineAngleSine([0, 0], facet1_tangent, [0, 0], facet2_tangent)) > NeighboredPolygon.corner_sharpness_threshold:
            facet = CornerFacet(facet1.center if facet1.name == "arc" else None, facet2.center if facet2.name == "arc" else None, facet1.radius if facet1.name == "arc" else None, facet2.radius if facet2.name == "arc" else None, facet1.pRight, corner_point, facet2.pLeft)
            if ret:
                return facet, abs(cornerareafraction - self.getFraction())
            elif abs(cornerareafraction - self.getFraction()) < NeighboredPolygon.curved_corner_area_threshold:
                self.setFacet(facet)
        else:
            return None, None
            # print("Failed curved corner:")
            # print(self)
            # print(CornerFacet(facet1.center if facet1.name == "arc" else None, facet2.center if facet2.name == "arc" else None, facet1.radius if facet1.name == "arc" else None, facet2.radius if facet2.name == "arc" else None, facet1.pRight, corner_point, facet2.pLeft))
            # print(cornerareafraction)
            # print(abs(cornerareafraction - self.getFraction()))

    #TODO add settings for only circles / only lines / corners

    def __str__(self):
        if self.hasFacet:
            return f"\nPoints: {self.points}\nFraction: {self.fraction}\nLeft neighbor: {self.left_neighbor.points if self.left_neighbor is not None else None}\nRight neighbor: {self.right_neighbor.points if self.right_neighbor is not None else None}\nFacet: {self.facet}\n"
        else:
            return f"\nPoints: {self.points}\nFraction: {self.fraction}\nLeft neighbor: {self.left_neighbor.points if self.left_neighbor is not None else None}\nRight neighbor: {self.right_neighbor.points if self.right_neighbor is not None else None}\n"
            