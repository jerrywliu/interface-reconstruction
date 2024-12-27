import math
import numpy as np

from facet import Facet, LinearFacet, ArcFacet, getNormal, isDegenFacet
from circular_facet import getArcArea, matchArcArea, getCenter, isMajorArc
from geoms import getArea, getDistance, lerp, pointLeftOfLine, pointRightOfLine, pointInPoly
from copy import deepcopy

class StrandPoly:

    def __init__(self, points):
        self.points = points
        #reset after zeroStrands
        self.merged_strands = []
        self.merged_strands_poly_lerps = []
        self.unmerged_strands = []
        self.unmerged_strands_poly_lerps = []
        #reset after updateStrands
        self.strands_sorted = [[] for _ in range(len(self.points))]
        self.strands_poly_lerps_pLefts = [[] for _ in range(len(self.points))]
        self.strands_poly_lerps_pRights = [[] for _ in range(len(self.points))]
        self.strands_stored = []

    def __str__(self):
        retString = "Points: {}".format(self.points)
        for i in range(len(self.points)):
            retString += "\nSide {} strands: {}, poly lerp reps: {}".format(i, str(list(map(lambda x : str(x), self.strands_sorted[i]))), self.strands_poly_lerps_sorted[i])
        return retString

    #Polygon-line intersects are represented in three ways:
    #1. Cartesian coordinates, 2. polygon edgewise lerp, 3. line lerp
    #Retursn are sorted by distance from l1
    def getPolyLineIntersects(self, l1, l2):
        assert not (l1[0] == l2[0] and l1[1] == l2[1]), "{}, {}".format(l1, l2)
        adjustcorneramount = 1e-14
        notmod = True
        while notmod:
            intersects = []
            poly_lerps = []
            line_lerps = []
            if l1[0] == l2[0]:
                for i in range(len(self.points)):
                    p1 = self.points[i]
                    p2 = self.points[(i+1) % len(self.points)]
                    if (p1[0] < l1[0] and p2[0] > l1[0]) or (p1[0] > l1[0] and p2[0] < l1[0]):
                        t = (l1[0] - p1[0])/(p2[0]-p1[0])
                        pinter = lerp(p1, p2, t)
                        intersects.append(pinter)
                        poly_lerps.append([i, t])
                        line_lerps.append((pinter[1]-l1[1])/(l2[1]-l1[1]))
            else:
                l = lambda x : l1[1] + (l2[1]-l1[1])*(x-l1[0])/(l2[0]-l1[0])
                for i in range(len(self.points)):
                    p1 = self.points[i]
                    p2 = self.points[(i+1) % len(self.points)]
                    if (p1[1] > l(p1[0]) and p2[1] < l(p2[0])) or (p1[1] < l(p1[0]) and p2[1] > l(p2[0])):
                        t = (p1[1]-l1[1]-(l2[1]-l1[1])*(p1[0]-l1[0])/(l2[0]-l1[0]))/((l2[1]-l1[1])*(p2[0]-p1[0])/(l2[0]-l1[0]) - (p2[1]-p1[1]))
                        pinter = lerp(p1, p2, t)
                        intersects.append(pinter)
                        poly_lerps.append([i, t])
                        line_lerps.append((pinter[0]-l1[0])/(l2[0]-l1[0]))

            #If not 0 mod 2, line intersects a corner, perturb line and rerun
            #TODO: how many times does this trigger? Is there a more elegant solution?
            if len(intersects) % 2 == 1:
                l2 = [l2[0]+adjustcorneramount, l2[1]+adjustcorneramount]
            else:
                notmod = False

        return [(a,b,c) for _,a,b,c in sorted(zip(line_lerps, intersects, poly_lerps, line_lerps))]

    #Polygon-circle intersects are represented in two ways:
    #1. Cartesian coordinates, 2. polygon edgewise lerp
    #Returns are unsorted (circle requires a reference point to sort)
    def getPolyCircleIntersects(self, center, radius):
        #Hyperparameter
        adjustcorneramount = 1e-14
        notmod = True
        while notmod:
            startAt = 1
            intersects = []
            poly_lerps = []
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i+1) % len(self.points)]
                p1in = getDistance(p1, center) <= abs(radius)
                p2in = getDistance(p2, center) <= abs(radius)

                a = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
                b = 2*((p1[0]-center[0])*(p2[0]-p1[0]) + (p1[1]-center[1])*(p2[1]-p1[1]))
                c = (p1[0]-center[0])**2 + (p1[1]-center[1])**2 - radius**2

                disc = b**2 - 4*a*c
                if disc > 0:
                    #Intersect 1
                    x1 = (-b - math.sqrt(disc))/(2*a)
                    if x1 <= 1 and x1 > 0:
                        inter1 = lerp(p1, p2, x1)
                        intersects.append(inter1)
                        poly_lerps.append([i, x1])
                        #Decide which arcs are inside poly and which are outside
                        if len(intersects) == 1 and p1in and not(p2in):
                            startAt = 0
                    #Intersect 2
                    x2 = (-b + math.sqrt(disc))/(2*a)
                    if x2 < 1 and x2 >= 0:
                        inter2 = lerp(p1, p2, x2)
                        intersects.append(inter2)
                        poly_lerps.append([i, x2])
                        #Decide which arcs are inside poly and which are outside
                        if len(intersects) == 1 and p1in and not(p2in):
                            startAt = 0
                        
            #If not 0 mod 2, circle intersects a corner, perturb circle and rerun
            #TODO: how many times does this trigger? Is there a more elegant solution?
            if len(intersects) % 2 == 1:
                center = [center[0]+adjustcorneramount, center[1]+adjustcorneramount]
            else:
                notmod = False

        #Adjust based on startAt
        if startAt == 1 and len(intersects) > 0:
            intersects = intersects[1:] + [intersects[0]]
            poly_lerps = poly_lerps[1:] + [poly_lerps[0]]

        #If radius is negative, reverse arcpoints
        if radius < 0:
            intersects.reverse()
            poly_lerps.reverse()

        return list(zip(intersects, poly_lerps))

    def pointInPoly(self, p):
        return pointInPoly(p, self.points)
        
    #Returns list of intersections of [linear, arc] facet with polygon, represented as facets
    #Returned in same pLeft, pRight orientation as facet
    def intersectFacet(self, facet):
        coinciding_distance_threshold = 1e-10
        coinciding_lerp_threshold = 1e-8
        ignore_facet_threshold = 1e-10
        intersectedFacets = []
        poly_lerps = []
        facet_lerps = []
        if facet.name == 'linear':
            intersectPoints = self.getPolyLineIntersects(facet.pLeft, facet.pRight)
            #Cartesian, polygon lerp, line lerp
            """
            for i in range(len(intersectPoints)//2):
                line_lerp1 = intersectPoints[2*i][2]
                line_lerp2 = intersectPoints[2*i+1][2]
                #left point
                if line_lerp1 < coinciding_lerp_threshold:
                    newfacet_pLeft = (facet.pLeft, None, 0)
                elif coinciding_lerp_threshold <= line_lerp1 and line_lerp1 < 1-coinciding_lerp_threshold:
                    newfacet_pLeft = intersectPoints[2*i]
                else:
                    #left point is beyond facet pRight: no more valid intersects
                    break
                #right point
                if coinciding_lerp_threshold < line_lerp2 and line_lerp2 <= 1-coinciding_lerp_threshold:
                    newfacet_pRight = intersectPoints[2*i+1]
                elif 1-coinciding_lerp_threshold < line_lerp2:
                    newfacet_pRight = (facet.pRight, None, 1)
                else:
                    #right point is beyond facet pLeft: this pair of intersects is not valid
                    continue
                if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                    newfacet = LinearFacet(newfacet_pLeft[0], newfacet_pRight[0])
                    intersectedFacets.append(newfacet)
                    poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                    facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
            """
            for i in range(len(intersectPoints)//2):
                cart1 = intersectPoints[2*i][0]
                cart2 = intersectPoints[2*i+1][0]
                line_lerp1 = intersectPoints[2*i][2]
                line_lerp2 = intersectPoints[2*i+1][2]
                #left point
                if getDistance(cart1, facet.pLeft) < coinciding_distance_threshold:
                    newfacet_pLeft = intersectPoints[2*i]
                elif line_lerp1 < 0:
                    newfacet_pLeft = (facet.pLeft, None, 0)
                elif 0 <= line_lerp1 and line_lerp1 < 1:
                    newfacet_pLeft = intersectPoints[2*i]
                else:
                    #left point is beyond facet pRight: no more valid intersects
                    break
                #right point
                if getDistance(cart2, facet.pRight) < coinciding_distance_threshold:
                    newfacet_pRight = intersectPoints[2*i+1]
                elif line_lerp2 > 1:
                    newfacet_pRight = (facet.pRight, None, 1)
                elif 0 < line_lerp2 and line_lerp2 <= 1:
                    newfacet_pRight = intersectPoints[2*i+1]
                else:
                    #right point is beyond facet pLeft: this pair of intersects is not valid
                    continue
                if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                    newfacet = LinearFacet(newfacet_pLeft[0], newfacet_pRight[0])
                    intersectedFacets.append(newfacet)
                    poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                    facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])

            return intersectedFacets, poly_lerps, facet_lerps

        elif facet.name == 'arc':
            #TODO: the code assumes that the arc is a minor arc. can add check and code for handling major arcs.
            intersectPoints = self.getPolyCircleIntersects(facet.center, facet.radius)
            #print(intersectPoints)
            def circleDistance(p):
                if not(pointLeftOfLine(p, facet.pLeft, facet.center)) == (facet.radius > 0):
                    return getDistance(p, facet.pLeft)
                else:
                    return -getDistance(p, facet.pLeft) + 4*facet.radius
            #Case: if entire circle lies in polygon (no intersects), then facet itself is intersection
            if len(intersectPoints) == 0 and self.pointInPoly(facet.pLeft) and self.pointInPoly(facet.pRight):
                case = 0
                newfacet_pLeft = (facet.pLeft, None, 0)
                newfacet_pRight = (facet.pRight, None, circleDistance(facet.pRight))
                if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                    newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                    intersectedFacets.append(newfacet)
                    poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                    facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
            #Cartesian, polygon lerp
            for i in range(len(intersectPoints)//2):
                cart1 = intersectPoints[2*i][0]
                cart2 = intersectPoints[2*i+1][0]
                testfacet = ArcFacet(facet.center, facet.radius, cart1, cart2)
                #case: [cart1, cart2] = [facet.pLeft, facet.pRight]
                if getDistance(cart1, facet.pLeft) < coinciding_distance_threshold and getDistance(cart2, facet.pRight) < coinciding_distance_threshold:
                    case = 1
                    newfacet_pLeft = (cart1, intersectPoints[2*i][1], 0)
                    newfacet_pRight = (cart2, intersectPoints[2*i+1][1], circleDistance(cart2))
                    if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                        newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                        intersectedFacets.append(newfacet)
                        poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                        facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                elif getDistance(cart1, facet.pLeft) < coinciding_distance_threshold:
                    case = 2
                    newfacet_pLeft = (cart1, intersectPoints[2*i][1], 0)
                    if testfacet.pointInArcRange(facet.pRight):
                        newfacet_pRight = (facet.pRight, None, circleDistance(facet.pRight))
                    else:
                        newfacet_pRight = (cart2, intersectPoints[2*i+1][1], circleDistance(cart2))
                    if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                        newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                        intersectedFacets.append(newfacet)
                        poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                        facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                elif getDistance(cart2, facet.pRight) < coinciding_distance_threshold:
                    case = 3
                    if testfacet.pointInArcRange(facet.pLeft):
                        newfacet_pLeft = (facet.pLeft, None, 0)
                    else:
                        newfacet_pLeft = (cart1, intersectPoints[2*i][1], circleDistance(cart1))
                    newfacet_pRight = (cart2, intersectPoints[2*i+1][1], circleDistance(cart2))
                    if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                        newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                        intersectedFacets.append(newfacet)
                        poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                        facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                else:
                    case = 4
                    inrange1 = facet.pointInArcRange(cart1)
                    inrange2 = facet.pointInArcRange(cart2)
                    if inrange1 and not(inrange2):
                        case = 5
                        newfacet_pLeft = (cart1, intersectPoints[2*i][1], circleDistance(cart1))
                        newfacet_pRight = (facet.pRight, None, circleDistance(facet.pRight))
                        if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                            newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                            intersectedFacets.append(newfacet)
                            poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                            facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                    elif not(inrange1) and inrange2:
                        case = 6
                        newfacet_pLeft = (facet.pLeft, None, 0)
                        newfacet_pRight = (cart2, intersectPoints[2*i+1][1], circleDistance(cart2))
                        if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                            newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                            intersectedFacets.append(newfacet)
                            poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                            facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                    elif inrange1 and inrange2:
                        #two cases: check if pLeft, pRight lie within range of cart1, cart2
                        if testfacet.pointInArcRange(facet.pLeft) and testfacet.pointInArcRange(facet.pRight):
                            #two facets
                            #pLeft to cart2
                            newfacet_pLeft = (facet.pLeft, None, 0)
                            newfacet_pRight = (cart2, intersectPoints[2*i+1][1], circleDistance(cart2))
                            if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                                newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                                intersectedFacets.append(newfacet)
                                poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                                facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                            #cart1 to pRight
                            newfacet_pLeft = (cart1, intersectPoints[2*i][1], circleDistance(cart1))
                            newfacet_pRight = (facet.pRight, None, circleDistance(facet.pRight))
                            if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                                newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                                intersectedFacets.append(newfacet)
                                poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                                facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                        else:
                            #one facet
                            newfacet_pLeft = (cart1, intersectPoints[2*i][1], circleDistance(cart1))
                            newfacet_pRight = (cart2, intersectPoints[2*i+1][1], circleDistance(cart2))
                            if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                                newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                                intersectedFacets.append(newfacet)
                                poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                                facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])
                    else: #not(inrange1) and not(inrange2)
                        #check if pLeft, pRight lie within range of cart1, cart2
                        testfacet = ArcFacet(facet.center, facet.radius, cart1, cart2)
                        if testfacet.pointInArcRange(facet.pLeft) and testfacet.pointInArcRange(facet.pRight):
                            newfacet_pLeft = (facet.pLeft, None, 0)
                            newfacet_pRight = (facet.pRight, None, circleDistance(facet.pRight))
                            if getDistance(newfacet_pLeft[0], newfacet_pRight[0]) > ignore_facet_threshold:
                                newfacet = ArcFacet(facet.center, facet.radius, newfacet_pLeft[0], newfacet_pRight[0])
                                intersectedFacets.append(newfacet)
                                poly_lerps.append([newfacet_pLeft[1], newfacet_pRight[1]])
                                facet_lerps.append([newfacet_pLeft[2], newfacet_pRight[2]])

            #sort by distances
            facet_distances = list(map(lambda x : x[0], facet_lerps))
            sorted_facets = [(a,b,c) for _,a,b,c in sorted(zip(facet_distances, intersectedFacets, poly_lerps, facet_lerps))]
            intersectedFacets = list(map(lambda x : x[0], sorted_facets))
            poly_lerps = list(map(lambda x : x[1], sorted_facets))
            facet_lerps = list(map(lambda x : x[2], sorted_facets))
            return intersectedFacets, poly_lerps, facet_lerps

        #print("Original facet: {}".format(facet))
        #print("Advected facets:")
        #for intersectedFacet in intersectedFacets:
        #    print(intersectedFacet)

        #if len(intersectedFacets) > 0:
        #    print(case)

    def intersectStrand(self, strand):
        intersectedStrands = [[]]
        strands_poly_lerps = [[]]
        strands_facet_lerps = [[]]
        facets = strand.facets
        for j in range(len(facets)):
            facet = facets[j]
            intersectedFacets, poly_lerps, facet_lerps = self.intersectFacet(facet)
            """
            print("Original facet:")
            print(facet)
            print("New facets:")
            for intersectedFacet in intersectedFacets:
                print(intersectedFacet)
            print("Quad points:")
            print(self.points)
            """
            for i in range(len(intersectedFacets)):
                #print(intersectedFacets[i])
                intersectedStrands[-1] += [intersectedFacets[i]] #addresses cases when multiple disconnected facets
                strands_poly_lerps[-1] += [poly_lerps[i]]
                strands_facet_lerps[-1] += [facet_lerps[i]]
                #print(poly_lerps[i])
                if poly_lerps[i][1] is not None or (j == len(facets)-1 and i == len(intersectedFacets)-1): #facet ends at intersection with polygon edge
                    intersectedStrands.append([])
                    strands_poly_lerps.append([])
                    strands_facet_lerps.append([])
        #remove the extra empty list at the end of intersectedStrands

        intersectedStrands = list(map(lambda x : Strand(x), intersectedStrands[:-1]))

        return intersectedStrands, strands_poly_lerps[:-1], strands_facet_lerps[:-1]

    def zeroStrands(self):
        self.merged_strands = []
        self.merged_strands_poly_lerps = []
        self.unmerged_strands = []
        self.unmerged_strands_poly_lerps = []

    def gatherStrands(self, strand, collinearity_threshold=1e-10):
        """
        print("gatherStrands: {}, {}".format(self.points, strand))
        print("Original length of self.strands: {}".format(len(self.strands)))
        for astrand in self.strands:
            print(astrand)
        """
        #set up a consistent gathering collection of strands to be updated at next call of updateStrands
        #for each strand, add info about endpoints: inner, or on edge
        #if on edge, represent as linear combination of endpoints
        #merge strands with inner endpoints
        intersectedStrands, intersectedStrands_poly_lerps, _ = self.intersectStrand(strand)

        #Test if strand can be merged with any other strands in self.unmerged_strands
        for initial_merge_strand, initial_merge_poly_lerps in list(zip(intersectedStrands, intersectedStrands_poly_lerps)):
            merge_strand = initial_merge_strand
            merge_poly_lerps = initial_merge_poly_lerps
            #print("Merge strand: {}".format(merge_strand))
            #print("Merge poly lerps: {}".format(merge_poly_lerps))
            merge_strand_pLeft = merge_strand.facets[0].pLeft
            merge_strand_pRight = merge_strand.facets[-1].pRight
            checkedLeft = -1
            checkedRight = -1
            i = 0
            while i < len(self.unmerged_strands) and (merge_poly_lerps[0][0] is None or merge_poly_lerps[-1][1] is None): #not checked all unmerged strands and endpoints are not on poly edges yet
                doCheckRight = True
                original_strand_pLeft = self.unmerged_strands[i].facets[0].pLeft
                original_strand_pRight = self.unmerged_strands[i].facets[-1].pRight
                #checkedLeft case
                if i > checkedLeft:
                    #print(getDistance(merge_strand_pRight, original_strand_pLeft))
                    if self.unmerged_strands_poly_lerps[i][0][0] is None and merge_poly_lerps[-1][1] is None and getDistance(merge_strand_pRight, original_strand_pLeft) < collinearity_threshold:
                        #merge_strand + original_strand

                        #Enforce C0 continuity
                        c0_cont_endpoint = lerp(merge_strand_pRight, original_strand_pLeft, 0.5)
                        last_merge_strand_facet = merge_strand.facets[-1]
                        last_merge_strand_facet = last_merge_strand_facet.update_endpoints(last_merge_strand_facet.pLeft, c0_cont_endpoint)
                        first_original_strand_facet = self.unmerged_strands[i].facets[0]
                        first_original_strand_facet = first_original_strand_facet.update_endpoints(c0_cont_endpoint, first_original_strand_facet.pRight)

                        #Merge
                        newfacets = merge_strand.facets[:-1] + [last_merge_strand_facet, first_original_strand_facet] + self.unmerged_strands[i].facets[1:]
                        self.unmerged_strands.pop(i)
                        merge_poly_lerps = merge_poly_lerps + self.unmerged_strands_poly_lerps[i]
                        merge_strand_pRight = original_strand_pRight
                        self.unmerged_strands_poly_lerps.pop(i)
                        merge_strand = Strand(newfacets)
                        i = -1
                        checkedRight = -1
                        doCheckRight = False
                    else:
                        checkedLeft = i
                #checkedRight case
                if i > checkedRight and doCheckRight:
                    #print(getDistance(merge_strand_pLeft, original_strand_pRight))
                    if self.unmerged_strands_poly_lerps[i][-1][1] is None and merge_poly_lerps[0][0] is None and getDistance(merge_strand_pLeft, original_strand_pRight) < collinearity_threshold:
                        #original_strand + merge_strand

                        #Enforce C0 continuity
                        c0_cont_endpoint = lerp(original_strand_pRight, merge_strand_pLeft, 0.5)
                        first_merge_strand_facet = merge_strand.facets[0]
                        first_merge_strand_facet = first_merge_strand_facet.update_endpoints(c0_cont_endpoint, first_merge_strand_facet.pRight)
                        last_original_strand_facet = self.unmerged_strands[i].facets[-1]
                        last_original_strand_facet = last_original_strand_facet.update_endpoints(last_original_strand_facet.pLeft, c0_cont_endpoint)

                        #Merge
                        newfacets = self.unmerged_strands[i].facets[:-1] + [last_original_strand_facet, first_merge_strand_facet] + merge_strand.facets[1:]
                        self.unmerged_strands.pop(i)
                        merge_poly_lerps = self.unmerged_strands_poly_lerps[i] + merge_poly_lerps
                        merge_strand_pLeft = original_strand_pLeft
                        self.unmerged_strands_poly_lerps.pop(i)
                        merge_strand = Strand(newfacets)
                        i = -1
                        checkedLeft = -1
                    else:
                        checkedRight = i
                i += 1

            if (merge_poly_lerps[0][0] is None or merge_poly_lerps[-1][1] is None): #still unmerged
                #add to unmerged
                self.unmerged_strands.append(merge_strand)
                self.unmerged_strands_poly_lerps.append(merge_poly_lerps)
            else:
                #add to merged
                self.merged_strands.append(merge_strand)
                self.merged_strands_poly_lerps.append(merge_poly_lerps)
        """
        print("New length of self.strands: {}".format(len(self.strands)))
        for astrand in self.strands:
            print(astrand)
        """

    def fixUnmergedStrands(self, collinearity_threshold=1e-6):
        while len(self.unmerged_strands) > 0:
            merge_strand = self.unmerged_strands.pop(0)
            merge_poly_lerps = self.unmerged_strands_poly_lerps.pop(0)
            merge_strand_pLeft = merge_strand.facets[0].pLeft
            merge_strand_pRight = merge_strand.facets[-1].pRight
            checkedLeft = -1
            checkedRight = -1
            i = 0
            while i < len(self.unmerged_strands) and (merge_poly_lerps[0][0] is None or merge_poly_lerps[-1][1] is None): #not checked all unmerged strands and endpoints are not on poly edges yet
                doCheckRight = True
                original_strand_pLeft = self.unmerged_strands[i].facets[0].pLeft
                original_strand_pRight = self.unmerged_strands[i].facets[-1].pRight
                #checkedLeft case
                if i > checkedLeft:
                    #print(getDistance(merge_strand_pRight, original_strand_pLeft))
                    if self.unmerged_strands_poly_lerps[i][0][0] is None and merge_poly_lerps[-1][1] is None and getDistance(merge_strand_pRight, original_strand_pLeft) < collinearity_threshold:
                        #merge_strand + original_strand

                        #Enforce C0 continuity
                        c0_cont_endpoint = lerp(merge_strand_pRight, original_strand_pLeft, 0.5)
                        last_merge_strand_facet = merge_strand.facets[-1]
                        last_merge_strand_facet = last_merge_strand_facet.update_endpoints(last_merge_strand_facet.pLeft, c0_cont_endpoint)
                        first_original_strand_facet = self.unmerged_strands[i].facets[0]
                        first_original_strand_facet = first_original_strand_facet.update_endpoints(c0_cont_endpoint, first_original_strand_facet.pRight)

                        #Merge
                        newfacets = merge_strand.facets[:-1] + [last_merge_strand_facet, first_original_strand_facet] + self.unmerged_strands[i].facets[1:]
                        self.unmerged_strands.pop(i)
                        merge_poly_lerps = merge_poly_lerps + self.unmerged_strands_poly_lerps[i]
                        merge_strand_pRight = original_strand_pRight
                        self.unmerged_strands_poly_lerps.pop(i)
                        merge_strand = Strand(newfacets)
                        i = -1
                        checkedRight = -1
                        doCheckRight = False
                    else:
                        checkedLeft = i
                #checkedRight case
                if i > checkedRight and doCheckRight:
                    #print(getDistance(merge_strand_pLeft, original_strand_pRight))
                    if self.unmerged_strands_poly_lerps[i][-1][1] is None and merge_poly_lerps[0][0] is None and getDistance(merge_strand_pLeft, original_strand_pRight) < collinearity_threshold:
                        #original_strand + merge_strand

                        #Enforce C0 continuity
                        c0_cont_endpoint = lerp(original_strand_pRight, merge_strand_pLeft, 0.5)
                        first_merge_strand_facet = merge_strand.facets[0]
                        first_merge_strand_facet = first_merge_strand_facet.update_endpoints(c0_cont_endpoint, first_merge_strand_facet.pRight)
                        last_original_strand_facet = self.unmerged_strands[i].facets[-1]
                        last_original_strand_facet = last_original_strand_facet.update_endpoints(last_original_strand_facet.pLeft, c0_cont_endpoint)

                        #Merge
                        newfacets = self.unmerged_strands[i].facets[:-1] + [last_original_strand_facet, first_merge_strand_facet] + merge_strand.facets[1:]
                        self.unmerged_strands.pop(i)
                        merge_poly_lerps = self.unmerged_strands_poly_lerps[i] + merge_poly_lerps
                        merge_strand_pLeft = original_strand_pLeft
                        self.unmerged_strands_poly_lerps.pop(i)
                        merge_strand = Strand(newfacets)
                        i = -1
                        checkedLeft = -1
                    else:
                        checkedRight = i
                i += 1

            if (merge_poly_lerps[0][0] is not None and merge_poly_lerps[-1][1] is not None):
                #add to merged
                self.merged_strands.append(merge_strand)
                self.merged_strands_poly_lerps.append(merge_poly_lerps)

    def updateStrands(self, merge_all=False):
        #use gathered info about strands from gatherStrands to update strands
        #test for all inner endpoints are dealt with?
        #format on edge strand endpoints in preparation for getArea

        #Try to merge unmerged strands using a higher collinearity threshold (1e-6)
        if len(self.unmerged_strands) > 1:
            self.fixUnmergedStrands()

        self.strands_sorted = [[] for _ in range(len(self.points))]
        self.strands_poly_lerps_pLefts = [[] for _ in range(len(self.points))]
        self.strands_poly_lerps_pRights = [[] for _ in range(len(self.points))]
        self.strands_stored = []
        for i in range(len(self.merged_strands)):
            strand = self.merged_strands[i]
            strand_poly_lerps = self.merged_strands_poly_lerps[i]
            if strand_poly_lerps[0][0] is not None: #pLeft of first facet is a polygon intersect. format: [vertex number, lerp]
                mergedStrand = strand.mergedStrand(merge_all=merge_all)
                strand_poly_lerp_pLeft = strand_poly_lerps[0][0] #pLeft of first facet
                strand_poly_lerp_pRight = strand_poly_lerps[-1][-1] #pRight of last facet
                if strand_poly_lerp_pRight is None:
                    print("Issue in updateStrands: no strand_poly_lerp_pRight")
                    print(self.points)
                    print(strand)
                    print(strand_poly_lerps)
                
                self.strands_sorted[strand_poly_lerp_pLeft[0]].append(mergedStrand)
                self.strands_poly_lerps_pLefts[strand_poly_lerp_pLeft[0]].append(strand_poly_lerp_pLeft[1])
                self.strands_poly_lerps_pRights[strand_poly_lerp_pLeft[0]].append(strand_poly_lerp_pRight)
        #Sort strands based on quad lerp
        for i in range(len(self.points)):
            sort_helper = np.argsort(self.strands_poly_lerps_pLefts[i])
            self.strands_sorted[i] = list(map(lambda x : self.strands_sorted[i][x], sort_helper))
            self.strands_poly_lerps_pLefts[i] = list(map(lambda x : self.strands_poly_lerps_pLefts[i][x], sort_helper))
            self.strands_poly_lerps_pRights[i] = list(map(lambda x : self.strands_poly_lerps_pRights[i][x], sort_helper))
            self.strands_stored += self.strands_sorted[i]
            """
            try:
                sort_helper = sorted(zip(self.strands_poly_lerps_sorted[i], self.strands_sorted[i]))
            except:
                print("Failure to sort")
                for strand in self.strands:
                    print(strand)
                print(self.strands_poly_lerps)
                print(1/0)
            self.strands_sorted[i] = [x for _,x in sort_helper]
            self.strands_poly_lerps_sorted[i] = [x for x,_ in sort_helper]
            self.strands_stored += self.strands_sorted[i]
            """

    def getArea(self):
        helper_strands_sorted = deepcopy(self.strands_sorted)
        helper_strands_poly_lerps_pLefts = deepcopy(self.strands_poly_lerps_pLefts)
        helper_strands_poly_lerps_pRights = deepcopy(self.strands_poly_lerps_pRights)

        #print(helper_strands_sorted)
        #print(helper_strands_poly_lerps_pLefts)
        #print(helper_strands_poly_lerps_pRights)
        
        def getFirstStrand():
            i = 0
            while i < len(helper_strands_sorted):
                if len(helper_strands_sorted[i]) == 0:
                    i += 1
                else:
                    return i
            return None
        
        ret = 0

        begin_i = getFirstStrand()

        while begin_i is not None:
            areapoly = []

            #Begin strand
            first_strand = helper_strands_sorted[begin_i].pop(0)
            areapoly.append(first_strand.pLeft)
            areapoly.append(first_strand.pRight)
            ret += first_strand.area
            first_strand_poly_lerp_pLeft = helper_strands_poly_lerps_pLefts[begin_i].pop(0)
            first_strand_poly_lerp_pRight = helper_strands_poly_lerps_pRights[begin_i].pop(0)
            tracing_i = first_strand_poly_lerp_pRight[0]

            begin_pLeft = first_strand_poly_lerp_pLeft

            poly_done = False
            while not(poly_done):
                j = 0
                addCornerPoint = True
                while j < len(helper_strands_sorted[tracing_i]):
                    if helper_strands_poly_lerps_pLefts[tracing_i][j] > first_strand_poly_lerp_pRight[1]:
                        #Valid next strand
                        first_strand = helper_strands_sorted[tracing_i].pop(j)
                        areapoly.append(first_strand.pLeft)
                        areapoly.append(first_strand.pRight)
                        ret += first_strand.area
                        first_strand_poly_lerp_pLeft = helper_strands_poly_lerps_pLefts[tracing_i].pop(j)
                        first_strand_poly_lerp_pRight = helper_strands_poly_lerps_pRights[tracing_i].pop(j)
                        tracing_i = first_strand_poly_lerp_pRight[0]
                        addCornerPoint = False
                        break
                    else:
                        j += 1
                if addCornerPoint:
                    next_tracing_i = (tracing_i+1) % len(self.points)
                    areapoly.append(self.points[next_tracing_i])
                    tracing_i = next_tracing_i
                    if tracing_i == begin_i:
                        ret += getArea(areapoly)
                        poly_done = True
                elif not(addCornerPoint) and tracing_i == begin_i and first_strand_poly_lerp_pLeft < begin_pLeft:
                    ret += getArea(areapoly)
                    poly_done = True

            begin_i = getFirstStrand()

            #print(areapoly)
                
        return ret/getArea(self.points)

def getStrandArea(facets, degenFacetThreshold=1e-10):
    area = 0
    points = []

    for facet in facets:
        points.append(facet.pLeft)
        if facet.name == 'arc' and getDistance(facet.pLeft, facet.pRight) > degenFacetThreshold:
            area += getArcArea(facet.pLeft, facet.pRight, facet.center, facet.radius)
    points.append(facets[-1].pRight)

    area += getArea(points)
    return area

class Strand:
    #facets is a list of Facets
    def __init__(self, facets):
        assert len(facets) > 0, "Empty list of facets"
        self.degenFacetThreshold = 1e-10

        self.facets = facets
        self.pLeft = facets[0].pLeft
        self.pRight = facets[-1].pRight
        
        self.area = getStrandArea(self.facets, self.degenFacetThreshold)

    def __str__(self):
        return str(list(map(lambda x : str(x), self.facets)))

    def advectedStrand(self, velocity, t, h, mode='RK4'):
        advectedFacets = []
        for facet in self.facets:
            advectedFacets.append(facet.advected(velocity, t, h, mode))
        return Strand(advectedFacets)

    #min_area from epsilon and maxRadius from matchArcArea: TODO reconcile
    def mergedStrand(self, normalThreshold=0.8, epsilon=1e-10, merge_all=False):
        if merge_all:
            merged_r = matchArcArea(getDistance(self.pLeft, self.pRight), self.area, epsilon)
            if abs(merged_r) == float('inf'):
                #Linearize
                merged_facet = LinearFacet(self.pLeft, self.pRight)
            else:
                merged_center = getCenter(self.pLeft, self.pRight, merged_r)
                merged_facet = ArcFacet(merged_center, merged_r, self.pLeft, self.pRight)
            return Strand([merged_facet])

        else:
            #Find the substrands to merge based on normal threshold
            merge_substrands = [[]]
            for i in range(len(self.facets)-1):
                facet1 = self.facets[i]
                normal1 = getNormal(facet1, facet1.pRight)
                merge_substrands[-1].append(facet1)

                facet2 = self.facets[i+1]
                normal2 = getNormal(facet2, facet2.pRight)
                cosine_similarity = normal1[0]*normal2[0]+normal1[1]*normal2[1]
                if cosine_similarity < normalThreshold and not(isDegenFacet(facet1, self.degenFacetThreshold) or isDegenFacet(facet2, self.degenFacetThreshold)):
                    #The cosine similarity is too low, end substrand and create a new one
                    merge_substrands.append([])
            merge_substrands[-1].append(self.facets[-1])
            
            #Merge substrands
            merged_facets = []
            for merge_substrand in merge_substrands:
                substrand_pLeft = merge_substrand[0].pLeft
                getUnmergedNormalLeft = getNormal(merge_substrand[0], substrand_pLeft)
                substrand_pRight = merge_substrand[-1].pRight
                getUnmergedNormalRight = getNormal(merge_substrand[-1], substrand_pRight)
                try:
                    merged_r = matchArcArea(getDistance(substrand_pLeft, substrand_pRight), getStrandArea(merge_substrand), epsilon)
                except:
                    print("matchArcArea({}, {}, {})".format(getDistance(substrand_pLeft, substrand_pRight), getStrandArea(merge_substrand), epsilon))
                    print(self)
                    print(1/0)
                if abs(merged_r) == float('inf'):
                    #Linearize
                    merged_facet = LinearFacet(substrand_pLeft, substrand_pRight)
                else:
                    merged_center = getCenter(substrand_pLeft, substrand_pRight, merged_r)
                    merged_facet = ArcFacet(merged_center, merged_r, substrand_pLeft, substrand_pRight)

                    getMergedNormalLeft = getNormal(merged_facet, substrand_pLeft)
                    getMergedNormalRight = getNormal(merged_facet, substrand_pRight)

                    cosine_similarityLeft = getUnmergedNormalLeft[0]*getMergedNormalLeft[0]+getUnmergedNormalLeft[1]*getMergedNormalLeft[1]
                    cosine_similarityRight = getUnmergedNormalRight[0]*getUnmergedNormalRight[0]+getUnmergedNormalRight[1]*getUnmergedNormalRight[1]
                    #if cosine_similarityLeft < 0.999 or cosine_similarityRight < 0.999:
                    #    print("Left cosine similarity: {}".format(cosine_similarityLeft))
                    #    print("Right cosine similarity: {}".format(cosine_similarityRight))
                    if cosine_similarityLeft < 0.995 or cosine_similarityRight < 0.995:
                        for merge_substrand_facet in merge_substrand[:-1]:
                            merged_facets.append(merge_substrand_facet)
                        merged_facet = merge_substrand[-1]

                    """
                    #Attempt to add regularization by checking how far the interior facet vertices lie from the merged arc: doesn't work
                    max_off_threshold = 2e-4
                    max_off = 0
                    for merge_substrand_facet in merge_substrand[:-2]:
                        max_off = max(max_off, abs(getDistance(merged_center, merge_substrand_facet.pRight) - abs(merged_r)))
                    if max_off > max_off_threshold:
                        for merge_substrand_facet in merge_substrand[:-1]:
                            merged_facets.append(merge_substrand_facet)
                        merged_facet = merge_substrand[-1]
                    """

                merged_facets.append(merged_facet)

            return Strand(merged_facets)