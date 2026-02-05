from main.structs.facets.base_facet import Facet, advectPoint
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


class CornerFacet(Facet):

    def __init__(
        self, centerLeft, centerRight, radiusLeft, radiusRight, pLeft, corner, pRight
    ):
        super().__init__("corner", pLeft, pRight)
        self.centerLeft = centerLeft
        self.centerRight = centerRight
        self.radiusLeft = radiusLeft
        self.radiusRight = radiusRight
        self.corner = corner

        # Left facet
        if self.centerLeft is None and self.radiusLeft is None:
            self.facetLeft = LinearFacet(pLeft=pLeft, pRight=corner)
        else:
            self.facetLeft = ArcFacet(
                center=centerLeft, radius=radiusLeft, pLeft=pLeft, pRight=corner
            )
        # Right facet
        if self.centerRight is None and self.radiusRight is None:
            self.facetRight = LinearFacet(pLeft=corner, pRight=pRight)
        else:
            self.facetRight = ArcFacet(
                center=centerRight, radius=radiusRight, pLeft=corner, pRight=pRight
            )

    def advected(self, velocity, t, h, mode="RK4"):
        # Advect each side as LinearFacet or ArcFacet
        # Left facet
        advectedFacetLeft = self.facetLeft.advected(
            velocity=velocity, t=t, h=h, mode=mode
        )
        if isinstance(advectedFacetLeft, LinearFacet):
            advectedCenterLeft = None
            advectedRadiusLeft = None
        elif isinstance(advectedFacetLeft, ArcFacet):
            advectedCenterLeft = advectedFacetLeft.center
            advectedRadiusLeft = advectedFacetLeft.radius
        else:
            print("Issue in corner advection: left facet")
            print(1 / 0)
        # Right facet
        advectedFacetRight = self.facetRight.advected(
            velocity=velocity, t=t, h=h, mode=mode
        )
        if isinstance(advectedFacetRight, LinearFacet):
            advectedCenterRight = None
            advectedRadiusRight = None
        elif isinstance(advectedFacetRight, ArcFacet):
            advectedCenterRight = advectedFacetRight.center
            advectedRadiusRight = advectedFacetRight.radius
        else:
            print("Issue in corner advection: right facet")
            print(1 / 0)

        # Returns CornerFacet
        assert advectedFacetLeft.pRight == advectedFacetRight.pLeft
        return CornerFacet(
            centerLeft=advectedCenterLeft,
            centerRight=advectedCenterRight,
            radiusLeft=advectedRadiusLeft,
            radiusRight=advectedRadiusRight,
            pLeft=advectedFacetLeft.pLeft,
            corner=advectedFacetLeft.pRight,
            pRight=advectedFacetRight.pRight,
        )

        """
        shiftpLeft = advectPoint(self.pLeft, velocity, t, h, mode=mode)
        shiftcorner = advectPoint(self.corner, velocity, t, h, mode=mode)
        shiftpRight = advectPoint(self.pRight, velocity, t, h, mode=mode)
        midleft = lerp(self.pLeft, self.corner, 0.5)
        if self.radiusLeft is not None:
            midleftconstant = math.sqrt((midleft[0]-self.centerLeft[0])**2 + (midleft[1]-self.centerLeft[1])**2)
            midleft = [self.centerLeft[0] + self.radiusLeft/midleftconstant*(midleft[0]-self.centerLeft[0]), self.centerLeft[1] + self.radiusLeft/midleftconstant*(midleft[1]-self.centerLeft[1])]
            #If the circular facet is a major arc
            if ((self.centerLeft[0]-self.pLeft[0])*(midleft[1]-self.pLeft[1])-(self.centerLeft[1]-self.pLeft[1])*(midleft[0]-self.pLeft[0]))*self.radiusLeft > 0:
                midleft = lerp(midleft, self.centerLeft, 2)
        shiftmidleft = advectPoint(midleft, velocity, t, h, mode=mode)
        midright = lerp(self.corner, self.pRight, 0.5)
        if self.radiusRight is not None:
            midrightconstant = math.sqrt((midright[0]-self.centerRight[0])**2 + (midright[1]-self.centerRight[1])**2)
            midright = [self.centerRight[0] + self.radiusRight/midrightconstant*(midright[0]-self.centerRight[0]), self.centerRight[1] + self.radiusRight/midrightconstant*(midright[1]-self.centerRight[1])]
            #If the circular facet is a major arc
            if ((self.centerRight[0]-self.corner[0])*(midright[1]-self.corner[1])-(self.centerRight[1]-self.corner[1])*(midright[0]-self.corner[0]))*self.radiusRight > 0:
                midright = lerp(midright, self.centerRight, 2)
        shiftmidright = advectPoint(midright, velocity, t, h, mode=mode)

        [shiftcirclecenterleft, shiftcircleradiusleft] = getCircumcircle(shiftpLeft, shiftmidleft, shiftcorner)
        if shiftcircleradiusleft is None or shiftcircleradiusleft > self.advect_collinearity_threshold:
            #Collinear: treat left edge as linear facet
            shiftcirclecenterleft = None
            shiftcircleradiusleft = None
        else:
            #Not collinear: treat left edge as arc facet
            if (shiftcirclecenterleft[0]-shiftpLeft[0])*(-(shiftcorner[1]-shiftpLeft[1])) + (shiftcirclecenterleft[1]-shiftpLeft[1])*(shiftcorner[0]-shiftpLeft[0]) < 0:
                #Circumcenter on right of line, need to invert radius
                shiftcircleradiusleft *= -1
        [shiftcirclecenterright, shiftcircleradiusright] = getCircumcircle(shiftcorner, shiftmidright, shiftpRight)
        if shiftcircleradiusright is None or shiftcircleradiusright > self.advect_collinearity_threshold:
            #Collinear: treat right edge as linear facet
            shiftcirclecenterright = None
            shiftcircleradiusright = None
        else:
            #Not collinear: treat right edge as arc facet
            if (shiftcirclecenterright[0]-shiftcorner[0])*(-(shiftpRight[1]-shiftcorner[1])) + (shiftcirclecenterright[1]-shiftcorner[1])*(shiftpRight[0]-shiftcorner[0]) < 0:
                print(f"How right of line is it? {(shiftcirclecenterright[0]-shiftcorner[0])*(-(shiftpRight[1]-shiftcorner[1])) + (shiftcirclecenterright[1]-shiftcorner[1])*(shiftpRight[0]-shiftcorner[0])}")
                #Circumcenter on right of line, need to invert radius
                shiftcircleradiusright *= -1
                print(shiftpLeft)
                print(shiftcirclecenterright)
                print(shiftcorner)
                print(shiftpRight)
                print(shiftmidright)

        return CornerFacet(shiftcirclecenterleft, shiftcirclecenterright, shiftcircleradiusleft, shiftcircleradiusright, shiftpLeft, shiftcorner, shiftpRight)
        """

    def getLeftTangent(self):
        return self.facetLeft.getLeftTangent()

    def getRightTangent(self):
        return self.facetRight.getRightTangent()

    def __str__(self):
        return "['{}', {}, {}, {}, {}, [{}, {}, {}]]".format(
            self.name,
            self.centerLeft,
            self.centerRight,
            self.radiusLeft,
            self.radiusRight,
            self.pLeft,
            self.corner,
            self.pRight,
        )

    # TODO: define equality for corner?

    def sample(self, n: int, mode: str = "arclength"):
        if n <= 1:
            return [self.pLeft]
        if n == 2:
            return [self.pLeft, self.pRight]

        left_n = n // 2 + 1
        right_n = n - left_n + 1
        if left_n < 2:
            left_n = 2
        if right_n < 2:
            right_n = 2

        left_points = self.facetLeft.sample(left_n, mode=mode)
        right_points = self.facetRight.sample(right_n, mode=mode)

        if left_points and right_points:
            return left_points[:-1] + right_points
        return left_points + right_points
