from main.geoms.circular_facet import getCircumcircle
from main.geoms.geoms import getDistance, lerp, pointRightOfLine
from main.structs.facets.base_facet import Facet, advectPoint


class LinearFacet(Facet):

    def __init__(self, pLeft, pRight, name="linear"):
        super().__init__(name, pLeft, pRight)
        self.curvature = 0
        self.midpoint = lerp(self.pLeft, self.pRight, 0.5)

    def advected(self, velocity, t, h, mode="RK4"):
        # Advect by using 3 control points: 2 endpoints + midpoint
        # Returns either LinearFacet or ArcFacet
        shiftpLeft = advectPoint(self.pLeft, velocity, t, h, mode=mode)
        shiftpRight = advectPoint(self.pRight, velocity, t, h, mode=mode)
        shiftmid = advectPoint(self.midpoint, velocity, t, h, mode=mode)
        [shiftcirclecenter, shiftcircleradius] = getCircumcircle(
            shiftpLeft, shiftmid, shiftpRight
        )
        if (
            shiftcircleradius is None
            or shiftcircleradius > self.advect_collinearity_threshold
        ):
            # Collinear: advect by creating linear facet
            return LinearFacet(shiftpLeft, shiftpRight)
        else:
            # Not collinear: advect by creating arc facet
            if pointRightOfLine(shiftcirclecenter, shiftpLeft, shiftpRight):
                # Circumcenter on right of line, need to invert radius
                shiftcircleradius *= -1

            # if getDistance(shiftpLeft, shiftpRight) < 1e-10:
            #    print("Mini facet: {}".format(self))
            from main.structs.facets.circular_facet import ArcFacet

            return ArcFacet(shiftcirclecenter, shiftcircleradius, shiftpLeft, shiftpRight)

    def update_endpoints(self, new_pLeft, new_pRight):
        return LinearFacet(new_pLeft, new_pRight)

    def getTangent(self, p):
        # TODO doesn't check if p is on line
        return [self.pRight[0] - self.pLeft[0], self.pRight[1] - self.pLeft[1]]

    def getLeftTangent(self):
        return self.getTangent(self.pLeft)

    def getRightTangent(self):
        return self.getTangent(self.pRight)

    def __str__(self):
        return "['{}', [{}, {}]]".format(self.name, self.pLeft, self.pRight)

    def __eq__(self, other_facet):
        if (
            isinstance(other_facet, self.__class__)
            and getDistance(other_facet.pLeft, self.pLeft) < self.equality_threshold
            and getDistance(other_facet.pRight, self.pRight) < self.equality_threshold
        ):
            return True
        return False

    def sample(self, n: int, mode: str = "arclength"):
        if n <= 1:
            return [self.pLeft]
        points = []
        for i in range(n):
            t = i / (n - 1)
            points.append(lerp(self.pLeft, self.pRight, t))
        return points
