import math

from main.geoms.circular_facet import (
    getArcArea,
    getCenter,
    getCircleLineIntersects,
    getCircumcircle,
    isMajorArc,
)
from main.geoms.geoms import (
    getArea,
    getDistance,
    lerp,
    pointInPoly,
    pointLeftOfLine,
    pointRightOfLine,
)
from main.structs.facets.base_facet import Facet, advectPoint, point_equality_threshold


class ArcFacet(Facet):

    def __init__(self, center, radius, pLeft, pRight):
        super().__init__("arc", pLeft, pRight)
        self.center = center
        if abs(radius) < getDistance(pLeft, pRight) / 2:
            print(
                "Radius is too small: radius {}, distance {}".format(
                    radius, getDistance(pLeft, pRight)
                )
            )
            if radius < 0:
                self.radius = -getDistance(pLeft, pRight) / 2
            else:
                self.radius = getDistance(pLeft, pRight) / 2
        else:
            self.radius = radius
        if self.radius == 0:
            self.curvature = float("inf")
        else:
            self.curvature = 1 / self.radius
        self.is_major_arc = isMajorArc(self.pLeft, self.pRight, self.center, self.radius)
        self.midpoint = self.getMidpoint()

    # TODO: deal with case when arc is perfectly pi radians? divide by zero issue
    def getMidpoint(self):
        mid = lerp(self.pLeft, self.pRight, 0.5)
        if self.is_major_arc:
            mid = lerp(mid, self.center, 2)
        midconstant = math.sqrt(
            (mid[0] - self.center[0]) ** 2 + (mid[1] - self.center[1]) ** 2
        )
        if midconstant < point_equality_threshold:
            # print("Error in facet.py: midconstant = 0")
            vect = [self.center[0] - self.pLeft[0], self.center[1] - self.pLeft[1]]
            mid = [self.center[0] - vect[1], self.center[1] - vect[0]]
        else:
            mid = [
                self.center[0]
                + abs(self.radius) / midconstant * (mid[0] - self.center[0]),
                self.center[1]
                + abs(self.radius) / midconstant * (mid[1] - self.center[1]),
            ]
        return mid

    # Return list of two ArcFacets
    def splitInTwo(self):
        return [
            ArcFacet(self.center, self.radius, self.pLeft, self.midpoint),
            ArcFacet(self.center, self.radius, self.midpoint, self.pRight),
        ]

    def pointInArcRange(self, p):  # endpoints are in arc range
        if not (self.is_major_arc):
            return (
                getDistance(p, self.pLeft) < point_equality_threshold
                or getDistance(p, self.pRight) < point_equality_threshold
                or (
                    (pointLeftOfLine(p, self.center, self.pLeft) == (self.radius > 0))
                    and (
                        pointRightOfLine(p, self.center, self.pRight)
                        == (self.radius > 0)
                    )
                )
            )
        else:
            return (
                getDistance(p, self.pLeft) < point_equality_threshold
                or getDistance(p, self.pRight) < point_equality_threshold
                or not (
                    (pointRightOfLine(p, self.center, self.pLeft) == (self.radius > 0))
                    and (
                        pointLeftOfLine(p, self.center, self.pRight)
                        == (self.radius > 0)
                    )
                )
            )

    # def getArcFacetPolyIntersectArea(self, poly):

    def getTangent(self, p):
        # TODO doesn't check if p is on arc
        return [self.center[1] - p[1], p[0] - self.center[0]]

    def getLeftTangent(self):
        return self.getTangent(self.pLeft)

    def getRightTangent(self):
        return self.getTangent(self.pRight)

    def advected(self, velocity, t, h, mode="RK4"):
        # Advect by using 3 control points: 2 endpoints + midpoint of arc
        # Returns either LinearFacet or ArcFacet
        shiftpLeft = advectPoint(self.pLeft, velocity, t, h, mode=mode)
        shiftpRight = advectPoint(self.pRight, velocity, t, h, mode=mode)
        shiftmid = advectPoint(self.midpoint, velocity, t, h, mode=mode)
        [shiftcirclecenter, shiftcircleradius] = getCircumcircle(
            shiftpLeft, shiftmid, shiftpRight
        )  # shiftcircleradius is always positive
        if (
            shiftcircleradius is None
            or shiftcircleradius > self.advect_collinearity_threshold
        ):
            # Collinear: advect by creating linear facet
            from main.structs.facets.linear_facet import LinearFacet

            return LinearFacet(shiftpLeft, shiftpRight)
        else:
            # Not collinear: advect by creating arc facet
            if pointLeftOfLine(
                shiftmid, shiftpLeft, shiftpRight
            ):
                shiftcircleradius *= -1

            # if getDistance(shiftpLeft, shiftpRight) < 1e-10:
            #    print("Mini facet: {}".format(self))

            return ArcFacet(shiftcirclecenter, shiftcircleradius, shiftpLeft, shiftpRight)

    # TODO: update_pLeft and update_pRight maintains curvature. Is there a balance between updating center and curvature?
    def update_endpoints(self, new_pLeft, new_pRight):
        if abs(self.radius) < getDistance(new_pLeft, new_pRight) / 2:
            if self.radius < 0:
                self.radius = -getDistance(new_pLeft, new_pRight) / 2
            else:
                self.radius = getDistance(new_pLeft, new_pRight) / 2
            self.center = lerp(new_pLeft, new_pRight, 0.5)
        else:
            self.center = getCenter(new_pLeft, new_pRight, self.radius)
        return ArcFacet(self.center, self.radius, new_pLeft, new_pRight)

    def getPolyIntersectArea(self, poly):  # needs to be fixed TODO

        # Hyperparameter
        adjustcorneramount = 1e-14
        notmod = True
        while notmod:
            startAt = 1
            foundArc = False
            intersectpoints = []
            arcpoints = []
            for i in range(len(poly)):
                curpoint = poly[i]
                nextpoint = poly[(i + 1) % len(poly)]
                curin = getDistance(curpoint, self.center) <= abs(self.radius)
                intersectpoints.append(curpoint)
                lineintersects = getCircleLineIntersects(
                    curpoint, nextpoint, self.center, abs(self.radius)
                )
                for intersect in lineintersects:
                    if self.pointInArcRange(intersect):
                        if len(arcpoints) == 0:
                            if curin:
                                startAt = 0
                            else:
                                # discard current intersectpoints
                                intersectpoints = []
                        intersectpoints.append(intersect)
                        arcpoints.append(intersect)
            # If not 0 mod 2, circle intersects a corner, perturb poly and rerun
            if len(arcpoints) % 2 == 1:
                poly = list(
                    map(
                        lambda x: [
                            x[0] + adjustcorneramount,
                            x[1] + adjustcorneramount,
                        ],
                        poly,
                    )
                )
            else:
                notmod = False

        area = 0
        # Adjust based on startAt
        if startAt == 1 and len(arcpoints) > 0:
            arcpoint1 = arcpoints[0]
            arcpoints.pop(0)
            arcpoints.append(arcpoint1)

        # Sum arc areas
        for i in range(0, len(arcpoints), 2):
            area += getArcArea(arcpoints[i], arcpoints[i + 1], self.center, abs(self.radius))
        area += getArea(intersectpoints)
        print(intersectpoints)

        if len(arcpoints) == 0:
            if pointInPoly(self.center, poly):
                # circle lies entirely inside poly
                return self.radius * self.radius * math.pi
            # circle lies entirely outside poly
            return 0

        if self.radius < 0:
            return getArea(poly) - area
        else:
            return area

    def __str__(self):
        return "['{}', {}, {}, [{}, {}]]".format(
            self.name, self.center, self.radius, self.pLeft, self.pRight
        )

    def __eq__(self, other_facet):
        if (
            isinstance(other_facet, self.__class__)
            and getDistance(other_facet.pLeft, self.pLeft) < self.equality_threshold
            and getDistance(other_facet.pRight, self.pRight) < self.equality_threshold
            and getDistance(other_facet.center, self.center) < self.equality_threshold
            and abs(other_facet.radius - self.radius) < self.equality_threshold
        ):
            return True
        return False

    def sample(self, n: int, mode: str = "arclength"):
        if n <= 1:
            return [self.pLeft]

        def _norm_angle(theta):
            two_pi = 2 * math.pi
            return theta % two_pi

        def _in_ccw_range(start, end, angle):
            if start <= end:
                return start <= angle <= end
            return angle >= start or angle <= end

        start_angle = math.atan2(
            self.pLeft[1] - self.center[1], self.pLeft[0] - self.center[0]
        )
        end_angle = math.atan2(
            self.pRight[1] - self.center[1], self.pRight[0] - self.center[0]
        )
        mid_angle = math.atan2(
            self.midpoint[1] - self.center[1], self.midpoint[0] - self.center[0]
        )

        start_n = _norm_angle(start_angle)
        end_n = _norm_angle(end_angle)
        mid_n = _norm_angle(mid_angle)

        two_pi = 2 * math.pi
        delta_ccw = (end_n - start_n) % two_pi
        if delta_ccw == 0:
            return [self.pLeft for _ in range(n)]

        if _in_ccw_range(start_n, end_n, mid_n):
            delta = delta_ccw
        else:
            delta = delta_ccw - two_pi

        points = []
        for i in range(n):
            t = i / (n - 1)
            angle = start_n + t * delta
            points.append(
                [
                    self.center[0] + abs(self.radius) * math.cos(angle),
                    self.center[1] + abs(self.radius) * math.sin(angle),
                ]
            )
        return points
