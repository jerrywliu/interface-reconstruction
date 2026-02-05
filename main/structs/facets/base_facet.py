from main.geoms.geoms import getDistance

point_equality_threshold = 1e-13

# Linear facet: ['linear', intersects]
# Corner facet: ['corner', intersects]
# Arc facet: ['arc', arccenter, arcradius, arcintersects]
# Curved corner facet: ['curvedcorner, prevcenter, nextcenter, prevradius, nextradius, intersects]


# Velocity should be a function of time and position: v(t, p)
def advectPoint(p, velocity, t, h, mode="RK4"):
    if mode == "RK1":
        v = velocity(t, p)
        pfinal = [p[0] + h * v[0], p[1] + h * v[1]]
    elif mode == "RK4":
        v1 = velocity(t, p)
        p2 = [p[0] + h / 2 * v1[0], p[1] + h / 2 * v1[1]]
        v2 = velocity(t + h / 2, p2)
        p3 = [p[0] + h / 2 * v2[0], p[1] + h / 2 * v2[1]]
        v3 = velocity(t + h / 2, p3)
        p4 = [p[0] + h * v3[0], p[1] + h * v3[1]]
        v4 = velocity(t + h, p4)

        pfinal = [
            p[0] + h / 6 * (v1[0] + 2 * v2[0] + 2 * v3[0] + v4[0]),
            p[1] + h / 6 * (v1[1] + 2 * v2[1] + 2 * v3[1] + v4[1]),
        ]
    return pfinal


# TODO move to linear and arc facet classes
def getNormal(facet, p):
    from main.structs.facets.circular_facet import ArcFacet
    from main.structs.facets.linear_facet import LinearFacet

    if isinstance(facet, ArcFacet):
        if getDistance(p, facet.center) == 0:
            print("getNormal({}, {})".format(facet, p))
        if facet.radius > 0:
            normal = [
                (p[0] - facet.center[0]) / getDistance(p, facet.center),
                (p[1] - facet.center[1]) / getDistance(p, facet.center),
            ]
        else:
            normal = [
                (facet.center[0] - p[0]) / getDistance(p, facet.center),
                (facet.center[1] - p[1]) / getDistance(p, facet.center),
            ]
        return normal
    # TODO 12/29/24: assumes everything else is linear. Need to fix when corner facets are implemented and when renaming.
    elif isinstance(facet, LinearFacet):
        if getDistance(facet.pLeft, facet.pRight) == 0:
            print("getNormal({}, {})".format(facet, p))
        tangent = [
            (facet.pRight[0] - facet.pLeft[0]) / getDistance(facet.pLeft, facet.pRight),
            (facet.pRight[1] - facet.pLeft[1]) / getDistance(facet.pLeft, facet.pRight),
        ]
        normal = [tangent[1], -tangent[0]]
        return normal
    # # TODO: corner facets?
    else:
        print("Improper facet in call to getNormal")
        print("getNormal({}, {})".format(facet, p))
        return None


def isDegenFacet(facet, threshold):
    if facet.name in ["linear", "arc"]:
        return getDistance(facet.pLeft, facet.pRight) < threshold
    else:
        print("Improper facet in isDegenFacet")
        print("isDegenFacet({}, {})".format(facet, threshold))


class Facet:

    def __init__(self, name, pLeft, pRight):
        self.name = name
        self.pLeft = pLeft
        self.pRight = pRight

        # If advected facet has radius larger than this, convert it to a line
        self.advect_collinearity_threshold = 1e6
        # If values are within this threshold, call them equal
        self.equality_threshold = 1e-6

    def __str__(self):
        return f"{self.name}"

    def advected(self, velocity):
        pass

    def getLeftTangent(self):
        pass

    def getRightTangent(self):
        pass

    def sample(self, n: int, mode: str = "arclength"):
        raise NotImplementedError("sample is not implemented for base Facet")


def __getattr__(name):
    if name == "LinearFacet":
        from main.structs.facets.linear_facet import LinearFacet

        return LinearFacet
    if name == "ArcFacet":
        from main.structs.facets.circular_facet import ArcFacet

        return ArcFacet
    if name == "CornerFacet":
        from main.structs.facets.corner_facet import CornerFacet

        return CornerFacet
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "Facet",
    "advectPoint",
    "getNormal",
    "isDegenFacet",
    "point_equality_threshold",
    "LinearFacet",
    "ArcFacet",
    "CornerFacet",
]
