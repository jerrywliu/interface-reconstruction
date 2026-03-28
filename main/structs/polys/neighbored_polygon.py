from main.geoms.corner_facet import getPolyCornerArea, getPolyCurvedCornerArea
from main.structs.polys.base_polygon import BasePolygon
from main.geoms.geoms import (
    getArea,
    getCentroid,
    getDistance,
    lineIntersect,
    pointInPoly,
    lineAngleSine,
)
from main.structs.facets.base_facet import Facet
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.geoms.circular_facet import (
    LinearFacetShortcut,
    getArcFacet,
    getCircleLineIntersects,
    getCircleCircleIntersects,
)
from main.geoms.linear_facet import (
    getLinearFacet,
    getPolyLineArea,
    getPolyLineIntersects,
    getLinearFacetFromNormal,
)


class NeighboredPolygon(BasePolygon):

    linearity_threshold = 1e-6  # if area fraction error in linear facet < this value, use linear facet at this cell
    optimization_threshold = 1e-10  # in optimizations
    linear_corner_area_threshold = 1e-4  # if area fraction error in corner facet < this value, use (straight edged) corner facet at this cell
    linear_corner_max_support_extrapolation = 4.0  # reject line-line corners whose apex is too far beyond the supporting facet endpoints
    linear_corner_max_cell_radius_ratio = 4.0  # reject line-line corners whose apex is too remote from the target cell
    corner_sharpness_threshold = 1e-2  # if abs(sine) of angle between corner edges < this value, it's too sharp to use
    curved_corner_curvature_threshold = 1e-2  # if adjacent curvatures > curvaturethreshold, try to fit a curved corner facet
    curved_corner_area_threshold = 1e-2  # if area fraction error in curved corner < curvedcornerthreshold, use curved corner at this cell
    linear_support_overwrite_names = {"default_linear"}
    corner_branch_support_area_threshold = 1e-2

    def __init__(self, points):
        super().__init__(points)

        self.facet_type = None
        self.left_neighbor = None
        self.right_neighbor = None

        # TODO List of unoriented neighbors, maybe useful later?
        self.unoriented_neighbors = []

    def setFacetType(self, facet_type):
        self.facet_type = facet_type

    # poly = NeighboredPolygon
    # orientation = "left", "right", other
    def setNeighbor(self, poly, orientation):
        if orientation == "left":
            self.left_neighbor = poly
        elif orientation == "right":
            self.right_neighbor = poly
        else:
            self.unoriented_neighbors.append(poly)

    # orientation = "left", "right", other
    def clearNeighbor(self, orientation):
        if orientation == "left":
            self.left_neighbor = None
        elif orientation == "right":
            self.right_neighbor = None
        else:
            self.unoriented_neighbors = []

    def hasLeftNeighbor(self):
        return self.left_neighbor is not None

    def hasRightNeighbor(self):
        return self.right_neighbor is not None

    def getLeftNeighbor(self):
        return self.left_neighbor

    def getRightNeighbor(self):
        return self.right_neighbor

    def fullyOriented(self):
        return self.hasLeftNeighbor() and self.hasRightNeighbor()

    # Finds orientation of neighbors based on 3x3 stencil, sets self.left_neighbor and self.right_neighbor for easy cases
    def findSafeOrientation(self, fit_1neighbor=False):
        # Inherit from BasePolygon
        orientation = super().findSafeOrientation(fit_1neighbor=fit_1neighbor)
        if orientation is None:
            return None
        else:
            [self.left_neighbor, self.right_neighbor] = orientation
            return orientation

    def fitCircularFacet(self, root_guess=None):
        # If both neighbors, try linear and circular TODO
        if self.hasLeftNeighbor() and self.hasRightNeighbor():
            root_fallback_args = (
                self.left_neighbor.points,
                self.points,
                self.right_neighbor.points,
                self.left_neighbor.getFraction(),
                self.getFraction(),
                self.right_neighbor.getFraction(),
                NeighboredPolygon.optimization_threshold,
            )
            try:
                facetline1, facetline2 = getLinearFacet(
                    self.left_neighbor.points,
                    self.right_neighbor.points,
                    self.left_neighbor.getFraction(),
                    self.right_neighbor.getFraction(),
                    NeighboredPolygon.optimization_threshold,
                )
            except RuntimeError as error:
                print(
                    f"fitCircularFacet fallback to ELVIRA after getLinearFacet failure: {error}"
                )
                self._set_default_plic_fallback()
                return
            if (
                abs(
                    self.getFraction()
                    - getPolyLineArea(self.points, facetline1, facetline2)
                    / self.getArea()
                )
                < NeighboredPolygon.linearity_threshold
                and (
                    getPolyLineArea(self.points, facetline1, facetline2)
                    / self.getArea()
                    > NeighboredPolygon.optimization_threshold
                )
                and (
                    getPolyLineArea(self.points, facetline1, facetline2)
                    / self.getArea()
                    < 1 - NeighboredPolygon.optimization_threshold
                )
            ):
                intersects = getPolyLineIntersects(self.points, facetline1, facetline2)
                self.setFacet(LinearFacet(intersects[0], intersects[-1]))
            else:
                try:
                    arccenter, arcradius, arcintersects = self._run_arc_fit_with_timeout(
                        self.left_neighbor.points,
                        self.points,
                        self.right_neighbor.points,
                        self.left_neighbor.getFraction(),
                        self.getFraction(),
                        self.right_neighbor.getFraction(),
                        NeighboredPolygon.optimization_threshold,
                    )
                    # TODO If failed: default to linear
                    if arccenter is None or arcradius is None or arcintersects is None:
                        arccenter, arcradius, arcintersects = self._try_arc_fit_root_fallbacks(
                            root_fallback_args, root_guess=root_guess
                        )
                        if arccenter is None or arcradius is None or arcintersects is None:
                            pass
                            # l1 = facetline1
                            # l2 = facetline2
                            # normal = [(-l2[1]+l1[1])/getDistance(l1, l2), (l2[0]-l1[0])/getDistance(l1, l2)]
                            # facetline1, facetline2 = getLinearFacetFromNormal(self.points, self.getFraction(), normal, NeighboredPolygon.optimization_threshold)
                            # self.setFacet(LinearFacet(facetline1, facetline2))
                    if arccenter is not None and arcradius is not None and arcintersects is not None:
                        self.setFacet(
                            ArcFacet(
                                arccenter,
                                arcradius,
                                arcintersects[0],
                                arcintersects[-1],
                            )
                        )
                        print(self)
                except LinearFacetShortcut as shortcut:
                    self.setFacet(LinearFacet(shortcut.pLeft, shortcut.pRight))
                except (RuntimeError, TimeoutError) as error:
                    arccenter, arcradius, arcintersects = self._try_arc_fit_root_fallbacks(
                        root_fallback_args, root_guess=root_guess
                    )
                    if arccenter is not None and arcradius is not None and arcintersects is not None:
                        self.setFacet(
                            ArcFacet(
                                arccenter,
                                arcradius,
                                arcintersects[0],
                                arcintersects[-1],
                            )
                        )
                    else:
                        print(f"fitCircularFacet fallback after getArcFacet failure: {error}")
                except Exception as error:
                    print(
                        f"fitCircularFacet fallback after unexpected getArcFacet failure: {error}"
                    )
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
            try:
                l1, l2 = getLinearFacet(
                    self.left_neighbor.points,
                    self.right_neighbor.points,
                    self.left_neighbor.getFraction(),
                    self.right_neighbor.getFraction(),
                    NeighboredPolygon.optimization_threshold,
                )
            except RuntimeError as error:
                print(
                    f"fitLinearFacet fallback to ELVIRA after getLinearFacet failure: {error}"
                )
                self._set_default_plic_fallback()
                return

            # Check if this facet matches middle area (and actually intersects middle poly in case of nearly empty or full cell)
            isValidLinearFacet = False
            if (
                abs(
                    self.getFraction()
                    - (getPolyLineArea(self.points, l1, l2) / self.getMaxArea())
                )
                < NeighboredPolygon.linearity_threshold
            ):
                intersects = getPolyLineIntersects(self.points, l1, l2)
                if len(intersects) > 0:
                    if doCollinearityCheck:
                        isValidLinearFacet = True
                        self.setFacet(LinearFacet(intersects[0], intersects[-1]))

            if not (isValidLinearFacet) and not (doCollinearityCheck):
                # Use this as normal and fit a linear facet
                normal = [
                    (-l2[1] + l1[1]) / getDistance(l1, l2),
                    (l2[0] - l1[0]) / getDistance(l1, l2),
                ]
                # TODO getLinearFacetFromNormal failed on areafraction=1e-8, threshold=1e-12
                try:
                    l1, l2 = getLinearFacetFromNormal(
                        self.points,
                        self.getFraction(),
                        normal,
                        NeighboredPolygon.optimization_threshold,
                    )
                except:
                    # TODO hack to fix getLinearFacetFromNormal for small areas
                    print(
                        f"getLinearFacetFromNormal({self.points}, {self.getFraction()}, {normal}, {NeighboredPolygon.optimization_threshold})"
                    )
                    l1, l2 = getLinearFacetFromNormal(
                        self.points,
                        self.getFraction() * 1e3,
                        normal,
                        NeighboredPolygon.optimization_threshold,
                    )

                self.setFacet(LinearFacet(l1, l2, name="default_linear"))

        else:
            print("Not enough neighbors: failed to make linear facet")

    def _set_default_plic_fallback(self):
        if self.has3x3Stencil():
            self.setFacet(self.runLVIRA(ret=True))
            return

        left_centroid = getCentroid(self.left_neighbor.points)
        right_centroid = getCentroid(self.right_neighbor.points)
        normal = [
            left_centroid[1] - right_centroid[1],
            right_centroid[0] - left_centroid[0],
        ]
        if abs(normal[0]) + abs(normal[1]) < 1e-14:
            normal = [1.0, 0.0]

        try:
            l1, l2 = getLinearFacetFromNormal(
                self.points,
                self.getFraction(),
                normal,
                NeighboredPolygon.optimization_threshold,
            )
        except:
            adjusted_fraction = min(max(self.getFraction() * 1e3, 0.0), 1.0)
            l1, l2 = getLinearFacetFromNormal(
                self.points,
                adjusted_fraction,
                normal,
                NeighboredPolygon.optimization_threshold,
            )

        self.setFacet(LinearFacet(l1, l2, name="default_linear"))

    # Extends lines l1 to l2, p1 to p2, calculates intersection if it exists, checks whether this corner matches area fraction, updates self.facet if it does
    # (l1, l2, p1, p2) = (l.l, l.r, r.r, r.l)
    def _is_local_linear_corner(self, l1, l2, p1, p2, corner_point):
        left_support_length = getDistance(l1, l2)
        right_support_length = getDistance(p1, p2)
        if left_support_length < 1e-14 or right_support_length < 1e-14:
            return False

        left_branch_length = getDistance(l2, corner_point)
        right_branch_length = getDistance(p2, corner_point)
        if (
            left_branch_length
            > NeighboredPolygon.linear_corner_max_support_extrapolation
            * left_support_length
        ) or (
            right_branch_length
            > NeighboredPolygon.linear_corner_max_support_extrapolation
            * right_support_length
        ):
            return False

        centroid = getCentroid(self.points)
        cell_radius = max(getDistance(centroid, point) for point in self.points)
        if (
            cell_radius > 1e-14
            and getDistance(centroid, corner_point)
            > NeighboredPolygon.linear_corner_max_cell_radius_ratio * cell_radius
        ):
            return False

        return True

    def _linear_facet_from_line(self, l1, l2, name="linear"):
        intersects = getPolyLineIntersects(self.points, l1, l2)
        if len(intersects) < 2:
            return None

        area_fraction = getPolyLineArea(self.points, l1, l2) / self.getMaxArea()
        if abs(area_fraction - self.getFraction()) >= NeighboredPolygon.linear_corner_area_threshold:
            return None

        return LinearFacet(intersects[0], intersects[-1], name=name)

    def _linear_facet_from_normal(self, normal, name="linear"):
        try:
            l1, l2 = getLinearFacetFromNormal(
                self.points,
                self.getFraction(),
                normal,
                NeighboredPolygon.optimization_threshold,
            )
        except Exception:
            return None

        intersects = getPolyLineIntersects(self.points, l1, l2)
        if len(intersects) < 2:
            return None
        return LinearFacet(intersects[0], intersects[-1], name=name)

    def _can_overwrite_with_linear_support(self):
        if not self.hasFacet():
            return True
        facet = self.getFacet()
        if isinstance(facet, ArcFacet):
            return True
        return facet.name in NeighboredPolygon.linear_support_overwrite_names

    @staticmethod
    def _line_area_residual_on_neighbor(branch, neighbor):
        area_fraction = (
            getPolyLineArea(neighbor.points, branch.pLeft, branch.pRight)
            / neighbor.getMaxArea()
        )
        return abs(area_fraction - neighbor.getFraction())

    def bestLinearCornerBranchForNeighbor(self, neighbor):
        if not self.hasFacet() or self.getFacet().name != "corner":
            return None, None

        corner_facet = self.getFacet()
        best_branch = None
        best_error = None
        for branch in [corner_facet.facetLeft, corner_facet.facetRight]:
            if not isinstance(branch, LinearFacet):
                continue
            error = self._line_area_residual_on_neighbor(branch, neighbor)
            if best_error is None or error < best_error:
                best_branch = branch
                best_error = error

        if (
            best_branch is None
            or best_error is None
            or best_error >= NeighboredPolygon.corner_branch_support_area_threshold
        ):
            return None, None
        return best_branch, best_error

    def _try_set_linear_facet_from_line(self, l1, l2, name="linear"):
        facet = self._linear_facet_from_line(l1, l2, name=name)
        if facet is None:
            return False
        self.setFacet(facet)
        return True

    def propagateCornerBranchFacets(self):
        if not self.hasFacet() or self.getFacet().name != "corner":
            return

        corner_facet = self.getFacet()
        left_neighbor = self.getLeftNeighbor()
        right_neighbor = self.getRightNeighbor()

        for neighbor in [left_neighbor, right_neighbor]:
            if neighbor is None or not neighbor._can_overwrite_with_linear_support():
                continue
            branch, _ = self.bestLinearCornerBranchForNeighbor(neighbor)
            if branch is None:
                continue
            normal = [
                -(branch.pRight[1] - branch.pLeft[1]),
                branch.pRight[0] - branch.pLeft[0],
            ]
            propagated_facet = neighbor._linear_facet_from_normal(
                normal, name="corner_branch_linear"
            )
            if propagated_facet is not None:
                neighbor.setFacet(propagated_facet)

    def _build_linear_corner_facet(self, l1, l2, p1, p2):
        corner_point, _, _ = lineIntersect(l1, l2, p1, p2)
        if corner_point is None:
            return None, None
        if not self._is_local_linear_corner(l1, l2, p1, p2, corner_point):
            return None, None

        corner = [l2, corner_point, p2]
        corner_area_fraction = (
            getPolyCornerArea(self.points, corner[0], corner[1], corner[2])
            / self.getMaxArea()
        )
        corner_area_error = abs(corner_area_fraction - self.getFraction())
        if corner_area_error >= NeighboredPolygon.linear_corner_area_threshold:
            return None, corner_area_error

        intersects1 = getPolyLineIntersects(self.points, corner[0], corner[1])
        intersects2 = getPolyLineIntersects(self.points, corner[1], corner[2])
        if not (
            len(intersects1) > 0
            and len(intersects2) > 0
            and abs(lineAngleSine(l1, l2, p1, p2))
            > NeighboredPolygon.corner_sharpness_threshold
        ):
            return None, corner_area_error

        return (
            CornerFacet(None, None, None, None, corner[0], corner[1], corner[2]),
            corner_area_error,
        )

    def checkCornerFacet(self, l1, l2, p1, p2):
        corner_facet, _ = self._build_linear_corner_facet(l1, l2, p1, p2)
        if corner_facet is None:
            return
        self.setFacet(corner_facet)
        self.propagateCornerBranchFacets()
        # print("checkCornerFacet formed corner:")
        # print(self)
        # print("Target area fraction: {}".format(self.getFraction()))

    # Facet 1: l1, l2, center, radius
    # Facet 2: p1, p2, center, radius
    # Returns closest intersection point, if there is one
    # TODO uses self.points[0] to decide which line-arc intersection is the corner point
    def checkCurvedCornerFacet(self, facet1: Facet, facet2: Facet, ret=False):
        facet1_is_line = isinstance(facet1, LinearFacet)
        facet2_is_line = isinstance(facet2, LinearFacet)
        facet1_is_arc = isinstance(facet1, ArcFacet)
        facet2_is_arc = isinstance(facet2, ArcFacet)

        if facet1_is_line and facet2_is_arc:
            intersects = getCircleLineIntersects(
                facet1.pLeft,
                facet1.pRight,
                facet2.center,
                facet2.radius,
                checkWithinLine=False,
            )
            if len(intersects) == 1:
                corner_point = intersects[0]
            elif len(intersects) > 1:
                if getDistance(self.points[0], intersects[0]) >= getDistance(
                    self.points[0], intersects[1]
                ):
                    corner_point = intersects[1]
                else:
                    corner_point = intersects[0]
            else:  # no intersects
                print("checkCurvedCornerFacet: no intersects between line and arc")
                return None, None

        elif facet1_is_arc and facet2_is_line:
            intersects = getCircleLineIntersects(
                facet2.pLeft,
                facet2.pRight,
                facet1.center,
                facet1.radius,
                checkWithinLine=False,
            )
            if len(intersects) == 1:
                corner_point = intersects[0]
            elif len(intersects) > 1:
                if getDistance(self.points[0], intersects[0]) >= getDistance(
                    self.points[0], intersects[1]
                ):
                    corner_point = intersects[1]
                else:
                    corner_point = intersects[0]
            else:  # no intersects
                print("checkCurvedCornerFacet: no intersects between line and arc")
                return None, None

        elif facet1_is_arc and facet2_is_arc:
            try:
                intersects = getCircleCircleIntersects(
                    facet1.center, facet2.center, facet1.radius, facet2.radius
                )
                if len(intersects) == 0:
                    print("checkCurvedCornerFacet: no intersects between arc and arc")
                    return None, None
                else:  # two intersects TODO what if more than 2?
                    assert len(intersects) == 2
                    if getDistance(self.points[0], intersects[0]) >= getDistance(
                        self.points[0], intersects[1]
                    ):
                        corner_point = intersects[1]
                    else:
                        corner_point = intersects[0]
            except:
                # Failed two arc curved corner
                print("checkCurvedCornerFacet: failed intersect between arc and arc")
                return None, None
        else:  # linear, linear
            # TODO: skip linear, linear case in curved corner
            # self.checkCornerFacet(facet1.pLeft, facet1.pRight, facet2.pRight, facet2.pLeft)
            return None, None

        try:
            cornerareafraction = (
                getPolyCurvedCornerArea(
                    self.points,
                    facet1.pRight,
                    corner_point,
                    facet2.pLeft,
                    facet1.radius if facet1_is_arc else None,
                    facet2.radius if facet2_is_arc else None,
                )
                / self.getMaxArea()
            )
        except RuntimeError as error:
            print(f"checkCurvedCornerFacet: failed curved area evaluation: {error}")
            return None, None
        facet1_tangent = facet1.getTangent(corner_point)
        facet2_tangent = facet2.getTangent(corner_point)
        # Check that corner is not too sharp
        # TODO Check if this corner facet intersects middle poly
        if (
            abs(lineAngleSine([0, 0], facet1_tangent, [0, 0], facet2_tangent))
            > NeighboredPolygon.corner_sharpness_threshold
        ):
            facet = CornerFacet(
                facet1.center if facet1_is_arc else None,
                facet2.center if facet2_is_arc else None,
                facet1.radius if facet1_is_arc else None,
                facet2.radius if facet2_is_arc else None,
                facet1.pRight,
                corner_point,
                facet2.pLeft,
            )
            if ret:
                return facet, abs(cornerareafraction - self.getFraction())
            elif (
                abs(cornerareafraction - self.getFraction())
                < NeighboredPolygon.curved_corner_area_threshold
            ):
                self.setFacet(facet)
        else:
            return None, None
            # print("Failed curved corner:")
            # print(self)
            # print(CornerFacet(facet1.center if facet1.name == "arc" else None, facet2.center if facet2.name == "arc" else None, facet1.radius if facet1.name == "arc" else None, facet2.radius if facet2.name == "arc" else None, facet1.pRight, corner_point, facet2.pLeft))
            # print(cornerareafraction)
            # print(abs(cornerareafraction - self.getFraction()))

    # TODO add settings for only circles / only lines / corners

    def __str__(self):
        if self.hasFacet:
            return f"\nPoints: {self.points}\nFraction: {self.fraction}\nLeft neighbor: {self.left_neighbor.points if self.left_neighbor is not None else None}\nRight neighbor: {self.right_neighbor.points if self.right_neighbor is not None else None}\nFacet: {self.facet}\n"
        else:
            return f"\nPoints: {self.points}\nFraction: {self.fraction}\nLeft neighbor: {self.left_neighbor.points if self.left_neighbor is not None else None}\nRight neighbor: {self.right_neighbor.points if self.right_neighbor is not None else None}\n"
