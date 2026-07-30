import signal
import threading

from main.geoms.geoms import (
    getArea,
    getCentroid,
    getDistance,
    getPolyLineArea,
    lineIntersect,
    pointInPoly,
)
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet
from main.algos.plic_normals import getELVIRANormal, getYoungsNormal
from main.geoms.linear_facet import (
    getLinearFacet,
    getLVIRALinearFacet,
    getPolyLineIntersects,
    getLinearFacetFromNormal,
)
from main.geoms.circular_facet import (
    LinearFacetShortcut,
    getArcFacet,
    getCircleIntersectArea,
    getArcFacetRoot,
    matchArcArea,
    getCenter,
)


class BasePolygon:

    fraction_tolerance = 1e-10
    C0_linear_tolerance = 1e-5  # should be a couple orders of magnitude higher than linearity_threshold in NeighboredPolygon

    linearity_threshold = 1e-6  # if area fraction error in linear facet < this value, use linear facet at this cell
    extreme_fraction_orientation_threshold = 5e-3
    three_neighbor_orientation_tolerance = 5e-3
    optimization_threshold = 1e-10  # in optimizations
    arc_fit_residual_threshold = 1e-5
    arc_fit_timeout_seconds = 2.0

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

        # Optional callback used by MergeMesh to retain facet-construction provenance.
        self._facet_assignment_callback = None
        self.last_safe_circle_fallback = None
        self.last_c0_fit_diagnostic = None

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
        previous_facet = self.facet
        self.facet = facet
        if self._facet_assignment_callback is not None:
            self._facet_assignment_callback(self, previous_facet, facet)

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

    @staticmethod
    def _run_arc_fit_with_timeout(*args):
        timeout_seconds = BasePolygon.arc_fit_timeout_seconds
        if (
            timeout_seconds is None
            or timeout_seconds <= 0
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")
            or not hasattr(signal, "setitimer")
            or not hasattr(signal, "ITIMER_REAL")
        ):
            return getArcFacet(*args)

        def _handle_timeout(signum, frame):
            raise TimeoutError(
                f"getArcFacet timed out after {timeout_seconds:.1f}s"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            return getArcFacet(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    @staticmethod
    def _try_arc_fit_root_fallback(*args):
        try:
            return getArcFacetRoot(*args)
        except Exception:
            return None, None, None

    @staticmethod
    def _arc_fit_max_fraction_error(polys, fractions, center, radius):
        max_error = 0.0
        for poly, target_fraction in zip(polys, fractions):
            fit_area, _ = getCircleIntersectArea(center, radius, poly)
            fit_fraction = fit_area / abs(getArea(poly))
            max_error = max(max_error, abs(fit_fraction - target_fraction))
        return max_error

    @staticmethod
    def _arc_facet_fraction_error(poly, target_fraction, center, radius, p_left, p_right):
        try:
            facet = ArcFacet(center, radius, p_left, p_right)
        except Exception:
            return float("inf")

        try:
            fit_area = facet.getPolyIntersectArea(poly)
        except Exception:
            return float("inf")
        fit_fraction = fit_area / abs(getArea(poly))
        return abs(fit_fraction - target_fraction)

    @staticmethod
    def _root_fallback_selection_key(mid_poly, center, radius, arcintersects, root_guess=None):
        facet = ArcFacet(center, radius, arcintersects[0], arcintersects[-1])
        midpoint = facet.midpoint
        centroid = getCentroid(mid_poly)
        midpoint_distance = getDistance(midpoint, centroid)
        inside_mid = pointInPoly(midpoint, mid_poly)
        if root_guess is None:
            return (0 if inside_mid else 1, midpoint_distance)

        guess_center = [root_guess[0], root_guess[1]]
        guess_distance = getDistance(center, guess_center) + abs(
            abs(radius) - abs(root_guess[2])
        )
        return (0 if inside_mid else 1, guess_distance, midpoint_distance)

    @staticmethod
    def _normalize_root_fallback_arc(polys, fractions, center, radius, arcintersects, root_guess=None):
        if arcintersects is None or len(arcintersects) < 2:
            return None, None, None

        p_first = list(arcintersects[0])
        p_last = list(arcintersects[-1])
        candidates = [
            (radius, p_first, p_last),
            (radius, p_last, p_first),
            (-radius, p_first, p_last),
            (-radius, p_last, p_first),
        ]

        best = None
        best_key = None
        mid_poly = polys[1]
        for cand_radius, cand_left, cand_right in candidates:
            try:
                key = BasePolygon._root_fallback_selection_key(
                    mid_poly,
                    center,
                    cand_radius,
                    [cand_left, cand_right],
                    root_guess=root_guess,
                )
            except Exception:
                continue
            if best_key is None or key < best_key:
                best_key = key
                best = (center, cand_radius, [cand_left, cand_right])

        if best is None:
            return None, None, None
        return best

    @staticmethod
    def _try_arc_fit_root_fallbacks(base_args, root_guess=None):
        valid_candidates = []

        def _collect(raw_candidate, guess_for_key=None):
            arccenter, arcradius, arcintersects = raw_candidate
            if (
                arccenter is None
                or arcradius is None
                or arcintersects is None
            ):
                return
            residual = BasePolygon._arc_fit_max_fraction_error(
                base_args[:3], base_args[3:6], arccenter, arcradius
            )
            if residual > BasePolygon.arc_fit_residual_threshold:
                return
            normalized = BasePolygon._normalize_root_fallback_arc(
                base_args[:3],
                base_args[3:6],
                arccenter,
                arcradius,
                arcintersects,
                root_guess=guess_for_key,
            )
            if normalized[0] is None:
                return
            valid_candidates.append(normalized)

        _collect(BasePolygon._try_arc_fit_root_fallback(*base_args), guess_for_key=root_guess)
        if root_guess is not None:
            _collect(
                BasePolygon._try_arc_fit_root_fallback(
                    *(tuple(base_args) + tuple(root_guess))
                ),
                guess_for_key=root_guess,
            )

        if not valid_candidates:
            return None, None, None

        return min(
            valid_candidates,
            key=lambda candidate: BasePolygon._root_fallback_selection_key(
                base_args[1],
                candidate[0],
                candidate[1],
                candidate[2],
                root_guess=root_guess,
            ),
        )

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

        # If three mixed neighbors and this cell is almost full/empty, use the pair
        # whose linear seed best matches the middle-cell fraction.
        elif (
            len(mixed_neighbors) == 3
            and min(self.getFraction(), 1 - self.getFraction())
            < BasePolygon.extreme_fraction_orientation_threshold
        ):
            best_orientation = None
            best_error = None
            for left_neighbor in mixed_neighbors:
                for right_neighbor in mixed_neighbors:
                    if left_neighbor is right_neighbor:
                        continue
                    try:
                        l1, l2 = getLinearFacet(
                            left_neighbor.points,
                            right_neighbor.points,
                            left_neighbor.getFraction(),
                            right_neighbor.getFraction(),
                            BasePolygon.optimization_threshold,
                        )
                    except RuntimeError:
                        continue
                    line_fraction = (
                        getPolyLineArea(self.points, l1, l2) / self.getMaxArea()
                    )
                    error = abs(self.getFraction() - line_fraction)
                    if best_error is None or error < best_error:
                        best_error = error
                        best_orientation = [left_neighbor, right_neighbor]
            if (
                best_orientation is not None
                and best_error is not None
                and best_error < BasePolygon.three_neighbor_orientation_tolerance
            ):
                return best_orientation

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

    def runELVIRA(self, ret=False):
        assert self.has3x3Stencil()
        normal = getELVIRANormal(self.stencil)
        l1, l2 = getLinearFacetFromNormal(
            self.points, self.getFraction(), normal, BasePolygon.optimization_threshold
        )
        intersects = getPolyLineIntersects(self.points, l1, l2)
        youngsFacet = LinearFacet(intersects[0], intersects[-1], name="ELVIRA")
        if ret:
            return youngsFacet
        else:
            self.setFacet(youngsFacet)

    def runLVIRA(self, ret=False):
        assert self.has3x3Stencil()
        l1, l2 = getLVIRALinearFacet(
            self.stencil,
            BasePolygon.optimization_threshold,
            initial_normals=[
                getELVIRANormal(self.stencil),
                getYoungsNormal(self.stencil),
            ],
        )
        intersects = getPolyLineIntersects(self.points, l1, l2)
        lvira_facet = LinearFacet(intersects[0], intersects[-1], name="LVIRA")
        if ret:
            return lvira_facet
        else:
            self.setFacet(lvira_facet)

    @staticmethod
    def _normalize_plic_fallback(policy):
        if policy is None:
            return None
        policies = {
            "youngs": "Youngs",
            "elvira": "ELVIRA",
            "lvira": "LVIRA",
        }
        policy_key = str(policy).lower()
        if policy_key not in policies:
            raise ValueError(
                f"Unknown plic_fallback={policy!r}; expected Youngs, ELVIRA, or LVIRA"
            )
        return policies[policy_key]

    def _run_plic_fallback(self, policy):
        policy = self._normalize_plic_fallback(policy)
        if policy == "Youngs":
            return self.runYoungs(ret=True)
        if policy == "ELVIRA":
            return self.runELVIRA(ret=True)
        if policy == "LVIRA":
            return self.runLVIRA(ret=True)
        return None

    def _fit_mass_matching_line(self, seed_left, seed_right):
        distance = getDistance(seed_left, seed_right)
        if distance <= 0:
            raise RuntimeError("Degenerate support line")
        normal = [
            (-seed_right[1] + seed_left[1]) / distance,
            (seed_right[0] - seed_left[0]) / distance,
        ]
        line_left, line_right = getLinearFacetFromNormal(
            self.points,
            self.getFraction(),
            normal,
            BasePolygon.optimization_threshold,
        )
        return LinearFacet(line_left, line_right, name="default_linear")

    def runSafeLinear(
        self,
        ret=False,
        check_threshold=False,
        default_to_youngs=False,
        default_to_elvira=True,
        fit_1neighbor=False,
    ):
        assert self.has3x3Stencil()
        orientation = self.findSafeOrientation(fit_1neighbor=fit_1neighbor)
        if orientation is None:
            # Default to PLIC
            if default_to_youngs:
                facet = self.runYoungs(ret=True)
            elif default_to_elvira:
                facet = self.runELVIRA(ret=True)
            else:
                facet = None
        else:
            left_neighbor: BasePolygon = orientation[0]
            right_neighbor: BasePolygon = orientation[1]
            try:
                l1, l2 = getLinearFacet(
                    left_neighbor.points,
                    right_neighbor.points,
                    left_neighbor.getFraction(),
                    right_neighbor.getFraction(),
                    BasePolygon.optimization_threshold,
                )
            except RuntimeError as error:
                print(f"runSafeLinear fallback to PLIC after getLinearFacet failure: {error}")
                if default_to_youngs:
                    facet = self.runYoungs(ret=True)
                elif default_to_elvira:
                    facet = self.runELVIRA(ret=True)
                else:
                    facet = None
                if ret:
                    return facet
                else:
                    if facet is not None:
                        self.setFacet(facet)
                    return
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
                    elif default_to_elvira:
                        facet = self.runELVIRA(ret=True)
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
        plic_fallback="LVIRA",
        arc_failure_fallback="local_linear",
        default_to_youngs=None,
        default_to_elvira=None,
        return_info=False,
    ):
        assert self.has3x3Stencil()
        if default_to_youngs is True:
            plic_fallback = "Youngs"
        elif default_to_elvira is True:
            plic_fallback = "ELVIRA"
        elif default_to_youngs is False and default_to_elvira is False:
            plic_fallback = None
        plic_fallback = self._normalize_plic_fallback(plic_fallback)

        arc_failure_fallback = str(arc_failure_fallback or "none").lower()
        if arc_failure_fallback not in {"local_linear", "plic", "none"}:
            raise ValueError(
                "Unknown arc_failure_fallback={!r}; expected local_linear, plic, or none".format(
                    arc_failure_fallback
                )
            )

        def finish(facet, fallback_record=None):
            self.last_safe_circle_fallback = fallback_record
            if ret:
                if return_info:
                    return facet, fallback_record
                return facet
            if facet is not None:
                self.setFacet(facet)
            if return_info:
                return fallback_record
            return None

        def plic_result(reason):
            facet = self._run_plic_fallback(plic_fallback)
            return finish(
                facet,
                {
                    "event_kind": "plic_fallback" if facet is not None else "missing_fallback",
                    "reason": reason,
                    "policy": plic_fallback or "",
                },
            )

        def arc_failure_result(seed_left, seed_right, reason):
            if arc_failure_fallback == "local_linear":
                try:
                    facet = self._fit_mass_matching_line(seed_left, seed_right)
                    return finish(
                        facet,
                        {
                            "event_kind": "local_linear_fallback",
                            "reason": reason,
                            "policy": "local_linear",
                        },
                    )
                except Exception:
                    return plic_result(f"{reason}_local_linear_failed")
            if arc_failure_fallback == "plic":
                return plic_result(reason)
            return finish(
                None,
                {
                    "event_kind": "missing_fallback",
                    "reason": reason,
                    "policy": "",
                },
            )

        orientation = self.findSafeOrientation(fit_1neighbor=False)
        if orientation is None:
            return plic_result("unresolved_orientation")

        left_neighbor: BasePolygon = orientation[0]
        right_neighbor: BasePolygon = orientation[1]
        try:
            l1, l2 = getLinearFacet(
                left_neighbor.points,
                right_neighbor.points,
                left_neighbor.getFraction(),
                right_neighbor.getFraction(),
                BasePolygon.optimization_threshold,
            )
        except RuntimeError as error:
            print(f"runSafeCircle fallback to PLIC after getLinearFacet failure: {error}")
            return plic_result("support_line_fit_failed")

        line_area_fraction = getPolyLineArea(self.points, l1, l2) / self.getArea()
        if (
            abs(self.getFraction() - line_area_fraction)
            < BasePolygon.linearity_threshold
            and line_area_fraction > BasePolygon.optimization_threshold
            and line_area_fraction < 1 - BasePolygon.optimization_threshold
        ):
            intersects = getPolyLineIntersects(self.points, l1, l2)
            return finish(LinearFacet(intersects[0], intersects[-1]))

        arc_args = (
            left_neighbor.points,
            self.points,
            right_neighbor.points,
            left_neighbor.getFraction(),
            self.getFraction(),
            right_neighbor.getFraction(),
            BasePolygon.optimization_threshold,
        )
        try:
            arccenter, arcradius, arcintersects = self._run_arc_fit_with_timeout(
                *arc_args
            )
            if arccenter is None or arcradius is None or arcintersects is None:
                arccenter, arcradius, arcintersects = self._try_arc_fit_root_fallbacks(
                    arc_args
                )
        except LinearFacetShortcut as shortcut:
            return finish(LinearFacet(shortcut.pLeft, shortcut.pRight))
        except (RuntimeError, TimeoutError) as error:
            arccenter, arcradius, arcintersects = self._try_arc_fit_root_fallbacks(
                arc_args
            )
            if arccenter is None or arcradius is None or arcintersects is None:
                print(
                    f"runSafeCircle local-line fallback after getArcFacet failure: {error}"
                )
                return arc_failure_result(l1, l2, "arc_fit_failed")
        except Exception as error:
            print(
                f"runSafeCircle local-line fallback after unexpected getArcFacet failure: {error}"
            )
            return arc_failure_result(l1, l2, "unexpected_arc_fit_failed")

        if arccenter is None or arcradius is None or arcintersects is None:
            return arc_failure_result(l1, l2, "arc_fit_failed")
        return finish(
            ArcFacet(arccenter, arcradius, arcintersects[0], arcintersects[-1])
        )

    def _facet_phase_area(self, facet):
        if isinstance(facet, LinearFacet):
            return getPolyLineArea(self.points, facet.pLeft, facet.pRight)
        if isinstance(facet, ArcFacet):
            return getCircleIntersectArea(
                facet.center, facet.radius, self.points
            )[0]
        raise TypeError(f"Unsupported C0 facet type: {type(facet).__name__}")

    # Given two endpoints of facet and area fraction, find a conservative curvature.
    def fitCurvature(
        self, pLeft, pRight, fraction_tolerance=fraction_tolerance, ret=False
    ):
        target_area = self.getArea()
        area_tolerance = fraction_tolerance * self.getMaxArea()
        try:
            original_error = abs(self._facet_phase_area(self.getFacet()) - target_area)
        except Exception:
            original_error = 0.0
        allowed_error = original_error + area_tolerance

        try:
            d = getDistance(pLeft, pRight)
            lineArea = getPolyLineArea(self.points, pLeft, pRight)
            if (
                abs(target_area - lineArea) / self.getMaxArea()
                < BasePolygon.C0_linear_tolerance
            ):
                facet = LinearFacet(pLeft, pRight)
            else:
                radius = matchArcArea(
                    d, target_area - lineArea, area_tolerance
                )
                if radius == float("inf") or radius == -float("inf"):
                    facet = LinearFacet(pLeft, pRight)
                else:
                    center = getCenter(pLeft, pRight, radius)
                    facet = ArcFacet(center, radius, pLeft, pRight)
        except Exception as error:
            self.last_c0_fit_diagnostic = {
                "selected_branch": "rejected_exception",
                "rejection_reason": type(error).__name__,
                "rejection_message": str(error),
                "target_area": target_area,
                "original_error": original_error,
                "candidate_error": float("inf"),
                "alternate_error": float("inf"),
                "selected_error": float("inf"),
                "allowed_error": allowed_error,
            }
            return None

        try:
            candidate_error = abs(self._facet_phase_area(facet) - target_area)
        except Exception:
            candidate_error = float("inf")

        alternate = None
        alternate_error = float("inf")
        if isinstance(facet, ArcFacet) and candidate_error > allowed_error:
            alternate_radius = -facet.radius
            try:
                alternate = ArcFacet(
                    getCenter(pLeft, pRight, alternate_radius),
                    alternate_radius,
                    pLeft,
                    pRight,
                )
                alternate_error = abs(
                    self._facet_phase_area(alternate) - target_area
                )
            except Exception:
                alternate = None
                alternate_error = float("inf")

        if candidate_error <= allowed_error:
            selected_facet = facet
            selected_error = candidate_error
            selected_branch = "analytic"
        elif alternate is not None and alternate_error <= allowed_error:
            selected_facet = alternate
            selected_error = alternate_error
            selected_branch = "opposite_signed_curvature"
        else:
            selected_facet = None
            selected_error = min(candidate_error, alternate_error)
            selected_branch = "rejected"

        self.last_c0_fit_diagnostic = {
            "selected_branch": selected_branch,
            "target_area": target_area,
            "original_error": original_error,
            "candidate_error": candidate_error,
            "alternate_error": alternate_error,
            "selected_error": selected_error,
            "allowed_error": allowed_error,
            "rejection_reason": (
                "area_residual" if selected_branch == "rejected" else ""
            ),
        }
        if ret:
            return selected_facet
        if selected_facet is not None:
            self.setFacet(selected_facet)

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
