from typing import Dict
import math
import numpy as np
from matplotlib.patches import Polygon as plt_polygon

from main.structs.meshes.base_mesh import BaseMesh
from main.structs.polys.base_polygon import BasePolygon
from main.structs.polys.neighbored_polygon import NeighboredPolygon
from main.geoms.geoms import (
    getDistance,
    mergePolys,
    getPolyIntersectArea,
    getArea,
    getPolyLineArea,
    lineIntersect,
    lerp,
    pointInPoly,
)
from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.corner_facet import getPolyCornerArea, getPolyCurvedCornerArea
from main.structs.facets.base_facet import advectPoint
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from util.logging.get_arc_facet_logger import arc_facet_log_context

"""
A class for meshes that merge cells.
Assumes a perturbed Cartesian grid of quads. Polygons are stored in a 2D list of NeighboredPolygon objects.

This class is used for the algorithms that merge cells:
- Linear corner
- Circular corner
"""


class MergeMesh(BaseMesh):

    default_corner_behavior_profile = "pre_f8_corner"
    default_rescue_profile = "exact_linear_support_only"

    rescue_profiles = {
        "full",
        "no_corner_rescues",
        "no_linear_corner_rescues",
        "no_curved_corner_rescues",
        "no_repeated_corner_rescues",
        "no_repeated_tiny_corner_rescues",
        "no_repeated_corner_component_rescues",
        "candidate_keep_12346_drop_9",
        "exact_linear_support_only",
    }

    corner_behavior_profiles = {
        "current": (True, True, True, True),
        "no_orientation_hint": (False, True, True, True),
        "no_locality_guard": (True, False, True, True),
        "legacy_branch_intersection": (True, True, False, True),
        "legacy_corner_acceptance": (True, False, False, True),
        "no_corner_branch_propagation": (True, True, True, False),
        "no_hint_legacy_acceptance": (False, False, False, True),
        "no_hint_no_branch_propagation": (False, True, True, False),
        "legacy_acceptance_no_branch_propagation": (True, False, False, False),
        "pre_f8_with_locality_guard": (False, True, False, False),
        "pre_f8_with_both_branch_requirement": (False, False, True, False),
        "pre_f8_corner": (False, False, False, False),
        "pre_f8_corner_late_hint": (False, False, False, False),
        "pre_f8_corner_greedy_retry": (False, False, False, False),
        "pre_f8_corner_greedy_continue": (False, False, False, False),
        "pre_f8_corner_late_hint_retry": (False, False, False, False),
    }
    late_orientation_hint_profiles = {
        "pre_f8_corner_late_hint",
        "pre_f8_corner_late_hint_retry",
    }
    greedy_orientation_retry_profiles = {
        "pre_f8_corner_greedy_retry",
        "pre_f8_corner_late_hint_retry",
    }
    greedy_orientation_continue_profiles = {
        "pre_f8_corner_greedy_continue",
    }

    def __init__(self, points, threshold, areas=None):
        super().__init__(points, threshold, areas)
        self.plic_fallback_records = []
        self.safe_circle_fallback_records = []
        self.facet_provenance_events = []
        self._provenance_event_order = 0
        self._provenance_stage = "initial"
        self._provenance_override = None
        # Let self.polys be a list of NeighboredPolygon objects
        self.polys: list[list[NeighboredPolygon]] = [
            [None] * (len(points[0]) - 1) for _ in range(len(points) - 1)
        ]
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                # Make quads
                poly = NeighboredPolygon(
                    [
                        points[x][y],
                        points[x + 1][y],
                        points[x + 1][y + 1],
                        points[x][y + 1],
                    ]
                )
                self.polys[x][y] = poly

        # Set adjacent polys
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                poly = self.polys[x][y]
                if x > 0:
                    poly.adjacent_polys.append(self.polys[x - 1][y])
                if y > 0:
                    poly.adjacent_polys.append(self.polys[x][y - 1])
                if x < len(self.polys) - 1:
                    poly.adjacent_polys.append(self.polys[x + 1][y])
                if y < len(self.polys[0]) - 1:
                    poly.adjacent_polys.append(self.polys[x][y + 1])

        # List of merged polys, index in list is used as id.
        # Each element is a list of (x, y)s corresponding to the Cartesian coordinates of the polys to be merged.
        self.merge_ids_to_coords = []
        # List of neighbor ids, index matches self.merge_ids_to_coords
        self.merge_id_to_neighbor_ids = []

        # Newest id that can be used
        self.next_merge_id = 0

        # Same shape as self.polys
        self.coords_to_merge_id = [
            [None for _ in range(len(self.polys[0]))] for _ in range(len(self.polys))
        ]

        # Dict of NeighboredPolygon objects, index matches self.merge_ids_to_coords
        self.merged_polys = dict()
        self.configure_corner_behavior(self.default_corner_behavior_profile)

    def configure_corner_behavior(self, profile=None):
        profile = str(profile or self.default_corner_behavior_profile).lower()
        if profile not in self.corner_behavior_profiles:
            raise ValueError(
                f"Unknown corner_behavior_profile={profile!r}; expected one of "
                f"{sorted(self.corner_behavior_profiles)}"
            )

        (
            self.use_three_neighbor_orientation_hint,
            self.use_linear_corner_locality_guard,
            self.require_both_linear_corner_branches,
            self.use_corner_branch_propagation,
        ) = self.corner_behavior_profiles[profile]
        self.use_late_three_neighbor_orientation_hint = (
            profile in self.late_orientation_hint_profiles
        )
        self.retry_greedy_orientations = (
            profile in self.greedy_orientation_retry_profiles
        )
        self.continue_greedy_orientation_conflicts = (
            profile in self.greedy_orientation_continue_profiles
        )
        self.corner_behavior_profile = profile

        polys = [poly for column in self.polys for poly in column]
        polys.extend(self.merged_polys.values())
        for poly in polys:
            poly.use_linear_corner_locality_guard = self.use_linear_corner_locality_guard
            poly.require_both_linear_corner_branches = (
                self.require_both_linear_corner_branches
            )
            poly.use_corner_branch_propagation = self.use_corner_branch_propagation

    @staticmethod
    def _facet_provenance_class(facet):
        if facet is None:
            return "missing"
        if isinstance(facet, CornerFacet):
            if isinstance(facet.facetLeft, LinearFacet) and isinstance(
                facet.facetRight, LinearFacet
            ):
                return "linear_corner"
            if isinstance(facet.facetLeft, ArcFacet) or isinstance(
                facet.facetRight, ArcFacet
            ):
                return "curved_corner"
            return "corner"
        if isinstance(facet, ArcFacet):
            return "circular"
        if isinstance(facet, LinearFacet):
            return "linear"
        return type(facet).__name__

    @classmethod
    def _facet_provenance_name(cls, facet):
        if facet is None:
            return ""
        return str(getattr(facet, "name", "") or "")

    def _attach_facet_provenance(self, poly, merge_id):
        poly._merge_id = merge_id
        poly._facet_assignment_callback = self._record_facet_assignment

    def _record_facet_assignment(self, poly, previous_facet, facet):
        merge_id = getattr(poly, "_merge_id", None)
        override = self._provenance_override or {}
        override_record = override.get(merge_id)
        if isinstance(override_record, dict):
            event_kind = override_record.get("event_kind", "facet_assignment")
            fallback_policy = override_record.get("policy", "")
            fallback_reason = override_record.get("reason", "")
        elif override_record is not None:
            event_kind = override_record[0]
            fallback_policy = override_record[1]
            fallback_reason = ""
        else:
            event_kind = "facet_assignment"
            fallback_policy = ""
            fallback_reason = ""
        self._provenance_event_order += 1
        self.facet_provenance_events.append(
            {
                "event_order": self._provenance_event_order,
                "merge_id": merge_id,
                "stage": self._provenance_stage,
                "event_kind": event_kind,
                "fallback_policy": fallback_policy,
                "fallback_reason": fallback_reason,
                "previous_facet_class": self._facet_provenance_class(previous_facet),
                "previous_facet_name": self._facet_provenance_name(previous_facet),
                "facet_class": self._facet_provenance_class(facet),
                "facet_name": self._facet_provenance_name(facet),
            }
        )

    def _record_stage_snapshots(self, stage, merge_ids):
        self._provenance_stage = stage
        for merge_id in tuple(merge_ids):
            poly = self.merged_polys.get(merge_id)
            if poly is None:
                continue
            self._provenance_event_order += 1
            facet = poly.getFacet()
            self.facet_provenance_events.append(
                {
                    "event_order": self._provenance_event_order,
                    "merge_id": merge_id,
                    "stage": stage,
                    "event_kind": "stage_snapshot",
                    "fallback_policy": "",
                    "fallback_reason": "",
                    "previous_facet_class": "",
                    "previous_facet_name": "",
                    "facet_class": self._facet_provenance_class(facet),
                    "facet_name": self._facet_provenance_name(facet),
                }
            )

    def _rewrite_latest_facet_provenance(
        self, merge_id, event_kind, policy, reason
    ):
        for event in reversed(self.facet_provenance_events):
            if event.get("merge_id") != merge_id:
                continue
            event["event_kind"] = event_kind
            event["fallback_policy"] = policy
            event["fallback_reason"] = reason
            return

    def _append_plic_fallback_record(self, setting, merge_id, poly, facet, policy):
        self.plic_fallback_records.append(
            {
                "setting": setting,
                "merge_id": merge_id,
                "policy": policy,
                "facet_name": getattr(facet, "name", ""),
                "num_vertices": len(poly.points),
            }
        )

    def _fit_deadend_facet(self, merge_id, prefer_safe_circle=False):
        merged_poly: NeighboredPolygon = self.merged_polys[merge_id]

        if prefer_safe_circle:
            merge_coords = self._get_merge_coords(merge_id)
            if len(merge_coords) == 1:
                x, y = merge_coords[0]
                stencil = self.get3x3Stencil(x, y)
                if stencil is not None:
                    merged_poly.set3x3Stencil(stencil)
                    try:
                        facet = merged_poly.runSafeCircle(
                            ret=True,
                            default_to_youngs=False,
                            default_to_elvira=False,
                        )
                    except Exception as error:
                        print(
                            f"Dead-end safe_circle fallback failed for merge_id={merge_id}: {error}"
                        )
                        facet = None
                    if facet is not None:
                        if facet.name == "arc":
                            facet.name = "deadend_arc"
                        elif facet.name == "linear":
                            facet.name = "linear_deadend"
                        merged_poly.setFacet(facet)
                        return

        merged_poly.fitLinearFacet()
        merged_poly.getFacet().name = "linear_deadend"

    def _get_merge_id(self, x, y):
        return self.coords_to_merge_id[x][y]

    def _get_merge_coords(self, merge_id):
        return self.merge_ids_to_coords[merge_id]

    def _get_neighbor_ids_from_merge_id(self, merge_id):
        return self.merge_id_to_neighbor_ids[merge_id]

    def _get_neighbor_ids_from_coords(self, x, y):
        return self._get_neighbor_ids_from_merge_id(self._get_merge_id(x, y))

    def _get_num_merge_ids(self):
        return self.next_merge_id

    @staticmethod
    def _poly_centroid(points):
        return [
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        ]

    @staticmethod
    def _poly_radius(points, centroid):
        return max(getDistance(point, centroid) for point in points)

    def _find_arc_fit_guess(self, merge_id):
        target_poly = self.merged_polys[merge_id]
        target_centroid = self._poly_centroid(target_poly.points)
        target_radius = self._poly_radius(target_poly.points, target_centroid)
        min_radius = max(4.0 * target_radius, 1.0)

        best_guess = None
        for candidate_id, candidate_poly in self.merged_polys.items():
            if candidate_id == merge_id:
                continue
            candidate_facet = candidate_poly.getFacet()
            if (
                candidate_facet is None
                or getattr(candidate_facet, "name", None) != "arc"
                or not hasattr(candidate_facet, "center")
                or not hasattr(candidate_facet, "radius")
            ):
                continue
            if abs(candidate_facet.radius) < min_radius:
                continue
            candidate_centroid = self._poly_centroid(candidate_poly.points)
            centroid_distance = getDistance(target_centroid, candidate_centroid)
            if best_guess is None or centroid_distance < best_guess[0]:
                best_guess = (
                    centroid_distance,
                    candidate_facet.center[0],
                    candidate_facet.center[1],
                    candidate_facet.radius,
                )

        if best_guess is None:
            return None
        return best_guess[1:]

    @staticmethod
    def _is_line_like_support_facet(facet):
        return isinstance(facet, LinearFacet)

    @staticmethod
    def _is_arc_like_support_facet(facet):
        return isinstance(facet, ArcFacet) and facet.name != "deadend_arc"

    @classmethod
    def _is_curved_corner_support_facet(cls, facet):
        return cls._is_line_like_support_facet(facet) or cls._is_arc_like_support_facet(
            facet
        )

    @staticmethod
    def _rounded_point_signature(point, digits=6):
        if point is None:
            return None
        return tuple(round(float(coord), digits) for coord in point)

    @classmethod
    def _repeated_corner_triplet_signature(cls, facet):
        if not isinstance(facet, CornerFacet):
            return None

        left_is_line = isinstance(facet.facetLeft, LinearFacet)
        right_is_line = isinstance(facet.facetRight, LinearFacet)
        if left_is_line == right_is_line:
            return None

        arc_branch = facet.facetRight if left_is_line else facet.facetLeft
        if not isinstance(arc_branch, ArcFacet):
            return None

        return (
            left_is_line,
            cls._rounded_point_signature(facet.pLeft),
            cls._rounded_point_signature(facet.corner),
            cls._rounded_point_signature(facet.pRight),
            cls._rounded_point_signature(arc_branch.center),
            round(float(arc_branch.radius), 6),
        )

    def _collect_same_corner_component(self, start_poly, signature):
        component = [start_poly]
        seen = {start_poly}

        current = start_poly
        while True:
            left_poly = current.getLeftNeighbor()
            if (
                left_poly is None
                or left_poly == current
                or left_poly in seen
                or not left_poly.hasFacet()
                or self._repeated_corner_triplet_signature(left_poly.getFacet())
                != signature
            ):
                break
            component.insert(0, left_poly)
            seen.add(left_poly)
            current = left_poly

        current = start_poly
        while True:
            right_poly = current.getRightNeighbor()
            if (
                right_poly is None
                or right_poly == current
                or right_poly in seen
                or not right_poly.hasFacet()
                or self._repeated_corner_triplet_signature(right_poly.getFacet())
                != signature
            ):
                break
            component.append(right_poly)
            seen.add(right_poly)
            current = right_poly

        return component

    def _propagate_exact_linear_supports(self, merge_ids, max_passes=3):
        for _ in range(max_passes):
            changed = False
            for merge_id in merge_ids:
                support_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if not support_poly.hasFacet() or not self._is_line_like_support_facet(
                    support_poly.getFacet()
                ):
                    continue

                support_facet = support_poly.getFacet()
                if support_facet.name == "corner_branch_linear":
                    continue
                for neighbor in [
                    support_poly.getLeftNeighbor(),
                    support_poly.getRightNeighbor(),
                ]:
                    if (
                        neighbor is None
                        or neighbor == support_poly
                        or not neighbor._can_overwrite_with_linear_support()
                    ):
                        continue
                    candidate = neighbor._linear_facet_from_line(
                        support_facet.pLeft,
                        support_facet.pRight,
                        name="linear_support",
                    )
                    if candidate is None:
                        continue
                    neighbor.setFacet(candidate)
                    changed = True
            if not changed:
                break

    def _rescue_corner_linear_bridge_cells(self, merge_ids):
        changed = False
        for merge_id in merge_ids:
            target_poly: NeighboredPolygon = self.merged_polys[merge_id]
            if (
                target_poly.hasFacet()
                and target_poly.getFacet().name not in NeighboredPolygon.linear_support_overwrite_names
            ):
                continue

            candidate_pairs = [
                (target_poly.getLeftNeighbor(), target_poly.getRightNeighbor()),
                (target_poly.getRightNeighbor(), target_poly.getLeftNeighbor()),
            ]
            for support_poly, corner_poly in candidate_pairs:
                if (
                    support_poly is None
                    or corner_poly is None
                    or not support_poly.hasFacet()
                    or not corner_poly.hasFacet()
                    or support_poly.getFacet().name != "linear_support"
                    or corner_poly.getFacet().name != "corner"
                ):
                    continue

                branch, branch_error = corner_poly.bestLinearCornerBranchForNeighbor(
                    target_poly
                )
                if branch is None:
                    continue

                support_error = NeighboredPolygon._line_area_residual_on_neighbor(
                    support_poly.getFacet(), target_poly
                )
                if branch_error >= support_error:
                    continue

                normal = [
                    -(branch.pRight[1] - branch.pLeft[1]),
                    branch.pRight[0] - branch.pLeft[0],
                ]
                candidate = target_poly._linear_facet_from_normal(
                    normal,
                    name="corner_branch_linear",
                )
                if candidate is None:
                    continue

                target_poly.setFacet(candidate)
                changed = True
                break

        return changed

    def _rescue_corner_arc_corner_triplets(self, merge_ids):
        changed = False
        for merge_id in merge_ids:
            middle_poly: NeighboredPolygon = self.merged_polys[merge_id]
            if not (
                middle_poly.hasFacet()
                and self._is_arc_like_support_facet(middle_poly.getFacet())
            ):
                continue

            left_poly = middle_poly.getLeftNeighbor()
            right_poly = middle_poly.getRightNeighbor()
            if (
                left_poly is None
                or right_poly is None
                or left_poly == middle_poly
                or right_poly == middle_poly
                or left_poly == right_poly
                or not left_poly.hasFacet()
                or not right_poly.hasFacet()
                or left_poly.getFacet().name != "corner"
                or right_poly.getFacet().name != "corner"
            ):
                continue

            left_corner = left_poly.getFacet()
            right_corner = right_poly.getFacet()
            left_branch = (
                left_corner.facetLeft
                if isinstance(left_corner.facetLeft, LinearFacet)
                and isinstance(left_corner.facetRight, ArcFacet)
                else None
            )
            right_branch = (
                right_corner.facetRight
                if isinstance(right_corner.facetRight, LinearFacet)
                and isinstance(right_corner.facetLeft, ArcFacet)
                else None
            )
            if left_branch is None or right_branch is None:
                continue

            corner_facet, _ = middle_poly._build_linear_corner_facet(
                left_branch.pLeft,
                left_branch.pRight,
                right_branch.pRight,
                right_branch.pLeft,
            )
            if corner_facet is None:
                continue

            middle_poly.setFacet(corner_facet)
            changed = True

        return changed

    def _rescue_repeated_tiny_corner_triplets(self, merge_ids):
        changed = False
        processed_polys = set()
        branch_residual_threshold = 1e-8
        max_arc_radius_ratio = 2.0

        for merge_id in merge_ids:
            start_poly: NeighboredPolygon = self.merged_polys[merge_id]
            if start_poly in processed_polys or not start_poly.hasFacet():
                continue

            signature = self._repeated_corner_triplet_signature(start_poly.getFacet())
            if signature is None:
                continue

            component = self._collect_same_corner_component(start_poly, signature)
            processed_polys.update(component)

            if len(component) != 3:
                continue

            owner_count = 0
            component_max_radius = 0.0
            branch_normals = []
            for poly in component:
                facet = poly.getFacet()
                if pointInPoly(facet.corner, poly.points):
                    owner_count += 1
                component_max_radius = max(
                    component_max_radius,
                    self._poly_radius(poly.points, self._poly_centroid(poly.points)),
                )
                branch, branch_error = poly.bestLinearCornerBranchForNeighbor(poly)
                if branch is None or branch_error is None:
                    branch_normals = []
                    break
                if branch_error >= branch_residual_threshold:
                    branch_normals = []
                    break
                branch_normals.append(
                    [
                        -(branch.pRight[1] - branch.pLeft[1]),
                        branch.pRight[0] - branch.pLeft[0],
                    ]
                )

            if owner_count != 1 or len(branch_normals) != len(component):
                continue

            facet = start_poly.getFacet()
            arc_branch = (
                facet.facetRight
                if isinstance(facet.facetRight, ArcFacet)
                else facet.facetLeft
            )
            if abs(arc_branch.radius) > max_arc_radius_ratio * component_max_radius:
                continue

            replacement_facets = []
            for poly, normal in zip(component, branch_normals):
                candidate = poly._linear_facet_from_normal(
                    normal,
                    name="corner_branch_linear",
                )
                if candidate is None:
                    replacement_facets = []
                    break
                replacement_facets.append((poly, candidate))

            if len(replacement_facets) != len(component):
                continue

            for poly, candidate in replacement_facets:
                poly.setFacet(candidate)
            changed = True

        return changed

    def _rescue_repeated_corner_components_as_linear_corners(self, merge_ids):
        changed = False
        processed_polys = set()
        owner_corner_threshold = 1e-8
        branch_residual_threshold = 1e-8

        for merge_id in merge_ids:
            start_poly: NeighboredPolygon = self.merged_polys[merge_id]
            if start_poly in processed_polys or not start_poly.hasFacet():
                continue

            signature = self._repeated_corner_triplet_signature(start_poly.getFacet())
            if signature is None:
                continue

            component = self._collect_same_corner_component(start_poly, signature)
            processed_polys.update(component)
            if len(component) != 3:
                continue

            owner_poly = None
            for poly in component:
                if pointInPoly(poly.getFacet().corner, poly.points):
                    if owner_poly is not None:
                        owner_poly = None
                        break
                    owner_poly = poly
            if owner_poly is None:
                continue

            left_support = component[0].getLeftNeighbor()
            right_support = component[-1].getRightNeighbor()
            if (
                left_support is None
                or right_support is None
                or not left_support.hasFacet()
                or not right_support.hasFacet()
                or not self._is_line_like_support_facet(left_support.getFacet())
                or not self._is_line_like_support_facet(right_support.getFacet())
            ):
                continue

            owner_candidate, owner_error = owner_poly._build_linear_corner_facet(
                left_support.getFacet().pLeft,
                left_support.getFacet().pRight,
                right_support.getFacet().pRight,
                right_support.getFacet().pLeft,
            )
            if (
                owner_candidate is None
                or owner_error is None
                or owner_error >= owner_corner_threshold
            ):
                continue

            saved_owner_facet = owner_poly.getFacet()
            owner_poly.setFacet(owner_candidate)
            branch_candidates = []
            success = True
            for poly in component:
                if poly == owner_poly:
                    continue
                branch, branch_error = owner_poly.bestLinearCornerBranchForNeighbor(poly)
                if (
                    branch is None
                    or branch_error is None
                    or branch_error >= branch_residual_threshold
                ):
                    success = False
                    break
                normal = [
                    -(branch.pRight[1] - branch.pLeft[1]),
                    branch.pRight[0] - branch.pLeft[0],
                ]
                candidate = poly._linear_facet_from_normal(
                    normal,
                    name="corner_branch_linear",
                )
                if candidate is None:
                    success = False
                    break
                branch_candidates.append((poly, candidate))

            if not success:
                owner_poly.setFacet(saved_owner_facet)
                continue

            for poly, candidate in branch_candidates:
                poly.setFacet(candidate)
            changed = True

        return changed

    def _collect_contiguous_line_like_neighbors(self, start_poly, direction, max_steps=3):
        chain = []
        seen = {start_poly}
        current = (
            start_poly.getLeftNeighbor() if direction == "left" else start_poly.getRightNeighbor()
        )
        previous = start_poly
        while (
            current is not None
            and current != previous
            and current not in seen
            and len(chain) < max_steps
            and current.hasFacet()
            and self._is_line_like_support_facet(current.getFacet())
        ):
            chain.append(current)
            seen.add(current)
            previous = current
            current = (
                current.getLeftNeighbor()
                if direction == "left"
                else current.getRightNeighbor()
            )
        return chain

    def _rescue_linear_corner_owner_intruder_arcs(self, merge_ids):
        changed = False
        owner_corner_threshold = 1e-8
        branch_residual_threshold = 1e-8

        for merge_id in merge_ids:
            target_poly: NeighboredPolygon = self.merged_polys[merge_id]
            if not (
                target_poly.hasFacet()
                and isinstance(target_poly.getFacet(), ArcFacet)
            ):
                continue

            left_chain = self._collect_contiguous_line_like_neighbors(
                target_poly, "left"
            )
            right_chain = self._collect_contiguous_line_like_neighbors(
                target_poly, "right"
            )
            if not left_chain or not right_chain:
                continue

            best_candidate = None
            saved_target_facet = target_poly.getFacet()
            for left_support in left_chain:
                for right_support in right_chain:
                    candidate, owner_error = target_poly._build_linear_corner_facet(
                        left_support.getFacet().pLeft,
                        left_support.getFacet().pRight,
                        right_support.getFacet().pRight,
                        right_support.getFacet().pLeft,
                    )
                    if (
                        candidate is None
                        or owner_error is None
                        or owner_error >= owner_corner_threshold
                    ):
                        continue

                    target_poly.setFacet(candidate)
                    branch_errors = []
                    success = True
                    for neighbor in [left_chain[0], right_chain[0]]:
                        branch, branch_error = target_poly.bestLinearCornerBranchForNeighbor(
                            neighbor
                        )
                        if (
                            branch is None
                            or branch_error is None
                            or branch_error >= branch_residual_threshold
                        ):
                            success = False
                            break
                        branch_errors.append(branch_error)
                    target_poly.setFacet(saved_target_facet)

                    if not success:
                        continue

                    score = (owner_error, sum(branch_errors))
                    if best_candidate is None or score < best_candidate[0]:
                        best_candidate = (score, candidate)

            if best_candidate is None:
                continue

            target_poly.setFacet(best_candidate[1])
            changed = True

        return changed

    def _collect_small_neighbor_loop(self, target_poly, max_size=6):
        cluster = [target_poly]
        seen = {target_poly}
        loop_found = False

        for get_neighbor in ("getLeftNeighbor", "getRightNeighbor"):
            current_poly = target_poly
            local_seen = {target_poly}
            local_path = []
            while len(local_path) < max_size:
                next_poly = getattr(current_poly, get_neighbor)()
                if next_poly is None or next_poly == current_poly:
                    break
                if next_poly in local_seen:
                    if next_poly is target_poly:
                        loop_found = True
                    break
                local_seen.add(next_poly)
                local_path.append(next_poly)
                current_poly = next_poly

            for poly in local_path:
                if poly not in seen:
                    seen.add(poly)
                    cluster.append(poly)

        if not loop_found or len(cluster) > max_size:
            return None
        return cluster

    def _fit_local_curved_corner_cluster(self, cluster_polys):
        if not cluster_polys:
            return None

        centroids = [self._poly_centroid(poly.points) for poly in cluster_polys]
        xmin = min(point[0] for point in centroids)
        xmax = max(point[0] for point in centroids)
        ymin = min(point[1] for point in centroids)
        ymax = max(point[1] for point in centroids)
        avg_radius = sum(
            self._poly_radius(poly.points, centroid)
            for poly, centroid in zip(cluster_polys, centroids)
        ) / len(cluster_polys)
        expansion = max(5.0 * avg_radius, 1e-12)

        candidate_lines = []
        candidate_arcs = []
        cluster_set = set(cluster_polys)
        for candidate_id, candidate_poly in self.merged_polys.items():
            if candidate_poly in cluster_set or not candidate_poly.hasFacet():
                continue

            candidate_centroid = self._poly_centroid(candidate_poly.points)
            if (
                candidate_centroid[0] < xmin - expansion
                or candidate_centroid[0] > xmax + expansion
                or candidate_centroid[1] < ymin - expansion
                or candidate_centroid[1] > ymax + expansion
            ):
                continue

            candidate_facet = candidate_poly.getFacet()
            if self._is_line_like_support_facet(candidate_facet):
                candidate_lines.append(candidate_facet)
            elif self._is_arc_like_support_facet(candidate_facet):
                candidate_arcs.append(candidate_facet)

        best_fit = None
        for line_facet in candidate_lines:
            for arc_facet in candidate_arcs:
                for facet1, facet2 in ((line_facet, arc_facet), (arc_facet, line_facet)):
                    candidate_assignments = []
                    candidate_errors = []
                    corner_in_poly = False
                    for poly in cluster_polys:
                        corner_facet, corner_error = poly.checkCurvedCornerFacet(
                            facet1, facet2, ret=True
                        )
                        if corner_facet is None or corner_error is None:
                            candidate_assignments = None
                            break
                        candidate_assignments.append((poly, corner_facet))
                        candidate_errors.append(corner_error)
                        if not corner_in_poly and pointInPoly(
                            corner_facet.corner, poly.points
                        ):
                            corner_in_poly = True

                    if (
                        candidate_assignments is None
                        or not candidate_errors
                        or not corner_in_poly
                    ):
                        continue

                    error_geomean = 1.0
                    for candidate_error in candidate_errors:
                        error_geomean *= candidate_error ** (
                            1 / len(candidate_errors)
                        )
                    if best_fit is None or error_geomean < best_fit[0]:
                        best_fit = (
                            error_geomean,
                            candidate_assignments,
                        )

        if (
            best_fit is None
            or best_fit[0] >= NeighboredPolygon.curved_corner_area_threshold
        ):
            return None

        return dict(best_fit[1])

    def _try_local_curved_corner_loop_rescue(self, target_poly):
        loop_cluster = self._collect_small_neighbor_loop(target_poly)
        if loop_cluster is None:
            return None
        return self._fit_local_curved_corner_cluster(loop_cluster)

    def _try_local_curved_corner_transition_rescue(self, target_poly):
        cluster = [target_poly]
        for neighbor in [target_poly.getLeftNeighbor(), target_poly.getRightNeighbor()]:
            if (
                neighbor is None
                or not neighbor.hasFacet()
                or not self._is_curved_corner_support_facet(neighbor.getFacet())
            ):
                continue

            neighbor_facet = neighbor.getFacet()
            if (
                self._is_line_like_support_facet(neighbor_facet)
                or abs(neighbor_facet.curvature)
                > NeighboredPolygon.curved_corner_curvature_threshold
            ):
                cluster.append(neighbor)

        if len(cluster) < 2:
            return None

        return self._fit_local_curved_corner_cluster(cluster)


    # Merge the polys corresponding to merge_ids
    # merge_ids = list of merge_ids
    def _merge(self, merge_ids):
        # Get coords of polys to be merged
        # Get new neighbor ids
        merge_coords = []
        neighbor_ids = set()
        for merge_id in merge_ids:
            merge_coords += self._get_merge_coords(merge_id)
            some_neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
            # For each neighbor
            for neighbor_id in some_neighbor_ids:
                if neighbor_id not in merge_ids:
                    neighbor_ids.add(neighbor_id)
                    # Replace old merge id with new one
                    self.merge_id_to_neighbor_ids[neighbor_id] = [
                        (self.next_merge_id if x == merge_id else x)
                        for x in self.merge_id_to_neighbor_ids[neighbor_id]
                    ]

        self.merge_ids_to_coords.append(merge_coords)
        self.merge_id_to_neighbor_ids.append(list(neighbor_ids))

        for merge_coord in merge_coords:
            [x, y] = merge_coord
            self.coords_to_merge_id[x][y] = self.next_merge_id

        self.next_merge_id += 1

    # Each time new fractions are set, run merging algorithm
    def setFractions(self, fractions):
        super().setFractions(fractions)

        def _helper_isMixed(x, y):
            if x < 0 or y < 0 or x >= len(self.polys) or y >= len(self.polys[0]):
                return False
            return self.polys[x][y].isMixed()

        # Set global values to initials
        self.merge_ids_to_coords = []
        self.merge_id_to_neighbor_ids = []
        self.next_merge_id = 0
        self.coords_to_merge_id = [
            [None for _ in range(len(self.polys[0]))] for _ in range(len(self.polys))
        ]
        self.merged_polys = dict()

        # Set each individual poly to its own merge id
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    self.merge_ids_to_coords.append([[x, y]])
                    self.merge_id_to_neighbor_ids.append([])
                    self.coords_to_merge_id[x][y] = self.next_merge_id
                    self.next_merge_id += 1

        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        # Locate mixed neighbors and set neighbor ids
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    merge_id = self._get_merge_id(x, y)
                    for dir in dirs:
                        neighbor_coords = [x + dir[0], y + dir[1]]
                        if _helper_isMixed(neighbor_coords[0], neighbor_coords[1]):
                            neighbor_id = self._get_merge_id(
                                neighbor_coords[0], neighbor_coords[1]
                            )
                            self.merge_id_to_neighbor_ids[merge_id].append(neighbor_id)

    # TODO add check for case of diagonal neighbors
    def merge1Neighbors(self):
        print("Merging 1 neighbors")
        process_queue = []

        # Find all 1 neighbors
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                merge_id = self._get_merge_id(x, y)
                if merge_id is not None:
                    neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
                    if len(neighbor_ids) == 1:
                        process_queue.append(merge_id)

        # Merge all 1 neighbors
        while process_queue:
            merge_id = process_queue.pop(0)
            [x, y] = self._get_merge_coords(merge_id)[0]
            neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
            # By here, this poly should only have one neighbor. If not, low resolution: case of a long chain of mixed cells?
            if (
                len(neighbor_ids) != 1 or self._get_merge_id(x, y) != merge_id
            ):  # second case is when merge_id has already been merged somehow
                raise ValueError(
                    f"Error in merge1Neighbors: neighbor_ids {neighbor_ids} has length != 1. Low resolution: long chain of mixed cells?"
                )
            # Check if neighbor only has two neighbors: if so, after merging, merged poly would again have a single neighbor, which could be a long chain of mixed cells. We want to avoid this.
            neighbor_id = neighbor_ids[0]
            if len(self._get_neighbor_ids_from_merge_id(neighbor_id)) < 3:
                # Probably a long chain of mixed cells. Instead of merging with neighbor cell, find adjacent empty/full cell that's closest to being mixed and merge with that instead.
                # For now, instead of trying to turn a full cell into a mixed cell, just skip it
                # merge_coords = self._get_merge_coords(merge_id)
                # # merge_coords should be shape [[x, y]]
                # [full_x, full_y] = self.getMostMixedAdjacentFullCell(merge_coords[0][0], merge_coords[0][1])
                # self._merge([merge_id, self._get_merge_id(full_x, full_y)])
                pass
            else:
                # No issue, merge with its single neighbor
                self._merge([merge_id, neighbor_ids[0]])

    def _helper_createMergedPolys(self, merge_coords_list):
        # Merge polys by coordinates
        merge_coords = merge_coords_list.copy()
        if len(merge_coords) == 1:
            merge_points = self.polys[merge_coords[0][0]][merge_coords[0][1]].points
        else:
            merge_points = []
            i = 0
            while i < len(merge_coords):
                try:
                    merge_points = mergePolys(
                        merge_points,
                        self.polys[merge_coords[i][0]][merge_coords[i][1]].points,
                    )
                    merge_coords.pop(i)
                    i = 0
                except:
                    i += 1

            if len(merge_coords) > 0:
                raise ValueError(
                    f"Error in _helper_createMergedPolys: number of polys to merge: {len(merge_coords)}"
                )

        ret_poly = NeighboredPolygon(merge_points)
        ret_poly.use_linear_corner_locality_guard = self.use_linear_corner_locality_guard
        ret_poly.require_both_linear_corner_branches = (
            self.require_both_linear_corner_branches
        )
        ret_poly.use_corner_branch_propagation = self.use_corner_branch_propagation
        total_area = sum(
            list(map(lambda x: self.polys[x[0]][x[1]].getArea(), merge_coords_list))
        )
        ret_poly.setArea(total_area)

        # Set adjacent polys
        adjacent_polys = []
        for merge_coord in merge_coords_list:
            for neighbor in self.polys[merge_coord[0]][merge_coord[1]].adjacent_polys:
                if neighbor not in adjacent_polys:
                    adjacent_polys.append(neighbor)
                    # Update neighbor's adjacent polys
                    # TODO JL 4/16/25: this try catch is a hack to fix a bug where the neighbor is not in the adjacent_polys list. Need to figure out why this is happening.
                    try:
                        neighbor.adjacent_polys.remove(
                            self.polys[merge_coord[0]][merge_coord[1]]
                        )
                    except:
                        # breakpoint()
                        # TODO JL 5/29/25: pass for now? we need to figure out why this is happening though
                        pass
                    neighbor.adjacent_polys.append(ret_poly)

        ret_poly.adjacent_polys = adjacent_polys

        return ret_poly

    # Resets self.merged_polys according to the latest values in self.merge_ids_to_coords
    # Coordinates only, no neighbors
    def createMergedPolys(self):
        # Set global variables to initials
        self.merged_polys = dict()

        # Add all merge ids in use to the queue
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                merge_id = self._get_merge_id(x, y)
                if merge_id is not None and merge_id not in self.merged_polys.keys():
                    merge_coords = self._get_merge_coords(merge_id).copy()
                    self.merged_polys[merge_id]: NeighboredPolygon = (
                        self._helper_createMergedPolys(merge_coords)
                    )
                    self._attach_facet_provenance(
                        self.merged_polys[merge_id], merge_id
                    )

    # Runs youngs on all mixed cells. Run setFractions and createMergedPolys before.
    def runYoungs(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_facet = mixed_poly.runYoungs(ret=True)
                    self.merged_polys[self._get_merge_id(x, y)].setFacet(mixed_facet)

    # Runs ELVIRA on all mixed cells. Run setFractions and createMergedPolys before.
    def runELVIRA(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_facet = mixed_poly.runELVIRA(ret=True)
                    self.merged_polys[self._get_merge_id(x, y)].setFacet(mixed_facet)

    # Runs LVIRA on all mixed cells. Run setFractions and createMergedPolys before.
    def runLVIRA(self):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_facet = mixed_poly.runLVIRA(ret=True)
                    self.merged_polys[self._get_merge_id(x, y)].setFacet(mixed_facet)

    # Runs linear on all mixed cells (default to Youngs). Run setFractions and createMergedPolys before.
    def runSafeLinear(
        self, check_threshold=True, default_to_youngs=True, fit_1neighbor=False
    ):
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_facet = mixed_poly.runSafeLinear(
                        ret=True,
                        check_threshold=check_threshold,
                        default_to_youngs=default_to_youngs,
                        fit_1neighbor=fit_1neighbor,
                    )
                    self.merged_polys[self._get_merge_id(x, y)].setFacet(mixed_facet)

    # Runs independent-cell circular reconstruction. Run setFractions and createMergedPolys before.
    def runSafeCircle(
        self, plic_fallback="LVIRA", arc_failure_fallback="local_linear"
    ):
        self.plic_fallback_records = []
        self.safe_circle_fallback_records = []
        self.facet_provenance_events = []
        self._provenance_event_order = 0
        self._provenance_stage = "safe_circle"
        self._provenance_override = None
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    merge_id = self._get_merge_id(x, y)
                    merge_coords = self._get_merge_coords(merge_id)
                    with arc_facet_log_context(
                        call_source="safe_circle",
                        grid_coords=[x, y],
                        merge_id=merge_id,
                        merge_coords=merge_coords,
                    ):
                        mixed_facet, fallback_record = mixed_poly.runSafeCircle(
                            ret=True,
                            plic_fallback=plic_fallback,
                            arc_failure_fallback=arc_failure_fallback,
                            return_info=True,
                        )

                    merged_poly = self.merged_polys[merge_id]
                    previous_override = self._provenance_override
                    if fallback_record is not None:
                        record = {
                            "setting": "safe_circle",
                            "merge_id": merge_id,
                            "policy": fallback_record.get("policy", ""),
                            "reason": fallback_record.get("reason", ""),
                            "event_kind": fallback_record.get("event_kind", ""),
                            "facet_name": getattr(mixed_facet, "name", ""),
                            "num_vertices": len(merged_poly.points),
                        }
                        self.safe_circle_fallback_records.append(record)
                        self._provenance_override = {merge_id: fallback_record}
                        if fallback_record.get("event_kind") == "plic_fallback":
                            self._append_plic_fallback_record(
                                "safe_circle",
                                merge_id,
                                merged_poly,
                                mixed_facet,
                                fallback_record.get("policy", ""),
                            )
                    try:
                        merged_poly.setFacet(mixed_facet)
                    finally:
                        self._provenance_override = previous_override

    # 1. Runs linear on all mixed cells
    # 2. Tries linear corners on all mixed cells
    # 3. Defaults to Youngs
    # Run setFractions and createMergedPolys before.
    def runSafeLinearCorner(self):
        # 1. Run linear on all mixed cells (implicitly sets neighbors for simple cases)
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_poly.set3x3Stencil(self.get3x3Stencil(x, y))
                    mixed_facet = mixed_poly.runSafeLinear(
                        ret=True, check_threshold=True, default_to_youngs=False
                    )
                    print(mixed_facet)
                    if mixed_facet is not None and mixed_facet.name == "linear":
                        # Linearity check passed, set facet
                        self.merged_polys[self._get_merge_id(x, y)].setFacet(
                            mixed_facet
                        )

        # 2. For mixed cells with left/right neighbors, try linear corners
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed():
                    mixed_poly: NeighboredPolygon = self.merged_polys[
                        self._get_merge_id(x, y)
                    ]
                    if mixed_poly.fullyOriented() and not (mixed_poly.hasFacet()):
                        print("Start corner check")
                        # Fully oriented but no linear facet: try linear corners
                        left: NeighboredPolygon = mixed_poly.getLeftNeighbor()
                        right: NeighboredPolygon = mixed_poly.getRightNeighbor()
                        print(f"Left: {left}")
                        print(f"Right: {right}")
                        # Loop through left until a linear facet is found
                        doneLeft = False
                        success = True
                        loopcounter = 0
                        while not (doneLeft) and loopcounter < 50:  # TODO
                            loopcounter += 1
                            # Handle cases: looped all the way around, or no left neighbor
                            if left == mixed_poly or left == right or left is None:
                                doneLeft = True
                                success = False
                            elif left.hasFacet() and left.getFacet().name in [
                                "Youngs",
                                "ELVIRA",
                                "LVIRA",
                                "linear",
                            ]:
                                doneLeft = True
                                success = True
                            else:
                                left: NeighboredPolygon = left.getLeftNeighbor()
                        if loopcounter >= 50:
                            success = False
                        # Either left = closest neighbor on left with linear facet, or success = False
                        # Loop through right until a linear facet is found
                        doneRight = not (success)
                        loopcounter = 0
                        while not (doneRight) and loopcounter < 50:
                            loopcounter += 1
                            # Handle cases: looped all the way around, or no right neighbor
                            if right == mixed_poly or right == left or right is None:
                                doneRight = True
                                success = False
                            elif right.hasFacet() and right.getFacet().name in [
                                "Youngs",
                                "ELVIRA",
                                "LVIRA",
                                "linear",
                            ]:
                                doneRight = True
                                success = True
                            else:
                                right: NeighboredPolygon = right.getRightNeighbor()
                        if loopcounter >= 50:
                            success = False
                        # If success, try linear corners
                        if success:
                            print(f"Trying linear corner for {mixed_poly.points}")
                            # Fits linear corner if possible, else no-op
                            mixed_poly.checkCornerFacet(
                                left.getFacet().pLeft,
                                left.getFacet().pRight,
                                right.getFacet().pRight,
                                right.getFacet().pLeft,
                            )
                            # If successful, set facet
                            if mixed_poly.hasFacet():
                                self.merged_polys[self._get_merge_id(x, y)].setFacet(
                                    mixed_poly.getFacet()
                                )

        # 3. Defaults to Youngs
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                if self.polys[x][y].isMixed() and not (
                    self.merged_polys[self._get_merge_id(x, y)].hasFacet()
                ):
                    mixed_poly: BasePolygon = self.polys[x][y]
                    mixed_facet = mixed_poly.runYoungs(ret=True)
                    self.merged_polys[self._get_merge_id(x, y)].setFacet(mixed_facet)

    # 1. Runs linear on all mixed cells
    # 2. Tries linear corners on all mixed cells
    # 3. Runs circular on all mixed cells
    # 4. Tries circular corners on all mixed cells
    # 5. Defaults to Youngs
    # def runSafeCircularCorner(self):

    def advectMergedFacets(self, velocity, t, dt, checkSize=2):
        print("Advecting facets and recalculating areas")
        advected_areas = [[0] * len(self.polys[0]) for _ in range(len(self.polys))]
        advected_facets = []
        # List to keep track of merge ids such that advected facet has been appended
        processed_merge_ids = []

        # Advect interface and intersect with neighbors
        for advectx in range(len(self.polys)):
            for advecty in range(len(self.polys[0])):
                if self.polys[advectx][advecty].getFraction() > self.threshold:
                    # shiftpoly
                    shiftpoly = list(
                        map(
                            lambda x: advectPoint(x, velocity, t, dt),
                            self.polys[advectx][advecty].points,
                        )
                    )
                    shiftbounds = [
                        min(list(map(lambda x: x[0], shiftpoly))),
                        min(list(map(lambda x: x[1], shiftpoly))),
                        max(list(map(lambda x: x[0], shiftpoly))),
                        max(list(map(lambda x: x[1], shiftpoly))),
                    ]

                    for testx in range(-checkSize, checkSize + 1):
                        for testy in range(-checkSize, checkSize + 1):
                            checkx = advectx - testx
                            checky = advecty - testy
                            if (
                                checkx >= 0
                                and checkx < len(self.polys)
                                and checky >= 0
                                and checky < len(self.polys[0])
                            ):
                                testpoly = self.polys[checkx][checky].points
                                testbounds = [
                                    min(list(map(lambda x: x[0], testpoly))),
                                    min(list(map(lambda x: x[1], testpoly))),
                                    max(list(map(lambda x: x[0], testpoly))),
                                    max(list(map(lambda x: x[1], testpoly))),
                                ]
                                if not (
                                    testbounds[2] <= shiftbounds[0]
                                    or shiftbounds[2] <= testbounds[0]
                                    or testbounds[3] <= shiftbounds[1]
                                    or shiftbounds[3] <= testbounds[1]
                                ):
                                    # bounding boxes intersect, could be nonzero intersection
                                    # TODO is this part still necessary?
                                    try:
                                        polyintersections = getPolyIntersectArea(
                                            shiftpoly, testpoly
                                        )
                                    except:
                                        print(
                                            "Failed polyintersect: getPolyIntersectArea({}, {})".format(
                                                shiftpoly, testpoly
                                            )
                                        )
                                        testpoly = list(
                                            map(
                                                lambda x: [x[0] + 1e-13, x[1] + 1e-13],
                                                testpoly,
                                            )
                                        )
                                        polyintersections = getPolyIntersectArea(
                                            shiftpoly, testpoly
                                        )
                                    if len(polyintersections) == 0:
                                        # No intersection
                                        continue
                                    # For each overlap region
                                    for polyintersection in polyintersections:
                                        advect_merge_id = self._get_merge_id(
                                            advectx, advecty
                                        )
                                        if advect_merge_id is None:
                                            # Full cell
                                            # TODO is this necessary?
                                            try:
                                                assert (
                                                    self.polys[advectx][
                                                        advecty
                                                    ].getFraction()
                                                    > 1 - self.threshold
                                                )
                                            except:
                                                print(
                                                    self.polys[advectx][advecty].points
                                                )
                                                print(
                                                    self.polys[advectx][
                                                        advecty
                                                    ].getFraction()
                                                )
                                                print(
                                                    self.polys[advectx][
                                                        advecty
                                                    ].getArea()
                                                )
                                                print(1 / 0)
                                            advected_areas[checkx][checky] += abs(
                                                getArea(polyintersection)
                                            )
                                        else:
                                            # Mixed cell with facet
                                            advectedfacet = self.merged_polys[
                                                advect_merge_id
                                            ].getFacet()

                                            # Linear or arc
                                            advectedfacet = advectedfacet.advected(
                                                velocity, t, dt
                                            )

                                            # Add to return
                                            if (
                                                advect_merge_id
                                                not in processed_merge_ids
                                            ):
                                                processed_merge_ids.append(
                                                    advect_merge_id
                                                )
                                                advected_facets.append(advectedfacet)
                                            if isinstance(advectedfacet, LinearFacet):
                                                polyintersectionarea = getPolyLineArea(
                                                    polyintersection,
                                                    advectedfacet.pLeft,
                                                    advectedfacet.pRight,
                                                )
                                            elif isinstance(advectedfacet, ArcFacet):
                                                # print(polyintersection)
                                                # print(advectedfacet)
                                                # polyintersectionarea = advectedfacet.getPolyIntersectArea(polyintersection)
                                                # print(polyintersectionarea)
                                                polyintersectionarea, _ = (
                                                    getCircleIntersectArea(
                                                        advectedfacet.center,
                                                        advectedfacet.radius,
                                                        polyintersection,
                                                    )
                                                )
                                                # print(polyintersectionarea)
                                            elif isinstance(advectedfacet, CornerFacet):
                                                polyintersectionarea = (
                                                    getPolyCurvedCornerArea(
                                                        polyintersection,
                                                        advectedfacet.pLeft,
                                                        advectedfacet.corner,
                                                        advectedfacet.pRight,
                                                        advectedfacet.radiusLeft,
                                                        advectedfacet.radiusRight,
                                                    )
                                                )
                                                # polyintersectionarea = getPolyCornerArea(polyintersection, advectedfacet.pLeft, advectedfacet.corner, advectedfacet.pRight)
                                            else:
                                                print(
                                                    "Unknown facet type in advectMergedFacets"
                                                )
                                                raise ValueError(advectedfacet.name)
                                            advected_areas[checkx][
                                                checky
                                            ] += polyintersectionarea  # TODO: abs here?
                                            # TODO necessary?
                                            if polyintersectionarea < 0:
                                                print("Negative polyintersectionarea")

                                            # TODO handle corner case here
                                            """
                                                elif len(predfacets[advectx][advecty]) == 2:
                                                    #Corner
                                                    advectedfacet1 = predfacets[advectx][advecty][0].advected(velocity, t, dt)
                                                    advectedfacet1r = advectedfacet1.radius if advectedfacet1.name == 'arc' else None
                                                    plot_advectedfacets.append(advectedfacet1)
                                                    advectedfacet2 = predfacets[advectx][advecty][1].advected(velocity, t, dt)
                                                    advectedfacet2r = advectedfacet2.radius if advectedfacet2.name == 'arc' else None
                                                    plot_advectedfacets.append(advectedfacet2)
                                                    nareas[checkx][checky] += getPolyCurvedCornerArea(polyintersection, advectedfacet1.pLeft, advectedfacet1.pRight, advectedfacet2.pRight, advectedfacet1r, advectedfacet2r)
                                                else:
                                                    print("More than two facets in this cell?")
                                            """

        # Update areas
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                advected_areas[x][y] = min(
                    max(advected_areas[x][y] / self.polys[x][y].getMaxArea(), 0), 1
                )

        self.setFractions(advected_areas)
        return advected_facets

    # JL 2024-06-04
    # Idea: only assumptions we make are that we can identify the intended left/right when mixed cell is "obvious" (2 mixed neighbors only and area fractions match standard Cartesian cases), and from there we only use other left/right neighbors deduced by "process of elimination".
    # Crucially, no guessing is done and no merging is done based on assumptions.
    def findSafeOrientations(self):

        class MergeIdWithNeighbors:

            def __init__(self, merge_id, neighbor_ids: list):
                self.merge_id = merge_id
                self.neighbor_ids = neighbor_ids.copy()
                self.left_neighbor_id = None
                self.right_neighbor_id = None

            def has_left(self):
                return self.left_neighbor_id is not None

            def has_right(self):
                return self.right_neighbor_id is not None

            def set_left(self, neighbor_id, set_neighbor=True):
                if neighbor_id not in self.neighbor_ids:
                    raise ValueError(
                        f"Error in set_left of findOrientations: neighbor_id {neighbor_id} is not in {self.neighbor_ids}"
                    )
                if self.has_left():
                    raise ValueError(
                        f"Error in set_left of findOrientations: left neighbor is already {self.left_neighbor_id}"
                    )
                self.neighbor_ids.remove(neighbor_id)
                self.left_neighbor_id = neighbor_id
                if self.fully_oriented():
                    for id in self.neighbor_ids:
                        self.remove_neighbor_id(id)
                if set_neighbor:
                    neighbor_obj = merge_id_to_obj[neighbor_id]
                    neighbor_obj.set_right(self.get_merge_id(), set_neighbor=False)

            def set_right(self, neighbor_id, set_neighbor=True):
                if neighbor_id not in self.neighbor_ids:
                    raise ValueError(
                        f"Error in set_right of findOrientations: neighbor_id {neighbor_id} is not in {self.neighbor_ids}"
                    )
                if self.has_right():
                    raise ValueError(
                        f"Error in set_right of findOrientations: right neighbor is already {self.right_neighbor_id}"
                    )
                self.neighbor_ids.remove(neighbor_id)
                self.right_neighbor_id = neighbor_id
                if self.fully_oriented():
                    for id in self.neighbor_ids:
                        self.remove_neighbor_id(id)
                if set_neighbor:
                    neighbor_obj = merge_id_to_obj[neighbor_id]
                    neighbor_obj.set_left(self.get_merge_id(), set_neighbor=False)

            def get_left(self):
                return self.left_neighbor_id

            def get_right(self):
                return self.right_neighbor_id

            # Unassigned neighbor ids
            def get_neighbor_ids(self):
                return self.neighbor_ids

            def get_merge_id(self):
                return self.merge_id

            # Always removes a whole neighbor interaction: when removing neighbor id, also removes its own merge id from neighbor
            def remove_neighbor_id(self, neighbor_id):
                neighbor_obj: MergeIdWithNeighbors = merge_id_to_obj[neighbor_id]
                if neighbor_id not in self.neighbor_ids:
                    raise ValueError(
                        f"Error in remove_neighbor_id of findOrientations: neighbor_id {neighbor_id} is not in neighbors {self.neighbor_ids}"
                    )
                elif self.merge_id not in neighbor_obj.neighbor_ids:
                    raise ValueError(
                        f"Error in remove_neighbor_id of findOrientations: merge_id {self.merge_id} is not in neighbor's neighbors {neighbor_obj.neighbor_ids}"
                    )
                self.neighbor_ids.remove(neighbor_id)
                neighbor_obj.neighbor_ids.remove(self.merge_id)

            def fully_oriented(self):
                return self.has_left() and self.has_right()

            # TODO
            def __str__(self):
                return f"Merge id: {self.merge_id}\nLeft neighbor id: {self.left_neighbor_id}\nRight neighbor id: {self.right_neighbor_id}\nCurrent neighbor candidates: {self.neighbor_ids}\n"

        merge_id_to_obj: Dict[int, MergeIdWithNeighbors] = dict()
        process_queue = []
        processed_merge_ids = []

        # Add all merge ids in use to the queue
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                merge_id = self._get_merge_id(x, y)
                if merge_id is not None and merge_id not in processed_merge_ids:
                    processed_merge_ids.append(merge_id)
                    neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
                    merge_id_with_neighbors = MergeIdWithNeighbors(
                        merge_id, neighbor_ids
                    )
                    merge_id_to_obj[merge_id] = merge_id_with_neighbors
                    process_queue.append(merge_id)

        def try_base_orientation_hint(merge_id_with_neighbors):
            if not self.use_three_neighbor_orientation_hint:
                return False
            if (
                merge_id_with_neighbors.has_left()
                or merge_id_with_neighbors.has_right()
                or len(merge_id_with_neighbors.get_neighbor_ids()) < 3
                or len(self._get_merge_coords(merge_id_with_neighbors.get_merge_id())) != 1
            ):
                return False

            x, y = self._get_merge_coords(merge_id_with_neighbors.get_merge_id())[0]
            base_poly = self.polys[x][y]
            base_poly.set3x3Stencil(self.get3x3Stencil(x, y))
            orientation = base_poly.findSafeOrientation(fit_1neighbor=False)
            if orientation is None:
                return False

            neighbor_ids_by_poly = {}
            for dx, dy in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < len(self.polys)
                    and 0 <= ny < len(self.polys[0])
                    and self.polys[nx][ny].isMixed()
                ):
                    neighbor_ids_by_poly[id(self.polys[nx][ny])] = self._get_merge_id(nx, ny)

            try:
                left_id = neighbor_ids_by_poly[id(orientation[0])]
                right_id = neighbor_ids_by_poly[id(orientation[1])]
            except KeyError:
                return False

            if left_id == right_id:
                return False
            if left_id not in merge_id_with_neighbors.get_neighbor_ids():
                return False
            if right_id not in merge_id_with_neighbors.get_neighbor_ids():
                return False
            if merge_id_to_obj[left_id].has_right() or merge_id_to_obj[right_id].has_left():
                return False

            merge_id_with_neighbors.set_left(left_id)
            merge_id_with_neighbors.set_right(right_id)
            return True

        # Add new MergeIdWithNeighbors object to merge_id_to_obj which corresponds to merging merge_id with neighbor_id
        def mergeObjs(merge_id, neighbor_id, use_neighbor_id_orientation=False):
            # If either id is not in merge_id_to_obj, throw error
            if merge_id not in merge_id_to_obj.keys():
                raise ValueError(f"{merge_id} merge id not in {merge_id_to_obj.keys()}")
            elif neighbor_id not in merge_id_to_obj.keys():
                raise ValueError(
                    f"{neighbor_id} neighbor id not in {merge_id_to_obj.keys()}"
                )

            merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[merge_id]
            neighbor_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                neighbor_id
            ]
            # For each of merge_id's (original) neighbors, replace merge_id with self.next_merge_id
            for merge_neighbor_id in self._get_neighbor_ids_from_merge_id(merge_id):
                if merge_neighbor_id != neighbor_id:
                    neighbor_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                        merge_neighbor_id
                    ]
                    if (
                        neighbor_with_neighbors.has_left()
                        and neighbor_with_neighbors.get_left() == merge_id
                    ):
                        neighbor_with_neighbors.left_neighbor_id = self.next_merge_id
                    if (
                        neighbor_with_neighbors.has_right()
                        and neighbor_with_neighbors.get_right() == merge_id
                    ):
                        neighbor_with_neighbors.right_neighbor_id = self.next_merge_id
                    for i in range(len(neighbor_with_neighbors.get_neighbor_ids())):
                        if neighbor_with_neighbors.get_neighbor_ids()[i] == merge_id:
                            neighbor_with_neighbors.get_neighbor_ids()[
                                i
                            ] = self.next_merge_id
            # For each of neighbor_id's (original) neighbors, replace neighbor_id with self.next_merge_id
            for neighbor_neighbor_id in self._get_neighbor_ids_from_merge_id(
                neighbor_id
            ):
                if neighbor_neighbor_id != merge_id:
                    neighbor_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                        neighbor_neighbor_id
                    ]
                    if (
                        neighbor_with_neighbors.has_left()
                        and neighbor_with_neighbors.get_left() == neighbor_id
                    ):
                        neighbor_with_neighbors.left_neighbor_id = self.next_merge_id
                    if (
                        neighbor_with_neighbors.has_right()
                        and neighbor_with_neighbors.get_right() == neighbor_id
                    ):
                        neighbor_with_neighbors.right_neighbor_id = self.next_merge_id
                    for i in range(len(neighbor_with_neighbors.get_neighbor_ids())):
                        if neighbor_with_neighbors.get_neighbor_ids()[i] == neighbor_id:
                            neighbor_with_neighbors.get_neighbor_ids()[
                                i
                            ] = self.next_merge_id

            # New list of neighbors is merge_id's neighbors + neighbor_id's neighbors
            # Reset left/right neighbors
            new_neighbor_ids = list(
                filter(
                    lambda x: x != merge_id and x != neighbor_id,
                    merge_id_with_neighbors.get_neighbor_ids()
                    + neighbor_id_with_neighbors.get_neighbor_ids(),
                )
            )
            new_merge_id_with_neighbors = MergeIdWithNeighbors(
                self.next_merge_id, new_neighbor_ids
            )
            if use_neighbor_id_orientation:
                # Danger: potential to ruin neighbor operation symmetry
                if (
                    neighbor_id_with_neighbors.has_left()
                    and neighbor_id_with_neighbors.get_left() != merge_id
                ):
                    new_merge_id_with_neighbors.left_neighbor_id = (
                        neighbor_id_with_neighbors.get_left()
                    )
                if (
                    neighbor_id_with_neighbors.has_right()
                    and neighbor_id_with_neighbors.get_right() != merge_id
                ):
                    new_merge_id_with_neighbors.right_neighbor_id = (
                        neighbor_id_with_neighbors.get_right()
                    )
            merge_id_to_obj[self.next_merge_id] = new_merge_id_with_neighbors

            # Remove merge_id and neighbor_id from merge_id_to_obj
            merge_id_to_obj.pop(merge_id)
            merge_id_to_obj.pop(neighbor_id)
            # Merge polygons and original neighbors
            self._merge([merge_id, neighbor_id])

            return new_merge_id_with_neighbors

        def doGreedyOrientations():
            iters_without_progress = 0
            while process_queue:
                merge_id = process_queue.pop(0)
                merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                    merge_id
                ]
                # 1 neighbor
                if len(merge_id_with_neighbors.get_neighbor_ids()) == 1:
                    neighbor_id = merge_id_with_neighbors.get_neighbor_ids()[0]
                    # If merge poly only has one unassigned neighbor and only needs to assign one more neighbor, do it
                    if (
                        merge_id_with_neighbors.has_left()
                        and not (merge_id_with_neighbors.has_right())
                        and not (merge_id_to_obj[neighbor_id].has_left())
                    ):
                        merge_id_with_neighbors.set_right(neighbor_id)
                        iters_without_progress = 0
                    elif (
                        merge_id_with_neighbors.has_right()
                        and not (merge_id_with_neighbors.has_left())
                        and not (merge_id_to_obj[neighbor_id].has_right())
                    ):
                        merge_id_with_neighbors.set_left(neighbor_id)
                        iters_without_progress = 0
                    # There should never be a case when neither orientation is set but only one neighbor
                    elif not (merge_id_with_neighbors.has_left()) and not (
                        merge_id_with_neighbors.has_right()
                    ):
                        print(
                            "Neither orientation is set but one neighbor: why did this happen?"
                        )
                        # TODO not sure how to handle this, merge into its neighbor for now
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                        # raise ValueError("Neither orientation is set but one neighbor: why did this happen?")
                    # This case should be dealt with via implementation of remove_neighbor_id
                    elif merge_id_with_neighbors.fully_oriented():
                        print("Fully oriented but one neighbor: why did this happen?")
                        # TODO is this an actual issue? For now, assume this cell is ok and just remove from queue
                        # process_queue.append(merge_id_with_neighbors)
                        # iters_without_progress += 1
                        # raise ValueError("Fully oriented but one neighbor: why did this happen?")
                    else:  # single neighbor but it's already got the neighbor we would want to assign merge_id to
                        print(
                            "Single neighbor but orientations are inconsistent, skip for now"
                        )
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                # 2 neighbors
                elif len(merge_id_with_neighbors.get_neighbor_ids()) == 2:
                    if not (merge_id_with_neighbors.has_left()) and not (
                        merge_id_with_neighbors.has_right()
                    ):
                        # If merge poly consists of only one poly and has exactly two neighbors
                        if len(self._get_merge_coords(merge_id)) == 1:
                            # Check if orientation is easy to figure out
                            [x, y] = self._get_merge_coords(merge_id)[0]
                            dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                            neighbor_dirs = []
                            nonneighbor_dirs = []

                            def _helper_in_bounds(x, y):
                                return not (
                                    x < 0
                                    or y < 0
                                    or x >= len(self.polys)
                                    or y >= len(self.polys[0])
                                )

                            for i, dir in enumerate(dirs):
                                neighbor_coords = [x + dir[0], y + dir[1]]
                                if (
                                    _helper_in_bounds(
                                        neighbor_coords[0], neighbor_coords[1]
                                    )
                                    and self.polys[neighbor_coords[0]][
                                        neighbor_coords[1]
                                    ].isMixed()
                                    and self._get_merge_id(
                                        neighbor_coords[0], neighbor_coords[1]
                                    )
                                    in merge_id_with_neighbors.get_neighbor_ids()
                                ):
                                    neighbor_dirs.append(i)
                                else:
                                    nonneighbor_dirs.append(i)
                            # TODO go by self._get_neighbor_ids_from_merge_ids instead?
                            # if more than two neighbors, then this cell used to have more than two mixed neighbors but some were eliminated, pass
                            if len(neighbor_dirs) > 2:
                                print(
                                    "Passing on case where cell used to have more than two mixed neighbors but only two are left, still not easily orientable"
                                )
                                process_queue.append(merge_id)
                                iters_without_progress += 1
                                continue
                            elif len(neighbor_dirs) < 2:
                                # TODO this case might never happen
                                print(
                                    "Passing on case where cell has fewer than two cells, not easily orientable"
                                )
                                process_queue.append(merge_id)
                                iters_without_progress += 1
                                continue

                            # Verifies neighbors = coordinate neighbors
                            # print(merge_id_with_neighbors.get_neighbor_ids())
                            # print(list(map(lambda d: self._get_merge_id(x+dirs[d][0], y+dirs[d][1]), neighbor_dirs)))

                            # otherwise, two mixed neighbors are either across from each other or adjacent
                            neighbor_modes = abs(
                                neighbor_dirs[0] - neighbor_dirs[1]
                            )  # = 1 or 2 or 3 (3 is same case as 1)
                            # 1,3 = adjacent; 2 = across
                            nonneighbor_statuses = []
                            for nonneighbor_dir in nonneighbor_dirs:
                                nonneighbor_coords = [
                                    x + dirs[nonneighbor_dir][0],
                                    y + dirs[nonneighbor_dir][1],
                                ]
                                if (
                                    _helper_in_bounds(
                                        nonneighbor_coords[0], nonneighbor_coords[1]
                                    )
                                    and self.polys[nonneighbor_coords[0]][
                                        nonneighbor_coords[1]
                                    ].isFull()
                                ):
                                    nonneighbor_statuses.append(1)
                                else:  # must be empty
                                    nonneighbor_statuses.append(0)
                            if neighbor_modes == 1 or neighbor_modes == 3:
                                clockwisemost = (
                                    min(neighbor_dirs) if neighbor_modes == 1 else 3
                                )
                                [c_x, c_y] = [
                                    x + dirs[clockwisemost][0],
                                    y + dirs[clockwisemost][1],
                                ]
                                c_merge_id = self._get_merge_id(c_x, c_y)
                                [cc_x, cc_y] = [
                                    x + dirs[(clockwisemost + 1) % 4][0],
                                    y + dirs[(clockwisemost + 1) % 4][1],
                                ]
                                cc_merge_id = self._get_merge_id(cc_x, cc_y)
                                if (
                                    nonneighbor_statuses == [0, 0]
                                    and not (merge_id_to_obj[c_merge_id].has_left())
                                    and not (merge_id_to_obj[cc_merge_id].has_right())
                                ):  # both empty, clockwise
                                    merge_id_with_neighbors.set_left(
                                        self._get_merge_id(cc_x, cc_y)
                                    )
                                    merge_id_with_neighbors.set_right(
                                        self._get_merge_id(c_x, c_y)
                                    )
                                    iters_without_progress = 0
                                elif (
                                    nonneighbor_statuses == [1, 1]
                                    and not (merge_id_to_obj[c_merge_id].has_right())
                                    and not (merge_id_to_obj[cc_merge_id].has_left())
                                ):  # both full, counterclockwise
                                    merge_id_with_neighbors.set_left(
                                        self._get_merge_id(c_x, c_y)
                                    )
                                    merge_id_with_neighbors.set_right(
                                        self._get_merge_id(cc_x, cc_y)
                                    )
                                    iters_without_progress = 0
                                else:
                                    print(
                                        f"Error in easy orientation with two adjacent neighbors: {nonneighbor_statuses}"
                                    )
                                    process_queue.append(merge_id)
                                    iters_without_progress += 1
                            else:  # neighbor_modes == 2
                                if nonneighbor_statuses == [0, 1]:
                                    full_index = 1
                                elif nonneighbor_statuses == [1, 0]:
                                    full_index = 0
                                else:
                                    print(
                                        f"Error in easy orientation with two opposite neighbors: {nonneighbor_statuses}"
                                    )
                                    process_queue.append(merge_id)
                                    iters_without_progress += 1
                                    if not self.continue_greedy_orientation_conflicts:
                                        break
                                [l_x, l_y] = [
                                    x + dirs[(nonneighbor_dirs[full_index] + 1) % 4][0],
                                    y + dirs[(nonneighbor_dirs[full_index] + 1) % 4][1],
                                ]
                                l_merge_id = self._get_merge_id(l_x, l_y)
                                [r_x, r_y] = [
                                    x + dirs[(nonneighbor_dirs[full_index] - 1) % 4][0],
                                    y + dirs[(nonneighbor_dirs[full_index] - 1) % 4][1],
                                ]
                                r_merge_id = self._get_merge_id(r_x, r_y)
                                if not (
                                    merge_id_to_obj[l_merge_id].has_right()
                                ) and not (merge_id_to_obj[r_merge_id].has_left()):
                                    merge_id_with_neighbors.set_left(l_merge_id)
                                    merge_id_with_neighbors.set_right(r_merge_id)
                                    iters_without_progress = 0
                                else:
                                    print(
                                        "Error in easy orientation with two opposite neighbors but inconsistent orientations"
                                    )
                                    process_queue.append(merge_id)
                                    iters_without_progress += 1
                        # Not easily orientable, pass
                        else:
                            print(
                                "Passing on case with two neighbors and two missing orientations"
                            )
                            process_queue.append(merge_id)
                            iters_without_progress += 1
                    elif merge_id_with_neighbors.fully_oriented():
                        # TODO this doesn't seem to happen
                        print("Fully oriented but two neighbors: why did this happen?")
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                    # One more orientation to be set and two neighbors, pass
                    else:
                        print(
                            "Passing on case with two neighbors and one missing orientation"
                        )
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                # 3+ neighbors
                elif len(merge_id_with_neighbors.get_neighbor_ids()) >= 3:
                    if merge_id_with_neighbors.fully_oriented():
                        raise ValueError(
                            f"Fully oriented but {len(merge_id_with_neighbors.get_neighbor_ids())} neighbors: why did this happen?"
                        )
                    elif try_base_orientation_hint(merge_id_with_neighbors):
                        iters_without_progress = 0
                    else:
                        print("Passing on case with 3+ neighbors")
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                # 0 neighbors
                else:
                    # if still not fully oriented, these are problematic and we want to do something
                    if not (merge_id_with_neighbors.fully_oriented()):
                        print("Zero neighbors and not full oriented, passing")
                        process_queue.append(merge_id)
                        iters_without_progress += 1

                # iters_without_progress = length + 1 (full cycle of queue without progress)
                if iters_without_progress >= len(process_queue) + 1:
                    print(
                        f"Rest of queue cannot be resolved: length {len(process_queue)}"
                    )
                    break

        for merge_id in process_queue.copy():
            if try_base_orientation_hint(merge_id_to_obj[merge_id]):
                continue

        doGreedyOrientations()

        # Cases at this point:
        # 1 neighbor candidate:
        # Two unfilled neighbors
        # One unfilled neighbor but orientations are inconsistent
        # 2 neighbor candidates:
        # Not an "easy orientation" case
        # 3+ neighbor candidates:
        # All
        # 0 neighbor candidates:
        # Not fully oriented

        # Find the vertices of the merged polygons
        self.createMergedPolys()

        # Merge ids that failed to be oriented
        failed_merge_ids = []
        added_merge_ids = dict()
        for merge_id in merge_id_to_obj.keys():
            merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[merge_id]
            if merge_id_with_neighbors.fully_oriented():
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                merged_poly.setNeighbor(
                    self.merged_polys[merge_id_with_neighbors.get_left()], "left"
                )
                merged_poly.setNeighbor(
                    self.merged_polys[merge_id_with_neighbors.get_right()], "right"
                )
                # Check if we used hack and had set left/right neighbor to itself to signify dead-end cell with single mixed neighbor
                if (
                    merge_id_with_neighbors.get_left() == merge_id
                    or merge_id_with_neighbors.get_right() == merge_id
                ):
                    merged_poly.setFacetType("linear_deadend")
            else:
                print("Final failed orientations:")
                print(merge_id_with_neighbors)
                # For all polys in failed merge id, add a lone polygon with a 3x3 stencil and run Young's
                # May still need this poly because it could be a neighbor of something else so can't directly remove it
                for merge_coords in self._get_merge_coords(
                    merge_id_with_neighbors.get_merge_id()
                ):
                    lone_base_poly = self.polys[merge_coords[0]][merge_coords[1]]
                    merged_poly = NeighboredPolygon(lone_base_poly.points)
                    merged_poly.setFraction(lone_base_poly.getFraction())
                    merged_poly.set3x3Stencil(
                        self.get3x3Stencil(merge_coords[0], merge_coords[1])
                    )

                    # TODO does this throw off the algorithm somehow because we're appending to merge_id_to_obj only? (Some invariant where merge_id_to_obj has to have same length as something else?)
                    self.merged_polys[self.next_merge_id] = merged_poly
                    self._attach_facet_provenance(merged_poly, self.next_merge_id)
                    self.coords_to_merge_id[merge_coords[0]][
                        merge_coords[1]
                    ] = self.next_merge_id
                    self.merge_ids_to_coords.append(merge_coords)
                    self.merge_id_to_neighbor_ids.append([])
                    added_merge_ids[self.next_merge_id] = merged_poly
                    self.next_merge_id += 1
                failed_merge_ids.append(merge_id)

        # Remove failed merge ids from list of polygons
        for failed_merge_id in failed_merge_ids:
            merge_id_to_obj.pop(failed_merge_id)
        # Add Young's polygons
        for added_merge_id in added_merge_ids.keys():
            merge_id_to_obj[added_merge_id] = added_merge_ids[added_merge_id]
        return list(merge_id_to_obj.keys())

    # TODO: figure out how to load algo by name
    """
    input: MergeMesh m, side effects on m

    1. Merge single-neighbor mixed cells until all (merged) polys have 2+ neighbors.
    2. For each poly with two neighbors, assume they're the true neighbors. Try to use orientation to determine left or right.
    Also add poly as a neighbor of those neighbors.
    3. For polys with 3+ neighbors, check if potential neighbors already have two neighbors set. If so, remove as potential neighbor. Readd to queue.

    Unclear if this will run into issues, especially when the interface exits the same edge twice.

    Returns merge ids #TODO poly objects instead?
    """

    def findOrientations(self):

        self.orientation_hint_records = []
        self.orientation_retry_records = []
        self.orientation_retry_passes = 0

        class MergeIdWithNeighbors:

            def __init__(self, merge_id, neighbor_ids: list):
                self.merge_id = merge_id
                self.neighbor_ids = neighbor_ids.copy()
                self.left_neighbor_id = None
                self.right_neighbor_id = None

            def has_left(self):
                return self.left_neighbor_id is not None

            def has_right(self):
                return self.right_neighbor_id is not None

            def set_left(self, neighbor_id, set_neighbor=True):
                if neighbor_id not in self.neighbor_ids:
                    raise ValueError(
                        f"Error in set_left of findOrientations: neighbor_id {neighbor_id} is not in {self.neighbor_ids}"
                    )
                if self.has_left():
                    raise ValueError(
                        f"Error in set_left of findOrientations: left neighbor is already {self.left_neighbor_id}"
                    )
                self.neighbor_ids.remove(neighbor_id)
                self.left_neighbor_id = neighbor_id
                if self.fully_oriented():
                    for id in self.neighbor_ids:
                        self.remove_neighbor_id(id)
                if set_neighbor:
                    neighbor_obj = merge_id_to_obj[neighbor_id]
                    neighbor_obj.set_right(self.get_merge_id(), set_neighbor=False)

            def set_right(self, neighbor_id, set_neighbor=True):
                if neighbor_id not in self.neighbor_ids:
                    raise ValueError(
                        f"Error in set_right of findOrientations: neighbor_id {neighbor_id} is not in {self.neighbor_ids}"
                    )
                if self.has_right():
                    raise ValueError(
                        f"Error in set_right of findOrientations: right neighbor is already {self.right_neighbor_id}"
                    )
                self.neighbor_ids.remove(neighbor_id)
                self.right_neighbor_id = neighbor_id
                if self.fully_oriented():
                    for id in self.neighbor_ids:
                        self.remove_neighbor_id(id)
                if set_neighbor:
                    neighbor_obj = merge_id_to_obj[neighbor_id]
                    neighbor_obj.set_left(self.get_merge_id(), set_neighbor=False)

            def get_left(self):
                return self.left_neighbor_id

            def get_right(self):
                return self.right_neighbor_id

            # Unassigned neighbor ids
            def get_neighbor_ids(self):
                return self.neighbor_ids

            def get_merge_id(self):
                return self.merge_id

            # Always removes a whole neighbor interaction: when removing neighbor id, also removes its own merge id from neighbor
            def remove_neighbor_id(self, neighbor_id):
                neighbor_obj: MergeIdWithNeighbors = merge_id_to_obj[neighbor_id]
                if neighbor_id not in self.neighbor_ids:
                    raise ValueError(
                        f"Error in remove_neighbor_id of findOrientations: neighbor_id {neighbor_id} is not in neighbors {self.neighbor_ids}"
                    )
                elif self.merge_id not in neighbor_obj.neighbor_ids:
                    raise ValueError(
                        f"Error in remove_neighbor_id of findOrientations: merge_id {self.merge_id} is not in neighbor's neighbors {neighbor_obj.neighbor_ids}"
                    )
                self.neighbor_ids.remove(neighbor_id)
                neighbor_obj.neighbor_ids.remove(self.merge_id)

            def fully_oriented(self):
                return self.has_left() and self.has_right()

            # TODO
            def __str__(self):
                return f"Merge id: {self.merge_id}\nLeft neighbor id: {self.left_neighbor_id}\nRight neighbor id: {self.right_neighbor_id}\nCurrent neighbor candidates: {self.neighbor_ids}\n"

        merge_id_to_obj: Dict[int, MergeIdWithNeighbors] = dict()
        process_queue = []
        processed_merge_ids = []

        # Add all merge ids in use to the queue
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                merge_id = self._get_merge_id(x, y)
                if merge_id is not None and merge_id not in processed_merge_ids:
                    processed_merge_ids.append(merge_id)
                    neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
                    merge_id_with_neighbors = MergeIdWithNeighbors(
                        merge_id, neighbor_ids
                    )
                    merge_id_to_obj[merge_id] = merge_id_with_neighbors
                    process_queue.append(merge_id)

        def try_base_orientation_hint(merge_id_with_neighbors, phase="early"):
            if phase == "early" and not self.use_three_neighbor_orientation_hint:
                return False
            if phase == "late" and not self.use_late_three_neighbor_orientation_hint:
                return False
            if (
                merge_id_with_neighbors.has_left()
                or merge_id_with_neighbors.has_right()
                or len(merge_id_with_neighbors.get_neighbor_ids()) < 3
                or len(self._get_merge_coords(merge_id_with_neighbors.get_merge_id())) != 1
            ):
                return False

            x, y = self._get_merge_coords(merge_id_with_neighbors.get_merge_id())[0]
            base_poly = self.polys[x][y]
            base_poly.set3x3Stencil(self.get3x3Stencil(x, y))
            orientation = base_poly.findSafeOrientation(fit_1neighbor=False)
            if orientation is None:
                return False

            neighbor_ids_by_poly = {}
            for dx, dy in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < len(self.polys)
                    and 0 <= ny < len(self.polys[0])
                    and self.polys[nx][ny].isMixed()
                ):
                    neighbor_ids_by_poly[id(self.polys[nx][ny])] = self._get_merge_id(nx, ny)

            try:
                left_id = neighbor_ids_by_poly[id(orientation[0])]
                right_id = neighbor_ids_by_poly[id(orientation[1])]
            except KeyError:
                return False

            if left_id == right_id:
                return False
            if left_id not in merge_id_with_neighbors.get_neighbor_ids():
                return False
            if right_id not in merge_id_with_neighbors.get_neighbor_ids():
                return False
            if merge_id_to_obj[left_id].has_right() or merge_id_to_obj[right_id].has_left():
                return False

            merge_id_with_neighbors.set_left(left_id)
            merge_id_with_neighbors.set_right(right_id)
            self.orientation_hint_records.append(
                {
                    "phase": phase,
                    "merge_id": merge_id_with_neighbors.get_merge_id(),
                    "cell": [x, y],
                    "left_merge_id": left_id,
                    "right_merge_id": right_id,
                }
            )
            return True

        # Add new MergeIdWithNeighbors object to merge_id_to_obj which corresponds to merging merge_id with neighbor_id
        def mergeObjs(merge_id, neighbor_id, use_neighbor_id_orientation=False):
            # If either id is not in merge_id_to_obj, throw error
            if merge_id not in merge_id_to_obj.keys():
                raise ValueError(f"{merge_id} merge id not in {merge_id_to_obj.keys()}")
            elif neighbor_id not in merge_id_to_obj.keys():
                raise ValueError(
                    f"{neighbor_id} neighbor id not in {merge_id_to_obj.keys()}"
                )

            merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[merge_id]
            neighbor_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                neighbor_id
            ]
            # For each of merge_id's (original) neighbors, replace merge_id with self.next_merge_id
            for merge_neighbor_id in self._get_neighbor_ids_from_merge_id(merge_id):
                if merge_neighbor_id != neighbor_id:
                    neighbor_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                        merge_neighbor_id
                    ]
                    if (
                        neighbor_with_neighbors.has_left()
                        and neighbor_with_neighbors.get_left() == merge_id
                    ):
                        neighbor_with_neighbors.left_neighbor_id = self.next_merge_id
                    if (
                        neighbor_with_neighbors.has_right()
                        and neighbor_with_neighbors.get_right() == merge_id
                    ):
                        neighbor_with_neighbors.right_neighbor_id = self.next_merge_id
                    for i in range(len(neighbor_with_neighbors.get_neighbor_ids())):
                        if neighbor_with_neighbors.get_neighbor_ids()[i] == merge_id:
                            neighbor_with_neighbors.get_neighbor_ids()[
                                i
                            ] = self.next_merge_id
            # For each of neighbor_id's (original) neighbors, replace neighbor_id with self.next_merge_id
            for neighbor_neighbor_id in self._get_neighbor_ids_from_merge_id(
                neighbor_id
            ):
                if neighbor_neighbor_id != merge_id:
                    neighbor_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                        neighbor_neighbor_id
                    ]
                    if (
                        neighbor_with_neighbors.has_left()
                        and neighbor_with_neighbors.get_left() == neighbor_id
                    ):
                        neighbor_with_neighbors.left_neighbor_id = self.next_merge_id
                    if (
                        neighbor_with_neighbors.has_right()
                        and neighbor_with_neighbors.get_right() == neighbor_id
                    ):
                        neighbor_with_neighbors.right_neighbor_id = self.next_merge_id
                    for i in range(len(neighbor_with_neighbors.get_neighbor_ids())):
                        if neighbor_with_neighbors.get_neighbor_ids()[i] == neighbor_id:
                            neighbor_with_neighbors.get_neighbor_ids()[
                                i
                            ] = self.next_merge_id

            # New list of neighbors is merge_id's neighbors + neighbor_id's neighbors
            # Reset left/right neighbors
            new_neighbor_ids = list(
                filter(
                    lambda x: x != merge_id and x != neighbor_id,
                    merge_id_with_neighbors.get_neighbor_ids()
                    + neighbor_id_with_neighbors.get_neighbor_ids(),
                )
            )
            new_merge_id_with_neighbors = MergeIdWithNeighbors(
                self.next_merge_id, new_neighbor_ids
            )
            if use_neighbor_id_orientation:
                # Danger: potential to ruin neighbor operation symmetry
                if (
                    neighbor_id_with_neighbors.has_left()
                    and neighbor_id_with_neighbors.get_left() != merge_id
                ):
                    new_merge_id_with_neighbors.left_neighbor_id = (
                        neighbor_id_with_neighbors.get_left()
                    )
                if (
                    neighbor_id_with_neighbors.has_right()
                    and neighbor_id_with_neighbors.get_right() != merge_id
                ):
                    new_merge_id_with_neighbors.right_neighbor_id = (
                        neighbor_id_with_neighbors.get_right()
                    )
            merge_id_to_obj[self.next_merge_id] = new_merge_id_with_neighbors

            # Remove merge_id and neighbor_id from merge_id_to_obj
            merge_id_to_obj.pop(merge_id)
            merge_id_to_obj.pop(neighbor_id)
            # Merge polygons and original neighbors
            self._merge([merge_id, neighbor_id])

            return new_merge_id_with_neighbors

        def doGreedyOrientations():
            iters_without_progress = 0
            while process_queue:
                merge_id = process_queue.pop(0)
                merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[
                    merge_id
                ]
                # 1 neighbor
                if len(merge_id_with_neighbors.get_neighbor_ids()) == 1:
                    neighbor_id = merge_id_with_neighbors.get_neighbor_ids()[0]
                    # If merge poly only has one unassigned neighbor and only needs to assign one more neighbor, do it
                    if (
                        merge_id_with_neighbors.has_left()
                        and not (merge_id_with_neighbors.has_right())
                        and not (merge_id_to_obj[neighbor_id].has_left())
                    ):
                        merge_id_with_neighbors.set_right(neighbor_id)
                        iters_without_progress = 0
                    elif (
                        merge_id_with_neighbors.has_right()
                        and not (merge_id_with_neighbors.has_left())
                        and not (merge_id_to_obj[neighbor_id].has_right())
                    ):
                        merge_id_with_neighbors.set_left(neighbor_id)
                        iters_without_progress = 0
                    # There should never be a case when neither orientation is set but only one neighbor
                    elif not (merge_id_with_neighbors.has_left()) and not (
                        merge_id_with_neighbors.has_right()
                    ):
                        print(
                            "Neither orientation is set but one neighbor: why did this happen?"
                        )
                        # TODO not sure how to handle this, merge into its neighbor for now
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                        # raise ValueError("Neither orientation is set but one neighbor: why did this happen?")
                    # This case should be dealt with via implementation of remove_neighbor_id
                    elif merge_id_with_neighbors.fully_oriented():
                        print("Fully oriented but one neighbor: why did this happen?")
                        # TODO is this an actual issue? For now, assume this cell is ok and just remove from queue
                        # process_queue.append(merge_id_with_neighbors)
                        # iters_without_progress += 1
                        # raise ValueError("Fully oriented but one neighbor: why did this happen?")
                    else:  # single neighbor but it's already got the neighbor we would want to assign merge_id to
                        print(
                            "Single neighbor but orientations are inconsistent, skip for now"
                        )
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                # 2 neighbors
                elif len(merge_id_with_neighbors.get_neighbor_ids()) == 2:
                    if not (merge_id_with_neighbors.has_left()) and not (
                        merge_id_with_neighbors.has_right()
                    ):
                        # If merge poly consists of only one poly and has exactly two neighbors
                        if len(self._get_merge_coords(merge_id)) == 1:
                            # Check if orientation is easy to figure out
                            [x, y] = self._get_merge_coords(merge_id)[0]
                            dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                            neighbor_dirs = []
                            nonneighbor_dirs = []

                            def _helper_in_bounds(x, y):
                                return not (
                                    x < 0
                                    or y < 0
                                    or x >= len(self.polys)
                                    or y >= len(self.polys[0])
                                )

                            for i, dir in enumerate(dirs):
                                neighbor_coords = [x + dir[0], y + dir[1]]
                                if (
                                    _helper_in_bounds(
                                        neighbor_coords[0], neighbor_coords[1]
                                    )
                                    and self.polys[neighbor_coords[0]][
                                        neighbor_coords[1]
                                    ].isMixed()
                                    and self._get_merge_id(
                                        neighbor_coords[0], neighbor_coords[1]
                                    )
                                    in merge_id_with_neighbors.get_neighbor_ids()
                                ):
                                    neighbor_dirs.append(i)
                                else:
                                    nonneighbor_dirs.append(i)
                            # TODO go by self._get_neighbor_ids_from_merge_ids instead?
                            # if more than two neighbors, then this cell used to have more than two mixed neighbors but some were eliminated, pass
                            if len(neighbor_dirs) > 2:
                                print(
                                    "Passing on case where cell used to have more than two mixed neighbors but only two are left, still not easily orientable"
                                )
                                process_queue.append(merge_id)
                                iters_without_progress += 1
                                continue
                            elif len(neighbor_dirs) < 2:
                                # TODO this case might never happen
                                print(
                                    "Passing on case where cell has fewer than two cells, not easily orientable"
                                )
                                process_queue.append(merge_id)
                                iters_without_progress += 1
                                continue

                            # Verifies neighbors = coordinate neighbors
                            # print(merge_id_with_neighbors.get_neighbor_ids())
                            # print(list(map(lambda d: self._get_merge_id(x+dirs[d][0], y+dirs[d][1]), neighbor_dirs)))

                            # otherwise, two mixed neighbors are either across from each other or adjacent
                            neighbor_modes = abs(
                                neighbor_dirs[0] - neighbor_dirs[1]
                            )  # = 1 or 2 or 3 (3 is same case as 1)
                            # 1,3 = adjacent; 2 = across
                            nonneighbor_statuses = []
                            for nonneighbor_dir in nonneighbor_dirs:
                                nonneighbor_coords = [
                                    x + dirs[nonneighbor_dir][0],
                                    y + dirs[nonneighbor_dir][1],
                                ]
                                if (
                                    _helper_in_bounds(
                                        nonneighbor_coords[0], nonneighbor_coords[1]
                                    )
                                    and self.polys[nonneighbor_coords[0]][
                                        nonneighbor_coords[1]
                                    ].isFull()
                                ):
                                    nonneighbor_statuses.append(1)
                                else:  # must be empty
                                    nonneighbor_statuses.append(0)
                            if neighbor_modes == 1 or neighbor_modes == 3:
                                clockwisemost = (
                                    min(neighbor_dirs) if neighbor_modes == 1 else 3
                                )
                                [c_x, c_y] = [
                                    x + dirs[clockwisemost][0],
                                    y + dirs[clockwisemost][1],
                                ]
                                c_merge_id = self._get_merge_id(c_x, c_y)
                                [cc_x, cc_y] = [
                                    x + dirs[(clockwisemost + 1) % 4][0],
                                    y + dirs[(clockwisemost + 1) % 4][1],
                                ]
                                cc_merge_id = self._get_merge_id(cc_x, cc_y)
                                if (
                                    nonneighbor_statuses == [0, 0]
                                    and not (merge_id_to_obj[c_merge_id].has_left())
                                    and not (merge_id_to_obj[cc_merge_id].has_right())
                                ):  # both empty, clockwise
                                    merge_id_with_neighbors.set_left(
                                        self._get_merge_id(cc_x, cc_y)
                                    )
                                    merge_id_with_neighbors.set_right(
                                        self._get_merge_id(c_x, c_y)
                                    )
                                    iters_without_progress = 0
                                elif (
                                    nonneighbor_statuses == [1, 1]
                                    and not (merge_id_to_obj[c_merge_id].has_right())
                                    and not (merge_id_to_obj[cc_merge_id].has_left())
                                ):  # both full, counterclockwise
                                    merge_id_with_neighbors.set_left(
                                        self._get_merge_id(c_x, c_y)
                                    )
                                    merge_id_with_neighbors.set_right(
                                        self._get_merge_id(cc_x, cc_y)
                                    )
                                    iters_without_progress = 0
                                else:
                                    print(
                                        f"Error in easy orientation with two adjacent neighbors: {nonneighbor_statuses}"
                                    )
                                    process_queue.append(merge_id)
                                    iters_without_progress += 1
                            else:  # neighbor_modes == 2
                                if nonneighbor_statuses == [0, 1]:
                                    full_index = 1
                                elif nonneighbor_statuses == [1, 0]:
                                    full_index = 0
                                else:
                                    print(
                                        f"Error in easy orientation with two opposite neighbors: {nonneighbor_statuses}"
                                    )
                                    process_queue.append(merge_id)
                                    iters_without_progress += 1
                                    break
                                [l_x, l_y] = [
                                    x + dirs[(nonneighbor_dirs[full_index] + 1) % 4][0],
                                    y + dirs[(nonneighbor_dirs[full_index] + 1) % 4][1],
                                ]
                                l_merge_id = self._get_merge_id(l_x, l_y)
                                [r_x, r_y] = [
                                    x + dirs[(nonneighbor_dirs[full_index] - 1) % 4][0],
                                    y + dirs[(nonneighbor_dirs[full_index] - 1) % 4][1],
                                ]
                                r_merge_id = self._get_merge_id(r_x, r_y)
                                if not (
                                    merge_id_to_obj[l_merge_id].has_right()
                                ) and not (merge_id_to_obj[r_merge_id].has_left()):
                                    merge_id_with_neighbors.set_left(l_merge_id)
                                    merge_id_with_neighbors.set_right(r_merge_id)
                                    iters_without_progress = 0
                                else:
                                    print(
                                        "Error in easy orientation with two opposite neighbors but inconsistent orientations"
                                    )
                                    process_queue.append(merge_id)
                                    iters_without_progress += 1
                        # Not easily orientable, pass
                        else:
                            print(
                                "Passing on case with two neighbors and two missing orientations"
                            )
                            process_queue.append(merge_id)
                            iters_without_progress += 1
                    elif merge_id_with_neighbors.fully_oriented():
                        # TODO this doesn't seem to happen
                        print("Fully oriented but two neighbors: why did this happen?")
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                    # One more orientation to be set and two neighbors, pass
                    else:
                        print(
                            "Passing on case with two neighbors and one missing orientation"
                        )
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                # 3+ neighbors
                elif len(merge_id_with_neighbors.get_neighbor_ids()) >= 3:
                    if merge_id_with_neighbors.fully_oriented():
                        raise ValueError(
                            f"Fully oriented but {len(merge_id_with_neighbors.get_neighbor_ids())} neighbors: why did this happen?"
                        )
                    elif try_base_orientation_hint(merge_id_with_neighbors):
                        iters_without_progress = 0
                    else:
                        print("Passing on case with 3+ neighbors")
                        process_queue.append(merge_id)
                        iters_without_progress += 1
                # 0 neighbors
                else:
                    # if still not fully oriented, these are problematic and we want to do something
                    if not (merge_id_with_neighbors.fully_oriented()):
                        print("Zero neighbors and not full oriented, passing")
                        process_queue.append(merge_id)
                        iters_without_progress += 1

                # iters_without_progress = length + 1 (full cycle of queue without progress)
                if iters_without_progress >= len(process_queue) + 1:
                    print(
                        f"Rest of queue cannot be resolved: length {len(process_queue)}"
                    )
                    break

        for merge_id in process_queue.copy():
            if try_base_orientation_hint(merge_id_to_obj[merge_id]):
                continue

        doGreedyOrientations()

        if self.use_late_three_neighbor_orientation_hint:
            hint_count_before = len(self.orientation_hint_records)
            for merge_id in process_queue.copy():
                try_base_orientation_hint(
                    merge_id_to_obj[merge_id], phase="late"
                )
            if len(self.orientation_hint_records) > hint_count_before:
                doGreedyOrientations()

        if self.retry_greedy_orientations:
            degree_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            unoriented = 0
            half_oriented = 0
            for merge_id in process_queue:
                obj = merge_id_to_obj[merge_id]
                degree_counts[min(len(obj.get_neighbor_ids()), 3)] += 1
                assigned_sides = int(obj.has_left()) + int(obj.has_right())
                unoriented += assigned_sides == 0
                half_oriented += assigned_sides == 1
            self.orientation_retry_records.append(
                {
                    "queue_size": len(process_queue),
                    "degree_0": degree_counts[0],
                    "degree_1": degree_counts[1],
                    "degree_2": degree_counts[2],
                    "degree_3plus": degree_counts[3],
                    "unoriented": unoriented,
                    "half_oriented": half_oriented,
                }
            )
            self.orientation_retry_passes += 1
            doGreedyOrientations()

        # Cases at this point:
        # 1 neighbor candidate:
        # Two unfilled neighbors
        # One unfilled neighbor but orientations are inconsistent
        # 2 neighbor candidates:
        # Not an "easy orientation" case
        # 3+ neighbor candidates:
        # All
        # 0 neighbor candidates:
        # Not fully oriented

        iters_without_progress = 0
        while process_queue:
            merge_id = process_queue.pop(0)
            merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[merge_id]
            if len(merge_id_with_neighbors.get_neighbor_ids()) == 0:
                # No orientations set
                if not (merge_id_with_neighbors.has_left()) and not (
                    merge_id_with_neighbors.has_right()
                ):
                    print("Weird case 1")
                    # Choose a random mixed neighbor and merge with it
                    neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
                    if len(neighbor_ids) > 0:
                        neighbor_id = neighbor_ids[
                            0
                        ]  # TODO make random, or choose the one in the direction most normal to the interface?
                        new_merge_id_with_neighbors = mergeObjs(
                            merge_id, neighbor_id, use_neighbor_id_orientation=True
                        )
                        # After merging, remove neighbor_id from the process queue if needed
                        if neighbor_id in process_queue:
                            process_queue.remove(neighbor_id)
                        # Add newly merged poly to process queue
                        process_queue.append(new_merge_id_with_neighbors.get_merge_id())
                        doGreedyOrientations()
                        iters_without_progress = 0
                    # Else, mixed cell originally had no mixed neighbors (typically because true mixed neighbors have fractions below threshold and are rounded to full/empty)
                    # Construct Young's linear facet. (This case should be handled at the very end.)
                # Has left neighbor
                elif merge_id_with_neighbors.has_left():
                    # TODO am I ok to set its right neighbor to itself
                    print("Weird case 1b: right")
                    merge_id_with_neighbors.right_neighbor_id = merge_id
                # Has right neighbor
                elif merge_id_with_neighbors.has_right():
                    # TODO am I ok to set its left neighbor to itself
                    print("Weird case 1c: left")
                    merge_id_with_neighbors.left_neighbor_id = merge_id
            elif merge_id_with_neighbors.has_left() and not (
                merge_id_with_neighbors.has_right()
            ):
                print("Weird case 2")
                left_neighbor_id = merge_id_with_neighbors.get_left()
                neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
                merge_coords = self._get_merge_coords(merge_id)[
                    0
                ]  # choose first poly within merged poly
                did_update = False

                # Construct Young's linear facet and use it to make guess about correct neighbor
                base_poly = self.polys[merge_coords[0]][merge_coords[1]]
                youngs_poly = BasePolygon(base_poly.points)
                youngs_poly.setFraction(base_poly.getFraction())
                youngs_poly.set3x3Stencil(
                    self.get3x3Stencil(merge_coords[0], merge_coords[1])
                )
                youngs_poly.runYoungs()
                youngs_poly_pLeft = youngs_poly.getFacet().pLeft
                youngs_poly_pRight = youngs_poly.getFacet().pRight

                # Metric to calculate how close a point is to a polygon's edges: equal to 1 if point lies on an edge
                # TODO not an ideal metric: poly = [[0, 0], [1, 0], [1, 1], [0, 1]], p1 = [0.5, 0.1] vs. p2 = [0.7, 0.1]. p2 closer
                # to edges but has larger value of metric
                def _helper_pointDistanceToPolyEdges(x, poly_points):
                    ret = float("inf")
                    for i in range(len(poly_points)):
                        p1 = poly_points[i]
                        p2 = poly_points[(i + 1) % len(poly_points)]
                        d = (getDistance(x, p1) + getDistance(x, p2)) / (
                            getDistance(p1, p2)
                        )
                        ret = min(d, ret)
                    return ret

                # For each neighboring poly, calculate metric; choose neighbor with min metric
                min_i = None
                min_metric = float("inf")
                for i, neighbor_id in enumerate(neighbor_ids):
                    # Don't want to choose the neighbor that's already set, or one that has a left neighbor already
                    if neighbor_id != left_neighbor_id and not (
                        merge_id_to_obj[neighbor_id].has_left()
                    ):
                        neighbor_coords = self._get_merge_coords(neighbor_id)[
                            0
                        ]  # choose first poly within merged poly
                        neighbor_poly_points = self.polys[neighbor_coords[0]][
                            neighbor_coords[1]
                        ].points
                        d = min(
                            _helper_pointDistanceToPolyEdges(
                                youngs_poly_pLeft, neighbor_poly_points
                            ),
                            _helper_pointDistanceToPolyEdges(
                                youngs_poly_pRight, neighbor_poly_points
                            ),
                        )
                        if d < min_metric:
                            print(f"Metric left: {d}")
                            min_i = i
                            min_metric = d
                            did_update = True
                # If we found one, set neighbor
                if did_update:
                    merge_id_with_neighbors.set_right(
                        neighbor_ids[min_i], set_neighbor=True
                    )

                """
                # This uses a hack: since we processed neighbors in counterclockwise order, this should give the mixed neighbor to the right of the left neighbor
                # TODO is this neighbor guaranteed to not already have a left neighbor?
                for i, neighbor_id in enumerate(neighbor_ids):
                    if neighbor_id == left_neighbor_id and not(merge_id_to_obj[neighbor_ids[(i+1) % len(neighbor_ids)]].has_left()):
                        merge_id_with_neighbors.set_right(neighbor_ids[(i+1) % len(neighbor_ids)], set_neighbor=True)
                        did_update = True
                """

                if did_update:
                    doGreedyOrientations()
                    iters_without_progress = 0
                else:
                    process_queue.append(merge_id_with_neighbors.get_merge_id())
                    iters_without_progress += 1
            elif (
                not (merge_id_with_neighbors.has_left())
                and merge_id_with_neighbors.has_right()
            ):
                print("Weird case 3")
                right_neighbor_id = merge_id_with_neighbors.get_right()
                neighbor_ids = self._get_neighbor_ids_from_merge_id(merge_id)
                merge_coords = self._get_merge_coords(merge_id)[
                    0
                ]  # choose first poly within merged poly
                did_update = False

                # Construct Young's linear facet and use it to make guess about correct neighbor
                print(merge_coords)
                base_poly = self.polys[merge_coords[0]][merge_coords[1]]
                youngs_poly = BasePolygon(base_poly.points)
                youngs_poly.setFraction(base_poly.getFraction())
                youngs_poly.set3x3Stencil(
                    self.get3x3Stencil(merge_coords[0], merge_coords[1])
                )
                youngs_poly.runYoungs()
                youngs_poly_pLeft = youngs_poly.getFacet().pLeft
                youngs_poly_pRight = youngs_poly.getFacet().pRight

                # Metric to calculate how close a point is to a polygon's edges: equal to 1 if point lies on an edge
                # TODO not an ideal metric: poly = [[0, 0], [1, 0], [1, 1], [0, 1]], p1 = [0.5, 0.1] vs. p2 = [0.7, 0.1]. p2 closer
                # to edges but has larger value of metric
                def _helper_pointDistanceToPolyEdges(x, poly_points):
                    ret = float("inf")
                    for i in range(len(poly_points)):
                        p1 = poly_points[i]
                        p2 = poly_points[(i + 1) % len(poly_points)]
                        d = (getDistance(x, p1) + getDistance(x, p2)) / (
                            getDistance(p1, p2)
                        )
                        ret = min(d, ret)
                    return ret

                # For each neighboring poly, calculate metric; choose neighbor with min metric
                min_i = None
                min_metric = float("inf")
                for i, neighbor_id in enumerate(neighbor_ids):
                    # Don't want to choose the neighbor that's already set, or one that has a right neighbor already
                    if neighbor_id != right_neighbor_id and not (
                        merge_id_to_obj[neighbor_id].has_right()
                    ):
                        neighbor_coords = self._get_merge_coords(neighbor_id)[
                            0
                        ]  # choose first poly within merged poly
                        neighbor_poly_points = self.polys[neighbor_coords[0]][
                            neighbor_coords[1]
                        ].points
                        d = min(
                            _helper_pointDistanceToPolyEdges(
                                youngs_poly_pLeft, neighbor_poly_points
                            ),
                            _helper_pointDistanceToPolyEdges(
                                youngs_poly_pRight, neighbor_poly_points
                            ),
                        )
                        if d < min_metric:
                            print(f"Metric right: {d}")
                            min_i = i
                            min_metric = d
                            did_update = True
                # If we found one, set neighbor
                if did_update:
                    merge_id_with_neighbors.set_left(
                        neighbor_ids[min_i], set_neighbor=True
                    )

                """
                # This uses a hack: since we processed neighbors in counterclockwise order, this should give the mixed neighbor to the left of the right neighbor
                # TODO is this neighbor guaranteed to not already have a right neighbor?
                for i, neighbor_id in enumerate(neighbor_ids):
                    if neighbor_id == right_neighbor_id and not(merge_id_to_obj[neighbor_ids[(i-1) % len(neighbor_ids)]].has_right()):
                        merge_id_with_neighbors.set_left(neighbor_ids[(i-1) % len(neighbor_ids)], set_neighbor=True)
                        did_update = True
                """

                if did_update:
                    doGreedyOrientations()
                    iters_without_progress = 0
                else:
                    process_queue.append(merge_id_with_neighbors.get_merge_id())
                    iters_without_progress += 1
            else:
                print("Weird case 4")
                process_queue.append(merge_id_with_neighbors.get_merge_id())
                iters_without_progress += 1
            if iters_without_progress >= len(process_queue) + 1:
                print(
                    f"Rest of meta queue cannot be resolved: length {len(process_queue)}"
                )
                break

        # very_ambiguous_ids = process_queue.copy()

        # # Attempts to merge cells that are still unresolved
        # while process_queue:
        #     merge_id_with_neighbors: MergeIdWithNeighbors = process_queue.pop(0)
        #     merge_id = merge_id_with_neighbors.get_merge_id()
        #     # TODO handle other cases (neither left nor right neighbors are set?)
        #     print(f"Unresolved cells: num neighbors = {len(merge_id_with_neighbors.get_neighbor_ids())}")
        #     # Choose a random mixed neighbor and merge with it
        #     neighbor_ids = list(filter(lambda x : x not in list(map(lambda y : y.get_merge_id(), very_ambiguous_ids)), merge_id_with_neighbors.get_neighbor_ids()))
        #     if len(neighbor_ids) == 0:
        #         neighbor_ids = list(filter(lambda x : x not in list(map(lambda y : y.get_merge_id(), very_ambiguous_ids)), self._get_neighbor_ids_from_merge_id(merge_id)))
        #         if len(neighbor_ids) == 0:
        #             #TODO no clue if this is possible
        #             raise ValueError("Is it possible that an entirely unoriented cell also has no neighbors not in process queue here?")
        #     neighbor_id = neighbor_ids[0]
        #     # TODO make random, or choose the one in the direction most normal to the interface?
        #     mergeObjs(merge_id, neighbor_id, use_neighbor_id_orientation=True)

        # Create polygon objects

        # Find the vertices of the merged polygons
        self.createMergedPolys()

        # Merge ids that failed to be oriented
        failed_merge_ids = []
        added_merge_ids = dict()
        for merge_id in merge_id_to_obj.keys():
            merge_id_with_neighbors: MergeIdWithNeighbors = merge_id_to_obj[merge_id]
            if merge_id_with_neighbors.fully_oriented():
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                merged_poly.setNeighbor(
                    self.merged_polys[merge_id_with_neighbors.get_left()], "left"
                )
                merged_poly.setNeighbor(
                    self.merged_polys[merge_id_with_neighbors.get_right()], "right"
                )
                # Check if we used hack and had set left/right neighbor to itself to signify dead-end cell with single mixed neighbor
                if (
                    merge_id_with_neighbors.get_left() == merge_id
                    or merge_id_with_neighbors.get_right() == merge_id
                ):
                    merged_poly.setFacetType("linear_deadend")
            else:
                print("Final failed orientations:")
                print(merge_id_with_neighbors)
                # For all polys in failed merge id, add a lone polygon with a 3x3 stencil and run Young's
                # May still need this poly because it could be a neighbor of something else so can't directly remove it
                for merge_coords in self._get_merge_coords(
                    merge_id_with_neighbors.get_merge_id()
                ):
                    lone_base_poly = self.polys[merge_coords[0]][merge_coords[1]]
                    merged_poly = NeighboredPolygon(lone_base_poly.points)
                    merged_poly.setFraction(lone_base_poly.getFraction())
                    merged_poly.set3x3Stencil(
                        self.get3x3Stencil(merge_coords[0], merge_coords[1])
                    )

                    # TODO does this throw off the algorithm somehow because we're appending to merge_id_to_obj only? (Some invariant where merge_id_to_obj has to have same length as something else?)
                    self.merged_polys[self.next_merge_id] = merged_poly
                    self._attach_facet_provenance(merged_poly, self.next_merge_id)
                    self.coords_to_merge_id[merge_coords[0]][
                        merge_coords[1]
                    ] = self.next_merge_id
                    self.merge_ids_to_coords.append(merge_coords)
                    self.merge_id_to_neighbor_ids.append([])
                    added_merge_ids[self.next_merge_id] = merged_poly
                    self.next_merge_id += 1
                failed_merge_ids.append(merge_id)

        # Remove failed merge ids from list of polygons
        for failed_merge_id in failed_merge_ids:
            merge_id_to_obj.pop(failed_merge_id)
        # Add Young's polygons
        for added_merge_id in added_merge_ids.keys():
            merge_id_to_obj[added_merge_id] = added_merge_ids[added_merge_id]
        return list(merge_id_to_obj.keys())

    # Update plt and vtk variables to account for merged polys
    def updatePlots(self):
        # plt variables
        self._plt_patches = []
        self._plt_patchareas = []
        self._plt_patchpartialareas = []
        # vtk variables: only plots mixed cells
        self._vtk_mixed_polys = []
        self._vtk_mixed_polyareas = []

        processed_merge_ids = []

        # Add all merge ids in use to the queue
        for x in range(len(self.polys)):
            for y in range(len(self.polys[0])):
                merge_id = self._get_merge_id(x, y)
                if merge_id is None:
                    # Full/empty cells
                    patch = plt_polygon(np.array(self.polys[x][y].points))
                    self._plt_patches.append(patch)
                    adjusted_fraction = self.polys[x][y].getFraction()
                    self._plt_patchareas.append(adjusted_fraction)
                    self._plt_patchpartialareas.append(
                        math.ceil(adjusted_fraction - math.floor(adjusted_fraction))
                    )
                elif merge_id not in processed_merge_ids:
                    # Mixed merged cell
                    # plt
                    merged_cell = self.merged_polys[merge_id]
                    patch = plt_polygon(np.array(merged_cell.points))
                    self._plt_patches.append(patch)
                    adjusted_fraction = merged_cell.getFraction()
                    self._plt_patchareas.append(adjusted_fraction)
                    self._plt_patchpartialareas.append(
                        math.ceil(adjusted_fraction - math.floor(adjusted_fraction))
                    )
                    # vtk
                    self._vtk_mixed_polys.append(merged_cell.points)
                    self._vtk_mixed_polyareas.append(adjusted_fraction)

        self._plt_patchareas = np.array(self._plt_patchareas)
        self._plt_patchpartialareas = np.array(self._plt_patchpartialareas)
        self._plt_patchinitialareas = np.array(self._plt_patchinitialareas)

    def _run_unresolved_plic_fallback(
        self, merged_poly: NeighboredPolygon, merge_id, setting, plic_fallback
    ):
        policy_lookup = {
            "youngs": "Youngs",
            "elvira": "ELVIRA",
            "lvira": "LVIRA",
        }
        policy_key = str(plic_fallback or "LVIRA").lower()
        if policy_key not in policy_lookup:
            raise ValueError(
                f"Unknown plic_fallback={plic_fallback!r}; expected Youngs, ELVIRA, or LVIRA"
            )

        policy = policy_lookup[policy_key]
        if policy == "Youngs":
            facet = merged_poly.runYoungs(ret=True)
        elif policy == "ELVIRA":
            facet = merged_poly.runELVIRA(ret=True)
        else:
            facet = merged_poly.runLVIRA(ret=True)

        previous_override = self._provenance_override
        self._provenance_override = {
            merge_id: {
                "event_kind": "plic_fallback",
                "policy": policy,
                "reason": "unresolved_orientation",
            },
        }
        try:
            merged_poly.setFacet(facet)
        finally:
            self._provenance_override = previous_override
        self._append_plic_fallback_record(
            setting, merge_id, merged_poly, facet, policy
        )

    # TODO why are we popping the merge ids that fail?
    def fitFacets(
        self,
        merge_ids,
        setting="circular",
        plic_fallback="LVIRA",
        rescue_profile="exact_linear_support_only",
        stage_callback=None,
    ):
        self.plic_fallback_records = []
        self.safe_circle_fallback_records = []
        self.facet_provenance_events = []
        self._provenance_event_order = 0
        self._provenance_stage = setting
        self._provenance_override = None
        normalized_plic_fallback = BasePolygon._normalize_plic_fallback(
            plic_fallback or "LVIRA"
        )
        for merge_id in merge_ids:
            self.merged_polys[merge_id].plic_fallback_policy = (
                normalized_plic_fallback
            )
        rescue_profile = str(rescue_profile or self.default_rescue_profile).lower()
        if rescue_profile not in self.rescue_profiles:
            raise ValueError(
                f"Unknown rescue_profile={rescue_profile!r}; "
                f"expected one of {sorted(self.rescue_profiles)}"
            )
        use_linear_corner_rescues = rescue_profile not in {
            "no_corner_rescues",
            "no_linear_corner_rescues",
        }
        use_curved_corner_rescues = rescue_profile not in {
            "no_corner_rescues",
            "no_curved_corner_rescues",
            "candidate_keep_12346_drop_9",
            "exact_linear_support_only",
        }
        use_repeated_corner_rescues = rescue_profile not in {
            "no_corner_rescues",
            "no_linear_corner_rescues",
            "no_repeated_corner_rescues",
            "exact_linear_support_only",
        }
        use_repeated_tiny_corner_rescues = rescue_profile not in {
            "no_corner_rescues",
            "no_linear_corner_rescues",
            "no_repeated_corner_rescues",
            "no_repeated_tiny_corner_rescues",
            "exact_linear_support_only",
        }
        use_repeated_corner_component_rescues = rescue_profile not in {
            "no_corner_rescues",
            "no_linear_corner_rescues",
            "no_repeated_corner_rescues",
            "no_repeated_corner_component_rescues",
            "candidate_keep_12346_drop_9",
            "exact_linear_support_only",
        }
        use_only_exact_linear_support = rescue_profile == "exact_linear_support_only"

        def emit_stage(stage):
            self._record_stage_snapshots(stage, merge_ids)
            if stage_callback is not None:
                stage_callback(stage, self, tuple(merge_ids))

        if setting == "linear":
            self._provenance_stage = "linear"
            i = 0
            while i < len(merge_ids):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if merged_poly.fullyOriented():
                    merged_poly.fitLinearFacet()
                elif merged_poly.has3x3Stencil():
                    self._run_unresolved_plic_fallback(
                        merged_poly, merge_id, setting, plic_fallback
                    )
                else:
                    print("Something wrong with facet fitting!")
                    print(self.merged_polys[merge_id])
                    merge_ids.pop(i)
                    i -= 1
                i += 1

        elif setting == "circular":
            self._provenance_stage = "circular"
            i = 0
            while i < len(merge_ids):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if merged_poly.fullyOriented():
                    if merged_poly.facet_type == "linear_deadend":
                        self._fit_deadend_facet(merge_id, prefer_safe_circle=True)
                    else:
                        root_guess = self._find_arc_fit_guess(merge_id)
                        with arc_facet_log_context(
                            call_source="circular",
                            merge_id=merge_id,
                            merge_coords=self._get_merge_coords(merge_id),
                        ):
                            merged_poly.fitCircularFacet(root_guess=root_guess)
                        if (
                            merged_poly.hasFacet()
                            and getattr(merged_poly.getFacet(), "name", "")
                            in {"Youngs", "ELVIRA", "LVIRA"}
                        ):
                            policy = merged_poly.getFacet().name
                            self._rewrite_latest_facet_provenance(
                                merge_id,
                                "plic_fallback",
                                policy,
                                "support_line_fit_failed",
                            )
                            self._append_plic_fallback_record(
                                setting,
                                merge_id,
                                merged_poly,
                                merged_poly.getFacet(),
                                policy,
                            )
                        # If circular facet fitter failed, default to linear
                        if not (merged_poly.hasFacet()):
                            previous_override = self._provenance_override
                            self._provenance_override = {
                                merge_id: {
                                    "event_kind": "local_linear_fallback",
                                    "policy": "local_linear",
                                    "reason": "arc_fit_failed",
                                }
                            }
                            try:
                                merged_poly.fitLinearFacet()
                            finally:
                                self._provenance_override = previous_override
                            if (
                                merged_poly.hasFacet()
                                and getattr(merged_poly.getFacet(), "name", "")
                                in {"Youngs", "ELVIRA", "LVIRA"}
                            ):
                                policy = merged_poly.getFacet().name
                                self._rewrite_latest_facet_provenance(
                                    merge_id,
                                    "plic_fallback",
                                    policy,
                                    "arc_fit_failed_local_linear_failed",
                                )
                                self._append_plic_fallback_record(
                                    setting,
                                    merge_id,
                                    merged_poly,
                                    merged_poly.getFacet(),
                                    policy,
                                )
                            elif merged_poly.hasFacet():
                                merged_poly.getFacet().name = "default_linear"
                elif merged_poly.has3x3Stencil():
                    self._run_unresolved_plic_fallback(
                        merged_poly, merge_id, setting, plic_fallback
                    )
                else:
                    print("Something wrong with facet fitting!")
                    merge_ids.pop(i)
                    i -= 1
                i += 1

        elif setting == "linear+corner":
            self._provenance_stage = "linear"
            # First, try fitting linear facets
            for i in range(len(merge_ids)):
                merged_poly: NeighboredPolygon = self.merged_polys[merge_ids[i]]
                if merged_poly.fullyOriented():
                    if merged_poly.facet_type == "linear_deadend":
                        merged_poly.fitLinearFacet()
                        merged_poly.getFacet().name = "linear_deadend"
                        print("linear+corner, 1st phase, force linear")
                        print(merged_poly)
                    else:
                        merged_poly.fitLinearFacet(doCollinearityCheck=True)

            print("Using corners")
            # For the ones not fit properly, try a corner
            for i in range(len(merge_ids)):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if not (merged_poly.hasFacet()):
                    # loop through left until you find a linear facet
                    doneLeft = False
                    left: NeighboredPolygon = merged_poly.getLeftNeighbor()
                    right: NeighboredPolygon = merged_poly.getRightNeighbor()
                    success = True
                    while not (doneLeft):
                        # Handle cases: left = itself (hack for when no left neighbor but properly oriented), looped all the way to the beginning, or no left neighbor
                        if left == merged_poly or left == right or left is None:
                            doneLeft = True
                            success = False
                        elif left.hasFacet() and left.getFacet().name == "linear":
                            # Linear facet on left
                            doneLeft = True
                            success = True
                        else:
                            if left.getLeftNeighbor() == left:
                                # Rare case with dead-end, break
                                success = False
                                break
                            else:
                                left: NeighboredPolygon = left.getLeftNeighbor()
                    # Either left = closest neighbor on left with proper linear facet or success = False
                    doneRight = not (success)
                    while not (doneRight):
                        if right == merged_poly or right == left or right is None:
                            doneRight = True
                            success = False
                        elif right.hasFacet() and right.getFacet().name == "linear":
                            # Linear facet on right
                            doneRight = True
                            success = True
                        else:
                            if right.getRightNeighbor() == right:
                                # Rare case with dead-end, break
                                success = False
                                break
                            else:
                                right: NeighboredPolygon = right.getRightNeighbor()
                    # If success, left/right are closest neighbors on left/right with proper linear facet
                    if success:
                        # Try corner
                        # print(f"Trying corner with left: {left.getFacet()} and right: {right.getFacet()}")
                        merged_poly.checkCornerFacet(
                            left.getFacet().pLeft,
                            left.getFacet().pRight,
                            right.getFacet().pRight,
                            right.getFacet().pLeft,
                        )
                        if not (merged_poly.hasFacet()):
                            # print("Failed to form corner facet")
                            # print(merged_poly)
                            pass
                        else:
                            pass
                            # print(merged_poly.getFacet())
                    else:
                        # print("Failed to find neighbors")
                        # print(merged_poly)
                        pass

            # # For anything left, fit a linear facet again #TODO save computation of linear facet and set it here
            self._provenance_stage = "final_fallback"
            i = 0
            while i < len(merge_ids):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if not (merged_poly.hasFacet()):
                    if merged_poly.fullyOriented():
                        merged_poly.fitLinearFacet()
                    elif merged_poly.has3x3Stencil():
                        # print("Youngs on poly")
                        # print(merged_poly)
                        self._run_unresolved_plic_fallback(
                            merged_poly, merge_id, setting, plic_fallback
                        )
                    else:
                        print("Something wrong with facet fitting!")
                        print(self.merged_polys[merge_id])
                        merge_ids.pop(i)
                        i -= 1
                i += 1

        elif setting == "circular+corner":
            self._provenance_stage = "linear"
            # First, try fitting linear facets
            for i in range(len(merge_ids)):
                merged_poly: NeighboredPolygon = self.merged_polys[merge_ids[i]]
                if merged_poly.fullyOriented():
                    if merged_poly.facet_type == "linear_deadend":
                        self._fit_deadend_facet(
                            merge_ids[i], prefer_safe_circle=True
                        )
                    else:
                        merged_poly.fitLinearFacet(doCollinearityCheck=True)

            emit_stage("linear")

            print("Using corners")
            self._provenance_stage = "linear_corners"
            # For the ones not fit properly, try a corner
            for i in range(len(merge_ids)):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if not (merged_poly.hasFacet()):
                    # loop through left until you find a linear facet
                    doneLeft = False
                    left: NeighboredPolygon = merged_poly.getLeftNeighbor()
                    right: NeighboredPolygon = merged_poly.getRightNeighbor()
                    success = True
                    while not (doneLeft):
                        # Handle cases: left = itself (hack for when no left neighbor but properly oriented), looped all the way to the beginning, or no left neighbor
                        if left == merged_poly or left == right or left is None:
                            doneLeft = True
                            success = False
                        elif left.hasFacet() and self._is_line_like_support_facet(
                            left.getFacet()
                        ):
                            # Linear facet on left
                            doneLeft = True
                            success = True
                        else:
                            if left.getLeftNeighbor() == left:
                                # Rare case with dead-end, break
                                success = False
                                break
                            else:
                                left: NeighboredPolygon = left.getLeftNeighbor()
                    # Either left = closest neighbor on left with proper linear facet or success = False
                    doneRight = not (success)
                    while not (doneRight):
                        if right == merged_poly or right == left or right is None:
                            doneRight = True
                            success = False
                        elif right.hasFacet() and self._is_line_like_support_facet(
                            right.getFacet()
                        ):
                            # Linear facet on right
                            doneRight = True
                            success = True
                        else:
                            if right.getRightNeighbor() == right:
                                # Rare case with dead-end, break
                                success = False
                                break
                            else:
                                right: NeighboredPolygon = right.getRightNeighbor()
                    # If success, left/right are closest neighbors on left/right with proper linear facet
                    if success:
                        # Try corner
                        # print(f"Trying corner with left: {left.getFacet()} and right: {right.getFacet()}")
                        merged_poly.checkCornerFacet(
                            left.getFacet().pLeft,
                            left.getFacet().pRight,
                            right.getFacet().pRight,
                            right.getFacet().pLeft,
                        )
                        if not (merged_poly.hasFacet()):
                            # print("Failed to form corner facet")
                            # print(merged_poly)
                            pass
                        else:
                            pass
                            # print(merged_poly.getFacet())
                    else:
                        # print("Failed to find neighbors")
                        # print(merged_poly)
                        pass

            emit_stage("linear_corners")

            # Try fitting circular facets
            self._provenance_stage = "circular"
            i = 0
            while i < len(merge_ids):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if not (merged_poly.hasFacet()):
                    if merged_poly.fullyOriented():
                        if merged_poly.facet_type == "linear_deadend":
                            self._fit_deadend_facet(
                                merge_id, prefer_safe_circle=True
                            )
                        else:
                            merged_poly.fitCircularFacet(
                                root_guess=self._find_arc_fit_guess(merge_id)
                            )
                i += 1

            emit_stage("circular")

            print("Using circular corners")
            self._provenance_stage = "curved_corners"
            # For the ones not fit properly, try a corner
            for i in range(len(merge_ids)):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]

                # If no facet or curvatures differ by large/small threshold
                def _helper_checkCurvatureChange(test_poly, checkBig=True):
                    test_left: NeighboredPolygon = test_poly.getLeftNeighbor()
                    test_right: NeighboredPolygon = test_poly.getRightNeighbor()
                    if checkBig:
                        return (
                            test_poly.hasFacet()
                            and self._is_curved_corner_support_facet(
                                test_poly.getFacet()
                            )
                            and (
                                (
                                    test_left.hasFacet()
                                    and self._is_curved_corner_support_facet(
                                        test_left.getFacet()
                                    )
                                    and abs(
                                        test_left.getFacet().curvature
                                        - test_poly.getFacet().curvature
                                    )
                                    > NeighboredPolygon.curved_corner_curvature_threshold
                                )
                                or not (test_left.hasFacet())
                            )
                            and (
                                (
                                    test_right.hasFacet()
                                    and self._is_curved_corner_support_facet(
                                        test_right.getFacet()
                                    )
                                    and abs(
                                        test_right.getFacet().curvature
                                        - test_poly.getFacet().curvature
                                    )
                                    > NeighboredPolygon.curved_corner_curvature_threshold
                                )
                                or not (test_right.hasFacet())
                            )
                        )
                    else:
                        return (
                            test_poly.hasFacet()
                            and self._is_curved_corner_support_facet(
                                test_poly.getFacet()
                            )
                            and (
                                (
                                    test_left.hasFacet()
                                    and self._is_curved_corner_support_facet(
                                        test_left.getFacet()
                                    )
                                    and abs(
                                        test_left.getFacet().curvature
                                        - test_poly.getFacet().curvature
                                    )
                                    < NeighboredPolygon.curved_corner_curvature_threshold
                                )
                                or not (test_left.hasFacet())
                            )
                            and (
                                (
                                    test_right.hasFacet()
                                    and self._is_curved_corner_support_facet(
                                        test_right.getFacet()
                                    )
                                    and abs(
                                        test_right.getFacet().curvature
                                        - test_poly.getFacet().curvature
                                    )
                                    < NeighboredPolygon.curved_corner_curvature_threshold
                                )
                                or not (test_right.hasFacet())
                            )
                        )

                def _helper_needs_transition_rescue(test_poly):
                    if not test_poly.hasFacet() or not self._is_line_like_support_facet(
                        test_poly.getFacet()
                    ):
                        return False
                    neighbors = [
                        test_poly.getLeftNeighbor(),
                        test_poly.getRightNeighbor(),
                    ]
                    return any(
                        neighbor is not None
                        and neighbor.hasFacet()
                        and self._is_arc_like_support_facet(neighbor.getFacet())
                        for neighbor in neighbors
                    )

                if (
                    not (merged_poly.hasFacet())
                    or _helper_checkCurvatureChange(merged_poly, checkBig=True)
                    or _helper_needs_transition_rescue(merged_poly)
                ):
                    curved_corner_applied = False
                    # if _helper_checkCurvatureChange(merged_poly, checkBig=True):
                    #     print("Potential curved corner:")
                    #     print(merged_poly)
                    # loop through left until you find a linear/circular facet
                    doneLeft = False
                    left: NeighboredPolygon = merged_poly.getLeftNeighbor()
                    right: NeighboredPolygon = merged_poly.getRightNeighbor()
                    success = True
                    while not (doneLeft):
                        # Handle cases: left = itself (hack for when no left neighbor but properly oriented), looped all the way to the beginning, or no left neighbor
                        if left == merged_poly or left == right or left is None:
                            doneLeft = True
                            success = False
                        elif _helper_checkCurvatureChange(left, checkBig=False):
                            # Linear/circular facet on left
                            doneLeft = True
                            success = True
                        else:
                            if left.getLeftNeighbor() == left:
                                # Rare case with dead-end, break
                                success = False
                                break
                            else:
                                left: NeighboredPolygon = left.getLeftNeighbor()
                    # Either left = closest neighbor on left with proper linear/circular facet or success = False
                    doneRight = not (success)
                    while not (doneRight):
                        if right == merged_poly or right == left or right is None:
                            doneRight = True
                            success = False
                        elif _helper_checkCurvatureChange(right, checkBig=False):
                            # Linear/circular facet on right
                            doneRight = True
                            success = True
                        else:
                            if right.getRightNeighbor() == right:
                                # Rare case with dead-end, break
                                success = False
                                break
                            else:
                                right: NeighboredPolygon = right.getRightNeighbor()
                    # If success, left/right are closest neighbors on left/right with proper linear/circular facet
                    if success:
                        # Try all polys between left and right
                        corner_testing_poly: NeighboredPolygon = left.getRightNeighbor()
                        corner_assignments = []
                        corner_area_fraction_errors = []
                        corner_in_poly = False
                        while corner_testing_poly is not None and corner_testing_poly != right:
                            # can cache these. right now each unfit cell is checked once which checks all of its unfit neighbors. TODO fix
                            corner_facet, corner_area_fraction_error = (
                                corner_testing_poly.checkCurvedCornerFacet(
                                    left.getFacet(), right.getFacet(), ret=True
                                )
                            )
                            corner_assignments.append(
                                (corner_testing_poly, corner_facet)
                            )
                            corner_area_fraction_errors.append(
                                corner_area_fraction_error
                            )
                            if (
                                not (corner_in_poly)
                                and corner_facet is not None
                                and pointInPoly(
                                    corner_facet.corner, corner_testing_poly.points
                                )
                            ):
                                corner_in_poly = True
                            next_poly = corner_testing_poly.getRightNeighbor()
                            if next_poly == corner_testing_poly:
                                corner_testing_poly = None
                                break
                            corner_testing_poly = next_poly
                        # Each poly passes checkCurvedCornerFacet test and corner lies in at least one poly
                        if (
                            corner_testing_poly == right
                            and None not in corner_area_fraction_errors
                            and corner_in_poly
                        ):
                            # Compute geometric mean of errors #TODO is this the best choice?
                            corner_area_fraction_error_geomean = 1
                            for (
                                corner_area_fraction_error
                            ) in corner_area_fraction_errors:
                                corner_area_fraction_error_geomean *= (
                                    corner_area_fraction_error
                                ) ** (1 / len(corner_area_fraction_errors))
                            if (
                                corner_area_fraction_error_geomean
                                < NeighboredPolygon.curved_corner_area_threshold
                            ):
                                for (
                                    assigned_poly,
                                    assigned_corner_facet,
                                ) in corner_assignments:
                                    assigned_poly.setFacet(assigned_corner_facet)
                                curved_corner_applied = True
                            else:
                                print(corner_area_fraction_errors)
                                print(corner_area_fraction_error_geomean)
                                print("Failed to fit curved corner")
                        else:
                            print(
                                "Failed to fit curved corners: issue with at least one checkCurvedCornerFacet"
                            )

                    if use_curved_corner_rescues and not curved_corner_applied:
                        self._provenance_stage = "curved_corner_loop_rescue"
                        rescue_assignments = self._try_local_curved_corner_loop_rescue(
                            merged_poly
                        )
                        if rescue_assignments is not None:
                            for rescue_poly, rescue_facet in rescue_assignments.items():
                                rescue_poly.setFacet(rescue_facet)
                            curved_corner_applied = True

                    if use_curved_corner_rescues and not curved_corner_applied:
                        self._provenance_stage = "curved_corner_transition_rescue"
                        rescue_assignments = (
                            self._try_local_curved_corner_transition_rescue(
                                merged_poly
                            )
                        )
                        if rescue_assignments is not None:
                            for rescue_poly, rescue_facet in rescue_assignments.items():
                                rescue_poly.setFacet(rescue_facet)

            if use_linear_corner_rescues:
                self._provenance_stage = "linear_corner_rescues"
                # Rebuild straight geometry into nearby weak cells using exact support lines
                # or accepted corner-branch normals before falling all the way to linear.
                if not use_only_exact_linear_support:
                    self._rescue_corner_arc_corner_triplets(merge_ids)
                if use_repeated_corner_rescues and use_repeated_tiny_corner_rescues:
                    self._rescue_repeated_tiny_corner_triplets(merge_ids)
                self._propagate_exact_linear_supports(merge_ids)
                if not use_only_exact_linear_support:
                    self._rescue_corner_linear_bridge_cells(merge_ids)

            # For anything left, fit a linear facet
            self._provenance_stage = "final_fallback"
            i = 0
            while i < len(merge_ids):
                merge_id = merge_ids[i]
                merged_poly: NeighboredPolygon = self.merged_polys[merge_id]
                if not (merged_poly.hasFacet()):
                    if merged_poly.fullyOriented():
                        merged_poly.fitLinearFacet()
                    elif merged_poly.has3x3Stencil():
                        self._run_unresolved_plic_fallback(
                            merged_poly, merge_id, setting, plic_fallback
                        )
                    else:
                        print("Something wrong with facet fitting!")
                        merge_ids.pop(i)
                        i -= 1
                i += 1

            if use_linear_corner_rescues:
                self._provenance_stage = "post_fallback_rescues"
                # After the fallback pass, some outer support cells have settled into
                # reliable line facets. Revisit repeated curved-corner triplets now
                # that those linear supports are available.
                if (
                    use_repeated_corner_rescues
                    and use_repeated_corner_component_rescues
                ):
                    self._rescue_repeated_corner_components_as_linear_corners(merge_ids)
                if not use_only_exact_linear_support:
                    self._rescue_linear_corner_owner_intruder_arcs(merge_ids)

            emit_stage("final")

        elif setting == "extra_corners":
            pass

        # Return merge_polys
        return list(map(lambda x: self.merged_polys[x], merge_ids))

    # TODO if gap is too big, don't make C0?
    def makeC0(self, merged_polys):
        # If facet name in this list, don't use it for fitting C0
        fixed_endpoint_facetnames = ["corner"]

        def _helper_checkValidFacet(poly: NeighboredPolygon):
            return (
                poly is not None
                and poly.hasFacet()
                and poly.facet.name not in fixed_endpoint_facetnames
            )

        if hasattr(self, "_provenance_stage"):
            self._provenance_stage = "c0"

        # List of C0 facets matching order of merged_polys, None if merged_poly's facet is not to be C0 adjusted
        C0_facets = []
        for i, _ in enumerate(merged_polys):
            merged_poly: NeighboredPolygon = merged_polys[i]
            if _helper_checkValidFacet(merged_poly):
                facet = merged_poly.facet
                # Left
                left_neighbor: NeighboredPolygon = merged_poly.getLeftNeighbor()
                if _helper_checkValidFacet(left_neighbor):
                    left_pRight = left_neighbor.facet.pRight
                    C0_pLeft = lerp(facet.pLeft, left_pRight, 0.5)
                else:
                    C0_pLeft = facet.pLeft
                # Right
                right_neighbor: NeighboredPolygon = merged_poly.getRightNeighbor()
                if _helper_checkValidFacet(right_neighbor):
                    right_pLeft = right_neighbor.facet.pLeft
                    C0_pRight = lerp(facet.pRight, right_pLeft, 0.5)
                else:
                    C0_pRight = facet.pRight
                # Get unique linear/arc facet with endpoints C0_pLeft/Right and whose curvature matches area constraint
                C0_facet = merged_poly.fitCurvature(C0_pLeft, C0_pRight, ret=True)
                C0_facets.append(C0_facet)
            else:
                C0_facets.append(None)

        # Replace each facet with its C0 adjusted version.
        for i, _ in enumerate(merged_polys):
            merged_poly: NeighboredPolygon = merged_polys[i]
            merge_id = getattr(merged_poly, "_merge_id", None)
            diagnostic = getattr(merged_poly, "last_c0_fit_diagnostic", None) or {}
            if C0_facets[i] is not None:
                previous_override = getattr(self, "_provenance_override", None)
                if merge_id is not None and hasattr(self, "_provenance_override"):
                    self._provenance_override = {
                        merge_id: {
                            "event_kind": "c0_adjustment",
                            "policy": diagnostic.get("selected_branch", ""),
                            "reason": "conservative_refit_accepted",
                        }
                    }
                try:
                    merged_poly.setFacet(C0_facets[i])
                finally:
                    if hasattr(self, "_provenance_override"):
                        self._provenance_override = previous_override
            elif (
                merge_id is not None
                and diagnostic.get("selected_branch", "").startswith("rejected")
                and hasattr(self, "_record_facet_assignment")
            ):
                previous_override = self._provenance_override
                self._provenance_override = {
                    merge_id: {
                        "event_kind": "c0_rejection",
                        "policy": diagnostic.get("selected_branch", ""),
                        "reason": diagnostic.get("rejection_reason", "area_residual"),
                    }
                }
                try:
                    original_facet = merged_poly.getFacet()
                    self._record_facet_assignment(
                        merged_poly, original_facet, original_facet
                    )
                finally:
                    self._provenance_override = previous_override

        return merged_polys
