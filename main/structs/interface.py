from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from main.geoms.geoms import getDistance
from main.structs.facets.base_facet import Facet
from main.structs.interface_geometry import composite_from_facet


@dataclass
class FacetRecord:
    cell_id: int
    facet: Facet
    record_id: Optional[Tuple[int, int]] = None
    left_cell_id: Optional[int] = None
    right_cell_id: Optional[int] = None
    left_record_id: Optional[Tuple[int, int]] = None
    right_record_id: Optional[Tuple[int, int]] = None
    left_joint_kind: str = "smooth"
    right_joint_kind: str = "smooth"
    forward: bool = True

    def left_point(self):
        return self.facet.pLeft if self.forward else self.facet.pRight

    def right_point(self):
        return self.facet.pRight if self.forward else self.facet.pLeft

    def left_joint(self):
        return self.left_joint_kind if self.forward else self.right_joint_kind

    def right_joint(self):
        return self.right_joint_kind if self.forward else self.left_joint_kind


@dataclass
class InterfaceComponent:
    records: List[FacetRecord]
    is_closed: bool = False


@dataclass
class Interface:
    components: List[InterfaceComponent]

    @classmethod
    def from_merge_mesh(
        cls,
        mesh,
        reconstructed_facets: Optional[Iterable[Facet]] = None,
        infer_missing_neighbors: bool = False,
    ) -> "Interface":
        """
        Build an Interface from a MergeMesh. Uses oriented neighbors when available.
        Optionally infers missing left/right neighbors using adjacency + endpoint distances.
        """
        if mesh is None or not hasattr(mesh, "merged_polys"):
            return cls(components=[])

        merge_items = list(mesh.merged_polys.items())
        id_to_poly = {merge_id: poly for merge_id, poly in merge_items}
        poly_to_id = {poly: merge_id for merge_id, poly in merge_items}

        facet_override = None
        if reconstructed_facets is not None:
            reconstructed_facets = list(reconstructed_facets)
            if len(reconstructed_facets) == len(merge_items):
                has_mesh_facets = any(poly.getFacet() is not None for _, poly in merge_items)
                if not has_mesh_facets:
                    facet_override = {
                        poly: facet
                        for (_, poly), facet in zip(merge_items, reconstructed_facets)
                    }

        records: Dict[Tuple[int, int], FacetRecord] = {}
        cell_to_records: Dict[int, List[Tuple[int, int]]] = {}
        for merge_id, poly in merge_items:
            facet = (
                facet_override.get(poly)
                if facet_override is not None
                else poly.getFacet()
            )
            if facet is None:
                continue
            composite = composite_from_facet(facet)
            record_ids = []
            for prim_idx, primitive in enumerate(composite.primitives):
                record_id = (merge_id, prim_idx)
                records[record_id] = FacetRecord(
                    record_id=record_id,
                    cell_id=merge_id,
                    facet=primitive,
                )
                record_ids.append(record_id)
            if not record_ids:
                continue
            cell_to_records[merge_id] = record_ids
            for prim_idx in range(len(record_ids) - 1):
                left_record = records[record_ids[prim_idx]]
                right_record = records[record_ids[prim_idx + 1]]
                joint_kind = (
                    composite.joints[prim_idx].kind
                    if prim_idx < len(composite.joints)
                    else "smooth"
                )
                left_record.right_record_id = right_record.record_id
                left_record.right_joint_kind = joint_kind
                right_record.left_record_id = left_record.record_id
                right_record.left_joint_kind = joint_kind

        for merge_id, poly in merge_items:
            record_ids = cell_to_records.get(merge_id)
            if not record_ids:
                continue
            poly = id_to_poly[merge_id]
            left_poly = poly.getLeftNeighbor() if hasattr(poly, "getLeftNeighbor") else None
            right_poly = (
                poly.getRightNeighbor() if hasattr(poly, "getRightNeighbor") else None
            )
            left_id = poly_to_id.get(left_poly)
            right_id = poly_to_id.get(right_poly)
            if left_id == merge_id:
                left_id = None
            if right_id == merge_id:
                right_id = None
            left_record = records[record_ids[0]]
            right_record = records[record_ids[-1]]
            if left_id in cell_to_records:
                left_record.left_cell_id = left_id
                left_record.left_record_id = cell_to_records[left_id][-1]
            if right_id in cell_to_records:
                right_record.right_cell_id = right_id
                right_record.right_record_id = cell_to_records[right_id][0]

        if infer_missing_neighbors:
            _infer_neighbors_from_adjacency(
                records, cell_to_records, id_to_poly, poly_to_id
            )

        return cls(components=_order_components(records))


def _infer_neighbors_from_adjacency(
    records: Dict[Tuple[int, int], FacetRecord],
    cell_to_records: Dict[int, List[Tuple[int, int]]],
    id_to_poly: Dict[int, object],
    poly_to_id: Dict[object, int],
) -> None:
    for merge_id, record_ids in cell_to_records.items():
        left_record = records[record_ids[0]]
        right_record = records[record_ids[-1]]
        if (
            left_record.left_cell_id is not None
            and right_record.right_cell_id is not None
        ):
            continue
        if getattr(left_record.facet, "name", None) == "linear_deadend":
            continue
        poly = id_to_poly[merge_id]
        if not hasattr(poly, "adjacent_polys"):
            continue
        candidates = []
        for neighbor in poly.adjacent_polys:
            neighbor_id = poly_to_id.get(neighbor)
            if neighbor_id is None or neighbor_id == merge_id:
                continue
            if neighbor_id not in cell_to_records:
                continue
            candidates.append(neighbor_id)
        if not candidates:
            continue

        def _best_left_neighbor(point):
            best_id = None
            best_dist = float("inf")
            for cand_id in candidates:
                cand_record = records[cell_to_records[cand_id][-1]]
                dist = getDistance(point, cand_record.right_point())
                if dist < best_dist:
                    best_dist = dist
                    best_id = cand_id
            return best_id, best_dist

        def _best_right_neighbor(point):
            best_id = None
            best_dist = float("inf")
            for cand_id in candidates:
                cand_record = records[cell_to_records[cand_id][0]]
                dist = getDistance(point, cand_record.left_point())
                if dist < best_dist:
                    best_dist = dist
                    best_id = cand_id
            return best_id, best_dist

        left_id, left_dist = _best_left_neighbor(left_record.left_point())
        right_id, right_dist = _best_right_neighbor(right_record.right_point())

        if left_record.left_cell_id is None and left_id in cell_to_records:
            left_record.left_cell_id = left_id
            left_record.left_record_id = cell_to_records[left_id][-1]
        if right_record.right_cell_id is None and right_id in cell_to_records:
            right_record.right_cell_id = right_id
            right_record.right_record_id = cell_to_records[right_id][0]

        if (
            left_record.left_cell_id is not None
            and right_record.right_cell_id is not None
            and left_record.left_cell_id == right_record.right_cell_id
            and len(candidates) > 1
        ):
            # Try to pick a different neighbor for the farther endpoint.
            if left_dist >= right_dist:
                alt = _second_best_left_neighbor(
                    left_record.left_point(),
                    candidates,
                    records,
                    cell_to_records,
                    exclude=right_record.right_cell_id,
                )
                if alt is not None:
                    left_record.left_cell_id = alt
                    left_record.left_record_id = cell_to_records[alt][-1]
            else:
                alt = _second_best_right_neighbor(
                    right_record.right_point(),
                    candidates,
                    records,
                    cell_to_records,
                    exclude=left_record.left_cell_id,
                )
                if alt is not None:
                    right_record.right_cell_id = alt
                    right_record.right_record_id = cell_to_records[alt][0]


def _second_best_left_neighbor(point, candidates, records, cell_to_records, exclude):
    best_id = None
    best_dist = float("inf")
    for cand_id in candidates:
        if cand_id == exclude:
            continue
        cand_record = records[cell_to_records[cand_id][-1]]
        dist = getDistance(point, cand_record.right_point())
        if dist < best_dist:
            best_dist = dist
            best_id = cand_id
    return best_id


def _second_best_right_neighbor(point, candidates, records, cell_to_records, exclude):
    best_id = None
    best_dist = float("inf")
    for cand_id in candidates:
        if cand_id == exclude:
            continue
        cand_record = records[cell_to_records[cand_id][0]]
        dist = getDistance(point, cand_record.left_point())
        if dist < best_dist:
            best_dist = dist
            best_id = cand_id
    return best_id


def _order_components(
    records: Dict[Tuple[int, int], FacetRecord]
) -> List[InterfaceComponent]:
    components: List[InterfaceComponent] = []
    visited = set()

    def _neighbor_ids(record: FacetRecord):
        neighbors = []
        if record.left_record_id is not None:
            neighbors.append(record.left_record_id)
        if record.right_record_id is not None and record.right_record_id not in neighbors:
            neighbors.append(record.right_record_id)
        return neighbors

    def _min_endpoint_distance(record: FacetRecord, point) -> float:
        return min(
            getDistance(record.facet.pLeft, point),
            getDistance(record.facet.pRight, point),
        )

    def _choose_forward_for_entry_point(record: FacetRecord, entry_point) -> bool:
        return getDistance(entry_point, record.facet.pLeft) <= getDistance(
            entry_point, record.facet.pRight
        )

    def _choose_start_orientation(start_record: FacetRecord):
        neighbors = _neighbor_ids(start_record)
        if not neighbors:
            return True, None
        if len(neighbors) == 1:
            neighbor = records[neighbors[0]]
            forward_score = _min_endpoint_distance(neighbor, start_record.facet.pRight)
            reverse_score = _min_endpoint_distance(neighbor, start_record.facet.pLeft)
            return forward_score <= reverse_score, neighbors[0]

        best = None
        for forward in (True, False):
            left_endpoint = start_record.facet.pLeft if forward else start_record.facet.pRight
            right_endpoint = start_record.facet.pRight if forward else start_record.facet.pLeft
            for next_id in neighbors:
                score = _min_endpoint_distance(records[next_id], right_endpoint)
                other_ids = [neighbor_id for neighbor_id in neighbors if neighbor_id != next_id]
                if other_ids:
                    score += _min_endpoint_distance(records[other_ids[0]], left_endpoint)
                candidate = (score, forward, next_id)
                if best is None or candidate[0] < best[0]:
                    best = candidate

        return best[1], best[2]

    start_ids = [
        record_id
        for record_id, record in records.items()
        if len(_neighbor_ids(record)) < 2
    ]

    def _walk(
        start_id: Tuple[int, int],
        start_forward: bool,
        start_next_id: Optional[Tuple[int, int]],
    ) -> InterfaceComponent:
        ordered: List[FacetRecord] = []
        current_id = start_id
        current_forward = start_forward
        previous_id = None
        closed = False
        while current_id is not None and current_id not in visited:
            record = records[current_id]
            record.forward = current_forward
            visited.add(current_id)
            ordered.append(record)

            neighbors = _neighbor_ids(record)
            if previous_id is None:
                next_id = start_next_id if start_next_id in neighbors else None
                if next_id is None and neighbors:
                    exit_point = record.right_point()
                    next_id = min(
                        neighbors,
                        key=lambda neighbor_id: _min_endpoint_distance(
                            records[neighbor_id], exit_point
                        ),
                    )
            else:
                remaining = [neighbor_id for neighbor_id in neighbors if neighbor_id != previous_id]
                if not remaining:
                    next_id = None
                elif len(remaining) == 1:
                    next_id = remaining[0]
                else:
                    exit_point = record.right_point()
                    next_id = min(
                        remaining,
                        key=lambda neighbor_id: _min_endpoint_distance(
                            records[neighbor_id], exit_point
                        ),
                    )

            if next_id is None:
                break
            if next_id == start_id:
                closed = True
                break
            if next_id in visited:
                break

            exit_point = record.right_point()
            next_record = records[next_id]
            next_forward = _choose_forward_for_entry_point(next_record, exit_point)
            previous_id = current_id
            current_id = next_id
            current_forward = next_forward
        return InterfaceComponent(records=ordered, is_closed=closed)

    for start_id in start_ids:
        if start_id in visited:
            continue
        start_forward, start_next_id = _choose_start_orientation(records[start_id])
        comp = _walk(start_id, start_forward, start_next_id)
        if comp.records:
            components.append(comp)

    for record_id in records.keys():
        if record_id in visited:
            continue
        start_forward, start_next_id = _choose_start_orientation(records[record_id])
        comp = _walk(record_id, start_forward, start_next_id)
        if comp.records:
            components.append(comp)

    return components
