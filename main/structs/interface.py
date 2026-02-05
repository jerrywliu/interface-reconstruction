from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from main.geoms.geoms import getDistance
from main.structs.facets.base_facet import Facet


@dataclass
class FacetRecord:
    cell_id: int
    facet: Facet
    left_cell_id: Optional[int] = None
    right_cell_id: Optional[int] = None
    forward: bool = True

    def left_point(self):
        return self.facet.pLeft if self.forward else self.facet.pRight

    def right_point(self):
        return self.facet.pRight if self.forward else self.facet.pLeft


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

        records: Dict[int, FacetRecord] = {}
        for merge_id, poly in merge_items:
            facet = (
                facet_override.get(poly)
                if facet_override is not None
                else poly.getFacet()
            )
            if facet is None:
                continue
            records[merge_id] = FacetRecord(cell_id=merge_id, facet=facet)

        for merge_id, record in records.items():
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
            if left_id in records:
                record.left_cell_id = left_id
            if right_id in records:
                record.right_cell_id = right_id

        if infer_missing_neighbors:
            _infer_neighbors_from_adjacency(records, id_to_poly, poly_to_id)

        return cls(components=_order_components(records))


def _infer_neighbors_from_adjacency(
    records: Dict[int, FacetRecord],
    id_to_poly: Dict[int, object],
    poly_to_id: Dict[object, int],
) -> None:
    for merge_id, record in records.items():
        if record.left_cell_id is not None or record.right_cell_id is not None:
            continue
        if getattr(record.facet, "name", None) == "linear_deadend":
            continue
        poly = id_to_poly[merge_id]
        if not hasattr(poly, "adjacent_polys"):
            continue
        candidates = []
        for neighbor in poly.adjacent_polys:
            neighbor_id = poly_to_id.get(neighbor)
            if neighbor_id is None or neighbor_id == merge_id:
                continue
            if neighbor_id not in records:
                continue
            candidates.append(neighbor_id)
        if not candidates:
            continue

        def _best_left_neighbor():
            best_id = None
            best_dist = float("inf")
            for cand_id in candidates:
                cand_facet = records[cand_id].facet
                dist = getDistance(record.facet.pLeft, cand_facet.pRight)
                if dist < best_dist:
                    best_dist = dist
                    best_id = cand_id
            return best_id, best_dist

        def _best_right_neighbor():
            best_id = None
            best_dist = float("inf")
            for cand_id in candidates:
                cand_facet = records[cand_id].facet
                dist = getDistance(record.facet.pRight, cand_facet.pLeft)
                if dist < best_dist:
                    best_dist = dist
                    best_id = cand_id
            return best_id, best_dist

        left_id, left_dist = _best_left_neighbor()
        right_id, right_dist = _best_right_neighbor()

        if record.left_cell_id is None:
            record.left_cell_id = left_id
        if record.right_cell_id is None:
            record.right_cell_id = right_id

        if (
            record.left_cell_id is not None
            and record.right_cell_id is not None
            and record.left_cell_id == record.right_cell_id
            and len(candidates) > 1
        ):
            # Try to pick a different neighbor for the farther endpoint.
            if left_dist >= right_dist:
                alt = _second_best_left_neighbor(
                    record.facet.pLeft,
                    candidates,
                    records,
                    exclude=record.right_cell_id,
                )
                if alt is not None:
                    record.left_cell_id = alt
            else:
                alt = _second_best_right_neighbor(
                    record.facet.pRight,
                    candidates,
                    records,
                    exclude=record.left_cell_id,
                )
                if alt is not None:
                    record.right_cell_id = alt


def _second_best_left_neighbor(point, candidates, records, exclude):
    best_id = None
    best_dist = float("inf")
    for cand_id in candidates:
        if cand_id == exclude:
            continue
        cand_facet = records[cand_id].facet
        dist = getDistance(point, cand_facet.pRight)
        if dist < best_dist:
            best_dist = dist
            best_id = cand_id
    return best_id


def _second_best_right_neighbor(point, candidates, records, exclude):
    best_id = None
    best_dist = float("inf")
    for cand_id in candidates:
        if cand_id == exclude:
            continue
        cand_facet = records[cand_id].facet
        dist = getDistance(point, cand_facet.pLeft)
        if dist < best_dist:
            best_dist = dist
            best_id = cand_id
    return best_id


def _order_components(records: Dict[int, FacetRecord]) -> List[InterfaceComponent]:
    components: List[InterfaceComponent] = []
    visited = set()

    start_ids = [
        cell_id
        for cell_id, record in records.items()
        if record.left_cell_id is None
    ]

    def _walk(start_id: int) -> InterfaceComponent:
        ordered: List[FacetRecord] = []
        current_id = start_id
        closed = False
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            record = records[current_id]
            ordered.append(record)
            next_id = record.right_cell_id
            if next_id is None:
                break
            if next_id == start_id:
                closed = True
                break
            if next_id in visited:
                break
            current_id = next_id
        return InterfaceComponent(records=ordered, is_closed=closed)

    for start_id in start_ids:
        if start_id in visited:
            continue
        comp = _walk(start_id)
        if comp.records:
            components.append(comp)

    for cell_id in records.keys():
        if cell_id in visited:
            continue
        comp = _walk(cell_id)
        if comp.records:
            components.append(comp)

    return components
