from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface import Interface, InterfaceComponent, FacetRecord
from main.structs.interface_geometry import composite_from_facet
from util.metrics.metrics import calculate_facet_gaps, interface_gap_stats


class _MeshPoly:
    def __init__(self, facet, points=None):
        self.facet = facet
        self.points = points or []
        self.left = None
        self.right = None
        self.adjacent_polys = []

    def getFacet(self):
        return self.facet

    def getLeftNeighbor(self):
        return self.left

    def getRightNeighbor(self):
        return self.right


def test_zero_gap_between_adjacent_segments():
    facet1 = LinearFacet([0.0, 0.0], [1.0, 0.0])
    facet2 = LinearFacet([1.0, 0.0], [2.0, 0.0])
    records = [
        FacetRecord(cell_id=0, facet=facet1, record_id=(0, 0), right_record_id=(1, 0)),
        FacetRecord(cell_id=1, facet=facet2, record_id=(1, 0), left_record_id=(0, 0)),
    ]
    interface = Interface(
        components=[InterfaceComponent(records=records, is_closed=False)]
    )

    stats = interface_gap_stats(interface)
    assert stats["count"] == 1
    assert stats["mean"] < 1e-12
    assert stats["max"] < 1e-12


def test_nonzero_gap_between_adjacent_segments():
    facet1 = LinearFacet([0.0, 0.0], [1.0, 0.0])
    facet2 = LinearFacet([1.5, 0.0], [2.5, 0.0])
    records = [
        FacetRecord(cell_id=0, facet=facet1, record_id=(0, 0), right_record_id=(1, 0)),
        FacetRecord(cell_id=1, facet=facet2, record_id=(1, 0), left_record_id=(0, 0)),
    ]
    interface = Interface(
        components=[InterfaceComponent(records=records, is_closed=False)]
    )

    stats = interface_gap_stats(interface)
    assert stats["count"] == 1
    assert abs(stats["mean"] - 0.5) < 1e-12
    assert abs(stats["max"] - 0.5) < 1e-12


def test_internal_corner_joint_has_zero_gap():
    corner = CornerFacet(
        centerLeft=None,
        centerRight=None,
        radiusLeft=None,
        radiusRight=None,
        pLeft=[0.0, 0.0],
        corner=[1.0, 0.0],
        pRight=[1.0, 1.0],
    )
    composite = composite_from_facet(corner)
    left_primitive, right_primitive = composite.primitives

    records = [
        FacetRecord(
            cell_id=0,
            facet=left_primitive,
            record_id=(0, 0),
            right_record_id=(0, 1),
            right_joint_kind="corner",
        ),
        FacetRecord(
            cell_id=0,
            facet=right_primitive,
            record_id=(0, 1),
            left_record_id=(0, 0),
            left_joint_kind="corner",
        ),
    ]
    interface = Interface(
        components=[InterfaceComponent(records=records, is_closed=False)]
    )

    stats = interface_gap_stats(interface)
    assert stats["count"] == 1
    assert stats["mean"] < 1e-12
    assert stats["max"] < 1e-12


def test_gap_inference_fills_fallback_bridge_in_partly_oriented_interface():
    first = _MeshPoly(LinearFacet([0.0, 0.0], [1.0, 0.0]))
    second = _MeshPoly(LinearFacet([1.0, 0.0], [2.0, 0.0]))
    fallback = _MeshPoly(LinearFacet([2.0, 0.0], [3.0, 0.0]))
    first.right = second
    second.left = first
    second.adjacent_polys = [first, fallback]
    fallback.adjacent_polys = [second]
    mesh = type("Mesh", (), {"merged_polys": {1: first, 2: second, 9: fallback}})()
    facets = [first.facet, second.facet, fallback.facet]

    without_inference = calculate_facet_gaps(
        mesh, facets, infer_missing_neighbors=False, return_stats=True
    )
    with_inference = calculate_facet_gaps(
        mesh, facets, infer_missing_neighbors=True, return_stats=True
    )

    assert without_inference["count"] == 1
    assert with_inference["count"] == 2
    assert with_inference["max"] < 1e-12


def test_gap_inference_uses_active_polygons_instead_of_stale_merge_entries():
    first = _MeshPoly(
        LinearFacet([0.0, 0.0], [1.0, 0.0]),
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )
    stale = _MeshPoly(
        LinearFacet([9.0, 0.0], [10.0, 0.0]),
        [[9.0, 0.0], [10.0, 0.0], [10.0, 1.0], [9.0, 1.0]],
    )
    fallback = _MeshPoly(
        LinearFacet([1.0, 0.0], [2.0, 0.0]),
        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]],
    )
    mesh = type(
        "Mesh", (), {"merged_polys": {1: first, 2: stale, 3: fallback}}
    )()

    stats = calculate_facet_gaps(
        mesh,
        [first.facet, fallback.facet],
        reconstructed_polys=[first, fallback],
        return_stats=True,
    )

    assert stats["count"] == 1
    assert stats["max"] < 1e-12


def test_gap_inference_splices_unresolved_replacements_into_oriented_shortcut():
    polys = [
        _MeshPoly(LinearFacet([float(i), 0.0], [float(i + 1), 0.0]))
        for i in range(5)
    ]
    polys.append(_MeshPoly(LinearFacet([5.0, 0.0], [0.0, 0.0])))
    first, second, third, fourth, fifth, sixth = polys
    first.right = fourth
    fourth.left = first
    fourth.right = fifth
    fifth.left = fourth
    fifth.right = sixth
    sixth.left = fifth
    sixth.right = first
    first.left = sixth
    first.adjacent_polys = [sixth, second, fourth]
    second.adjacent_polys = [first, third]
    third.adjacent_polys = [second, fourth]
    fourth.adjacent_polys = [first, third, fifth]
    fifth.adjacent_polys = [fourth, sixth]
    sixth.adjacent_polys = [fifth, first]
    mesh = type(
        "Mesh",
        (),
        {"merged_polys": {index: poly for index, poly in enumerate(polys)}},
    )()

    stats = calculate_facet_gaps(
        mesh,
        [poly.facet for poly in polys],
        reconstructed_polys=polys,
        return_stats=True,
    )

    assert stats["count"] == 6
    assert stats["max"] < 1e-12
