from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface import Interface, InterfaceComponent, FacetRecord
from main.structs.interface_geometry import composite_from_facet
from util.metrics.metrics import interface_gap_stats


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
