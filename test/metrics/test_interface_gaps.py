#!/usr/bin/env python3
"""
Quick sanity checks for interface gap metrics.

Run with:
python -m util.metrics.test.test_interface_gaps
"""

from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface import Interface, InterfaceComponent, FacetRecord
from util.metrics.metrics import interface_gap_stats


def test_zero_gap():
    facet1 = LinearFacet([0.0, 0.0], [1.0, 0.0])
    facet2 = LinearFacet([1.0, 0.0], [2.0, 0.0])
    records = [FacetRecord(cell_id=0, facet=facet1), FacetRecord(cell_id=1, facet=facet2)]
    interface = Interface(components=[InterfaceComponent(records=records, is_closed=False)])
    stats = interface_gap_stats(interface)
    print("Zero gap stats:", stats)


def test_nonzero_gap():
    facet1 = LinearFacet([0.0, 0.0], [1.0, 0.0])
    facet2 = LinearFacet([1.5, 0.0], [2.5, 0.0])
    records = [FacetRecord(cell_id=0, facet=facet1), FacetRecord(cell_id=1, facet=facet2)]
    interface = Interface(components=[InterfaceComponent(records=records, is_closed=False)])
    stats = interface_gap_stats(interface)
    print("Nonzero gap stats:", stats)


if __name__ == "__main__":
    test_zero_gap()
    test_nonzero_gap()
