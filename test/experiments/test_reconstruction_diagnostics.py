from main.structs.facets.linear_facet import LinearFacet
from util.reconstruction_diagnostics import _component_rows


class _Cell:
    def getFraction(self):
        return 0.5


class _Poly:
    def __init__(self, facet):
        self._facet = facet

    def getFacet(self):
        return self._facet

    def getFraction(self):
        return 0.5

    def fullyOriented(self):
        return False

    def has3x3Stencil(self):
        return True


class _Mesh:
    def __init__(self):
        fallback = LinearFacet([0.0, 0.5], [1.0, 0.5], name="LVIRA")
        self.merged_polys = {0: _Poly(None), 1: _Poly(fallback)}
        self.merge_ids_to_coords = [[(0, 0)], [(0, 0)]]
        self.coords_to_merge_id = [[1]]
        self.polys = [[_Cell()]]
        self.facet_provenance_events = [
            {
                "event_order": 1,
                "merge_id": 1,
                "stage": "plic_fallback",
                "facet_class": "linear",
            }
        ]
        self.plic_fallback_records = [{"merge_id": 1, "policy": "LVIRA"}]


def test_component_rows_only_include_active_merge_component():
    rows, components, _ = _component_rows(_Mesh(), case_index=4)

    assert len(rows) == 1
    assert rows[0]["cell_id"] == "0,0"
    assert rows[0]["merge_id"] == 1
    assert rows[0]["final_facet_class"] == "linear"
    assert rows[0]["construction_path"] == "plic_fallback"
    assert rows[0]["fallback_policy"] == "LVIRA"
    assert components == [(1, [(0, 0)])]
