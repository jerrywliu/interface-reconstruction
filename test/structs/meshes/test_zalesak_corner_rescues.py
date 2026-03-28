import contextlib
import io

from experiments.static import zalesak
from main.structs.facets.corner_facet import CornerFacet
from main.structs.meshes.merge_mesh import MergeMesh


class _DummyPoly:
    def __init__(self, facet):
        self._facet = facet
        self._left = None
        self._right = None

    def set_neighbors(self, left, right):
        self._left = left
        self._right = right

    def getLeftNeighbor(self):
        return self._left

    def getRightNeighbor(self):
        return self._right

    def hasFacet(self):
        return self._facet is not None

    def getFacet(self):
        return self._facet


def _run_cases(case_indices):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        _, _, _, case_records = zalesak.main(
            config_setting="static/zalesak",
            save_name="test_zalesak_corner_rescues",
            num_cases=max(case_indices) + 1,
            resolution=1.5,
            facet_algo="circular+corner",
            perturb_wiggle=0.3,
            perturb_seed=0,
            case_indices=case_indices,
            return_case_records=True,
            write_outputs=False,
            make_plots=False,
            metrics_output_dir=None,
        )
    return {record["case_index"]: record for record in case_records}


def test_case12_intruder_arc_is_rescued_without_regressing_case3():
    case_records = _run_cases([3, 12])

    assert case_records[3]["hausdorff"] < 1e-6
    assert case_records[12]["hausdorff"] < 2e-2
    assert case_records[12]["facet_gap"] < 5e-3


def test_collect_same_corner_component_breaks_two_cycle():
    mesh = MergeMesh.__new__(MergeMesh)
    facet = CornerFacet(
        None,
        [1.0, 1.0],
        None,
        1.0,
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    )

    poly_a = _DummyPoly(facet)
    poly_b = _DummyPoly(facet)
    poly_a.set_neighbors(poly_b, poly_b)
    poly_b.set_neighbors(poly_a, poly_a)

    signature = mesh._repeated_corner_triplet_signature(facet)
    component = mesh._collect_same_corner_component(poly_a, signature)

    assert component == [poly_b, poly_a]
