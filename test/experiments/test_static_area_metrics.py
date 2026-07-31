import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.static import squares, zalesak
from main.geoms.geoms import getArea, getPolyLineArea
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from util.metrics.area_metrics import (
    AreaMetricError,
    facet_area_in_polygon,
    facet_from_geometry,
)


UNIT_CELL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
WITNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "area_metrics"
    / "july_area_metric_witnesses.json"
)


def _witness(benchmark):
    records = json.loads(WITNESS_PATH.read_text(encoding="utf-8"))["witnesses"]
    return next(record for record in records if record["benchmark"] == benchmark)


class _Polygon:
    def __init__(self, points=UNIT_CELL):
        self.points = points


def test_exact_linear_arc_and_corner_areas():
    line = LinearFacet([0.3, 0.0], [0.3, 1.0])
    assert facet_area_in_polygon(UNIT_CELL, line) == pytest.approx(0.3)

    arc = ArcFacet([0.5, 0.5], 0.25, [0.75, 0.5], [0.5, 0.75])
    assert facet_area_in_polygon(UNIT_CELL, arc) == pytest.approx(math.pi / 16)
    complement_arc = ArcFacet(
        [0.5, 0.5], -0.25, [0.75, 0.5], [0.5, 0.75]
    )
    assert facet_area_in_polygon(UNIT_CELL, complement_arc) == pytest.approx(
        1 - math.pi / 16
    )

    corner = CornerFacet(
        centerLeft=None,
        centerRight=None,
        radiusLeft=None,
        radiusRight=None,
        pLeft=[0.5, 1.0],
        corner=[0.5, 0.5],
        pRight=[1.0, 0.5],
    )
    assert facet_area_in_polygon(UNIT_CELL, corner) == pytest.approx(0.25)


def test_july_square_corner_witness_is_not_replaced_by_its_chord():
    witness = _witness("squares")
    geometry = witness["facet_geometry"]
    legacy_area = getPolyLineArea(
        witness["polygon"], geometry["p_left"], geometry["p_right"]
    )
    corrected_area = facet_area_in_polygon(witness["polygon"], geometry)

    assert legacy_area == pytest.approx(witness["legacy_local_area"])
    assert abs(legacy_area - witness["target_area"]) > 3.0
    assert corrected_area == pytest.approx(witness["target_area"], abs=2.0e-10)


def test_july_zalesak_arc_witness_uses_the_signed_supporting_circle():
    witness = _witness("zalesak")
    facet = facet_from_geometry(witness["facet_geometry"])
    legacy_area = facet.getPolyIntersectArea(witness["polygon"])
    legacy_clamped = min(max(legacy_area, 0.0), abs(getArea(witness["polygon"])))
    corrected_area = facet_area_in_polygon(witness["polygon"], facet)

    assert legacy_area == pytest.approx(witness["legacy_local_area_before_clamp"])
    assert legacy_clamped == pytest.approx(4.0)
    assert corrected_area == pytest.approx(witness["target_area"], abs=3.0e-10)


@pytest.mark.parametrize("driver", [squares, zalesak])
def test_driver_area_metric_preserves_valid_facet_area(driver):
    area = driver.reconstructed_mixed_area(
        [_Polygon()],
        [LinearFacet([0.3, 0.0], [0.3, 1.0])],
        case_index=6,
    )

    assert area == pytest.approx(0.3)


@pytest.mark.parametrize("driver", [squares, zalesak])
def test_driver_area_metric_rejects_missing_facet(driver):
    with pytest.raises(AreaMetricError, match="facet 0 is missing"):
        driver.reconstructed_mixed_area([_Polygon()], [None], case_index=7)


@pytest.mark.parametrize("driver", [squares, zalesak])
def test_driver_area_metric_rejects_short_facet_list(driver):
    with pytest.raises(
        AreaMetricError, match="got 1 facets for 2 polygons"
    ):
        driver.reconstructed_mixed_area(
            [_Polygon(), _Polygon()],
            [LinearFacet([0.5, 0.0], [0.5, 1.0])],
            case_index=8,
        )


@pytest.mark.parametrize("driver", [squares, zalesak])
def test_driver_area_metric_propagates_geometry_evaluation_failure(
    driver, monkeypatch
):
    def fail_evaluation(_polygon, _facet):
        raise ValueError("invalid reconstructed geometry")

    monkeypatch.setattr(driver, "facet_area_in_polygon", fail_evaluation)
    with pytest.raises(
        AreaMetricError, match="failed to evaluate reconstructed facet 0"
    ) as exc_info:
        driver.reconstructed_mixed_area(
            [_Polygon()],
            [LinearFacet([0.5, 0.0], [0.5, 1.0])],
            case_index=9,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_square_driver_uses_active_polygons_returned_by_reconstruction(
    tmp_path, monkeypatch
):
    active = _Polygon()
    stale = _Polygon()
    facet = LinearFacet([0.3, 0.0], [0.3, 1.0])
    mesh = SimpleNamespace(
        merged_polys={0: active, 1: stale},
        plic_fallback_records=[],
        reconstruction_diagnostic_summary={},
        initializeFractions=lambda fractions: None,
    )
    output_dirs = {
        "base": str(tmp_path),
        "metrics": str(tmp_path / "metrics"),
        "vtk": str(tmp_path / "vtk"),
        "vtk_true": str(tmp_path / "vtk" / "true"),
        "plt_areas": str(tmp_path / "plt" / "areas"),
        "plt_partial": str(tmp_path / "plt" / "partial"),
    }
    for directory in output_dirs.values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        squares,
        "read_yaml",
        lambda path: {
            "TEST": {"SAVE_NAME": "unused"},
            "MESH": {"GRID_SIZE": 100.0, "RESOLUTION": 1.0},
            "GEOMS": {"FACET_ALGO": "linear", "THRESHOLD": 1.0e-10, "DO_C0": False},
        },
    )
    monkeypatch.setattr(squares, "setupOutputDirs", lambda *args, **kwargs: output_dirs)
    monkeypatch.setattr(squares, "write_run_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "make_points_from_config", lambda config: [])
    monkeypatch.setattr(squares, "MergeMesh", lambda *args, **kwargs: mesh)
    monkeypatch.setattr(squares, "writeMesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "initializePoly", lambda *args, **kwargs: [])
    monkeypatch.setattr(squares, "plotAreas", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "plotPartialAreas", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "writeFacets", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "append_case_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "append_case_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(squares, "calculate_facet_gaps", lambda *args: 0.0)
    monkeypatch.setattr(squares, "hausdorff_interface", lambda *args: 0.0)

    def reconstruct(*args, **kwargs):
        assert kwargs["return_polys"] is True
        return [facet], [active]

    seen = {}

    def true_area(polygons, geometry):
        seen["truth_polygons"] = list(polygons)
        return 0.3

    def reconstructed_area(polygons, facets, *, case_index):
        seen["reconstructed_polygons"] = list(polygons)
        assert list(facets) == [facet]
        return 0.3

    monkeypatch.setattr(squares, "runReconstruction", reconstruct)
    monkeypatch.setattr(squares, "true_area_over_active_polygons", true_area)
    monkeypatch.setattr(squares, "reconstructed_mixed_area", reconstructed_area)

    areas, gaps, hausdorff = squares.main(
        "static/square", num_squares=1, save_name="unit"
    )

    assert seen == {
        "truth_polygons": [active],
        "reconstructed_polygons": [active],
    }
    assert areas == [0.0]
    assert gaps == [0.0]
    assert hausdorff == [0.0]
