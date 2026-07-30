import json

import numpy as np

from experiments.static import generate_section6_maintext_figures as figures
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from util.write_facets import writeFacets


def _grid_segments():
    return np.asarray(
        [
            [[1.0, 0.0], [1.0, 2.0]],
            [[0.0, 1.0], [2.0, 1.0]],
        ],
        dtype=float,
    )


def test_write_facets_saves_exact_arc_and_corner_metadata(tmp_path):
    path = tmp_path / "facets.vtp"
    facets = [
        CornerFacet(
            None,
            None,
            None,
            None,
            [0.25, 0.25],
            [0.75, 0.75],
            [1.75, 0.25],
        ),
        ArcFacet([1.0, 1.0], 1.2, [0.2, 1.0], [1.0, 0.2]),
    ]

    writeFacets(facets, str(path))
    metadata = json.loads(path.with_suffix(".facet_metadata.json").read_text())

    assert metadata["schema_version"] == 2
    assert len(metadata["primitives"]) == 3
    assert len(metadata["corners"]) == 1
    arc = metadata["primitives"][-1]
    assert arc["kind"] == "arc"
    assert arc["center"] == [1.0, 1.0]
    assert arc["radius"] == 1.2
    assert arc["orientation"] in {"ccw", "cw"}
    corner = metadata["corners"][0]
    assert corner["apex"] == [0.75, 0.75]
    assert [
        corner["left_primitive"]["p_left"],
        corner["right_primitive"]["p_right"],
    ] == [[0.25, 0.25], [1.75, 0.25]]


def test_corner_boundary_crossings_are_deduplicated_at_mesh_vertices():
    line = {
        "kind": "line",
        "source_name": "corner",
        "p_left": [0.25, 0.25],
        "p_right": [1.75, 1.75],
    }
    metadata = {
        "schema_version": 2,
        "corners": [
            {
                "apex": [1.75, 1.75],
                "left_primitive": line,
                "right_primitive": {
                    "kind": "line",
                    "source_name": "corner",
                    "p_left": [1.75, 1.75],
                    "p_right": [1.75, 0.25],
                },
            }
        ],
    }

    crossings = figures._corner_boundary_crossings(metadata, _grid_segments())

    assert len(crossings) == 2
    assert any(np.allclose(point, [1.0, 1.0]) for point in crossings)
    assert any(np.allclose(point, [1.75, 1.0]) for point in crossings)


def test_arc_corner_boundary_crossings_respect_arc_orientation():
    arc = {
        "kind": "arc",
        "source_name": "corner",
        "center": [1.0, 1.0],
        "radius": 1.0,
        "p_left": [0.0, 1.0],
        "p_right": [1.0, 0.0],
    }
    metadata = {
        "schema_version": 2,
        "corners": [
            {
                "apex": [1.0, 0.0],
                "left_primitive": arc,
                "right_primitive": {
                    "kind": "line",
                    "source_name": "corner",
                    "p_left": [1.0, 0.0],
                    "p_right": [1.0, -0.5],
                },
            }
        ],
    }

    mesh = np.asarray(
        [
            [[0.5, 0.0], [0.5, 2.0]],
            [[0.0, 0.5], [2.0, 0.5]],
        ],
        dtype=float,
    )
    crossings = figures._corner_boundary_crossings(metadata, mesh)

    assert len(crossings) == 2
    assert any(np.allclose(point, [0.5, 1.0 - np.sqrt(0.75)]) for point in crossings)
    assert any(np.allclose(point, [1.0 - np.sqrt(0.75), 0.5]) for point in crossings)
