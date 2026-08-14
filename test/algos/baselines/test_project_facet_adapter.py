import math

import pytest

from main.algos.baselines.external_geometry import (
    ExternalArcPrimitive,
    ExternalLinePrimitive,
)
from main.algos.baselines.project_facet_adapter import (
    external_primitive_from_facet_metadata,
    external_primitive_from_project_facet,
    external_primitives_from_facet_metadata,
)
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


def test_live_line_facet_retains_exact_endpoints():
    primitive = external_primitive_from_project_facet(
        LinearFacet([0.25, 0.5], [1.75, 2.0])
    )

    assert isinstance(primitive, ExternalLinePrimitive)
    assert primitive.p_left == (0.25, 0.5)
    assert primitive.p_right == (1.75, 2.0)


@pytest.mark.parametrize(
    "radius,p_right",
    ((1.0, [0.0, 1.0]), (-1.0, [0.0, -1.0])),
)
def test_live_arc_facet_retains_endpoints_midpoint_and_orientation(radius, p_right):
    facet = ArcFacet([0.0, 0.0], radius, [1.0, 0.0], p_right)
    primitive = external_primitive_from_project_facet(facet)

    assert isinstance(primitive, ExternalArcPrimitive)
    assert primitive.radius == pytest.approx(1.0)
    assert primitive.p_left == pytest.approx(facet.pLeft)
    assert primitive.p_right == pytest.approx(facet.pRight)
    assert primitive.point(0.5) == pytest.approx(facet.midpoint)


def test_schema_v2_arc_uses_saved_signed_span():
    record = {
        "index": 3,
        "facet_index": 2,
        "primitive_index": 0,
        "kind": "arc",
        "source_name": "arc",
        "p_left": [1.0, 0.0],
        "p_right": [0.0, -1.0],
        "center": [0.0, 0.0],
        "radius": -1.0,
        "signed_delta": -0.5 * math.pi,
    }

    primitive = external_primitive_from_facet_metadata(record)

    assert primitive.sweep_angle == pytest.approx(-0.5 * math.pi)
    assert primitive.p_right == pytest.approx(record["p_right"])
    assert primitive.metadata["serialized_index"] == 3


def test_complete_metadata_requires_schema_v2_and_active_primitives():
    with pytest.raises(ValueError, match="schema version 2"):
        external_primitives_from_facet_metadata({"schema_version": 1, "primitives": []})
    with pytest.raises(ValueError, match="no active primitives"):
        external_primitives_from_facet_metadata({"schema_version": 2, "primitives": []})
