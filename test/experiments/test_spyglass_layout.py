import math

import numpy as np
import pytest

from experiments.static.generate_section6_maintext_figures import (
    REPRESENTATIVE_CASES,
    _circle_case_params,
    _endpoint_variant_specs,
    _endpoint_visibility_spec,
    _ellipse_case_params,
    _inset_bounds,
    _outer_spyglass_axes,
    _panel_spyglass_spec,
    _resolution_panel_spec,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_outer_spyglass_does_not_overlap_main_axes(side):
    left, bottom, width, height = _outer_spyglass_axes(side)

    assert 0.0 <= bottom < bottom + height <= 1.0
    if side == "left":
        assert left + width < 0.0
    else:
        assert left > 1.0


def test_panel_spyglass_uses_outer_side_for_each_column():
    spec = {"inset": {"kind": "square_corner"}, "inset_axes": [0.1, 0.1, 0.3, 0.3]}

    left = _panel_spyglass_spec(spec, 0)
    right = _panel_spyglass_spec(spec, 1)

    assert left["inset_side"] == "left"
    assert right["inset_side"] == "right"
    assert left["inset_connector"] == "frame"
    assert right["inset_connector"] == "frame"
    assert "inset_axes" not in left
    assert "inset_axes" not in right
    assert "inset_side" not in spec


def test_paired_endpoint_variants_keep_spyglass_labels():
    assert _endpoint_variant_specs("paired") == [
        ("with_endpoints", "_with_endpoints", True),
        ("clean", "_clean", False),
    ]

    original = {"inset": {"kind": "square_corner"}}
    clean = _endpoint_visibility_spec(original, show_main_endpoints=False)

    assert clean["show_main_endpoints"] is False
    assert clean["show_inset_endpoints"] is True
    assert "show_main_endpoints" not in original

    resolution = _resolution_panel_spec({**clean, "case_index": 22})
    assert resolution["show_main_endpoints"] is False
    assert resolution["show_inset_endpoints"] is True


def test_circle_spyglass_targets_the_upper_right_interface():
    spec = REPRESENTATIVE_CASES["circles"]
    bounds = _inset_bounds("circles", spec)
    params = _circle_case_params(spec["case_index"])
    target = params["center"] + params["radius"] / math.sqrt(2.0)

    assert bounds is not None
    assert np.allclose(
        [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0],
        target,
    )


def test_ellipse_spyglass_targets_a_maximum_curvature_tip():
    spec = REPRESENTATIVE_CASES["ellipses"]
    bounds = _inset_bounds("ellipses", spec)
    params = _ellipse_case_params(spec["case_index"])
    target = params["center"] + params["major_axis"] * np.asarray(
        [math.cos(params["theta"]), math.sin(params["theta"])]
    )

    assert bounds is not None
    assert np.allclose(
        [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0],
        target,
    )
