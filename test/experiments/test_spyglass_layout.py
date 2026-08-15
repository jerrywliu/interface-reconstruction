import numpy as np
import pytest

from experiments.static.generate_section6_maintext_figures import (
    _boundary_stencil_bounds,
    _boundary_stencil_indices,
    _line_boundary_comparison,
    _endpoint_variant_specs,
    _endpoint_visibility_spec,
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


def test_line_boundary_comparison_uses_reconstructed_crossing():
    recon_segments = np.asarray(
        [
            [
                [47.44636747263007, 0.0],
                [47.674955280912926, 3.6332829047239024],
            ]
        ]
    )

    reconstructed, exact, offset = _line_boundary_comparison(
        case_index=6,
        edge="bottom",
        bounds=(0.0, 100.0, 0.0, 100.0),
        recon_segments=recon_segments,
    )

    assert reconstructed == pytest.approx([47.44636747263007, 0.0])
    assert exact == pytest.approx([47.44636798427017, 0.0])
    assert offset == pytest.approx(5.11640102e-7)


def test_bottom_boundary_stencil_is_three_columns_by_two_available_rows():
    indices = _boundary_stencil_indices(
        cell_index=(15, 0),
        edge="bottom",
        grid_shape=(32, 32),
        stencil_shape=(3, 2),
    )

    assert indices == [
        (14, 0),
        (15, 0),
        (16, 0),
        (14, 1),
        (15, 1),
        (16, 1),
    ]


def test_bottom_boundary_stencil_bounds_include_off_domain_context():
    polygons = []
    for iy in range(2):
        for ix in range(3):
            polygons.append(
                np.asarray(
                    [
                        [float(ix), float(iy)],
                        [float(ix), float(iy + 1)],
                        [float(ix + 1), float(iy + 1)],
                        [float(ix + 1), float(iy)],
                    ]
                )
            )

    bounds = _boundary_stencil_bounds(
        cell_polygons=polygons,
        edge="bottom",
        stencil_shape=(3, 2),
        off_domain_fraction=0.35,
        padding_fraction=0.0,
    )

    assert bounds == pytest.approx((0.0, 3.0, -0.35, 2.0))
