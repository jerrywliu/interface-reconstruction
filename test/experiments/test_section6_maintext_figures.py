import numpy as np

from experiments.static.generate_section6_maintext_figures import (
    MAIN_RESOLUTION_LEGEND_GAP_POINTS,
    REPRESENTATIVE_CASES,
    RESOLUTION_QUANT_SPECS,
    _build_pooled_method_curves_by_resolution,
    _generate_resolution_quantitative_panel,
)


def test_circle_maintext_uses_facet_gap():
    assert RESOLUTION_QUANT_SPECS["circles"]["metrics"] == (
        "hausdorff",
        "facet_gap",
    )


def test_maintext_layout_exceptions_and_ellipse_spyglass_are_explicit():
    assert MAIN_RESOLUTION_LEGEND_GAP_POINTS == {"lines": 5.0, "circles": 11.0}
    assert REPRESENTATIVE_CASES["ellipses"]["inset"] == {
        "kind": "ellipse_curvature_tip",
        "half_span": 5.0,
    }


def test_pooled_resolution_curves_use_all_case_values():
    exp_data = {
        "linear": {
            "hausdorff": {
                0.32: {
                    0.0: {"value": [1.0, 2.0, 3.0]},
                    0.1: {"value": [10.0, 20.0, 30.0]},
                }
            }
        }
    }
    curves = _build_pooled_method_curves_by_resolution(exp_data, "hausdorff")
    assert np.allclose(curves["linear"]["median"], [6.5])
    assert np.allclose(curves["linear"]["p25"], [2.25])
    assert np.allclose(curves["linear"]["p75"], [17.5])


def test_ellipse_resolution_panel_uses_metric_specific_limits_and_marks_order(
    tmp_path, monkeypatch
):
    curves_by_metric = {
        "hausdorff": {
            "circular": {
                "x_values": np.asarray([0.32, 0.64, 1.28]),
                "median": np.asarray([1.0e-2, 5.0e-3, 2.5e-3]),
                "p25": np.asarray([9.0e-3, 4.5e-3, 2.2e-3]),
                "p75": np.asarray([1.1e-2, 5.5e-3, 2.8e-3]),
            }
        },
        "facet_gap": {
            "circular": {
                "x_values": np.asarray([0.32, 0.64, 1.28]),
                "median": np.asarray([1.0e-2, 1.25e-3, 1.5625e-4]),
                "p25": np.asarray([9.0e-3, 1.1e-3, 1.4e-4]),
                "p75": np.asarray([1.1e-2, 1.4e-3, 1.7e-4]),
            }
        },
    }
    monkeypatch.setattr(
        "experiments.static.generate_section6_maintext_figures._build_method_curves_by_resolution",
        lambda _data, metric: curves_by_metric[metric],
    )
    saved = []
    monkeypatch.setattr(
        "experiments.static.generate_section6_maintext_figures._save_figure",
        lambda figure, _path: saved.append(figure),
    )

    _generate_resolution_quantitative_panel(
        "ellipses",
        {"circular": {}},
        ["circular"],
        ("hausdorff", "facet_gap"),
        tmp_path / "ellipse.png",
    )

    axes = saved[0].axes[:2]
    assert not np.allclose(axes[0].get_ylim(), axes[1].get_ylim())
    assert all(axis.get_xscale() == "log" for axis in axes)
    assert np.allclose(axes[0].get_xticks(), [32.0, 64.0, 128.0])
    assert all(axis.title.get_fontweight() == "normal" for axis in axes)
    assert all(axis.yaxis.label.get_fontfamily() == ["serif"] for axis in axes)
    assert axes[0].get_lines()[0].get_marker() == "v"
    assert axes[0].get_lines()[0].get_color() == "#D55E00"
    assert axes[0].get_lines()[0].get_label() == "Ours (circular, graph-coordinated)"
    assert any(text.get_text() == "3" for text in axes[1].texts)
    assert any(text.get_text() == "1" for text in axes[1].texts)
    assert all("fit" not in label for label in axes[1].get_legend_handles_labels()[1])
