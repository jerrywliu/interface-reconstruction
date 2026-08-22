import numpy as np

from experiments.static.generate_section6_maintext_figures import (
    RESOLUTION_QUANT_SPECS,
    _generate_resolution_quantitative_panel,
)


def test_circle_maintext_uses_facet_gap():
    assert RESOLUTION_QUANT_SPECS["circles"]["metrics"] == (
        "hausdorff",
        "facet_gap",
    )


def test_ellipse_resolution_panel_matches_vertical_limits_and_marks_order(
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
    assert np.allclose(axes[0].get_ylim(), axes[1].get_ylim())
    assert all(axis.get_xscale() == "log" for axis in axes)
    assert np.allclose(axes[0].get_xticks(), [32.0, 64.0, 128.0])
    assert any(text.get_text() == "3" for text in axes[1].texts)
    assert any(text.get_text() == "1" for text in axes[1].texts)
    assert all("fit" not in label for label in axes[1].get_legend_handles_labels()[1])
