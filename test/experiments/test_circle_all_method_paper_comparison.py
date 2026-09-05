import json

import pytest

from experiments.baselines import plot_circle_all_method_paper_comparison as circle
from experiments.baselines import plot_ellipse_all_method_paper_comparison as ellipse


def test_circle_paper_figure_reuses_exact_ellipse_style_contract():
    assert circle.plot_paper_figure is ellipse.plot_paper_figure
    assert circle.CIRCLE_TRIANGLE_ANCHORS == {
        "native_symmetric_hausdorff": (0.52, 0.36),
        "geometric_curvature_mean_absolute_error": (0.52, 0.42),
    }
    assert ellipse.ERROR_PANELS == (
        (
            "native_symmetric_hausdorff",
            "(a) Hausdorff error",
            "Hausdorff error",
            3.0,
            (0.10, 0.32),
        ),
        (
            "geometric_curvature_mean_absolute_error",
            "(b) Curvature MAE",
            "Curvature MAE",
            1.0,
            (0.45, 0.66),
        ),
        (
            "facet_gap",
            "(c) Facet-gap error",
            "Facet-gap error",
            3.0,
            (0.70, 0.54),
        ),
    )
    assert tuple(method["id"] for method in ellipse.PAPER_METHODS) == (
        "plvira",
        "pcic_center",
        "quasi",
        "ours_per_cell",
        "ours_graph",
        "ours_c0",
    )


def test_circle_manifest_rejects_other_benchmarks(tmp_path, monkeypatch):
    manifest = {"benchmark": "ellipses"}
    monkeypatch.setattr(circle, "validate_frozen_manifest", lambda _: manifest)

    with pytest.raises(ValueError, match="expected 'circles'"):
        circle.validate_circle_manifest(tmp_path)

    manifest["benchmark"] = "circles"
    assert circle.validate_circle_manifest(tmp_path) == manifest


def test_paper_manifest_records_circle_identity(tmp_path, monkeypatch):
    captured = {}

    def fake_write(path, **kwargs):
        captured.update(kwargs)
        path.write_text(json.dumps(kwargs, default=str), encoding="utf-8")

    monkeypatch.setattr(circle, "write_paper_manifest", fake_write)
    path = tmp_path / "manifest.json"
    fake_write(
        path,
        artifact_kind="paper-ready Cartesian circle comparison",
        benchmark="circles",
    )

    assert path.exists()
    assert captured["benchmark"] == "circles"
    assert "circle" in captured["artifact_kind"]
