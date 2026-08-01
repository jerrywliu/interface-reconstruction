from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from experiments.static import figure_generation_provenance as provenance
from experiments.static import generate_plic_baseline_stencil_figure as plic_figure
from experiments.static import generate_section6_maintext_figures as maintext_figs
from experiments.static import run_appendix_c0_study as c0_study
from experiments.static import run_appendix_resolution_visuals as resolution_visuals


def test_frozen_profile_cli_args_are_explicit():
    profile = provenance.frozen_reconstruction_profile()

    assert profile == {
        "plic_fallback": "LVIRA",
        "corner_behavior_profile": "pre_f8_corner",
        "rescue_profile": "exact_linear_support_only",
    }
    assert provenance.reconstruction_cli_args("ellipses", profile) == [
        "--plic_fallback",
        "LVIRA",
        "--corner_behavior_profile",
        "pre_f8_corner",
    ]
    assert provenance.reconstruction_cli_args("zalesak", profile)[-2:] == [
        "--rescue_profile",
        "exact_linear_support_only",
    ]


def test_generation_provenance_filters_generated_roots(monkeypatch):
    responses = {
        ("status", "--short"): (
            " M experiments/static/example.py\n"
            "?? results/static/generated/manifest.json\n"
        ),
        ("rev-parse", "HEAD"): "abc123",
        ("branch", "--show-current"): "codex/test",
    }
    monkeypatch.setattr(
        provenance,
        "_git_output",
        lambda args: responses[tuple(args)],
    )

    record = provenance.generation_provenance(
        profile=provenance.frozen_reconstruction_profile(),
        profile_application="test",
    )

    assert record["source_commit"] == "abc123"
    assert record["source_dirty"] is True
    assert record["source_status"] == [" M experiments/static/example.py"]
    assert record["reconstruction_profile"]["plic_fallback"] == "LVIRA"


def test_plic_stencil_figure_renders_without_external_latex(tmp_path, monkeypatch):
    monkeypatch.setattr(
        plic_figure,
        "generation_provenance",
        lambda **_kwargs: {
            "source_commit": "a" * 40,
            "source_dirty": False,
            "source_status": [],
        },
    )
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)

    metadata = plic_figure.build_figure(
        tmp_path / "plic_stencil",
        case_index=4,
        cell_x=14,
        cell_y=13,
        resolution=0.32,
        wiggle=0.3,
        seed=0,
    )

    assert metadata["center_cell_hausdorff_over_h"]["LVIRA"] < 1.0e-6
    assert (tmp_path / "plic_stencil.pdf").is_file()
    assert (tmp_path / "plic_stencil.svg").is_file()
    assert (tmp_path / "plic_stencil.png").is_file()


def test_resolution_runner_generates_paired_vector_artifacts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        resolution_visuals,
        "_generate_figure",
        lambda exp_spec, out_path, main_endpoint_visibility: calls.append(
            (Path(out_path), main_endpoint_visibility)
        ),
    )

    outputs = resolution_visuals._generate_endpoint_variant_figures(
        {"name": "zalesak"},
        tmp_path,
        "paired",
    )

    assert [path.name for path, _ in calls] == [
        "zalesak_resolution_cartesian_vs_perturbed_with_endpoints.png",
        "zalesak_resolution_cartesian_vs_perturbed_clean.png",
    ]
    assert [visible for _, visible in calls] == [True, False]
    assert outputs["with_endpoints"]["pdf"].endswith(
        "zalesak_resolution_cartesian_vs_perturbed_with_endpoints.pdf"
    )
    assert outputs["clean"]["pdf"].endswith(
        "zalesak_resolution_cartesian_vs_perturbed_clean.pdf"
    )
    assert outputs["clean"]["png_review_300dpi"].endswith("_clean.png")


def test_resolution_runner_generates_allowlisted_hybrid_artifact(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        resolution_visuals,
        "_generate_figure",
        lambda exp_spec, out_path, main_endpoint_visibility: calls.append(
            (Path(out_path), main_endpoint_visibility)
        ),
    )

    exp_spec = {
        "name": "ellipses",
        "resolutions": [0.16, 0.32, 0.64],
    }
    outputs = resolution_visuals._generate_endpoint_variant_figures(
        exp_spec,
        tmp_path,
        resolution_visuals.HYBRID_ENDPOINT_MODE,
    )

    assert [path.name for path, _ in calls] == [
        "ellipses_resolution_cartesian_vs_perturbed_with_endpoints.png",
        "ellipses_resolution_cartesian_vs_perturbed_clean.png",
        "ellipses_resolution_cartesian_vs_perturbed_hybrid_endpoints_n16_n32.png",
    ]
    assert calls[-1][1] == {16: True, 32: True, 64: False}
    assert set(outputs) == {
        "with_endpoints",
        "clean",
        "hybrid_endpoints_n16_n32",
    }
    assert resolution_visuals._endpoint_visibility_manifest(
        exp_spec, resolution_visuals.HYBRID_ENDPOINT_MODE
    )["hybrid_endpoints_n16_n32"] == {
        "main_endpoint_visibility_by_resolution": {
            "16": True,
            "32": True,
            "64": False,
        },
        "show_inset_endpoints": True,
    }


def test_resolution_runner_rejects_hybrid_for_unapproved_experiment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        resolution_visuals,
        "_generate_figure",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="not allowlisted for squares"):
        resolution_visuals._generate_endpoint_variant_figures(
            {"name": "squares"},
            tmp_path,
            resolution_visuals.HYBRID_ENDPOINT_MODE,
        )


def test_resolution_endpoint_visibility_can_hide_only_the_finest_main_panels():
    visibility = {16: True, 32: True, 64: False}
    base_spec = {"inset": {"kind": "zalesak_corner"}}

    specs = [
        resolution_visuals._resolution_endpoint_visibility_spec(
            base_spec,
            resolution=resolution,
            main_endpoint_visibility=visibility,
        )
        for resolution in (0.16, 0.32, 0.64)
    ]

    assert [spec["show_main_endpoints"] for spec in specs] == [True, True, False]
    assert all(spec["show_inset_endpoints"] is True for spec in specs)
    assert "show_main_endpoints" not in base_spec


def test_resolution_endpoint_visibility_requires_an_explicit_mapping_entry():
    with pytest.raises(ValueError, match=r"N=64"):
        resolution_visuals._resolution_endpoint_visibility_spec(
            {"inset": None},
            resolution=0.64,
            main_endpoint_visibility={16: True, 32: True},
        )


def test_c0_runner_generates_paired_representatives(tmp_path, monkeypatch):
    monkeypatch.setattr(c0_study, "_load_rows", lambda _path: [{}])
    monkeypatch.setattr(
        c0_study.sweeps,
        "_build_metric_index",
        lambda _rows: {"zalesak": {"circular": {}}},
    )
    monkeypatch.setattr(
        maintext_figs,
        "_generate_quantitative_panel",
        lambda **_kwargs: None,
    )
    representative_calls = []
    monkeypatch.setattr(
        maintext_figs,
        "_generate_representative_figure",
        lambda **kwargs: representative_calls.append(kwargs),
    )

    outputs = c0_study._generate_plots(
        tmp_path / "metrics.csv",
        tmp_path,
        endpoint_variants="paired",
    )

    assert len(representative_calls) == 2
    assert [call["spec"]["show_main_endpoints"] for call in representative_calls] == [
        True,
        False,
    ]
    assert all(
        call["spec"]["show_inset_endpoints"] is True for call in representative_calls
    )
    assert outputs["representative"]["zalesak"]["with_endpoints"]["pdf"].endswith(
        "zalesak_appendix_c0_representative_with_endpoints.pdf"
    )
    assert outputs["representative"]["zalesak"]["clean"]["pdf"].endswith(
        "zalesak_appendix_c0_representative_clean.pdf"
    )


def test_c0_ellipse_spyglass_targets_accepted_endpoint_refits():
    ellipse_spec = next(
        spec for spec in c0_study.APPENDIX_EXPERIMENTS if spec["name"] == "ellipses"
    )["representative"]

    assert ellipse_spec["inset"] == {"kind": "ellipse_continuity"}
    assert maintext_figs._inset_bounds("ellipses", ellipse_spec) == (
        72.0,
        78.0,
        48.0,
        54.0,
    )


def _scatter_collections(axis):
    return [
        collection
        for collection in axis.collections
        if isinstance(collection, PathCollection)
    ]


def test_clean_panel_keeps_corner_diamond_and_spyglass_labels():
    mesh_segments = np.asarray([[[0.0, 0.0], [1.0, 0.0]]])
    true_segments = np.asarray([[[0.0, 0.4], [1.0, 0.4]]])
    recon_segments = np.asarray([[[0.0, 0.5], [1.0, 0.5]]])
    endpoint_points = np.asarray([[0.0, 0.5], [1.0, 0.5]])
    corner_tip_points = np.asarray([[0.5, 0.5]])
    corner_boundary_points = np.asarray([[0.25, 0.5], [0.75, 0.5]])

    figure, axis = plt.subplots()
    maintext_figs._plot_panel(
        axis,
        exp_name="squares",
        spec={
            "case_index": 0,
            "inset": {"kind": "square_corner"},
            "inset_bounds": (0.0, 1.0, 0.0, 1.0),
            "inset_axes": (0.55, 0.05, 0.4, 0.4),
            "true_fill_vertices": np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
            "show_main_endpoints": False,
            "show_inset_endpoints": True,
        },
        algo="linear+corner",
        mesh_segments=mesh_segments,
        true_segments=true_segments,
        recon_segments=recon_segments,
        endpoint_points=endpoint_points,
        corner_tip_points=corner_tip_points,
        corner_boundary_points=corner_boundary_points,
        title="clean",
        bounds=(0.0, 1.0, 0.0, 1.0),
    )

    assert len(_scatter_collections(axis)) == 1
    assert len(_scatter_collections(axis.child_axes[0])) == 3
    plt.close(figure)


def test_endpoint_panel_adds_open_circles_without_duplicate_diamond():
    figure, axis = plt.subplots()
    maintext_figs._plot_panel(
        axis,
        exp_name="squares",
        spec={
            "case_index": 0,
            "inset": None,
            "true_fill_vertices": np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
            "show_main_endpoints": True,
            "show_inset_endpoints": True,
        },
        algo="linear+corner",
        mesh_segments=np.asarray([[[0.0, 0.0], [1.0, 0.0]]]),
        true_segments=np.asarray([[[0.0, 0.4], [1.0, 0.4]]]),
        recon_segments=np.asarray([[[0.0, 0.5], [1.0, 0.5]]]),
        endpoint_points=np.asarray([[0.0, 0.5], [1.0, 0.5]]),
        corner_tip_points=np.asarray([[0.5, 0.5]]),
        corner_boundary_points=np.asarray([[0.25, 0.5], [0.75, 0.5]]),
        title="with endpoints",
        bounds=(0.0, 1.0, 0.0, 1.0),
    )

    scatters = _scatter_collections(axis)
    assert len(scatters) == 3
    assert np.allclose(scatters[0].get_facecolors()[0, :3], [1.0, 1.0, 1.0])
    assert len(scatters[-1].get_offsets()) == 1
    plt.close(figure)
