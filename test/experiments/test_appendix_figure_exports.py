from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

from experiments.static import figure_generation_provenance as provenance
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


def test_resolution_runner_generates_paired_vector_artifacts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        resolution_visuals,
        "_generate_figure",
        lambda exp_spec, out_path, show_main_endpoints: calls.append(
            (Path(out_path), show_main_endpoints)
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
