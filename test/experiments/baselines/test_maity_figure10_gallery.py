import math
from pathlib import Path

import pytest

from experiments.baselines import build_maity_figure10_gallery as gallery
from experiments.baselines.project_benchmarks import DOMAIN_SIZE


def test_source_case_matches_clean_figure_derived_parameters_and_scaling():
    case = gallery.source_case()
    parameters = case.parameters

    assert parameters["center"] == [0.5 * DOMAIN_SIZE, 0.55 * DOMAIN_SIZE]
    assert parameters["major_axis"] / DOMAIN_SIZE == pytest.approx(0.35)
    assert parameters["minor_axis"] / DOMAIN_SIZE == pytest.approx(0.15)
    assert parameters["aspect_ratio"] == pytest.approx(7.0 / 3.0)
    assert parameters["theta"] == pytest.approx(math.radians(40.0))
    assert parameters["benchmark_id"] == gallery.BENCHMARK_ID


def test_equation_case_uses_published_semiaxes_with_inferred_pose():
    case = gallery.source_case("equation")
    parameters = case.parameters

    assert parameters["center"] == [0.5 * DOMAIN_SIZE, 0.55 * DOMAIN_SIZE]
    assert parameters["major_axis"] / DOMAIN_SIZE == pytest.approx(math.sqrt(0.12))
    assert parameters["minor_axis"] / DOMAIN_SIZE == pytest.approx(math.sqrt(0.02))
    assert parameters["theta"] == pytest.approx(math.radians(40.0))
    assert parameters["benchmark_id"] == gallery.EQUATION_BENCHMARK_ID


def test_source_case_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="unknown Maity ellipse geometry"):
        gallery.source_case("unknown")


def test_text_equation_and_digitized_fit_remain_explicit_provenance():
    assert gallery.TEXT_A_SQUARED == pytest.approx(0.12)
    assert gallery.TEXT_B_SQUARED == pytest.approx(0.02)
    assert gallery.DIGITIZED_FIT["center"] == pytest.approx((5.16, 5.49))
    assert gallery.DIGITIZED_FIT["major_axis"] == pytest.approx(3.54)
    assert gallery.DIGITIZED_FIT["minor_axis"] == pytest.approx(1.45)
    assert gallery.DIGITIZED_FIT["theta_degrees"] == pytest.approx(39.99)


def test_gallery_roster_uses_approved_paper_colors_and_unchanged_variants():
    methods = {method["id"]: method for method in gallery.METHODS}

    assert tuple(methods) == (
        "plvira",
        "pcic_center",
        "quasi",
        "ours_per_cell",
        "ours_graph",
        "ours_c0",
    )
    assert methods["plvira"]["color"] == gallery.PAPER_HIGH_ORDER_COLORS["plvira"]
    assert methods["pcic_center"]["color"] == gallery.PAPER_HIGH_ORDER_COLORS["pcic_center"]
    assert methods["quasi"]["color"] == gallery.PAPER_HIGH_ORDER_COLORS["quasi"]
    assert methods["ours_per_cell"]["facet_algo"] == "safe_circle"
    assert methods["ours_graph"]["facet_algo"] == "circular"
    assert methods["ours_c0"]["facet_algo"] == "circular"
    assert methods["ours_c0"]["do_c0"] is True


def test_expected_pdf_deliverable_names_are_unique():
    names = [
        "maity_figure10_all_methods_gallery.pdf",
        "maity_figure10_all_methods_gallery_no_endpoints_2col.pdf",
    ]
    names.extend(
        f"maity_fig10_{method['id']}_N{resolution}.pdf"
        for method in gallery.METHODS
        for resolution in gallery.SOURCE_RESOLUTIONS
    )

    assert len(names) == 14
    assert len(set(names)) == 14
    assert all(Path(name).suffix == ".pdf" for name in names)
