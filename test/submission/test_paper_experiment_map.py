from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = REPO_ROOT / "docs" / "PAPER_EXPERIMENT_MAP.md"
PROVENANCE_PATH = REPO_ROOT / "submission" / "figure_provenance.csv"

EXPECTED_TIKZ = {
    "fig:regular_cases": (
        "new_sections/topology_identification.tex",
        "figs/tikz/topology_regular_cases.tex",
        "figs/tikz/topology_styles.tex",
    ),
    "fig:merging_ambiguous_cases": (
        "new_sections/topology_identification.tex",
        "figs/tikz/topology_merging_cases.tex",
        "figs/tikz/topology_styles.tex",
    ),
    "fig:orientation_dependencies": (
        "new_sections/topology_identification.tex",
        "figs/tikz/topology_orientation_dependencies.tex",
        "figs/tikz/topology_styles.tex",
    ),
    "fig:linear_facet_fitting": (
        "new_sections/appendix/algorithms/linear_facets.tex",
        "figs/tikz/appendix_linear_facet_fitting.tex",
        "figs/tikz/algorithm_styles.tex",
    ),
    "fig:circular_facet_fitting": (
        "new_sections/appendix/algorithms/circular_facets.tex",
        "figs/tikz/appendix_circular_facet_fitting.tex",
        "figs/tikz/algorithm_styles.tex",
    ),
    "fig:circle_quad_intersect": (
        "new_sections/appendix/algorithms/circular_facets.tex",
        "figs/tikz/appendix_circle_intersect_area.tex",
        "figs/tikz/algorithm_styles.tex",
    ),
    "fig:corner_facet_fitting": (
        "new_sections/appendix/algorithms/corner_facets.tex",
        "figs/tikz/appendix_corner_facet_fitting.tex",
        "figs/tikz/algorithm_styles.tex",
    ),
}


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    next_heading = text.find("\n## ", start + len(heading))
    return text[start:] if next_heading < 0 else text[start:next_heading]


def _required_external_root(variable: str, marker: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to run the active-submission cross-check")
    root = Path(value).resolve()
    assert (root / marker).is_file(), root
    return root


def _strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _active_tex_graph(paper_root: Path) -> tuple[set[Path], list[str]]:
    prefix = "interface-reconstruction-paper/"
    pending = [paper_root / "interface-reconstruction.tex"]
    visited: set[Path] = set()
    graphics: list[str] = []

    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        assert path.is_file(), path
        visited.add(path)
        text = _strip_tex_comments(path.read_text(encoding="utf-8"))
        graphics.extend(
            Path(match).name
            for match in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
        )
        for included in re.findall(r"\\input\{([^}]+)\}", text):
            relative = (
                included[len(prefix) :] if included.startswith(prefix) else included
            )
            target = paper_root / relative
            if target.suffix == "":
                target = target.with_suffix(".tex")
            pending.append(target)

    return visited, graphics


def _resolved_config() -> dict:
    final_root = _required_external_root(
        "FINAL_ROOT", "submission_config.resolved.json"
    )
    return json.loads((final_root / "submission_config.resolved.json").read_text())


def test_all_26_pdf_includes_map_exactly_once():
    text = MAP_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    with PROVENANCE_PATH.open(newline="", encoding="utf-8") as stream:
        paper_files = [row["paper_file"] for row in csv.DictReader(stream)]

    assert len(paper_files) == 26
    assert len(set(paper_files)) == 26
    for paper_file in paper_files:
        assert sum(f"`{paper_file}`" in line for line in lines) == 1, paper_file

    if os.environ.get("PAPER_ROOT"):
        paper_root = _required_external_root(
            "PAPER_ROOT", "interface-reconstruction.tex"
        )
        _, graphics = _active_tex_graph(paper_root)
        assert len(graphics) == 26
        assert set(graphics) == set(paper_files)


def test_all_seven_active_tikz_figures_and_dependencies_are_mapped():
    text = MAP_PATH.read_text(encoding="utf-8")
    section = _section(text, "## Author-Drawn TikZ Figure Map")
    assert section.count("| **Open:**") == 7
    assert "`graphicx`, `tikz`" in section
    assert "`arrows.meta`, `calc`, and `positioning`" in section

    for label, dependencies in EXPECTED_TIKZ.items():
        assert section.count(f"`{label}`") == 1
        for dependency in dependencies:
            assert section.count(f"`{dependency}`") >= 1

    if os.environ.get("PAPER_ROOT"):
        paper_root = _required_external_root(
            "PAPER_ROOT", "interface-reconstruction.tex"
        )
        tex_graph, _ = _active_tex_graph(paper_root)
        relative_graph = {path.relative_to(paper_root).as_posix() for path in tex_graph}
        expected_sources = {values[1] for values in EXPECTED_TIKZ.values()}
        active_tikz_sources = {
            path
            for path in relative_graph
            if path.startswith("figs/tikz/") and not path.endswith("_styles.tex")
        }
        assert active_tikz_sources == expected_sources
        for style in {values[2] for values in EXPECTED_TIKZ.values()}:
            assert style in relative_graph

        entrypoint = (paper_root / "interface-reconstruction.tex").read_text()
        assert r"\usepackage{graphicx}" in entrypoint
        assert r"\usepackage{tikz}" in entrypoint
        assert r"\usetikzlibrary{arrows.meta,calc,positioning}" in entrypoint


def test_table_map_uses_only_sealed_configuration_as_final_authority():
    text = MAP_PATH.read_text(encoding="utf-8")
    section = _section(text, "## Manuscript Table Map")
    assert "$FINAL_ROOT/submission_config.resolved.json" in section
    assert "submission/submission_config.json" not in section
    assert "main/geoms/linear_facet.py" in section
    assert "step < 1e-6" in section


def test_methods_table_gate():
    paper_root = _required_external_root("PAPER_ROOT", "interface-reconstruction.tex")
    config = _resolved_config()
    manuscript = (paper_root / "new_sections/problem_setup.tex").read_text()

    assert r"\label{tab:methods_compare}" in manuscript
    for method in ("Youngs", "ELVIRA", "LVIRA", "Garimella", "Evrard", "Maity", "Ours"):
        assert method in manuscript
    production = config["production_method"]
    assert production["unresolved_orientation_fallback"] == "LVIRA"
    assert production["corner_behavior_profile"] == "pre_f8_corner"
    assert production["rescue_profile"] == "exact_linear_support_only"


def test_numerical_parameters_table_gate():
    paper_root = _required_external_root("PAPER_ROOT", "interface-reconstruction.tex")
    config = _resolved_config()
    manuscript = (paper_root / "new_sections/appendix/algorithms.tex").read_text()

    expected_rows = (
        r"General optimization tolerance & $10^{-10}$",
        r"Straight-support residual & $10^{-10}$",
        r"Linear target-cell residual & $10^{-6}$",
        r"Circular residual & $10^{-5}$",
        r"Straight-corner and exact-support residual & $10^{-4}$",
        r"Near-parallel tangent threshold & $|\sin\theta|\leq 10^{-2}$",
        r"Curved-corner threshold & $10^{-2}$",
        r"LVIRA angular step tolerance & $10^{-6}$",
    )
    for row in expected_rows:
        assert row in manuscript

    numerics = config["numerics"]
    assert numerics["optimization_threshold"] == 1e-10
    assert numerics["linearity_threshold"] == 1e-6
    assert numerics["arc_fit_residual_threshold"] == 1e-5
    assert numerics["linear_corner_area_threshold"] == 1e-4
    assert numerics["corner_sharpness_threshold"] == 1e-2
    assert numerics["curved_corner_area_threshold"] == 1e-2

    lvira_source = (REPO_ROOT / "main/geoms/linear_facet.py").read_text()
    assert re.search(r"if\s+step\s*<\s*1e-6\s*:", lvira_source)


def test_benchmark_table_gate():
    paper_root = _required_external_root("PAPER_ROOT", "interface-reconstruction.tex")
    config = _resolved_config()
    manuscript = (
        paper_root / "new_sections/appendix/static_benchmarks/overview.tex"
    ).read_text()

    expected_fragments = (
        r"25$ equally spaced orientations $\theta\in[0,2\pi)$",
        r"Square & Side length $s\in[10,30]$",
        r"Circle & $R=10$, $\kappa=0.1$",
        r"Ellipse & $a=30$, $b\in[10,20]$",
        r"Zalesak & $R=15$, slot width $W=5$, slot height $y_{\mathrm{top}}=10$",
    )
    for fragment in expected_fragments:
        assert fragment in manuscript

    grid = config["benchmark_grid"]
    assert grid["seed"] == 0
    assert grid["trials_per_setting"] == 25
    assert config["benchmarks"]["circles"]["radius"] == 10.0
    assert config["benchmarks"]["zalesak"]["radius"] == 15.0
    assert config["benchmarks"]["zalesak"]["slot_width"] == 5.0
    assert config["benchmarks"]["zalesak"]["slot_top_relative_to_center"] == 10.0

    expected_seeds = {
        "lines.py": 42,
        "squares.py": 42,
        "circles.py": 41,
        "ellipses.py": 42,
        "zalesak.py": 43,
    }
    for filename, seed in expected_seeds.items():
        source = (REPO_ROOT / "experiments/static" / filename).read_text()
        assert re.search(rf"^RANDOM_SEED\s*=\s*{seed}$", source, re.MULTILINE)
