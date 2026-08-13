#!/usr/bin/env python3
"""Generate non-destructive vector prototypes for the August paper revision."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

from experiments.static import generate_section6_maintext_figures as maintext


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT.parent / "interface-reconstruction-paper"
SEALED_ROOT = (
    REPO_ROOT
    / "results/submission/final_figures_87c40309d16c_20260803_final"
)
DEFAULT_OUTPUT = REPO_ROOT / "results/submission/revision_prototypes_20260812"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

MESH_EDGE = "#9ca3af"
MIXED_FILL = "#dbeafe"
ACTIVE_FILL = "#dcece5"
ACTIVE_EDGE = "#2d7d64"
GRAPH_EDGE = "#374151"
GRAPH_NODE = "#d1843d"
TRUE_INTERFACE = "#111827"


def _save_vector_figure(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_perturbed_grid(ax, *, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.zeros((6, 6, 2), dtype=float)
    for i in range(6):
        for j in range(6):
            points[i, j] = [i, j]
            if 0 < i < 5 and 0 < j < 5:
                points[i, j] += rng.uniform(-0.13, 0.13, size=2)
    for i in range(5):
        for j in range(5):
            poly = np.asarray(
                [
                    points[i, j],
                    points[i + 1, j],
                    points[i + 1, j + 1],
                    points[i, j + 1],
                ]
            )
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor="white",
                    edgecolor=MESH_EDGE,
                    linewidth=0.55,
                    zorder=1,
                )
            )
    return points


def _cell_polygon(points: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.asarray(
        [
            points[i, j],
            points[i + 1, j],
            points[i + 1, j + 1],
            points[i, j + 1],
        ]
    )


def _finish_schematic_axis(ax, title: str, *, title_size: float = 10.5) -> None:
    ax.set_xlim(-0.15, 5.15)
    ax.set_ylim(-0.15, 5.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=8)


def generate_stencil_comparison(output_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15))
    active_triplet = [(1, 1), (2, 2), (3, 3)]
    for panel, (ax, title) in enumerate(
        zip(
            axes,
            [
                "(a) LVIRA\n" + r"$3\!\times\!3$ least-squares stencil",
                "(b) Ours (linear)\noriented three-cell support",
                "(c) Ours (circular)\noriented three-cell support",
            ],
        )
    ):
        points = _draw_perturbed_grid(ax, seed=4)
        if panel == 0:
            for i in range(1, 4):
                for j in range(1, 4):
                    ax.add_patch(
                        Polygon(
                            _cell_polygon(points, i, j),
                            closed=True,
                            facecolor=MIXED_FILL,
                            edgecolor="#4b6b8a",
                            linewidth=0.75,
                            zorder=2,
                        )
                    )
            ax.add_patch(
                Polygon(
                    _cell_polygon(points, 2, 2),
                    closed=True,
                    facecolor="#b7d6f4",
                    edgecolor="#1f4f7a",
                    linewidth=1.55,
                    zorder=3,
                )
            )
            ax.plot([0.45, 4.55], [0.65, 4.35], color=TRUE_INTERFACE, lw=1.5, zorder=4)
            ax.text(2.5, 2.5, r"$P_M$", ha="center", va="center", fontsize=10, zorder=5)
        else:
            for index, (i, j) in enumerate(active_triplet):
                ax.add_patch(
                    Polygon(
                        _cell_polygon(points, i, j),
                        closed=True,
                        facecolor=ACTIVE_FILL,
                        edgecolor=ACTIVE_EDGE,
                        linewidth=1.35,
                        zorder=2,
                    )
                )
                center = _cell_polygon(points, i, j).mean(axis=0)
                ax.text(
                    center[0],
                    center[1],
                    [r"$P_L$", r"$P_M$", r"$P_R$"][index],
                    ha="center",
                    va="center",
                    fontsize=10,
                    zorder=5,
                )
            centers = np.asarray(
                [_cell_polygon(points, i, j).mean(axis=0) for i, j in active_triplet]
            )
            ax.plot(
                centers[:, 0],
                centers[:, 1],
                color=ACTIVE_EDGE,
                linestyle=(0, (2.0, 1.8)),
                linewidth=1.0,
                zorder=3,
            )
            if panel == 1:
                ax.plot([0.45, 4.55], [0.65, 4.35], color=TRUE_INTERFACE, lw=1.6, zorder=4)
            else:
                theta = np.linspace(-0.74, 0.74, 160)
                x = 2.5 + 3.25 * np.sin(theta)
                y = -0.22 + 3.25 * np.cos(theta)
                ax.plot(x, y, color=TRUE_INTERFACE, lw=1.6, zorder=4)
        _finish_schematic_axis(ax, title, title_size=9.5)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.88, wspace=0.08)
    base = output_dir / "conceptual_stencil_comparison"
    _save_vector_figure(fig, base)
    return {
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "kind": "conceptual schematic",
    }


def generate_candidate_graph_overlay(output_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.75))
    node_cells = [(0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2)]
    candidate_edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 2)]
    for panel, ax in enumerate(axes):
        points = _draw_perturbed_grid(ax, seed=11)
        centers = []
        for node_index, (i, j) in enumerate(node_cells):
            poly = _cell_polygon(points, i, j)
            center = poly.mean(axis=0)
            centers.append(center)
            fill = "#b9d8c5" if node_index in (0, 3, 4, 5) or panel == 1 else "#f2c38f"
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor=fill,
                    edgecolor="#4b5563",
                    linewidth=0.8,
                    zorder=2,
                )
            )
        centers = np.asarray(centers)
        if panel == 0:
            edges = candidate_edges
            title = "(a) Candidate graph on the mesh"
        else:
            edges = [(0, 1), (1, 4), (4, 5), (5, 2), (2, 3)]
            title = "(b) Propagated interface walk"
        for edge_index, (source, target) in enumerate(edges):
            a, b = centers[source], centers[target]
            if panel == 1:
                patch = FancyArrowPatch(
                    a,
                    b,
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color=ACTIVE_EDGE,
                    linewidth=1.15,
                    shrinkA=7,
                    shrinkB=7,
                    zorder=4,
                )
                ax.add_patch(patch)
            else:
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=GRAPH_EDGE,
                    linestyle=(0, (3, 2)),
                    linewidth=1.05,
                    zorder=4,
                )
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=38,
            facecolors=[
                "#2d7d64" if index in (0, 3, 4, 5) or panel == 1 else GRAPH_NODE
                for index in range(len(centers))
            ],
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        curve_x = np.linspace(0.05, 4.15, 160)
        curve_y = 1.52 + 0.56 * np.exp(-((curve_x - 2.0) / 0.78) ** 2)
        ax.plot(curve_x, curve_y, color=TRUE_INTERFACE, linewidth=1.4, zorder=3)
        _finish_schematic_axis(ax, title)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.88, wspace=0.08)
    base = output_dir / "candidate_graph_mesh_overlay"
    _save_vector_figure(fig, base)
    return {
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "kind": "conceptual mesh/graph overlay",
    }


def _circle_inset_bounds(case_index: int, half_span: float = 4.5) -> tuple[float, ...]:
    params = maintext._circle_case_params(case_index)
    center = params["center"] + np.asarray([params["radius"], 0.0])
    return (
        center[0] - half_span,
        center[0] + half_span,
        center[1] - half_span,
        center[1] + half_span,
    )


def _ellipse_inset_bounds(case_index: int, half_span: float = 5.0) -> tuple[float, ...]:
    params = maintext._ellipse_case_params(case_index)
    direction = np.asarray([math.cos(params["theta"]), math.sin(params["theta"])])
    center = params["center"] + params["major_axis"] * direction
    return (
        center[0] - half_span,
        center[0] + half_span,
        center[1] - half_span,
        center[1] + half_span,
    )


def generate_enlarged_spyglasses(output_dir: Path) -> dict:
    plots_root = SEALED_ROOT / "provenance/release_input_snapshot/plots"
    maintext.PLOTS_ROOT = plots_root
    outputs = {}
    for exp_name, source_spec in maintext.REPRESENTATIVE_CASES.items():
        spec = dict(source_spec)
        spec["inset_size"] = 0.42
        spec["inset_gap"] = 0.05
        spec["inset_bottom"] = 0.02
        if exp_name == "circles":
            spec["inset"] = {"kind": "revision_prototype"}
            spec["inset_bounds"] = _circle_inset_bounds(spec["case_index"])
        elif exp_name == "ellipses":
            spec["inset"] = {"kind": "revision_prototype"}
            spec["inset_bounds"] = _ellipse_inset_bounds(spec["case_index"])
        spec = maintext._endpoint_visibility_spec(spec, show_main_endpoints=False)
        png_path = output_dir / "enlarged_spyglasses" / f"{exp_name}_maintext_enlarged_spyglass.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        maintext._generate_representative_figure(exp_name, spec, png_path)
        outputs[exp_name] = {
            "pdf": str(png_path.with_suffix(".pdf")),
            "png": str(png_path),
            "case_index": spec["case_index"],
            "sealed_plots_root": str(plots_root),
        }
    return outputs


def _pdf_size(path: Path) -> tuple[float, float]:
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Page size:"):
            fields = line.split()
            return float(fields[2]), float(fields[4])
    raise RuntimeError(f"Could not read PDF page size for {path}")


def _trim_for_box(path: Path, box: tuple[float, float, float, float]) -> str:
    """Convert a top-left fractional crop box to graphicx trim dimensions."""
    width, height = _pdf_size(path)
    x0, y0, x1, y1 = box
    left = x0 * width
    right = (1.0 - x1) * width
    top = y0 * height
    bottom = (1.0 - y1) * height
    return f"{left:.3f}bp {bottom:.3f}bp {right:.3f}bp {top:.3f}bp"


def _compile_standalone(tex_path: Path) -> Path:
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(tex_path.parent),
            str(tex_path),
        ],
        check=True,
        cwd=tex_path.parent,
        stdout=subprocess.DEVNULL,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-r", "180", str(pdf_path), str(pdf_path.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for suffix in (".aux", ".log"):
        tex_path.with_suffix(suffix).unlink(missing_ok=True)
    return pdf_path


def generate_resolution_layouts(output_dir: Path) -> dict:
    source_root = SEALED_ROOT / "candidates/figure_root/resolution"
    names = ["lines", "squares", "circles", "ellipses", "zalesak"]
    display = {
        "lines": "Lines",
        "squares": "Squares",
        "circles": "Circles",
        "ellipses": "Ellipses",
        "zalesak": "Zalesak",
    }
    source_paths = {
        name: source_root
        / name
        / "summary_plots"
        / f"{name}_resolution_cartesian_vs_perturbed_clean.pdf"
        for name in names
    }
    row_boxes = [(.51, .10, .995, .345), (.51, .41, .995, .665), (.51, .72, .995, .995)]
    spyglass_row_boxes = [(.53, .08, .88, .32), (.53, .41, .88, .65), (.53, .72, .88, .98)]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for columns, selected_rows in ((2, [1, 2]), (3, [0, 1, 2])):
        tex_path = output_dir / f"benchmark_resolution_5x{columns}.tex"
        lines = [
            r"\documentclass[border=2pt]{standalone}",
            r"\usepackage{graphicx}",
            r"\usepackage{array}",
            r"\begin{document}",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{0.92}",
            r"\begin{tabular}{>{\raggedleft\arraybackslash}p{0.70in}" + "c" * columns + "}",
            " & " + " & ".join(rf"\textbf{{$N={n}$}}" for n in ([32, 64] if columns == 2 else [16, 32, 64])) + r"\\",
        ]
        for name in names:
            source = source_paths[name]
            boxes = spyglass_row_boxes if name in {"squares", "zalesak"} else row_boxes
            cells = []
            for row in selected_rows:
                trim = _trim_for_box(source, boxes[row])
                width = "2.52in" if columns == 2 else "1.74in"
                cells.append(
                    rf"\includegraphics[width={width},trim={{{trim}}},clip]{{{source}}}"
                )
            lines.append(rf"\textbf{{{display[name]}}} & " + " & ".join(cells) + r"\\[-1pt]")
        lines.extend([r"\end{tabular}", r"\end{document}"])
        tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pdf_path = _compile_standalone(tex_path)
        results[f"5x{columns}"] = {
            "pdf": str(pdf_path),
            "png": str(pdf_path.with_suffix(".png")),
            "sources": [str(source_paths[name]) for name in names],
        }
    return results


def generate_compact_c0_layout(output_dir: Path) -> dict:
    c0_root = SEALED_ROOT / "candidates/c0_root"
    ellipse_metrics = c0_root / "summary_plots/ellipses_appendix_c0_2x2.pdf"
    zalesak_metrics = c0_root / "summary_plots/zalesak_appendix_c0_2x2.pdf"
    ellipse_rep = c0_root / "representative_cases/ellipses_appendix_c0_representative_clean.pdf"
    zalesak_rep = c0_root / "representative_cases/zalesak_appendix_c0_representative_clean.pdf"
    tex_path = output_dir / "compact_c0_one_page.tex"
    output_dir.mkdir(parents=True, exist_ok=True)
    ellipse_trim = _trim_for_box(ellipse_metrics, (.50, .0, 1.0, 1.0))
    zalesak_trim = _trim_for_box(zalesak_metrics, (.50, .0, 1.0, 1.0))
    tex = rf"""\documentclass[border=3pt]{{standalone}}
\usepackage{{graphicx}}
\usepackage{{array}}
\begin{{document}}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.96}}
\begin{{tabular}}{{>{{\raggedleft\arraybackslash}}p{{0.62in}}cc}}
 & \textbf{{Resolution metrics}} & \textbf{{Representative reconstruction}}\\
\textbf{{Ellipse}} &
\includegraphics[width=2.60in,trim={{{ellipse_trim}}},clip]{{{ellipse_metrics}}} &
\includegraphics[width=3.45in]{{{ellipse_rep}}}\\[-2pt]
\textbf{{Zalesak}} &
\includegraphics[width=2.60in,trim={{{zalesak_trim}}},clip]{{{zalesak_metrics}}} &
\includegraphics[width=3.45in]{{{zalesak_rep}}}\\
\end{{tabular}}
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")
    pdf_path = _compile_standalone(tex_path)
    return {
        "pdf": str(pdf_path),
        "png": str(pdf_path.with_suffix(".png")),
        "sources": [
            str(ellipse_metrics),
            str(ellipse_rep),
            str(zalesak_metrics),
            str(zalesak_rep),
        ],
        "note": "Uses the resolution columns from the aggregate metric panels.",
    }


def _sha256(path: Path) -> str:
    result = subprocess.run(
        ["shasum", "-a", "256", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()[0]


def write_manifest(output_dir: Path, outputs: dict) -> None:
    generated = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.suffix in {".pdf", ".png", ".svg", ".tex"}:
            generated.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": 1,
        "sealed_source": str(SEALED_ROOT.relative_to(REPO_ROOT)),
        "paper_assets_inspected": str(PAPER_ROOT / "figs/cameraready"),
        "outputs": outputs,
        "generated_files": generated,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "stencil_comparison": generate_stencil_comparison(output_dir),
        "candidate_graph_overlay": generate_candidate_graph_overlay(output_dir),
        "enlarged_spyglasses": generate_enlarged_spyglasses(output_dir),
        "resolution_layouts": generate_resolution_layouts(output_dir),
        "compact_c0_layout": generate_compact_c0_layout(output_dir),
    }
    write_manifest(output_dir, outputs)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
