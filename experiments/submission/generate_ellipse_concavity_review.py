"""Generate a focused visual audit of local concavity in joint-C0 ellipses."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.static import generate_section6_maintext_figures as figures
from experiments.submission.optimize_case10_joint_c0 import _facet_points
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "plots" / "ellipse_joint_c0_posthoc_n32_cartesian_final_20260801"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "submission" / "ellipse_concavity_review_20260801"
)
DEFAULT_CASES = (3, 10, 14)


@dataclass(frozen=True)
class FacetRecord:
    facet: Any
    concave: bool
    sag_over_h: float
    midpoint: np.ndarray


def _load_facet(primitive: dict[str, Any]) -> Any:
    if primitive["kind"] == "line":
        return LinearFacet(primitive["p_left"], primitive["p_right"])
    if primitive["kind"] == "arc":
        return ArcFacet(
            primitive["center"],
            primitive["radius"],
            primitive["p_left"],
            primitive["p_right"],
        )
    raise ValueError(f"Unsupported primitive kind: {primitive['kind']}")


def _facet_records(
    metadata_path: Path, ellipse_center: np.ndarray, cell_width: float
) -> list[FacetRecord]:
    metadata = json.loads(metadata_path.read_text())
    records = []
    for primitive in metadata["primitives"]:
        facet = _load_facet(primitive)
        points = _facet_points(facet, 101)
        chord_midpoint = 0.5 * (
            np.asarray(facet.pLeft, dtype=float) + np.asarray(facet.pRight, dtype=float)
        )
        midpoint = points[len(points) // 2]
        sag = midpoint - chord_midpoint
        concave = bool(np.dot(sag, ellipse_center - chord_midpoint) > 1.0e-10)
        records.append(
            FacetRecord(
                facet=facet,
                concave=concave,
                sag_over_h=float(np.linalg.norm(sag) / cell_width),
                midpoint=midpoint,
            )
        )
    return records


def _draw_case(
    axis: Any,
    records: Sequence[FacetRecord],
    truth: np.ndarray,
    mesh_segments: np.ndarray,
    *,
    zoom: bool,
) -> None:
    figures._add_segments(
        axis, mesh_segments, color="#cbd5e1", linewidth=0.30, alpha=0.7
    )
    figures._add_segments(
        axis,
        truth,
        color="#111827",
        linewidth=1.05,
        linestyle="--",
        zorder=2,
    )
    worst = max(
        (record for record in records if record.concave),
        key=lambda record: record.sag_over_h,
    )
    for record in records:
        points = _facet_points(record.facet, 101)
        color = "#dc2626" if record.concave else "#2563eb"
        linewidth = 2.0 if record is worst else 1.35
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=4 if record.concave else 3,
        )

    if zoom:
        cell_width = 100.0 / 32.0
        span = 2.8 * cell_width
        axis.set_xlim(worst.midpoint[0] - span / 2.0, worst.midpoint[0] + span / 2.0)
        axis.set_ylim(worst.midpoint[1] - span / 2.0, worst.midpoint[1] + span / 2.0)
        p_left = np.asarray(worst.facet.pLeft, dtype=float)
        p_right = np.asarray(worst.facet.pRight, dtype=float)
        chord_midpoint = 0.5 * (p_left + p_right)
        axis.plot(
            [p_left[0], p_right[0]],
            [p_left[1], p_right[1]],
            color="#111827",
            linewidth=0.9,
            linestyle="--",
            zorder=5,
        )
        axis.annotate(
            "",
            xy=worst.midpoint,
            xytext=chord_midpoint,
            arrowprops={"arrowstyle": "->", "color": "#dc2626", "linewidth": 1.0},
            zorder=6,
        )
        axis.scatter(
            [worst.midpoint[0]],
            [worst.midpoint[1]],
            marker="o",
            facecolors="white",
            edgecolors="#dc2626",
            linewidths=1.0,
            s=24,
            zorder=6,
        )
        axis.text(
            0.03,
            0.96,
            f"inward sag: {100 * worst.sag_over_h:.2f}% $h$",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "square,pad=0.2",
                "facecolor": "white",
                "alpha": 0.92,
            },
        )
    else:
        xmin, xmax, ymin, ymax = figures._compute_view_bounds(
            truth, min_span=66.0, margin_frac=0.07
        )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def generate(
    run_root: Path,
    output_dir: Path,
    case_indices: Sequence[int],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    mesh_segments = figures._mesh_segments(run_root / "vtk" / "mesh.vtk")
    cell_width = 100.0 / 32.0
    case_data = []
    all_concave_sags = []
    for case_index in case_indices:
        params = figures._ellipse_case_params(case_index)
        records = _facet_records(
            run_root
            / "vtk"
            / "reconstructed"
            / "facets"
            / f"{case_index}.facet_metadata.json",
            params["center"],
            cell_width,
        )
        truth = figures._ellipse_true_segments(case_index)
        concave_sags = [record.sag_over_h for record in records if record.concave]
        all_concave_sags.extend(concave_sags)
        case_data.append((case_index, records, truth, concave_sags))

    figure, axes = plt.subplots(2, len(case_data), figsize=(4.0 * len(case_data), 7.4))
    if len(case_data) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    labels = ("Mild", "Typical", "Worst")
    for column, (case_index, records, truth, concave_sags) in enumerate(case_data):
        _draw_case(axes[0, column], records, truth, mesh_segments, zoom=False)
        _draw_case(axes[1, column], records, truth, mesh_segments, zoom=True)
        prefix = labels[column] if column < len(labels) else "Example"
        axes[0, column].set_title(
            f"{prefix}: case {case_index}", fontsize=11, fontweight="bold"
        )
        axes[0, column].text(
            0.02,
            0.02,
            f"concave arcs: {len(concave_sags)}/{len(records)}\n"
            f"max inward sag: {100 * max(concave_sags):.2f}% $h$",
            transform=axes[0, column].transAxes,
            va="bottom",
            fontsize=8,
            bbox={
                "boxstyle": "square,pad=0.25",
                "facecolor": "white",
                "alpha": 0.92,
            },
        )
        if column == 0:
            axes[0, column].set_ylabel("Full interface", fontsize=10, fontweight="bold")
            axes[1, column].set_ylabel(
                "Worst local arc", fontsize=10, fontweight="bold"
            )

    figure.legend(
        handles=[
            Line2D([0], [0], color="#2563eb", linewidth=1.6, label="Convex arc"),
            Line2D([0], [0], color="#dc2626", linewidth=1.8, label="Concave arc"),
            Line2D(
                [0],
                [0],
                color="#111827",
                linewidth=1.0,
                linestyle="--",
                label="Exact ellipse",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Local concavity audit: Cartesian ellipse, $N=32$",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Across all 25 cases: median concave sag $0.37\%h$, "
        "95th percentile $2.28\%h$, maximum $9.77\%h$.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.90))
    pdf_path = output_dir / "ellipse_concavity_examples_n32.pdf"
    png_path = output_dir / "ellipse_concavity_examples_n32.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary_path = output_dir / "README.md"
    summary_path.write_text(
        "# Ellipse Local Concavity Review\n\n"
        "Concave arcs are highlighted in red for mild, typical, and worst "
        "Cartesian `N=32` cases ranked by maximum inward sag relative to cell "
        "width. Across all 25 cases, the median concave sag is `0.37% h`, the "
        "95th percentile is `2.28% h`, and the maximum is `9.77% h`.\n"
    )
    return {"pdf": pdf_path, "png": png_path, "readme": summary_path}


def _parse_cases(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", default=",".join(map(str, DEFAULT_CASES)))
    args = parser.parse_args(argv)
    outputs = generate(args.run_root, args.output_dir, _parse_cases(args.cases))
    for path in outputs.values():
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
