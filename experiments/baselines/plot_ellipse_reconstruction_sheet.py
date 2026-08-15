#!/usr/bin/env python3
"""Plot matched ellipse reconstruction windows across all comparison methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.baselines.project_benchmarks import (
    DOMAIN_SIZE,
    canonical_benchmark_cases,
)
from main.algos.baselines.external_geometry import (
    ExternalBaselineResult,
    ExternalPrimitive,
    ExternalReconstructionStatus,
)
from main.algos.baselines.project_facet_adapter import (
    external_primitives_from_facet_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = (
    {
        "id": "ours_per_cell",
        "label": "Ours: circular (per-cell)",
        "color": "#0072B2",
        "project_key": "per_cell_circular",
    },
    {
        "id": "ours_graph",
        "label": "Ours: circular (graph-coordinated)",
        "color": "#009E73",
        "project_key": "graph_coordinated_circular",
    },
    {
        "id": "ours_c0",
        "label": "Ours: circular (graph + joint C0)",
        "color": "#D55E00",
        "project_key": "graph_coordinated_circular_joint_c0",
    },
    {
        "id": "plvira",
        "label": "PLVIRA",
        "color": "#CC79A7",
        "external_key": "plvira_input",
        "filename": "ellipses_N{resolution}_case{case_index:02d}.json",
    },
    {
        "id": "pcic_center",
        "label": "PCIC (center translation)",
        "color": "#6B7280",
        "external_key": "pcic_input",
        "filename": (
            "translate_center_ellipses_N{resolution}_case{case_index:02d}.json"
        ),
    },
    {
        "id": "quasi",
        "label": "QUASI (frozen port)",
        "color": "#7E57C2",
        "external_key": "quasi_input",
        "filename": "ellipses_N{resolution}_case{case_index:02d}.json",
    },
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _sample_count(primitive: ExternalPrimitive, cell_size: float) -> int:
    return min(300, max(24, int(math.ceil(primitive.length() / cell_size * 12))))


def _project_geometry_path(
    run_prefix: str, project_key: str, resolution: int, case_index: int
) -> Path:
    run_name = f"{run_prefix}_{project_key}_n{resolution}"
    return (
        REPO_ROOT
        / "plots"
        / run_name
        / "vtk"
        / "reconstructed"
        / "facets"
        / f"{case_index}.facet_metadata.json"
    )


def _load_project_primitives(path: Path) -> tuple[ExternalPrimitive, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(external_primitives_from_facet_metadata(payload))


def _load_external_primitives(
    path: Path,
) -> tuple[tuple[ExternalPrimitive, ...], int, int]:
    result = ExternalBaselineResult.from_json(path)
    active_statuses = {
        ExternalReconstructionStatus.RECONSTRUCTED,
        ExternalReconstructionStatus.PAPER_FALLBACK,
    }
    primitives = tuple(
        primitive
        for record in result.cells.values()
        if record.status in active_statuses
        for primitive in record.primitives()
    )
    mixed = len(result.cells)
    reconstructed = sum(record.status in active_statuses for record in result.cells.values())
    return primitives, reconstructed, mixed


def _high_curvature_tip(parameters: Mapping[str, Any]) -> tuple[float, float]:
    center = np.asarray(parameters["center"], dtype=float)
    theta = float(parameters["theta"])
    direction = np.asarray([math.cos(theta), math.sin(theta)])
    first = center + float(parameters["major_axis"]) * direction
    second = center - float(parameters["major_axis"]) * direction
    selected = first if first[0] >= second[0] else second
    return float(selected[0]), float(selected[1])


def _plot_grid(
    axis: Any,
    resolution: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    cell_size = DOMAIN_SIZE / resolution
    x_values = np.arange(
        math.floor(xlim[0] / cell_size) * cell_size,
        xlim[1] + cell_size,
        cell_size,
    )
    y_values = np.arange(
        math.floor(ylim[0] / cell_size) * cell_size,
        ylim[1] + cell_size,
        cell_size,
    )
    axis.vlines(x_values, ylim[0], ylim[1], color="#d1d5db", linewidth=0.35, zorder=0)
    axis.hlines(y_values, xlim[0], xlim[1], color="#d1d5db", linewidth=0.35, zorder=0)


def _plot_primitives(
    axis: Any,
    primitives: Sequence[ExternalPrimitive],
    *,
    color: str,
    cell_size: float,
    endpoints: bool,
    linestyle: str = "-",
    linewidth: float = 1.15,
    zorder: int = 2,
) -> None:
    endpoint_values: list[tuple[float, float]] = []
    for primitive in primitives:
        points = np.asarray(primitive.sample(_sample_count(primitive, cell_size)))
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
        if endpoints:
            endpoint_values.extend((primitive.p_left, primitive.p_right))
    if endpoint_values:
        points = np.asarray(endpoint_values)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=8,
            facecolors="white",
            edgecolors=color,
            linewidths=0.55,
            zorder=zorder + 1,
        )


def plot_sheet(
    *,
    output: Path,
    case_index: int,
    resolutions: Sequence[int],
    run_prefix: str,
    plvira_input: Path,
    pcic_input: Path,
    quasi_input: Path,
    window_half_width: float,
) -> None:
    case = canonical_benchmark_cases("ellipses", (case_index,))[0]
    truth = case.truth_primitives()
    tip = _high_curvature_tip(case.parameters)
    xlim = (tip[0] - window_half_width, tip[0] + window_half_width)
    ylim = (tip[1] - window_half_width, tip[1] + window_half_width)
    inputs = {
        "plvira_input": plvira_input,
        "pcic_input": pcic_input,
        "quasi_input": quasi_input,
    }

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
    figure, axes = plt.subplots(
        len(METHODS),
        len(resolutions),
        figsize=(3.1 * len(resolutions), 2.35 * len(METHODS)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    source_files = []
    for row_index, method in enumerate(METHODS):
        for column_index, resolution in enumerate(resolutions):
            axis = axes[row_index, column_index]
            cell_size = DOMAIN_SIZE / resolution
            _plot_grid(axis, resolution, xlim, ylim)
            _plot_primitives(
                axis,
                truth,
                color="#111827",
                cell_size=cell_size,
                endpoints=False,
                linestyle="--",
                linewidth=0.9,
                zorder=1,
            )
            if "project_key" in method:
                source = _project_geometry_path(
                    run_prefix,
                    method["project_key"],
                    resolution,
                    case_index,
                )
                primitives = _load_project_primitives(source)
                reconstructed = mixed = None
            else:
                source = (
                    inputs[method["external_key"]]
                    / "geometry"
                    / method["filename"].format(
                        resolution=resolution, case_index=case_index
                    )
                )
                primitives, reconstructed, mixed = _load_external_primitives(source)
            source_files.append(source)
            _plot_primitives(
                axis,
                primitives,
                color=method["color"],
                cell_size=cell_size,
                endpoints=True,
            )
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(labelsize=6, length=2)
            if row_index == 0:
                axis.set_title(f"N = {resolution}", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(method["label"], fontsize=7)
            if row_index == len(METHODS) - 1:
                axis.set_xlabel("x")
            if reconstructed is not None and mixed:
                axis.text(
                    0.03,
                    0.96,
                    f"coverage {100.0 * reconstructed / mixed:.2f}%",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="#374151",
                )

    figure.suptitle(
        f"Ellipse case {case_index}: matched reconstruction near maximum curvature",
        fontsize=11,
        y=0.995,
    )
    figure.text(
        0.5,
        0.978,
        "Dashed black: analytic ellipse. Open circles: native facet endpoints.",
        ha="center",
        va="top",
        fontsize=7,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=0.5, w_pad=0.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=360)
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "analysis_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "case_index": case_index,
        "case_parameters": dict(case.parameters),
        "resolutions": list(resolutions),
        "window_center": list(tip),
        "window_half_width": window_half_width,
        "source_files": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in source_files
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-index", required=True, type=int)
    parser.add_argument("--resolutions", required=True, type=_parse_ints)
    parser.add_argument("--project-run-prefix", required=True)
    parser.add_argument("--plvira-input", required=True, type=Path)
    parser.add_argument("--pcic-input", required=True, type=Path)
    parser.add_argument("--quasi-input", required=True, type=Path)
    parser.add_argument("--window-half-width", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_sheet(
        output=args.output,
        case_index=args.case_index,
        resolutions=args.resolutions,
        run_prefix=args.project_run_prefix,
        plvira_input=args.plvira_input,
        pcic_input=args.pcic_input,
        quasi_input=args.quasi_input,
        window_half_width=args.window_half_width,
    )


if __name__ == "__main__":
    main()
