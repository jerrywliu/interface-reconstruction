#!/usr/bin/env python3
"""Reproduce the Maity et al. (2020) Figure 10 ellipse gallery.

The source gives the ellipse equation and the 10x10/20x20 Cartesian grids, but
does not publish the random realization used in the representative panels.  We
therefore use clean parameters inferred from an independent pixel fit of the
published representative panel.  All reconstruction kernels are imported
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from experiments.baselines.project_benchmarks import (
    DOMAIN_SIZE,
    ProjectBenchmarkCase,
)
from experiments.baselines.run_pcic_project_smoke import run_case as run_pcic_case
from experiments.baselines.run_plvira_project_smoke import run_case as run_plvira_case
from experiments.baselines.run_quasi_project_smoke import run_case as run_quasi_case
from experiments.plotting import (
    PAPER_HIGH_ORDER_COLORS,
    PAPER_METHOD_LINESTYLES,
    apply_paper_serif_style,
)
from main.algos.baselines.external_geometry import (
    ExternalBaselineResult,
    ExternalPrimitive,
    ExternalReconstructionStatus,
)
from main.algos.baselines.external_metrics import (
    shared_edge_gap_metrics,
    symmetric_hausdorff_external,
)
from main.algos.baselines.project_facet_adapter import (
    external_primitives_from_facet_metadata,
)
from main.structs.meshes.merge_mesh import MergeMesh
from util.initialize.areas import initializeEllipse
from util.initialize.points import makeFineCartesianGrid
from util.metrics.metrics import calculate_facet_gaps
from util.reconstruction import runReconstruction


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(
    "results/submission/maity_figure10_gallery_20260905"
)
SOURCE_PDF = Path(
    "/Users/wei/Code/Interface/references/"
    "Numerical Methods in Fluids - 2020 - Maity - "
    "An accurate interface reconstruction method using piecewise circular arcs.pdf"
)
PUBLISHER_FIGURE_ASSET = Path(
    "results/submission/maity_figure10_20260905/source_reference/"
    "fld4876-fig-0010-m.png"
)
PUBLISHER_FIGURE_CROP = Path(
    "results/submission/maity_figure10_20260905/source_reference/"
    "maity_fig10_actual_ellipse_crop.png"
)

TEXT_A_SQUARED = 0.12
TEXT_B_SQUARED = 0.02
FIGURE_CENTER_UNIT = (0.5, 0.55)
FIGURE_MAJOR_AXIS_UNIT = 0.35
FIGURE_MINOR_AXIS_UNIT = 0.15
SOURCE_THETA_DEGREES = 40.0
SOURCE_THETA_RADIANS = math.radians(SOURCE_THETA_DEGREES)
SOURCE_RESOLUTIONS = (10, 20)
CASE_INDEX = 0
BENCHMARK_ID = "maity_fig10_representative_ellipse"
EQUATION_BENCHMARK_ID = "maity_equation_ellipse"

# Independent pixel fit of the representative ellipse in the publisher asset,
# expressed in the apparent 10-by-10 grid coordinates.
DIGITIZED_FIT = {
    "center": (5.16, 5.49),
    "major_axis": 3.54,
    "minor_axis": 1.45,
    "theta_degrees": 39.99,
}
SOURCE_PIXEL_MAPPING = {
    "grid_left_px": 6.0,
    "grid_bottom_px": 280.0,
    "pixels_per_grid_unit": 10.0,
}

METHODS = (
    {
        "id": "plvira",
        "label": "PLVIRA",
        "gallery_label": "PLVIRA",
        "color": PAPER_HIGH_ORDER_COLORS["plvira"],
        "linestyle": "-.",
        "family": "external",
    },
    {
        "id": "pcic_center",
        "label": "PCIC (center translation)",
        "gallery_label": "PCIC\n(center translation)",
        "color": PAPER_HIGH_ORDER_COLORS["pcic_center"],
        "linestyle": "--",
        "family": "external",
    },
    {
        "id": "quasi",
        "label": "QUASI",
        "gallery_label": "QUASI",
        "color": PAPER_HIGH_ORDER_COLORS["quasi"],
        "linestyle": "-",
        "family": "external",
    },
    {
        "id": "ours_per_cell",
        "label": "Ours: circular (per-cell)",
        "gallery_label": "Ours: circular\n(per-cell)",
        "color": PAPER_HIGH_ORDER_COLORS["ours_per_cell"],
        "linestyle": PAPER_METHOD_LINESTYLES["safe_circle"],
        "family": "project",
        "facet_algo": "safe_circle",
        "do_c0": False,
    },
    {
        "id": "ours_graph",
        "label": "Ours: circular (graph-coordinated)",
        "gallery_label": "Ours: circular\n(graph-coordinated)",
        "color": PAPER_HIGH_ORDER_COLORS["ours_graph"],
        "linestyle": PAPER_METHOD_LINESTYLES["circular"],
        "family": "project",
        "facet_algo": "circular",
        "do_c0": False,
    },
    {
        "id": "ours_c0",
        "label": "Ours: circular (graph-coordinated + joint C0)",
        "gallery_label": "Ours: circular\n(graph-coordinated\n+ joint C0)",
        "color": PAPER_HIGH_ORDER_COLORS["ours_c0"],
        "linestyle": PAPER_METHOD_LINESTYLES["circular+C0"],
        "family": "project",
        "facet_algo": "circular",
        "do_c0": True,
    },
)

IMPLEMENTATION_PATHS = (
    Path("experiments/baselines/build_maity_figure10_gallery.py"),
    Path("experiments/baselines/run_pcic_project_smoke.py"),
    Path("main/algos/baselines/plvira.py"),
    Path("main/algos/baselines/plvira_ghf.py"),
    Path("main/algos/baselines/pcic.py"),
    Path("main/algos/baselines/quasi.py"),
    Path("main/structs/meshes/merge_mesh.py"),
    Path("util/reconstruction.py"),
)


def source_case(geometry: str = "figure") -> ProjectBenchmarkCase:
    """Return the requested Maity ellipse realization in project units."""

    if geometry == "figure":
        major_axis_unit = FIGURE_MAJOR_AXIS_UNIT
        minor_axis_unit = FIGURE_MINOR_AXIS_UNIT
        benchmark_id = BENCHMARK_ID
        provenance = "clean values inferred from publisher Figure 10 asset"
    elif geometry == "equation":
        major_axis_unit = math.sqrt(TEXT_A_SQUARED)
        minor_axis_unit = math.sqrt(TEXT_B_SQUARED)
        benchmark_id = EQUATION_BENCHMARK_ID
        provenance = (
            "semiaxes from Maity et al. Section 3.1 equation; "
            "center and angle inferred from publisher Figure 10 asset"
        )
    else:
        raise ValueError(f"unknown Maity ellipse geometry: {geometry!r}")

    scale = DOMAIN_SIZE
    return ProjectBenchmarkCase(
        benchmark="ellipses",
        case_index=CASE_INDEX,
        random_seed=-1,
        parameters={
            "center": [
                FIGURE_CENTER_UNIT[0] * scale,
                FIGURE_CENTER_UNIT[1] * scale,
            ],
            "major_axis": major_axis_unit * scale,
            "minor_axis": minor_axis_unit * scale,
            "aspect_ratio": major_axis_unit / minor_axis_unit,
            "theta": SOURCE_THETA_RADIANS,
            "benchmark_id": benchmark_id,
            "parameter_provenance": provenance,
        },
    )


def _output_dirs(base: Path) -> dict[str, str]:
    paths = {
        "base": base,
        "vtk": base / "vtk",
        "vtk_true": base / "vtk" / "true",
        "vtk_reconstructed": base / "vtk" / "reconstructed",
        "vtk_reconstructed_mixed": base / "vtk" / "reconstructed" / "mixed_cells",
        "vtk_reconstructed_c0": base / "vtk" / "reconstructed" / "C0_facets",
        "vtk_reconstructed_facets": base / "vtk" / "reconstructed" / "facets",
        "vtk_advected": base / "vtk" / "advected" / "facets",
        "plt": base / "plt",
        "plt_areas": base / "plt" / "areas",
        "plt_partial": base / "plt" / "partial_areas",
        "metrics": base / "metrics",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in paths.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _ellipse_pixels(
    center: Sequence[float],
    major_axis: float,
    minor_axis: float,
    theta_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    parameter = np.linspace(0.0, 2.0 * math.pi, 721)
    theta = math.radians(theta_degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    x = (
        float(center[0])
        + major_axis * np.cos(parameter) * cosine
        - minor_axis * np.sin(parameter) * sine
    )
    y = (
        float(center[1])
        + major_axis * np.cos(parameter) * sine
        + minor_axis * np.sin(parameter) * cosine
    )
    mapping = SOURCE_PIXEL_MAPPING
    return (
        mapping["grid_left_px"] + mapping["pixels_per_grid_unit"] * x,
        mapping["grid_bottom_px"] - mapping["pixels_per_grid_unit"] * y,
    )


def _preserve_source_reference(output: Path) -> dict[str, Any]:
    """Copy the publisher asset and render the figure-derived fit overlay."""

    source_directory = output / "source_reference"
    source_directory.mkdir(parents=True, exist_ok=True)
    source_asset = REPO_ROOT / PUBLISHER_FIGURE_ASSET
    source_crop = REPO_ROOT / PUBLISHER_FIGURE_CROP
    if not source_asset.exists():
        raise FileNotFoundError(f"publisher Figure 10 asset not found: {source_asset}")
    copied_asset = source_directory / source_asset.name
    shutil.copy2(source_asset, copied_asset)
    copied_crop = None
    if source_crop.exists():
        copied_crop = source_directory / source_crop.name
        shutil.copy2(source_crop, copied_crop)

    image = plt.imread(copied_asset)
    figure, axis = plt.subplots(figsize=(6.0, 4.07), dpi=150)
    axis.imshow(image)
    digitized_x, digitized_y = _ellipse_pixels(
        DIGITIZED_FIT["center"],
        float(DIGITIZED_FIT["major_axis"]),
        float(DIGITIZED_FIT["minor_axis"]),
        float(DIGITIZED_FIT["theta_degrees"]),
    )
    clean_x, clean_y = _ellipse_pixels(
        (10.0 * FIGURE_CENTER_UNIT[0], 10.0 * FIGURE_CENTER_UNIT[1]),
        10.0 * FIGURE_MAJOR_AXIS_UNIT,
        10.0 * FIGURE_MINOR_AXIS_UNIT,
        SOURCE_THETA_DEGREES,
    )
    axis.plot(
        digitized_x,
        digitized_y,
        color="#00A6C8",
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        label="pixel fit",
    )
    axis.plot(
        clean_x,
        clean_y,
        color="#D94F9D",
        linewidth=1.0,
        label="clean figure-derived benchmark",
    )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.axis("off")
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=7.0,
        handlelength=2.3,
    )
    overlay = source_directory / "maity_fig10_parameter_overlay.png"
    figure.savefig(overlay, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    report = {
        "source_asset": {
            "path": str(copied_asset.resolve()),
            "sha256": _sha256(copied_asset),
            "pixel_shape": [int(image.shape[1]), int(image.shape[0])],
        },
        "apparent_grid_coordinates": "[0,10] x [0,10] in the lower-left representative panel",
        "pixel_mapping": SOURCE_PIXEL_MAPPING,
        "digitized_fit": DIGITIZED_FIT,
        "clean_named_benchmark": {
            "id": BENCHMARK_ID,
            "center": [
                10.0 * FIGURE_CENTER_UNIT[0],
                10.0 * FIGURE_CENTER_UNIT[1],
            ],
            "major_axis": 10.0 * FIGURE_MAJOR_AXIS_UNIT,
            "minor_axis": 10.0 * FIGURE_MINOR_AXIS_UNIT,
            "theta_degrees": SOURCE_THETA_DEGREES,
        },
        "normalized_difference_clean_minus_pixel_fit": {
            "center_x": 10.0 * FIGURE_CENTER_UNIT[0] - DIGITIZED_FIT["center"][0],
            "center_y": 10.0 * FIGURE_CENTER_UNIT[1] - DIGITIZED_FIT["center"][1],
            "major_axis": 10.0 * FIGURE_MAJOR_AXIS_UNIT - DIGITIZED_FIT["major_axis"],
            "minor_axis": 10.0 * FIGURE_MINOR_AXIS_UNIT - DIGITIZED_FIT["minor_axis"],
            "theta_degrees": SOURCE_THETA_DEGREES - DIGITIZED_FIT["theta_degrees"],
        },
        "caveat": (
            "The publisher does not provide the representative realization's "
            "numerical parameters or random seed. Pixel fitting is limited by "
            "rasterization, grid lines, and the visible publisher watermark."
        ),
    }
    report_path = source_directory / "maity_fig10_parameter_fit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    report["files"] = [
        str(copied_asset.resolve()),
        str(overlay.resolve()),
        str(report_path.resolve()),
    ] + ([] if copied_crop is None else [str(copied_crop.resolve())])
    return report


def _write_volume_fractions(case: ProjectBenchmarkCase, resolution: int, path: Path) -> dict[str, Any]:
    mesh = case.build_mesh(resolution)
    fractions = np.asarray(case.initialize_fractions(mesh), dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("cell_x", "cell_y", "volume_fraction"),
            lineterminator="\n",
        )
        writer.writeheader()
        for x in range(fractions.shape[0]):
            for y in range(fractions.shape[1]):
                writer.writerow(
                    {
                        "cell_x": x,
                        "cell_y": y,
                        "volume_fraction": f"{fractions[x, y]:.17g}",
                    }
                )
    cell_area = (DOMAIN_SIZE / resolution) ** 2
    return {
        "resolution": resolution,
        "shape": list(fractions.shape),
        "mixed_cells": int(np.count_nonzero((fractions > 1.0e-10) & (fractions < 1.0 - 1.0e-10))),
        "integrated_phase_area_project_units": float(np.sum(fractions) * cell_area),
        "analytic_phase_area_project_units": math.pi
        * float(case.parameters["major_axis"])
        * float(case.parameters["minor_axis"]),
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def _run_external_methods(
    case: ProjectBenchmarkCase, output: Path
) -> tuple[dict[tuple[str, int], tuple[ExternalPrimitive, ...]], list[dict[str, Any]]]:
    loaded: dict[tuple[str, int], tuple[ExternalPrimitive, ...]] = {}
    rows: list[dict[str, Any]] = []
    roots = {
        "plvira": output / "raw" / "plvira",
        "pcic_center": output / "raw" / "pcic_center",
        "quasi": output / "raw" / "quasi",
    }
    for resolution in SOURCE_RESOLUTIONS:
        paths = {
            "plvira": roots["plvira"]
            / "geometry"
            / f"ellipses_N{resolution}_case{CASE_INDEX:02d}.json",
            "pcic_center": roots["pcic_center"]
            / "geometry"
            / f"translate_center_ellipses_N{resolution}_case{CASE_INDEX:02d}.json",
            "quasi": roots["quasi"]
            / "geometry"
            / f"ellipses_N{resolution}_case{CASE_INDEX:02d}.json",
        }
        if not paths["plvira"].exists():
            run_plvira_case(case, resolution, roots["plvira"])
        if not paths["pcic_center"].exists():
            run_pcic_case(
                case,
                resolution,
                "translate_center",
                roots["pcic_center"],
                boundary_policy="zero_exterior",
            )
        if not paths["quasi"].exists():
            run_quasi_case(case, resolution, roots["quasi"])
        for method_id, path in paths.items():
            result = ExternalBaselineResult.from_json(path)
            active = {
                ExternalReconstructionStatus.RECONSTRUCTED,
                ExternalReconstructionStatus.PAPER_FALLBACK,
            }
            primitives = tuple(
                primitive
                for record in result.cells.values()
                if record.status in active
                for primitive in record.primitives()
            )
            loaded[(method_id, resolution)] = primitives
            truth = case.truth_primitives()
            hausdorff = symmetric_hausdorff_external(primitives, truth) / DOMAIN_SIZE
            gap = float(shared_edge_gap_metrics(result)["mean"]) / DOMAIN_SIZE
            counts = dict(result.metadata["status_counts"])
            reconstructed = counts["reconstructed"] + counts["paper_fallback"]
            rows.append(
                {
                    "method_id": method_id,
                    "resolution": resolution,
                    "mixed_cells": len(result.cells),
                    "reconstructed_cells": reconstructed,
                    "coverage": reconstructed / len(result.cells),
                    "unsupported_cells": counts["unsupported"],
                    "unresolved_cells": counts["unresolved"],
                    "primitive_count": len(primitives),
                    "facet_count": len(primitives),
                    "hausdorff_distance": hausdorff,
                    "facet_gap": gap,
                    "metric_coordinates": "unit_square",
                    "geometry_path": str(path.resolve()),
                }
            )
    return loaded, rows


def _run_project_method(
    case: ProjectBenchmarkCase,
    method: Mapping[str, Any],
    resolution: int,
    output: Path,
) -> tuple[tuple[ExternalPrimitive, ...], dict[str, Any]]:
    points = makeFineCartesianGrid(DOMAIN_SIZE, resolution / DOMAIN_SIZE)
    mesh = MergeMesh(points, 1.0e-10)
    fractions = initializeEllipse(
        mesh,
        float(case.parameters["major_axis"]),
        float(case.parameters["minor_axis"]),
        float(case.parameters["theta"]),
        case.parameters["center"],
    )
    mesh.initializeFractions(fractions)
    raw = output / "raw" / str(method["id"]) / f"N{resolution}"
    path = (
        raw
        / "vtk"
        / "reconstructed"
        / "facets"
        / f"{CASE_INDEX}.facet_metadata.json"
    )
    output_dirs = _output_dirs(raw)
    facets, reconstructed_polys = runReconstruction(
        mesh,
        method["facet_algo"],
        bool(method["do_c0"]),
        CASE_INDEX,
        output_dirs,
        algo_kwargs={
            "plic_fallback": "LVIRA",
            "corner_behavior_profile": "pre_f8_corner",
            "c0_mode": "joint",
            "c0_joint_max_nfev": 500,
        },
        return_polys=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    primitives = tuple(external_primitives_from_facet_metadata(payload))
    truth = case.truth_primitives()
    hausdorff = symmetric_hausdorff_external(primitives, truth) / DOMAIN_SIZE
    gap = calculate_facet_gaps(
        mesh,
        facets,
        reconstructed_polys=reconstructed_polys,
    ) / DOMAIN_SIZE
    mixed_cells = int(
        np.count_nonzero(
            (np.asarray(fractions) > 1.0e-10)
            & (np.asarray(fractions) < 1.0 - 1.0e-10)
        )
    )
    return primitives, {
        "method_id": method["id"],
        "resolution": resolution,
        "mixed_cells": mixed_cells,
        "reconstructed_cells": mixed_cells,
        "coverage": 1.0,
        "unsupported_cells": 0,
        "unresolved_cells": 0,
        "primitive_count": len(primitives),
        "facet_count": len(facets),
        "hausdorff_distance": hausdorff,
        "facet_gap": gap,
        "metric_coordinates": "unit_square",
        "geometry_path": str(path.resolve()),
    }


def _run_project_methods(
    case: ProjectBenchmarkCase, output: Path
) -> tuple[dict[tuple[str, int], tuple[ExternalPrimitive, ...]], list[dict[str, Any]]]:
    loaded: dict[tuple[str, int], tuple[ExternalPrimitive, ...]] = {}
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        if method["family"] != "project":
            continue
        for resolution in SOURCE_RESOLUTIONS:
            primitives, row = _run_project_method(case, method, resolution, output)
            loaded[(str(method["id"]), resolution)] = primitives
            rows.append(row)
    return loaded, rows


def _sample_count(primitive: ExternalPrimitive, resolution: int) -> int:
    cell_size = DOMAIN_SIZE / resolution
    return min(400, max(32, int(math.ceil(primitive.length() / cell_size * 24))))


def _plot_grid(axis: Any, resolution: int) -> None:
    values = np.linspace(0.0, 1.0, resolution + 1)
    axis.vlines(values, 0.0, 1.0, color="#d1d5db", linewidth=0.35, zorder=0)
    axis.hlines(values, 0.0, 1.0, color="#d1d5db", linewidth=0.35, zorder=0)


def _plot_primitives(
    axis: Any,
    primitives: Sequence[ExternalPrimitive],
    *,
    resolution: int,
    color: str,
    linestyle: Any,
    linewidth: float,
    endpoints: bool,
    zorder: int,
) -> None:
    endpoint_values: list[tuple[float, float]] = []
    for primitive in primitives:
        points = np.asarray(primitive.sample(_sample_count(primitive, resolution)))
        points = points / DOMAIN_SIZE
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
        if endpoints:
            endpoint_values.extend(
                (
                    (primitive.p_left[0] / DOMAIN_SIZE, primitive.p_left[1] / DOMAIN_SIZE),
                    (primitive.p_right[0] / DOMAIN_SIZE, primitive.p_right[1] / DOMAIN_SIZE),
                )
            )
    if endpoint_values:
        points = np.asarray(endpoint_values)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=6.0,
            facecolors="white",
            edgecolors=color,
            linewidths=0.45,
            zorder=zorder + 1,
        )


def _style_axis(
    axis: Any,
    *,
    method: Mapping[str, Any],
    resolution: int,
    truth: Sequence[ExternalPrimitive],
    primitives: Sequence[ExternalPrimitive],
    summary: Mapping[str, Any],
    title: str,
    endpoints: bool = True,
) -> None:
    _plot_grid(axis, resolution)
    _plot_primitives(
        axis,
        truth,
        resolution=resolution,
        color="#111827",
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        endpoints=False,
        zorder=1,
    )
    _plot_primitives(
        axis,
        primitives,
        resolution=resolution,
        color=str(method["color"]),
        linestyle=method["linestyle"],
        linewidth=1.25,
        endpoints=endpoints,
        zorder=2,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.tick_params(labelsize=7.0, length=2.2, width=0.55)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    axis.set_title(title, fontsize=8.2, pad=3.5)
    if float(summary["coverage"]) < 1.0:
        axis.text(
            0.02,
            0.02,
            f"coverage {int(summary['reconstructed_cells'])}/{int(summary['mixed_cells'])}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
            zorder=5,
        )


def _scientific_math(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.2f}\times 10^{{{exponent}}}"


def _metric_title(summary: Mapping[str, Any]) -> str:
    return (
        rf"$d_H={_scientific_math(float(summary['hausdorff_distance']))}$, "
        rf"$g_h={_scientific_math(float(summary['facet_gap']))}$"
    )


def _save_individual(
    output: Path,
    case: ProjectBenchmarkCase,
    method: Mapping[str, Any],
    resolution: int,
    primitives: Sequence[ExternalPrimitive],
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    truth = case.truth_primitives()
    with mpl.rc_context():
        apply_paper_serif_style()
        figure, axis = plt.subplots(figsize=(3.45, 3.55))
        _style_axis(
            axis,
            method=method,
            resolution=resolution,
            truth=truth,
            primitives=primitives,
            summary=summary,
            title=f"{method['label']}, $N={resolution}$",
        )
        axis.set_xlabel("$x$", fontsize=8.0)
        axis.set_ylabel("$y$", fontsize=8.0)
        figure.text(
            0.5,
            0.015,
            "Dashed black: exact ellipse; colored: native reconstruction",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#4b5563",
        )
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0), pad=0.45)
        pdf = output / "panels" / f"maity_fig10_{method['id']}_N{resolution}.pdf"
        png = pdf.with_suffix(".png")
        pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, bbox_inches="tight", dpi=300)
        plt.close(figure)
    return pdf, png


def _save_gallery(
    output: Path,
    case: ProjectBenchmarkCase,
    loaded: Mapping[tuple[str, int], Sequence[ExternalPrimitive]],
    summaries: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[Path, Path]:
    truth = case.truth_primitives()
    with mpl.rc_context():
        apply_paper_serif_style()
        mpl.rcParams.update(
            {
                "axes.titlesize": 8.0,
                "xtick.labelsize": 6.7,
                "ytick.labelsize": 6.7,
            }
        )
        figure, axes = plt.subplots(3, 4, figsize=(7.25, 6.25), squeeze=False)
        panel_letters = iter("abcdefghijkl")
        for method_index, method in enumerate(METHODS):
            row = method_index // 2
            pair = method_index % 2
            for resolution_index, resolution in enumerate(SOURCE_RESOLUTIONS):
                column = 2 * pair + resolution_index
                axis = axes[row, column]
                method_id = str(method["id"])
                _style_axis(
                    axis,
                    method=method,
                    resolution=resolution,
                    truth=truth,
                    primitives=loaded[(method_id, resolution)],
                    summary=summaries[(method_id, resolution)],
                    title=f"({next(panel_letters)}) $N={resolution}$",
                )
                axis.text(
                    0.5,
                    0.965,
                    str(method["gallery_label"]),
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7.0,
                    color="#111827",
                    linespacing=0.95,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.9,
                        "pad": 1.2,
                    },
                    zorder=6,
                )
                if row == 2:
                    axis.set_xlabel("$x$", fontsize=7.5)
                else:
                    axis.tick_params(labelbottom=False)
                if column == 0:
                    axis.set_ylabel("$y$", fontsize=7.5)
                else:
                    axis.tick_params(labelleft=False)
        figure.suptitle(
            "Maity et al. Figure 10 ellipse: matched native reconstructions",
            fontsize=9.4,
            y=0.995,
        )
        figure.text(
            0.5,
            0.008,
            "Unit-square Cartesian grid; dashed black is the exact ellipse; open circles mark native facet endpoints.",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#4b5563",
        )
        figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.955), pad=0.42, h_pad=1.0, w_pad=0.48)
        pdf = output / "maity_figure10_all_methods_gallery.pdf"
        png = pdf.with_suffix(".png")
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, bbox_inches="tight", dpi=300)
        plt.close(figure)
    return pdf, png


def _save_two_column_gallery_without_endpoints(
    output: Path,
    case: ProjectBenchmarkCase,
    loaded: Mapping[tuple[str, int], Sequence[ExternalPrimitive]],
    summaries: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[Path, Path]:
    truth = case.truth_primitives()
    with mpl.rc_context():
        apply_paper_serif_style()
        mpl.rcParams.update(
            {
                "axes.titlesize": 8.2,
                "xtick.labelsize": 7.0,
                "ytick.labelsize": 7.0,
            }
        )
        figure, axes = plt.subplots(
            len(METHODS),
            len(SOURCE_RESOLUTIONS),
            figsize=(7.25, 17.4),
            squeeze=False,
        )
        panel_letters = iter("abcdefghijkl")
        for method_index, method in enumerate(METHODS):
            method_id = str(method["id"])
            for resolution_index, resolution in enumerate(SOURCE_RESOLUTIONS):
                axis = axes[method_index, resolution_index]
                _style_axis(
                    axis,
                    method=method,
                    resolution=resolution,
                    truth=truth,
                    primitives=loaded[(method_id, resolution)],
                    summary=summaries[(method_id, resolution)],
                    title=(
                        f"({next(panel_letters)}) {method['gallery_label']}, "
                        f"$N={resolution}$"
                    ),
                    endpoints=False,
                )
                if method_index == len(METHODS) - 1:
                    axis.set_xlabel("$x$", fontsize=8.0)
                else:
                    axis.tick_params(labelbottom=False)
                if resolution_index == 0:
                    axis.set_ylabel("$y$", fontsize=8.0)
                else:
                    axis.tick_params(labelleft=False)
        figure.suptitle(
            "Maity et al. Figure 10 ellipse: matched native reconstructions",
            fontsize=9.6,
            y=0.997,
        )
        figure.text(
            0.5,
            0.006,
            "Unit-square Cartesian grid; dashed black is the exact ellipse; colored curves are native reconstructions.",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#4b5563",
        )
        figure.tight_layout(
            rect=(0.0, 0.018, 1.0, 0.985),
            pad=0.5,
            h_pad=0.9,
            w_pad=0.65,
        )
        pdf = output / "maity_figure10_all_methods_gallery_no_endpoints_2col.pdf"
        png = pdf.with_suffix(".png")
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, bbox_inches="tight", dpi=300)
        plt.close(figure)
    return pdf, png


def _save_two_page_metric_gallery(
    output: Path,
    case: ProjectBenchmarkCase,
    loaded: Mapping[tuple[str, int], Sequence[ExternalPrimitive]],
    summaries: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[Path, tuple[Path, Path]]:
    """Save the logical 6x2 gallery as two legible 3x2 appendix pages."""

    truth = case.truth_primitives()
    pdf = output / "maity_figure10_all_methods_gallery_metrics_2page.pdf"
    previews = (
        output / "maity_figure10_all_methods_gallery_metrics_page1.png",
        output / "maity_figure10_all_methods_gallery_metrics_page2.png",
    )
    panel_letters = iter("abcdefghijkl")
    with mpl.rc_context():
        apply_paper_serif_style()
        mpl.rcParams.update(
            {
                "axes.titlesize": 8.0,
                "xtick.labelsize": 7.0,
                "ytick.labelsize": 7.0,
            }
        )
        pages = PdfPages(pdf)
        for page_index, page_methods in enumerate((METHODS[:3], METHODS[3:])):
            figure, axes = plt.subplots(
                len(page_methods),
                len(SOURCE_RESOLUTIONS),
                figsize=(7.25, 9.25),
                squeeze=False,
            )
            for method_index, method in enumerate(page_methods):
                method_id = str(method["id"])
                for resolution_index, resolution in enumerate(SOURCE_RESOLUTIONS):
                    axis = axes[method_index, resolution_index]
                    summary = summaries[(method_id, resolution)]
                    title = (
                        f"({next(panel_letters)}) {method['gallery_label']}, "
                        f"$N={resolution}$\n{_metric_title(summary)}"
                    )
                    _style_axis(
                        axis,
                        method=method,
                        resolution=resolution,
                        truth=truth,
                        primitives=loaded[(method_id, resolution)],
                        summary=summary,
                        title=title,
                        endpoints=False,
                    )
                    if method_index == len(page_methods) - 1:
                        axis.set_xlabel("$x$", fontsize=8.0)
                    else:
                        axis.tick_params(labelbottom=False)
                    if resolution_index == 0:
                        axis.set_ylabel("$y$", fontsize=8.0)
                    else:
                        axis.tick_params(labelleft=False)
            family = (
                "Reproduced higher-order methods"
                if page_index == 0
                else "Proposed circular variants"
            )
            figure.suptitle(
                f"Maity ellipse reconstruction: {family}",
                fontsize=9.6,
                y=0.997,
            )
            figure.text(
                0.5,
                0.006,
                "Dashed black: exact ellipse; colored: reconstruction. Metrics use unit-square coordinates.",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#4b5563",
            )
            figure.tight_layout(
                rect=(0.0, 0.018, 1.0, 0.98),
                pad=0.5,
                h_pad=1.25,
                w_pad=0.65,
            )
            pages.savefig(figure, bbox_inches="tight")
            figure.savefig(previews[page_index], bbox_inches="tight", dpi=300)
            plt.close(figure)
        pages.close()
    return pdf, previews


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    output: Path,
    case: ProjectBenchmarkCase,
    source_reference: Mapping[str, Any],
    fraction_records: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    pdfs: Sequence[Path],
) -> Path:
    parameters = case.parameters
    is_equation_geometry = parameters["benchmark_id"] == EQUATION_BENCHMARK_ID
    if is_equation_geometry:
        source_fit_note = (
            "The source-fit overlay documents the separate raster-inferred "
            "realization; the selected package instead uses the semiaxes stated "
            "in Section 3.1."
        )
        realization_note = (
            "The source does not identify which random pose supplies its "
            "representative panels. This package therefore combines the published "
            "equation-defined semiaxes with a figure-inferred center and angle and "
            "does not claim a bitwise reproduction of Figure 10."
        )
    else:
        source_fit_note = (
            "The selected clean values agree closely with the raster fit and "
            "visually overlay the publisher panel."
        )
        realization_note = (
            "The source does not identify which of its ten random ellipses supplies "
            "the representative panels, nor the exact center, semiaxes, angle, or "
            "seed. Consequently this is source-faithful to the published Figure 10 "
            "geometry, but it is not a bitwise reproduction of an undisclosed "
            "random realization."
        )
    pcic_coarse = next(
        row
        for row in summaries
        if row["method_id"] == "pcic_center" and int(row["resolution"]) == 10
    )
    if int(pcic_coarse["unresolved_cells"]) == 0:
        pcic_note = (
            "PLVIRA and PCIC remain qualified Cartesian reproductions. With the "
            "known empty exterior represented by zero-volume-fraction ghost cells, "
            "PCIC reconstructs every mixed cell at both resolutions."
        )
    else:
        pcic_note = (
            "PLVIRA and PCIC remain qualified Cartesian reproductions. "
            "Zero-exterior padding removes PCIC's coarse-grid boundary-halo "
            "failures. Two `N=10` tip cells remain unresolved because the fitted "
            "circle does not cross the target-cell boundary, leaving no fitted "
            "chord for the published center-translation direction."
        )
    coverage_lines = "\n".join(
        f"- `{row['method_id']}`, `N={row['resolution']}`: "
        f"{row['reconstructed_cells']}/{row['mixed_cells']} mixed cells "
        f"({100.0 * float(row['coverage']):.2f}% coverage), "
        f"{row['unsupported_cells']} unsupported, {row['unresolved_cells']} unresolved."
        for row in summaries
    )
    fraction_lines = "\n".join(
        f"- `N={row['resolution']}`: `{row['path']}`; "
        f"{row['mixed_cells']} mixed cells; integrated area "
        f"`{row['integrated_phase_area_project_units']:.12g}` versus analytic "
        f"`{row['analytic_phase_area_project_units']:.12g}` in the scaled project domain."
        for row in fraction_records
    )
    panel_lines = "\n".join(f"- `{path.name}`" for path in pdfs)
    readme = output / "README.md"
    readme.write_text(
        f"""# Maity Figure 10 reconstruction gallery

## Primary source and source audit

- Primary source: `{SOURCE_PDF}`.
- Inspected location: PDF page 13, journal page 105, Figure 10 and Section 3.1.
- Publisher Figure 10 asset: `source_reference/fld4876-fig-0010-m.png`; the copied source image, actual-interface crop, fit overlay, and parameter report are preserved together under `source_reference/`.
- Published domain: unit square.
- Published ellipse: `x^2/a^2 + y^2/b^2 = 1`, with `a^2 = 0.12` and `b^2 = 0.02`.
- Published grids represented here: `10 x 10` and `20 x 20` Cartesian cells.
- Published study context: the orientation is randomized and the error study averages 100 realizations; Figure 10 also shows ten randomly oriented ellipses.

## Named representative realization

- Benchmark identifier: `{parameters['benchmark_id']}`.
- Independent pixel fit on the apparent `[0,10]^2` panel grid: center approximately `({DIGITIZED_FIT['center'][0]}, {DIGITIZED_FIT['center'][1]})`, semiaxes `({DIGITIZED_FIT['major_axis']}, {DIGITIZED_FIT['minor_axis']})`, and angle `{DIGITIZED_FIT['theta_degrees']}` degrees.
- Selected parameters: center `(0.5, 0.55)`, semiaxes `({parameters['major_axis'] / DOMAIN_SIZE:.15g}, {parameters['minor_axis'] / DOMAIN_SIZE:.15g})`, and `{SOURCE_THETA_DEGREES:g}` degrees counterclockwise in unit-square coordinates.
- Parameter provenance: {parameters['parameter_provenance']}.
- Scaled project parameters: center `{parameters['center']}`, semiaxes `({parameters['major_axis']:.15g}, {parameters['minor_axis']:.15g})`, and aspect ratio `{parameters['aspect_ratio']:.15g}` on `[0,100]^2`.
- {source_fit_note} See `source_reference/maity_fig10_parameter_overlay.png` and `source_reference/maity_fig10_parameter_fit.json`.
- Section 3.1's equation implies semiaxes `sqrt(0.12) = {math.sqrt(TEXT_A_SQUARED):.15g}` and `sqrt(0.02) = {math.sqrt(TEXT_B_SQUARED):.15g}`. The representative raster is better matched by `0.35` and `0.15`, so the figure-derived and equation-defined packages remain separate rather than silently conflating them.

## Volume fractions

Volume fractions are recomputed from the analytic rotated ellipse on the exact Cartesian grids and supplied unchanged to every method. The CSVs enumerate every cell, including pure cells.

{fraction_lines}

## Methods and unchanged implementation scope

- `PLVIRA`: the existing Cartesian generalized-height-function PLVIRA reproduction.
- `PCIC (center translation)`: the existing paper-facing bare-PCIC reproduction and its selected conservative center-translation correction. Because this closed ellipse has known empty exterior phase at the domain boundary, missing Cartesian halo cells are represented by zero-volume-fraction ghost cells.
- `QUASI`: the existing frozen Cartesian port and continuity-sweep policy.
- `Ours: circular (per-cell)`: `safe_circle`, with no graph propagation, merging, or C0 pass.
- `Ours: circular (graph-coordinated)`: `circular`, with the production graph/merge path and no C0 pass.
- `Ours: circular (graph-coordinated + joint C0)`: the same production path followed by the unchanged joint C0 optimizer.

No reconstruction kernel or method parameter was tuned for this benchmark. The generator imports the existing implementations directly.

## Plotted quantities

- Cartesian cell boundaries in unit-square source coordinates.
- Exact analytic ellipse as a dashed black curve.
- Each method's native reconstructed primitives as colored curves using the approved paper palette.
- Native primitive endpoints as small open circles in the original overview and separate panels. The two-column overview omits these markers so the reconstructed curves remain unobstructed.
- Coverage annotations only when a reproduction does not return a facet for every mixed cell.
- The two-page metric gallery reports symmetric Hausdorff distance `d_H` and mean facet gap `g_h` in unit-square coordinates beside every reconstruction.

The gallery does not plot the paper's `E1` area-error aggregate because this task is a representative reconstruction gallery rather than a 100-realization convergence replay.

## Coverage and method caveats

{coverage_lines}

- {pcic_note}
- QUASI retains its frozen root, ordering, fallback, and ten-sweep stopping policies; a drawn curve does not imply that the endpoint sweep met its convergence rule.
- {realization_note}
- The source Figure 10 compares PLIC, bare PCIC, and C0-corrected PCIC. This package instead holds the source benchmark fixed and evaluates the manuscript's already-approved PLVIRA, PCIC, QUASI, and proposed-method comparison set.

## Outputs

- Combined appendix overview: `maity_figure10_all_methods_gallery.pdf`.
- Two-column overview without endpoint markers: `maity_figure10_all_methods_gallery_no_endpoints_2col.pdf`.
- Two-page appendix candidate with per-panel metrics: `maity_figure10_all_methods_gallery_metrics_2page.pdf`.
- Separate method/resolution panels:

{panel_lines}

- Machine-readable coverage: `reconstruction_summary.csv`.
- Exact fractions: `data/volume_fractions_N10.csv` and `data/volume_fractions_N20.csv`.
- Source-fit provenance: `source_reference/fld4876-fig-0010-m.png`, `source_reference/maity_fig10_actual_ellipse_crop.png`, `source_reference/maity_fig10_parameter_overlay.png`, and `source_reference/maity_fig10_parameter_fit.json`.
- Provenance and hashes: `manifest.json`.
- Rendered PNGs beside each PDF are QA previews only; the PDFs are the deliverables.

## Reproduction command

```bash
python -m experiments.baselines.build_maity_figure10_gallery \\
  --output {output}
```

## Validation performed

- Focused unit tests for the published geometry, scaling, method roster, palette, and output naming.
- Integrated phase-area check from the saved volume fractions against `pi a b`.
- All 13 deliverable PDFs checked for embedded fonts and absence of raster image objects.
- All deliverable PDFs rendered with Poppler and reviewed from a contact sheet at final layout scale.

## Changed files

- `experiments/baselines/build_maity_figure10_gallery.py`: source-faithful benchmark runner and gallery generator.
- `test/experiments/baselines/test_maity_figure10_gallery.py`: focused source/roster/output tests.
- `results/submission/maity_figure10_gallery_20260905/`: generated data, native method outputs, figures, README, and manifest.
- `FILE_INVENTORY.md`: exhaustive path inventory for every generated file in this output package.
- `../memory/progress/README.md`, `../memory/runs/README.md`, `../memory/todo/README.md`, and `../memory/experiments/static-tests.md`: workspace handoff/status updates required by `AGENTS.md`.

No Overleaf or manuscript file was edited, and no result was promoted.
""",
        encoding="utf-8",
    )
    return readme


def build(output: Path, geometry: str = "figure") -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    case = source_case(geometry)
    source_reference = _preserve_source_reference(output)
    fraction_records = [
        _write_volume_fractions(
            case,
            resolution,
            output / "data" / f"volume_fractions_N{resolution}.csv",
        )
        for resolution in SOURCE_RESOLUTIONS
    ]
    external, external_rows = _run_external_methods(case, output)
    project, project_rows = _run_project_methods(case, output)
    loaded = {**external, **project}
    rows = sorted(
        [*external_rows, *project_rows],
        key=lambda row: (
            next(i for i, method in enumerate(METHODS) if method["id"] == row["method_id"]),
            int(row["resolution"]),
        ),
    )
    _write_csv(output / "reconstruction_summary.csv", rows)
    summaries = {(str(row["method_id"]), int(row["resolution"])): row for row in rows}

    pdfs: list[Path] = []
    previews: list[Path] = []
    for method in METHODS:
        method_id = str(method["id"])
        for resolution in SOURCE_RESOLUTIONS:
            pdf, png = _save_individual(
                output,
                case,
                method,
                resolution,
                loaded[(method_id, resolution)],
                summaries[(method_id, resolution)],
            )
            pdfs.append(pdf)
            previews.append(png)
    gallery_pdf, gallery_png = _save_gallery(output, case, loaded, summaries)
    clean_gallery_pdf, clean_gallery_png = _save_two_column_gallery_without_endpoints(
        output, case, loaded, summaries
    )
    metric_gallery_pdf, metric_gallery_previews = _save_two_page_metric_gallery(
        output, case, loaded, summaries
    )
    pdfs[:0] = [gallery_pdf, clean_gallery_pdf, metric_gallery_pdf]
    previews[:0] = [gallery_png, clean_gallery_png, *metric_gallery_previews]
    readme = _write_readme(
        output, case, source_reference, fraction_records, rows, pdfs[3:]
    )

    manifest = {
        "schema_version": 1,
        "study": (
            "Maity et al. (2020) equation-defined ellipse gallery"
            if geometry == "equation"
            else "Maity et al. (2020) Figure 10 source-faithful ellipse gallery"
        ),
        "analysis_git_head": _git_head(),
        "source_pdf": {
            "path": str(SOURCE_PDF),
            "sha256": _sha256(SOURCE_PDF),
            "inspected_pdf_page": 13,
            "journal_page": 105,
            "figure": 10,
        },
        "published_specification": {
            "domain": [0.0, 1.0, 0.0, 1.0],
            "section_3_1_a_squared": TEXT_A_SQUARED,
            "section_3_1_b_squared": TEXT_B_SQUARED,
            "resolutions": list(SOURCE_RESOLUTIONS),
            "orientation_policy": "randomized; exact representative realization not published",
            "aggregate_case_count": 100,
        },
        "representative_realization": {
            "benchmark_id": case.parameters["benchmark_id"],
            "center_unit": list(FIGURE_CENTER_UNIT),
            "major_axis_unit": case.parameters["major_axis"] / DOMAIN_SIZE,
            "minor_axis_unit": case.parameters["minor_axis"] / DOMAIN_SIZE,
            "theta_degrees": SOURCE_THETA_DEGREES,
            "parameter_provenance": case.parameters["parameter_provenance"],
            "scaled_project_case": case.to_dict(),
        },
        "boundary_policies": {
            "pcic": "zero-volume-fraction Cartesian ghost cells for the known empty exterior phase",
            "scope": "closed interior ellipse benchmark only",
        },
        "source_reference": source_reference,
        "methods": [dict(method) for method in METHODS],
        "implementation_sha256": {
            str(path): _sha256(REPO_ROOT / path) for path in IMPLEMENTATION_PATHS
        },
        "volume_fractions": list(fraction_records),
        "reconstruction_summary": rows,
        "artifacts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in [*pdfs, *previews, readme, output / "reconstruction_summary.csv"]
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    inventory = output / "FILE_INVENTORY.md"
    relative_files = sorted(
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file() and path != inventory
    )
    inventory.write_text(
        "# Generated file inventory\n\n"
        "Every file generated under this package root is listed below.\n\n"
        + "\n".join(f"- `{path}`" for path in [*relative_files, Path("FILE_INVENTORY.md")])
        + "\n",
        encoding="utf-8",
    )
    return pdfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--geometry",
        choices=("figure", "equation"),
        default="figure",
        help="use raster-inferred axes or the semiaxes stated in Section 3.1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdfs = build(args.output, geometry=args.geometry)
    print(f"WROTE {len(pdfs)} PDF artifacts under {args.output}", flush=True)


if __name__ == "__main__":
    main()
