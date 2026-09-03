#!/usr/bin/env python3
"""Render the frozen extended ellipse comparison for manuscript use."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import (
    add_convergence_order_triangle,
    apply_paper_serif_style,
)
from submission.pdf_vector_qa import inspect_pdf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / (
    "experiments/baselines/results/"
    "ellipse_all_method_25case_extended_comparison_20260814"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT
EXPECTED_RESOLUTIONS = (32, 50, 64, 100, 128, 150, 256, 300)
EXPECTED_CASE_INDICES = tuple(range(25))
GAP_DISPLAY_FLOOR = 1.0e-12
PAIR_TIE_RELATIVE_TOLERANCE = 1.0e-12
PAIR_TIE_ABSOLUTE_TOLERANCE = 1.0e-15

PAPER_METHOD_SPECS = (
    {
        "id": "plvira",
        "label": "PLVIRA",
        "linestyle": "--",
        "linewidth": 1.8,
        "marker": "o",
    },
    {
        "id": "pcic_center",
        "label": "PCIC (center translation)",
        "linestyle": "-.",
        "linewidth": 1.8,
        "marker": "s",
    },
    {
        "id": "quasi",
        "label": "QUASI",
        "linestyle": ":",
        "linewidth": 1.8,
        "marker": "D",
    },
    {
        "id": "ours_per_cell",
        "label": "Ours (per-cell)",
        "linestyle": "--",
        "linewidth": 1.9,
        "marker": "^",
    },
    {
        "id": "ours_graph",
        "label": "Ours (graph-coordinated)",
        "linestyle": "-",
        "linewidth": 2.3,
        "marker": "v",
    },
    {
        "id": "ours_c0",
        "label": "Ours (graph-coordinated + joint C0)",
        "linestyle": "-",
        "linewidth": 2.5,
        "marker": "P",
    },
)

PAPER_PALETTES = {
    "current": {
        "plvira": "#6c757d",
        "pcic_center": "#495057",
        "quasi": "#212529",
        "ours_per_cell": "#f59e0b",
        "ours_graph": "#d97706",
        "ours_c0": "#b91c1c",
    },
    "b19_categorical": {
        "plvira": "#B14E5E",
        "pcic_center": "#B3811B",
        "quasi": "#2D7D64",
        "ours_per_cell": "#7C5AA6",
        "ours_graph": "#2F6FA3",
        "ours_c0": "#D55E00",
    },
    "colorblind_categorical": {
        "plvira": "#CC79A7",
        "pcic_center": "#009E73",
        "quasi": "#56B4E9",
        "ours_per_cell": "#E69F00",
        "ours_graph": "#0072B2",
        "ours_c0": "#D55E00",
    },
    "grouped": {
        "plvira": "#8A8A8A",
        "pcic_center": "#4D4D4D",
        "quasi": "#111111",
        "ours_per_cell": "#7C5AA6",
        "ours_graph": "#2F6FA3",
        "ours_c0": "#D55E00",
    },
}


def paper_methods(palette: str = "current") -> tuple[dict[str, Any], ...]:
    if palette not in PAPER_PALETTES:
        raise ValueError(f"unknown paper palette: {palette}")
    colors = PAPER_PALETTES[palette]
    return tuple(
        {**method, "color": colors[method["id"]]} for method in PAPER_METHOD_SPECS
    )


PAPER_METHODS = paper_methods()

ERROR_PANELS = (
    (
        "native_symmetric_hausdorff",
        "(a) Native Hausdorff",
        "Native Hausdorff",
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
        "(c) Facet gap",
        "Facet gap",
        3.0,
        (0.70, 0.54),
    ),
)

WIN_METRICS = (
    ("native_symmetric_hausdorff", "lower"),
    ("geometric_curvature_mean_absolute_error", "lower"),
    ("facet_gap", "lower"),
    ("reconstruction_coverage", "higher"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_file_records(records: Sequence[Mapping[str, Any]]) -> list[str]:
    issues = []
    for record in records:
        path = Path(str(record["path"]))
        if not path.is_file():
            issues.append(f"missing file: {path}")
            continue
        observed = sha256_file(path)
        expected = str(record["sha256"])
        if observed != expected:
            issues.append(
                f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
            )
    return issues


def validate_frozen_manifest(input_root: Path) -> dict[str, Any]:
    manifest_path = input_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported frozen manifest schema: {manifest_path}")
    if payload.get("case_row_count") != len(PAPER_METHODS) * len(
        EXPECTED_RESOLUTIONS
    ) * len(EXPECTED_CASE_INDICES):
        raise ValueError("frozen manifest does not describe the 1,200-row study")
    if tuple(payload.get("cells_per_side", ())) != EXPECTED_RESOLUTIONS:
        raise ValueError(
            "frozen manifest resolution grid does not match the paper study"
        )
    if tuple(payload.get("case_indices", ())) != EXPECTED_CASE_INDICES:
        raise ValueError("frozen manifest case grid does not match the paper study")
    expected_method_ids = {method["id"] for method in PAPER_METHODS}
    observed_method_ids = {
        method["id"] for method in payload.get("selected_methods", ())
    }
    if observed_method_ids != expected_method_ids:
        raise ValueError(
            "frozen manifest methods do not match the paper comparison: "
            f"{sorted(observed_method_ids)}"
        )

    # Analysis source paths may legitimately evolve after a result package is
    # frozen. The manifest retains those historical hashes; the immutable data
    # inputs and packaged artifacts are the files that must still verify now.
    records = []
    for key in ("inputs", "artifacts"):
        records.extend(payload.get(key, ()))
    issues = verify_file_records(records)
    if issues:
        raise ValueError("frozen manifest verification failed:\n" + "\n".join(issues))

    for required_name in ("case_metrics.csv", "summary.csv"):
        required_path = (input_root / required_name).resolve()
        if not any(
            Path(str(record["path"])).resolve() == required_path for record in records
        ):
            raise ValueError(f"frozen manifest does not hash {required_path}")
    return payload


def load_case_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_fields = (
        "native_symmetric_hausdorff",
        "geometric_curvature_mean_absolute_error",
        "facet_gap",
        "normalized_conservation_residual",
    )
    for raw in read_csv(path):
        mixed_cells = int(raw["mixed_cells"])
        reconstructed_cells = int(raw["reconstructed_cells"])
        if mixed_cells <= 0 or not 0 <= reconstructed_cells <= mixed_cells:
            raise ValueError(
                f"invalid coverage counts for {raw['method_id']} N="
                f"{raw['cells_per_side']} case={raw['case_index']}"
            )
        row: dict[str, Any] = {
            "method_id": raw["method_id"],
            "case_index": int(raw["case_index"]),
            "cells_per_side": int(raw["cells_per_side"]),
            "mixed_cells": mixed_cells,
            "reconstructed_cells": reconstructed_cells,
            "reconstruction_coverage": reconstructed_cells / mixed_cells,
        }
        for field in numeric_fields:
            value = float(raw[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"invalid {field} for {raw['method_id']} N="
                    f"{raw['cells_per_side']} case={raw['case_index']}: {value}"
                )
            row[field] = value
        rows.append(row)

    expected_keys = {
        (method["id"], resolution, case_index)
        for method in PAPER_METHODS
        for resolution in EXPECTED_RESOLUTIONS
        for case_index in EXPECTED_CASE_INDICES
    }
    observed_keys = {
        (row["method_id"], row["cells_per_side"], row["case_index"]) for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            "case metrics do not match the frozen study grid; "
            f"missing={missing[:8]} extra={extra[:8]} rows={len(rows)}"
        )
    return rows


def compute_summary(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]] = PAPER_METHODS,
) -> list[dict[str, Any]]:
    result = []
    resolutions = sorted({int(row["cells_per_side"]) for row in rows})
    for method in methods:
        for resolution in resolutions:
            selected = [
                row
                for row in rows
                if row["method_id"] == method["id"]
                and int(row["cells_per_side"]) == resolution
            ]
            if not selected:
                raise ValueError(f"no rows for {method['id']} at N={resolution}")
            item: dict[str, Any] = {
                "method_id": method["id"],
                "cells_per_side": resolution,
                "case_count": len(selected),
            }
            for metric, *_ in ERROR_PANELS:
                values = np.asarray([float(row[metric]) for row in selected])
                item[f"{metric}_median"] = float(np.median(values))
                item[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                item[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            item["reconstruction_coverage"] = sum(
                int(row["reconstructed_cells"]) for row in selected
            ) / sum(int(row["mixed_cells"]) for row in selected)
            result.append(item)
    return result


def validate_frozen_summary(
    computed: Sequence[Mapping[str, Any]], summary_path: Path
) -> None:
    frozen = {
        (row["method_id"], int(row["cells_per_side"])): row
        for row in read_csv(summary_path)
    }
    if len(frozen) != len(computed):
        raise ValueError(
            f"frozen summary has {len(frozen)} rows, expected {len(computed)}"
        )
    fields = [
        f"{metric}_{statistic}"
        for metric, *_ in ERROR_PANELS
        for statistic in ("median", "q25", "q75")
    ] + ["reconstruction_coverage"]
    for row in computed:
        key = row["method_id"], int(row["cells_per_side"])
        if key not in frozen:
            raise ValueError(f"missing frozen summary row: {key}")
        for field in fields:
            expected = float(frozen[key][field])
            observed = float(row[field])
            if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(
                    f"frozen summary mismatch for {key} {field}: "
                    f"expected {expected}, observed {observed}"
                )


def compute_paired_win_counts(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]] = PAPER_METHODS,
) -> list[dict[str, Any]]:
    by_method = {
        method["id"]: {
            (int(row["cells_per_side"]), int(row["case_index"])): row
            for row in rows
            if row["method_id"] == method["id"]
        }
        for method in methods
    }
    result: list[dict[str, Any]] = []
    scopes: list[int | str] = [*EXPECTED_RESOLUTIONS, "all"]
    for scope in scopes:
        for method_a, method_b in itertools.combinations(methods, 2):
            keys_a = set(by_method[method_a["id"]])
            keys_b = set(by_method[method_b["id"]])
            keys = sorted(keys_a & keys_b)
            if scope != "all":
                keys = [key for key in keys if key[0] == scope]
            if not keys:
                raise ValueError(
                    f"no paired rows for {method_a['id']} and {method_b['id']} "
                    f"at {scope}"
                )
            for metric, preferred in WIN_METRICS:
                a_wins = 0
                b_wins = 0
                ties = 0
                for key in keys:
                    value_a = float(by_method[method_a["id"]][key][metric])
                    value_b = float(by_method[method_b["id"]][key][metric])
                    if math.isclose(
                        value_a,
                        value_b,
                        rel_tol=PAIR_TIE_RELATIVE_TOLERANCE,
                        abs_tol=PAIR_TIE_ABSOLUTE_TOLERANCE,
                    ):
                        ties += 1
                    elif (value_a < value_b) == (preferred == "lower"):
                        a_wins += 1
                    else:
                        b_wins += 1
                result.append(
                    {
                        "cells_per_side": scope,
                        "metric": metric,
                        "preferred_value": preferred,
                        "method_a_id": method_a["id"],
                        "method_a_label": method_a["label"],
                        "method_b_id": method_b["id"],
                        "method_b_label": method_b["label"],
                        "paired_count": len(keys),
                        "method_a_wins": a_wins,
                        "ties": ties,
                        "method_b_wins": b_wins,
                    }
                )
    return result


def _display_error(values: np.ndarray, metric: str) -> np.ndarray:
    if metric != "facet_gap":
        return values
    return np.where(values == 0.0, GAP_DISPLAY_FLOOR, values)


def plot_paper_figure(
    summary: Sequence[Mapping[str, Any]],
    pdf_path: Path,
    png_path: Path,
    *,
    triangle_anchors: Mapping[str, tuple[float, float]] | None = None,
    methods: Sequence[Mapping[str, Any]] = PAPER_METHODS,
    figure_size: tuple[float, float] = (11.2, 7.4),
    large_text: bool = False,
) -> None:
    method_by_id = {method["id"]: method for method in methods}
    resolutions = sorted({int(row["cells_per_side"]) for row in summary})
    with mpl.rc_context():
        apply_paper_serif_style()
        if large_text:
            mpl.rcParams.update(
                {
                    "font.size": 10.5,
                    "axes.labelsize": 11.2,
                    "axes.titlesize": 11.2,
                    "legend.fontsize": 10.0,
                }
            )
        tick_fontsize = 9.5 if large_text else 7.5
        note_fontsize = 9.5 if large_text else 7.5
        triangle_fontsize = 9.5 if large_text else 7.0
        figure, axes = plt.subplots(2, 2, figsize=figure_size, sharex=True)
        for axis, (metric, title, ylabel, order, anchor) in zip(
            axes.ravel()[:3], ERROR_PANELS
        ):
            for method in methods:
                selected = sorted(
                    (row for row in summary if row["method_id"] == method["id"]),
                    key=lambda row: int(row["cells_per_side"]),
                )
                x = np.asarray(
                    [int(row["cells_per_side"]) for row in selected], dtype=float
                )
                median = _display_error(
                    np.asarray([float(row[f"{metric}_median"]) for row in selected]),
                    metric,
                )
                q25 = _display_error(
                    np.asarray([float(row[f"{metric}_q25"]) for row in selected]),
                    metric,
                )
                q75 = _display_error(
                    np.asarray([float(row[f"{metric}_q75"]) for row in selected]),
                    metric,
                )
                axis.fill_between(
                    x,
                    q25,
                    q75,
                    color=method["color"],
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
                axis.plot(
                    x,
                    median,
                    color=method["color"],
                    linestyle=method["linestyle"],
                    linewidth=method["linewidth"],
                    marker=method["marker"],
                    markersize=4.2,
                    label=method["label"],
                    zorder=2,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(True, which="major", alpha=0.30)
            axis.tick_params(labelsize=tick_fontsize)
            add_convergence_order_triangle(
                axis,
                order,
                order_label=f"{order:g}",
                anchor=(triangle_anchors or {}).get(metric, anchor),
                width=0.11,
                color="#4b5563",
                fontsize=triangle_fontsize,
            )

        gap_axis = axes[1, 0]
        gap_axis.text(
            0.98,
            0.95,
            r"Exact zeros shown at $10^{-12}$",
            transform=gap_axis.transAxes,
            ha="right",
            va="top",
            fontsize=note_fontsize,
            color="#4b5563",
        )

        coverage_axis = axes[1, 1]
        for method in methods:
            selected = sorted(
                (row for row in summary if row["method_id"] == method["id"]),
                key=lambda row: int(row["cells_per_side"]),
            )
            coverage_axis.plot(
                [int(row["cells_per_side"]) for row in selected],
                [100.0 * float(row["reconstruction_coverage"]) for row in selected],
                color=method["color"],
                linestyle=method["linestyle"],
                linewidth=method["linewidth"],
                marker=method["marker"],
                markersize=4.2,
                zorder=2,
            )
        coverage_values = [
            100.0 * float(row["reconstruction_coverage"]) for row in summary
        ]
        coverage_axis.set_xscale("log")
        coverage_axis.set_ylim(max(0.0, min(coverage_values) - 0.35), 100.15)
        coverage_axis.set_title("(d) Mixed-cell coverage")
        coverage_axis.set_ylabel("Coverage (%)")
        coverage_axis.grid(True, which="major", alpha=0.30)
        coverage_axis.tick_params(labelsize=tick_fontsize)

        for axis in axes.ravel():
            axis.set_xticks(resolutions, tuple(str(value) for value in resolutions))
        for axis in axes[1, :]:
            axis.set_xlabel("Cells per side, N")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        # Matplotlib fills multirow legends column-first. Interleave the source
        # order so the rendered first row is baselines and the second is ours.
        legend_method_order = (
            methods[0],
            methods[3],
            methods[1],
            methods[4],
            methods[2],
            methods[5],
        )
        legend_handles = [
            handle_by_label[method["label"]] for method in legend_method_order
        ]
        legend_labels = [method["label"] for method in legend_method_order]
        legend = figure.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 0.005),
        )
        legend.get_frame().set_edgecolor("#d1d5db")
        legend.get_frame().set_linewidth(0.8)

        figure.tight_layout(rect=(0.0, 0.105, 1.0, 1.0), h_pad=2.0, w_pad=1.6)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    if set(method_by_id) != {row["method_id"] for row in summary}:
        raise ValueError("paper figure did not receive all required methods")


def inspect_pdf_document(pdf_path: Path) -> dict[str, Any]:
    vector_report = inspect_pdf(pdf_path, require_fonts=True)
    if not vector_report.passed:
        raise ValueError("vector PDF QA failed: " + "; ".join(vector_report.issues))

    info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    pages = None
    page_size_points = None
    for line in info.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        elif line.startswith("Page size:"):
            fields = line.split()
            page_size_points = [float(fields[2]), float(fields[4])]
    if pages != 1 or page_size_points is None:
        raise ValueError(
            f"expected a one-page PDF with a reported page size, got pages={pages} "
            f"size={page_size_points}"
        )

    extracted_text = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    required_labels = [method["label"] for method in PAPER_METHODS]
    missing_labels = [label for label in required_labels if label not in extracted_text]
    prohibited_labels = ["topology + merging", "independent cells"]
    present_prohibited = [
        label for label in prohibited_labels if label in extracted_text.lower()
    ]
    if missing_labels or present_prohibited:
        raise ValueError(
            "PDF label QA failed; "
            f"missing={missing_labels} prohibited={present_prohibited}"
        )
    return {
        "pages": pages,
        "page_size_points": page_size_points,
        "required_labels_present": required_labels,
        "prohibited_labels_absent": prohibited_labels,
        "vector_qa": asdict(vector_report) | {"passed": vector_report.passed},
    }


def write_hash_ledger(path: Path, artifacts: Sequence[Path]) -> None:
    lines = [f"{sha256_file(artifact)}  {artifact.name}" for artifact in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_hash_ledger(path: Path) -> list[str]:
    issues = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            expected, filename = line.split("  ", 1)
        except ValueError:
            issues.append(f"malformed hash line {line_number}: {line}")
            continue
        artifact = path.parent / filename
        if not artifact.is_file():
            issues.append(f"missing hashed artifact: {artifact}")
            continue
        observed = sha256_file(artifact)
        if observed != expected:
            issues.append(
                f"SHA-256 mismatch for {artifact}: expected {expected}, observed {observed}"
            )
    return issues


def tracked_worktree_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def write_paper_manifest(
    path: Path,
    *,
    input_root: Path,
    frozen_manifest: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    win_counts: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Path],
    pdf_audit: Mapping[str, Any],
    artifact_kind: str = "paper-ready extended Cartesian ellipse comparison",
    benchmark: str = "ellipses",
    generator_sources: Sequence[Path] = (),
) -> None:
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_paths = list(generator_sources) or [Path(__file__).resolve()]
    for shared_source in (
        REPO_ROOT / "experiments/plotting.py",
        REPO_ROOT / "submission/pdf_vector_qa.py",
    ):
        if shared_source not in source_paths:
            source_paths.append(shared_source)
    payload = {
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "benchmark": benchmark,
        "generator_git_head": git_head,
        "tracked_worktree_clean": tracked_worktree_is_clean(),
        "generator_sources": [file_record(source) for source in source_paths],
        "frozen_input": {
            "root": str(input_root.resolve()),
            "manifest": file_record(input_root / "manifest.json"),
            "manifest_git_head": frozen_manifest["analysis_git_head"],
            "manifest_verified": True,
            "historical_analysis_sources": frozen_manifest["analysis_sources"],
            "case_metrics": file_record(input_root / "case_metrics.csv"),
            "summary": file_record(input_root / "summary.csv"),
        },
        "study_grid": {
            "methods": [method["id"] for method in PAPER_METHODS],
            "cells_per_side": list(EXPECTED_RESOLUTIONS),
            "case_indices": list(EXPECTED_CASE_INDICES),
            "case_row_count": len(case_rows),
            "summary_row_count": len(summary),
        },
        "display": {
            "method_labels": {
                method["id"]: method["label"] for method in PAPER_METHODS
            },
            "error_summary": "median and interquartile range over 25 matched cases",
            "coverage_summary": (
                "reconstructed mixed cells divided by mixed cells across the 25-case "
                "resolution group"
            ),
            "facet_gap_zero_display_floor": GAP_DISPLAY_FLOOR,
            "order_triangles": {
                "native_symmetric_hausdorff": 3,
                "geometric_curvature_mean_absolute_error": 1,
                "facet_gap": 3,
            },
        },
        "paired_win_counts": {
            "row_count": len(win_counts),
            "scopes": [*EXPECTED_RESOLUTIONS, "all"],
            "tie_relative_tolerance": PAIR_TIE_RELATIVE_TOLERANCE,
            "tie_absolute_tolerance": PAIR_TIE_ABSOLUTE_TOLERANCE,
        },
        "pdf_audit": pdf_audit,
        "artifacts": [file_record(artifact) for artifact in artifacts],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    frozen_manifest = validate_frozen_manifest(input_root)
    case_rows = load_case_metrics(input_root / "case_metrics.csv")
    summary = compute_summary(case_rows)
    validate_frozen_summary(summary, input_root / "summary.csv")
    win_counts = compute_paired_win_counts(case_rows)

    pdf_path = output_root / "ellipse_all_methods_metrics_paper.pdf"
    png_path = output_root / "ellipse_all_methods_metrics_paper.png"
    win_path = output_root / "ellipse_all_methods_paired_win_counts.csv"
    hash_path = output_root / "ellipse_all_methods_metrics_paper.sha256"
    manifest_path = output_root / "ellipse_all_methods_metrics_paper.manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    plot_paper_figure(summary, pdf_path, png_path)
    write_csv(win_path, win_counts)
    write_hash_ledger(hash_path, (pdf_path, png_path, win_path))
    hash_issues = verify_hash_ledger(hash_path)
    if hash_issues:
        raise ValueError(
            "generated artifact hash verification failed:\n" + "\n".join(hash_issues)
        )
    pdf_audit = inspect_pdf_document(pdf_path)
    write_paper_manifest(
        manifest_path,
        input_root=input_root,
        frozen_manifest=frozen_manifest,
        case_rows=case_rows,
        summary=summary,
        win_counts=win_counts,
        artifacts=(pdf_path, png_path, win_path, hash_path),
        pdf_audit=pdf_audit,
    )

    for artifact in (pdf_path, png_path, win_path, hash_path, manifest_path):
        print(artifact)
    print(
        "QA: frozen hashes verified; 1-page vector PDF; no raster objects; "
        "all fonts embedded; approved labels present"
    )


if __name__ == "__main__":
    main()
