"""Visualize why guarded C0 correction leaves gaps on Cartesian ellipses."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import generate_section6_maintext_figures as figures


DEFAULT_CASES = (10, 12, 23)
CASE_LABELS = {
    10: "Best case",
    12: "Typical case",
    23: "Worst tail",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rejected_merge_ids(run_root: Path) -> dict[int, set[int]]:
    rejected: dict[int, set[int]] = {}
    with (run_root / "metrics" / "merge_events.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["event_kind"] != "c0_rejection":
                continue
            rejected.setdefault(int(row["case_index"]), set()).add(int(row["merge_id"]))
    return rejected


def _case_geometry(run_root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["case_index"]): row
        for row in _read_jsonl(run_root / "metrics" / "case_geometry.jsonl")
    }


def _case_audits(run_root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["case_index"]): row
        for row in _read_jsonl(run_root / "metrics" / "c0_join_audit.jsonl")
    }


def _record_segments(
    records: Sequence[Mapping[str, Any]], spacing: float
) -> np.ndarray:
    chunks = []
    for record in records:
        primitive = figures._primitive_from_metadata(dict(record))
        points = np.asarray(
            primitive.sample_by_max_spacing(max(spacing, 1.0e-3)), dtype=float
        )
        if len(points) >= 2:
            chunks.append(np.stack([points[:-1], points[1:]], axis=1))
    if not chunks:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


def _case_plot_data(
    run_root: Path,
    case_index: int,
    mesh_segments: np.ndarray,
    rejected_ids: set[int],
) -> dict[str, Any]:
    facet_path = run_root / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    metadata = figures._read_facet_metadata(facet_path)
    if metadata is None:
        raise FileNotFoundError(f"Missing exact facet metadata: {facet_path}")
    step = figures._mesh_step_from_segments(mesh_segments) or 1.0
    accepted_records = [
        record
        for record in metadata["primitives"]
        if int(record["facet_index"]) not in rejected_ids
    ]
    rejected_records = [
        record
        for record in metadata["primitives"]
        if int(record["facet_index"]) in rejected_ids
    ]
    return {
        "accepted": _record_segments(accepted_records, step / 8.0),
        "rejected": _record_segments(rejected_records, step / 8.0),
        "true": figures._load_true_segments("ellipses", run_root.name, case_index),
        "step": step,
    }


def _style_axis(axis) -> None:
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#4b5563")


def _draw_geometry(
    axis,
    *,
    mesh_segments: np.ndarray,
    data: Mapping[str, Any],
    joins: Sequence[Mapping[str, Any]],
    show_mesh: bool,
) -> None:
    if show_mesh:
        figures._add_segments(
            axis,
            mesh_segments,
            color="#cbd5e1",
            linewidth=0.32,
            alpha=0.70,
        )
    figures._add_segments(
        axis,
        data["true"],
        color="#111827",
        linewidth=1.15,
        linestyle="--",
        zorder=2,
    )
    figures._add_segments(
        axis,
        data["accepted"],
        color="#2563eb",
        linewidth=1.75,
        zorder=3,
    )
    figures._add_segments(
        axis,
        data["rejected"],
        color="#dc2626",
        linewidth=2.15,
        zorder=4,
    )
    for join in joins:
        first = np.asarray(join["first_endpoint_after_c0"], dtype=float)
        second = np.asarray(join["second_endpoint_after_c0"], dtype=float)
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color="#dc2626",
            linestyle=(0, (2.0, 1.5)),
            linewidth=1.0,
            zorder=5,
        )
    _style_axis(axis)


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(
    run_root: Path,
    output_pdf: Path,
    case_indices: Sequence[int] = DEFAULT_CASES,
) -> tuple[Path, Path, Path]:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    run_root = run_root.resolve()
    output_pdf = output_pdf.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")
    output_csv = output_pdf.with_name(f"{output_pdf.stem}_summary.csv")

    mesh_segments = figures._mesh_segments(run_root / "vtk" / "mesh.vtk")
    rejected_by_case = _rejected_merge_ids(run_root)
    audits = _case_audits(run_root)
    geometry = _case_geometry(run_root)

    figure, axes = plt.subplots(2, len(case_indices), figsize=(11.2, 6.9))
    summary_rows = []
    for column, case_index in enumerate(case_indices):
        audit = audits[case_index]
        rejected_ids = rejected_by_case.get(case_index, set())
        data = _case_plot_data(run_root, case_index, mesh_segments, rejected_ids)
        bad_joins = sorted(
            (
                join
                for join in audit["joins"]
                if join["gap_after_c0"] is not None
                and float(join["gap_after_c0"]) > 1.0e-8
            ),
            key=lambda join: float(join["gap_after_c0"]),
            reverse=True,
        )
        worst = bad_joins[0]
        max_gap = float(worst["gap_after_c0"])

        full_axis = axes[0, column]
        _draw_geometry(
            full_axis,
            mesh_segments=mesh_segments,
            data=data,
            joins=bad_joins,
            show_mesh=True,
        )
        xmin, xmax, ymin, ymax = figures._compute_view_bounds(
            data["true"], min_span=66.0, margin_frac=0.08
        )
        full_axis.set_xlim(xmin, xmax)
        full_axis.set_ylim(ymin, ymax)
        full_axis.set_title(
            f"{CASE_LABELS.get(case_index, 'Case')} {case_index}",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )
        full_axis.text(
            0.02,
            0.02,
            f"rejected facets: {len(rejected_ids)}\n"
            f"non-C0 joins: {len(bad_joins)}/{len(audit['joins'])}\n"
            f"maximum gap: {max_gap:.3e}",
            transform=full_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
            bbox={
                "boxstyle": "square,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#94a3b8",
                "linewidth": 0.55,
                "alpha": 0.94,
            },
            zorder=8,
        )

        zoom_axis = axes[1, column]
        _draw_geometry(
            zoom_axis,
            mesh_segments=mesh_segments,
            data=data,
            joins=[worst],
            show_mesh=True,
        )
        first = np.asarray(worst["first_endpoint_after_c0"], dtype=float)
        second = np.asarray(worst["second_endpoint_after_c0"], dtype=float)
        midpoint = 0.5 * (
            np.asarray(worst["first_endpoint"], dtype=float)
            + np.asarray(worst["second_endpoint"], dtype=float)
        )
        endpoint_specs = (
            (first, int(worst["first_merge_id"])),
            (second, int(worst["second_merge_id"])),
        )
        for point, merge_id in endpoint_specs:
            rejected = merge_id in rejected_ids
            zoom_axis.scatter(
                point[0],
                point[1],
                s=36,
                marker="o",
                facecolors="#dc2626" if rejected else "white",
                edgecolors="#991b1b" if rejected else "#2563eb",
                linewidths=1.1,
                zorder=7,
            )
        zoom_axis.scatter(
            midpoint[0],
            midpoint[1],
            s=42,
            marker="x",
            color="#059669",
            linewidths=1.4,
            zorder=8,
        )
        center = 0.5 * (first + second)
        span = max(0.5, min(6.5, 8.0 * max_gap))
        zoom_axis.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        zoom_axis.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
        zoom_axis.set_title(
            f"Worst retained join (gap {max_gap:.3e})",
            fontsize=9.0,
            pad=6,
        )

        case_geometry = geometry[case_index]
        summary_rows.append(
            {
                "case_index": case_index,
                "aspect_ratio": case_geometry["aspect_ratio"],
                "theta": case_geometry["theta"],
                "eligible_joins": len(audit["joins"]),
                "rejected_facets": len(rejected_ids),
                "non_c0_joins": len(bad_joins),
                "mean_post_c0_join_gap": statistics.mean(
                    float(join["gap_after_c0"]) for join in audit["joins"]
                ),
                "max_post_c0_join_gap": max_gap,
                "worst_first_merge_id": worst["first_merge_id"],
                "worst_second_merge_id": worst["second_merge_id"],
            }
        )

    figure.legend(
        handles=[
            Line2D([0], [0], color="#111827", linestyle="--", label="Analytic ellipse"),
            Line2D([0], [0], color="#2563eb", label="Accepted C0 refit"),
            Line2D([0], [0], color="#dc2626", label="Original facet retained"),
            Line2D(
                [0],
                [0],
                color="#dc2626",
                linestyle=(0, (2.0, 1.5)),
                label="Residual endpoint gap",
            ),
            Line2D(
                [0],
                [0],
                color="#059669",
                marker="x",
                linestyle="None",
                label="Proposed common midpoint",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=8.0,
    )
    figure.suptitle(
        "Why guarded C0 correction is not globally continuous on Cartesian N=32 ellipses",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.055,
        "A rejected cell keeps its conservative original facet. Its neighbor may accept the midpoint-constrained refit, leaving the highlighted join open.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#374151",
    )
    figure.tight_layout(rect=(0.02, 0.10, 0.98, 0.95), h_pad=1.25, w_pad=0.8)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    _write_summary(output_csv, summary_rows)
    return output_pdf, output_png, output_csv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default=",".join(map(str, DEFAULT_CASES)),
        help="comma-separated deterministic ellipse case indices",
    )
    args = parser.parse_args(argv)
    cases = tuple(int(part.strip()) for part in args.cases.split(",") if part.strip())
    for path in generate(args.run_root, args.output, cases):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
