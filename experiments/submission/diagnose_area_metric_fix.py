"""Recompute corrected area errors for the two July metric witnesses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.submission.conservation_analyzer import analyze_saved_case


SELECTIONS = (
    {
        "benchmark": "Squares",
        "case_index": 2,
        "run_root": (
            "plots/"
            "perturb_sweep_squares_linearpluscorner_r0p5_w0p0_s0_corner_pre_f8_corner"
        ),
        "live_run_root": "plots/submission_area_metric_fix_square",
    },
    {
        "benchmark": "Zalesak",
        "case_index": 1,
        "run_root": (
            "plots/"
            "perturb_sweep_zalesak_circularpluscorner_r0p5_w0p0_s0_corner_"
            "pre_f8_corner_exact_linear_support_only"
        ),
        "live_run_root": "plots/submission_area_metric_fix_zalesak",
    },
)


def collect_rows(repo_root: Path) -> list[dict[str, object]]:
    rows = []
    for selection in SELECTIONS:
        run_root = repo_root / selection["run_root"]
        analysis = analyze_saved_case(
            run_root,
            int(selection["case_index"]),
            repo_root=repo_root,
        )
        summary = analysis.summary
        legacy = float(summary["legacy_reported_area_error"])
        corrected = float(summary["global_relative_phase_area_error"])
        live_metrics = read_case_metrics(
            repo_root / selection["live_run_root"], int(selection["case_index"])
        )
        rows.append(
            {
                **selection,
                "source_commit": summary["source_commit"],
                "legacy_area_error": legacy,
                "corrected_area_error": corrected,
                "legacy_over_corrected": legacy / corrected,
                "prescribed_phase_area": summary["prescribed_phase_area"],
                "corrected_reconstructed_phase_area": summary[
                    "reconstructed_phase_area"
                ],
                "max_merged_zone_absolute_residual": summary[
                    "max_merged_zone_absolute_residual"
                ],
                "live_driver_area_error": live_metrics.get("area_error"),
                "live_driver_hausdorff": live_metrics.get("hausdorff"),
                "live_driver_facet_gap": live_metrics.get("facet_gap"),
            }
        )
    return rows


def read_case_metrics(run_root: Path, case_index: int) -> dict[str, float]:
    path = run_root / "metrics" / "case_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["case_index"]) == case_index:
                return {
                    field: float(row[field])
                    for field in ("area_error", "hausdorff", "facet_gap")
                    if row.get(field) not in (None, "")
                }
    return {}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    x_positions = range(len(rows))
    width = 0.34
    legacy = [float(row["legacy_area_error"]) for row in rows]
    corrected = [float(row["corrected_area_error"]) for row in rows]
    axis.bar(
        [x - width / 2 for x in x_positions],
        legacy,
        width,
        label="Legacy metric",
        color="#c65d46",
    )
    axis.bar(
        [x + width / 2 for x in x_positions],
        corrected,
        width,
        label="Geometry-faithful metric",
        color="#297a70",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Relative phase-area error")
    axis.set_xticks(
        list(x_positions),
        [f"{row['benchmark']} case {row['case_index']}" for row in rows],
    )
    axis.grid(axis="y", which="both", color="#d7d7d7", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    for index, (old, new) in enumerate(zip(legacy, corrected)):
        axis.text(index - width / 2, old * 1.35, f"{old:.3g}", ha="center")
        axis.text(index + width / 2, new * 1.6, f"{new:.2e}", ha="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def write_readme(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Static area-metric fix",
        "",
        "The reconstruction is unchanged. The corrected metric evaluates each "
        "line, signed supporting circle, or two-branch corner as fitted.",
        "",
        "| Witness | Legacy error | Corrected replay | Patched live driver |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        live = row.get("live_driver_area_error")
        live_text = "not run" if live is None else f"{float(live):.3e}"
        lines.append(
            f"| {row['benchmark']} case {row['case_index']} | "
            f"{float(row['legacy_area_error']):.6g} | "
            f"{float(row['corrected_area_error']):.3e} | {live_text} |"
        )
    lines.extend(
        [
            "",
            "The replay uses the exact July mesh seed and structured facet records. "
            "The live smoke reruns preserve the historical Hausdorff and facet-gap "
            "values to roundoff while replacing only the reported area error.",
            "",
            "`area_error_before_after.pdf` is vector-only; the PNG is a preview for "
            "artifact sharing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/submission/area_metric_fix"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )

    rows = collect_rows(repo_root)
    write_csv(output_dir / "area_error_before_after.csv", rows)
    plot_comparison(output_dir / "area_error_before_after.pdf", rows)
    write_readme(output_dir / "README.md", rows)
    for row in rows:
        print(
            f"{row['benchmark']} case {row['case_index']}: "
            f"{row['legacy_area_error']:.6g} -> "
            f"{row['corrected_area_error']:.6g}"
        )


if __name__ == "__main__":
    main()
