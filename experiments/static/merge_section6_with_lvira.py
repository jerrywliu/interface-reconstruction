#!/usr/bin/env python3
"""
Merge refreshed ELVIRA/LVIRA baseline rows into the canonical Section 6 static CSV.

This script keeps the existing non-baseline rows from the current camera-ready
Section 6 sources, drops any existing ELVIRA/LVIRA baseline rows, and replaces
them with fresh explicit ELVIRA/LVIRA rows from a new rerun bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_MERGED_SOURCE = (
    REPO_ROOT
    / "results"
    / "static"
    / "camera_ready"
    / "static_cameraready_plotrefresh_elvira_lvira_backfill_20260327"
    / "csv"
    / "section6_plotrefresh_merged.csv"
)

CANONICAL_SOURCES = {
    "lines": REPO_ROOT
    / "results"
    / "static"
    / "camera_ready"
    / "static_cameraready_debug_20260303"
    / "csv"
    / "perturbed_sweep.csv",
    "circles": CURRENT_MERGED_SOURCE,
    "ellipses": CURRENT_MERGED_SOURCE,
    "squares": REPO_ROOT
    / "results"
    / "static"
    / "camera_ready"
    / "static_cameraready_corner_validation_20260308_143412"
    / "csv"
    / "perturbed_sweep.csv",
    "zalesak": REPO_ROOT
    / "results"
    / "static"
    / "camera_ready"
    / "static_cameraready_zalesak_rerun_20260319_083025"
    / "csv"
    / "zalesak_merged.csv",
}

FIELDNAMES = [
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "metric_key",
    "metric_value",
    "save_name",
]


def _load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rows_for_experiment(rows: list[dict], experiment: str) -> list[dict]:
    return [row for row in rows if row.get("experiment") == experiment]


def _load_replacement_rows(bundle: Path) -> dict[str, list[dict]]:
    csv_dir = bundle / "csv" / "by_experiment"
    replacements = {}
    for experiment in CANONICAL_SOURCES:
        csv_path = csv_dir / f"{experiment}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing rerun CSV for {experiment}: {csv_path}")
        rows = _load_rows(csv_path)
        if not rows:
            raise RuntimeError(f"Rerun CSV for {experiment} is empty: {csv_path}")
        replacements[experiment] = rows
    return replacements


def _merge_rows(bundle: Path) -> tuple[list[dict], dict]:
    replacements = _load_replacement_rows(bundle)
    merged_rows: list[dict] = []
    manifest = {"canonical_sources": {}, "replacement_sources": {}, "row_counts": {}}

    for experiment, source_csv in CANONICAL_SOURCES.items():
        old_rows = _rows_for_experiment(_load_rows(source_csv), experiment)
        kept_old_rows = [
            row for row in old_rows if row.get("algo") not in {"ELVIRA", "LVIRA"}
        ]
        new_rows = replacements[experiment]
        merged_rows.extend(kept_old_rows)
        merged_rows.extend(new_rows)
        manifest["canonical_sources"][experiment] = str(source_csv)
        manifest["replacement_sources"][experiment] = str(
            (bundle / "csv" / "by_experiment" / f"{experiment}.csv").resolve()
        )
        manifest["row_counts"][experiment] = {
            "old_total": len(old_rows),
            "old_kept": len(kept_old_rows),
            "new_added": len(new_rows),
        }

    return merged_rows, manifest


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge explicit ELVIRA/LVIRA rerun rows into the Section 6 CSV."
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Sharded rerun bundle root containing csv/by_experiment/*.csv",
    )
    parser.add_argument(
        "--out-bundle",
        required=True,
        type=Path,
        help="Output bundle root for merged CSV and refreshed plots",
    )
    parser.add_argument(
        "--generate-maintext",
        action="store_true",
        help="Also regenerate the Section 6 main-text figures from the merged CSV.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Pass --notify through to summary plot generation.",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    out_bundle = args.out_bundle.resolve()
    out_bundle.mkdir(parents=True, exist_ok=True)

    merged_rows, manifest = _merge_rows(bundle)
    merged_csv = out_bundle / "csv" / "section6_plotrefresh_merged.csv"
    _write_csv(merged_rows, merged_csv)

    summary_dir = out_bundle / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_cmd = [
        sys.executable,
        "-m",
        "experiments.static.run_perturbed_sweeps",
        "--plot_from_csv",
        str(merged_csv),
        "--summary_dir",
        str(summary_dir),
    ]
    if args.notify:
        plot_cmd.append("--notify")
    _run(plot_cmd)

    if args.generate_maintext:
        maintext_dir = out_bundle / "maintext"
        _run(
            [
                sys.executable,
                "-m",
                "experiments.static.generate_section6_maintext_figures",
                "--csv",
                str(merged_csv),
                "--out_dir",
                str(maintext_dir),
            ]
        )

    manifest["bundle"] = str(bundle)
    manifest["merged_csv"] = str(merged_csv)
    manifest["summary_dir"] = str(summary_dir)
    manifest["generate_maintext"] = args.generate_maintext
    (out_bundle / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
