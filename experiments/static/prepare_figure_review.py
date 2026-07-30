#!/usr/bin/env python3
"""Prepare merged metrics and plot artifacts for a Section 6 figure review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


KEY_FIELDS = (
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "metric_key",
)


def _metric_key(row: dict[str, str]) -> tuple:
    return (
        row["experiment"],
        row["algo"],
        float(row["resolution"]),
        float(row["wiggle"]),
        int(row["seed"]),
        row["metric_key"],
    )


def _setting_key(row: dict[str, str]) -> tuple:
    return _metric_key(row)[:-1]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists (pass --force): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def prepare_review(
    *,
    run_root: Path,
    baseline_csv: Path,
    current_plots_root: Path,
    archive_plots_root: Path,
    force: bool,
) -> Path:
    current_csv = run_root / "perturbed_sweep.csv"
    review_root = run_root / "figure_review"
    _prepare_output_dir(review_root, force=force)

    baseline_fields, baseline_rows = _read_csv(baseline_csv)
    _, current_rows = _read_csv(current_csv)
    baseline_by_key = {_metric_key(row): row for row in baseline_rows}
    current_by_key = {_metric_key(row): row for row in current_rows}
    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError("Baseline CSV contains duplicate metric keys")
    if len(current_by_key) != len(current_rows):
        raise ValueError("Current CSV contains duplicate metric keys")

    missing_keys = sorted(set(current_by_key) - set(baseline_by_key))
    if missing_keys:
        raise ValueError(
            f"Current CSV contains {len(missing_keys)} keys absent from the baseline"
        )

    merged_rows = []
    for baseline_row in baseline_rows:
        key = _metric_key(baseline_row)
        current_row = current_by_key.get(key)
        if current_row is None:
            merged_rows.append(baseline_row)
            continue
        replacement = dict(baseline_row)
        replacement["metric_value"] = current_row["metric_value"]
        merged_rows.append(replacement)

    merged_csv = review_root / "current_run_section6_merged.csv"
    with merged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=baseline_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    current_sources: dict[tuple, str] = {}
    for row in current_rows:
        setting = _setting_key(row)
        source_name = row["save_name"]
        previous = current_sources.setdefault(setting, source_name)
        if previous != source_name:
            raise ValueError(f"Multiple source directories for current setting: {setting}")

    canonical_names: dict[tuple, str] = {}
    for row in baseline_rows:
        setting = _setting_key(row)
        canonical_name = row["save_name"]
        previous = canonical_names.setdefault(setting, canonical_name)
        if previous != canonical_name:
            raise ValueError(f"Multiple canonical directories for setting: {setting}")

    plots_union = review_root / "plots_union"
    plots_union.mkdir()
    source_counts = {"current": 0, "archive": 0}
    missing_artifacts = []
    for setting, canonical_name in sorted(canonical_names.items()):
        if setting in current_sources:
            source = current_plots_root / current_sources[setting]
            source_kind = "current"
        else:
            source = archive_plots_root / canonical_name
            source_kind = "archive"
        if not source.is_dir():
            missing_artifacts.append(str(source))
            continue
        (plots_union / canonical_name).symlink_to(source.resolve(), target_is_directory=True)
        source_counts[source_kind] += 1

    if missing_artifacts:
        raise FileNotFoundError(
            "Missing run artifacts:\n" + "\n".join(missing_artifacts)
        )

    manifest = {
        "run_root": str(run_root),
        "current_csv": str(current_csv),
        "baseline_csv": str(baseline_csv),
        "merged_csv": str(merged_csv),
        "key_fields": list(KEY_FIELDS),
        "baseline_rows": len(baseline_rows),
        "current_rows": len(current_rows),
        "replaced_rows": len(current_by_key),
        "merged_rows": len(merged_rows),
        "plots_union": str(plots_union),
        "symlink_sources_added": source_counts,
        "missing_run_artifacts": missing_artifacts,
    }
    manifest_path = review_root / "review_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--baseline_csv", type=Path, required=True)
    parser.add_argument("--current_plots_root", type=Path, required=True)
    parser.add_argument("--archive_plots_root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = prepare_review(
        run_root=args.run_root.resolve(),
        baseline_csv=args.baseline_csv.resolve(),
        current_plots_root=args.current_plots_root.resolve(),
        archive_plots_root=args.archive_plots_root.resolve(),
        force=args.force,
    )
    print(manifest)


if __name__ == "__main__":
    main()
