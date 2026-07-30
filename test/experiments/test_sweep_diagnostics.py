import csv
import json
from pathlib import Path

import pytest

from experiments.static.sweep_diagnostics import (
    DiagnosticBundleError,
    archive_run_bundle,
    consolidate_run_diagnostics,
    prepare_diagnostic_bundle,
    remove_archived_run_source,
)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_run_bundle(root):
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True)
    manifest = {
        "source_commit": "abc123",
        "source_branch": "main",
        "parameters": {
            "plic_fallback": "LVIRA",
            "rescue_profile": "no_curved_corner_rescues",
            "corner_behavior_profile": "pre_f8_corner",
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (metrics_dir / "case_geometry.jsonl").write_text(
        json.dumps({"case_index": 3, "center": [1.0, 2.0]}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        metrics_dir / "case_metrics.csv",
        ["case_index", "hausdorff"],
        [{"case_index": 3, "hausdorff": 1.0e-8}],
    )
    _write_csv(
        metrics_dir / "cell_metrics.csv",
        ["case_index", "cell_id", "final_facet_class"],
        [
            {
                "case_index": 3,
                "cell_id": "1,2",
                "final_facet_class": "curved_corner",
            }
        ],
    )
    _write_csv(
        metrics_dir / "merge_events.csv",
        ["case_index", "event_order", "stage"],
        [{"case_index": 3, "event_order": 0, "stage": "fit"}],
    )
    _write_csv(
        metrics_dir / "unresolved_plic_fallbacks.csv",
        ["case_index", "merge_id", "policy"],
        [{"case_index": 3, "merge_id": 7, "policy": "LVIRA"}],
    )


def test_consolidates_run_diagnostics_with_run_context(tmp_path):
    run_dir = tmp_path / "plots" / "run-a"
    output_dir = tmp_path / "release" / "diagnostics"
    _make_run_bundle(run_dir)
    prepare_diagnostic_bundle(output_dir)

    counts = consolidate_run_diagnostics(
        run_dir,
        output_dir,
        {
            "experiment": "zalesak",
            "algo": "circular+corner",
            "resolution": 1.5,
            "wiggle": 0.3,
            "seed": 0,
            "save_name": "run-a",
        },
    )

    assert counts == {
        "case_geometry_rows": 1,
        "case_metrics_rows": 1,
        "cell_metrics_rows": 1,
        "merge_events_rows": 1,
        "unresolved_plic_fallbacks_rows": 1,
    }

    with (output_dir / "case_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["experiment"] == "zalesak"
    assert row["algo"] == "circular+corner"
    assert row["source_commit"] == "abc123"
    assert row["plic_fallback"] == "LVIRA"
    assert row["corner_behavior_profile"] == "pre_f8_corner"
    assert row["case_index"] == "3"

    geometry = json.loads(
        (output_dir / "case_geometry.jsonl").read_text(encoding="utf-8")
    )
    assert geometry["save_name"] == "run-a"
    assert geometry["case_index"] == 3

    with (output_dir / "run_inventory.csv").open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        inventory = next(csv.DictReader(stream))
    assert inventory["case_metrics_rows"] == "1"
    assert inventory["cell_metrics_rows"] == "1"


def test_missing_required_diagnostic_fails(tmp_path):
    run_dir = tmp_path / "plots" / "incomplete"
    run_dir.mkdir(parents=True)
    output_dir = prepare_diagnostic_bundle(tmp_path / "diagnostics")

    with pytest.raises(DiagnosticBundleError, match="missing required diagnostics"):
        consolidate_run_diagnostics(
            run_dir,
            output_dir,
            {
                "experiment": "lines",
                "algo": "linear",
                "resolution": 0.32,
                "wiggle": 0.0,
                "seed": 0,
                "save_name": "incomplete",
            },
        )


def test_consolidation_replaces_historical_missing_row_with_plic_fallback(tmp_path):
    run_dir = tmp_path / "plots" / "run-a"
    output_dir = tmp_path / "release" / "diagnostics"
    _make_run_bundle(run_dir)
    metrics_dir = run_dir / "metrics"
    case_fields = [
        "case_index",
        "num_mixed_cells",
        "num_plic_fallback_cells",
        "num_final_linear_cells",
        "num_final_missing_cells",
        "fraction_plic_fallback_cells",
        "fraction_final_linear_cells",
        "hausdorff",
    ]
    _write_csv(
        metrics_dir / "case_metrics.csv",
        case_fields,
        [
            {
                "case_index": 3,
                "num_mixed_cells": 2,
                "num_plic_fallback_cells": 1,
                "num_final_linear_cells": 1,
                "num_final_missing_cells": 1,
                "fraction_plic_fallback_cells": 0.5,
                "fraction_final_linear_cells": 0.5,
                "hausdorff": 1.0e-8,
            }
        ],
    )
    cell_fields = [
        "case_index",
        "cell_id",
        "merge_id",
        "is_merged",
        "final_facet_class",
        "construction_path",
        "fallback_policy",
        "event_count",
    ]
    _write_csv(
        metrics_dir / "cell_metrics.csv",
        cell_fields,
        [
            {
                "case_index": 3,
                "cell_id": "1,2",
                "merge_id": 7,
                "is_merged": 0,
                "final_facet_class": "missing",
                "construction_path": "direct_fit",
                "fallback_policy": "",
                "event_count": 0,
            },
            {
                "case_index": 3,
                "cell_id": "1,2",
                "merge_id": 8,
                "is_merged": 0,
                "final_facet_class": "linear",
                "construction_path": "plic_fallback",
                "fallback_policy": "LVIRA",
                "event_count": 1,
            },
        ],
    )
    prepare_diagnostic_bundle(output_dir)

    counts = consolidate_run_diagnostics(
        run_dir,
        output_dir,
        {
            "experiment": "zalesak",
            "algo": "circular+corner",
            "resolution": 1.5,
            "wiggle": 0.3,
            "seed": 0,
            "save_name": "run-a",
        },
    )

    assert counts["cell_metrics_rows"] == 1
    with (output_dir / "cell_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        cell = next(csv.DictReader(stream))
    assert cell["final_facet_class"] == "linear"
    assert cell["construction_path"] == "plic_fallback"
    assert cell["fallback_policy"] == "LVIRA"

    with (output_dir / "case_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        case = next(csv.DictReader(stream))
    assert case["num_mixed_cells"] == "1"
    assert case["num_plic_fallback_cells"] == "1"
    assert case["num_final_linear_cells"] == "1"
    assert case["num_final_missing_cells"] == "0"
    assert case["fraction_plic_fallback_cells"] == "1.0"
    assert case["fraction_final_linear_cells"] == "1.0"


def test_archived_run_bundle_is_self_contained_and_inventory_is_release_relative(
    tmp_path,
):
    source = tmp_path / "plots" / "run-a"
    release_root = tmp_path / "release"
    diagnostics = prepare_diagnostic_bundle(release_root / "diagnostics")
    _make_run_bundle(source)
    preview = source / "plt" / "areas" / "0.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"review-only raster preview")

    archived = archive_run_bundle(source, release_root / "raw_runs")
    consolidate_run_diagnostics(
        archived,
        diagnostics,
        {
            "experiment": "zalesak",
            "algo": "circular+corner",
            "resolution": 1.5,
            "wiggle": 0.3,
            "seed": 0,
            "save_name": "run-a",
        },
        inventory_root=release_root,
    )

    assert archived == release_root / "raw_runs" / "run-a"
    assert (archived / "run_manifest.json").is_file()
    assert archived.stat().st_mode & 0o222 == 0
    assert (archived / "run_manifest.json").stat().st_mode & 0o222 == 0
    assert not (archived / "plt").exists()

    archived_manifest = json.loads(
        (archived / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert archived_manifest["source_commit"] == "abc123"

    with (diagnostics / "run_inventory.csv").open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        inventory = next(csv.DictReader(stream))
    assert inventory["run_bundle"] == "raw_runs/run-a"
    assert (release_root / inventory["run_bundle"]).is_dir()

    remove_archived_run_source(source, tmp_path / "plots", archived)
    assert not source.exists()
    assert (archived / "run_manifest.json").is_file()


def test_remove_archived_run_source_refuses_paths_outside_source_root(tmp_path):
    source = tmp_path / "elsewhere" / "run-a"
    archived = tmp_path / "release" / "raw_runs" / "run-a"
    _make_run_bundle(source)
    archive_run_bundle(source, archived.parent)

    with pytest.raises(DiagnosticBundleError, match="outside temporary source root"):
        remove_archived_run_source(source, tmp_path / "plots", archived)

    assert source.is_dir()


def test_archive_collision_preserves_existing_release_bundle(tmp_path):
    source = tmp_path / "plots" / "run-a"
    destination = tmp_path / "release" / "raw_runs" / "run-a"
    _make_run_bundle(source)
    destination.mkdir(parents=True)
    sentinel = destination / "existing.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(DiagnosticBundleError, match="already exists"):
        archive_run_bundle(source, destination.parent)

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert not any(destination.parent.glob(".run-a.copying-*"))
