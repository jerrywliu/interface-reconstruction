import csv
import json
from pathlib import Path

import pytest

from submission.summarize_final_diagnostics import (
    DiagnosticSummaryError,
    summarize_release,
)


CONTEXT_FIELDS = [
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
]


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _context(experiment="zalesak", algo="circular+corner", save_name="run-z"):
    return {
        "experiment": experiment,
        "algo": algo,
        "resolution": "0.64",
        "wiggle": "0.1",
        "seed": "0",
        "save_name": save_name,
    }


def _build_release(root: Path):
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True)
    (root / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "planned_run_count": 2,
                "successful_run_count": 2,
                "failure_count": 0,
            }
        ),
        encoding="utf-8",
    )

    circle = _context("circles", "circular", "run-c")
    zalesak = _context()
    cases = [
        {**circle, "case_index": "0", "num_mixed_cells": "2", "num_merged_components": "1"},
        {**zalesak, "case_index": "0", "num_mixed_cells": "4", "num_merged_components": "1"},
    ]
    _write_csv(
        diagnostics / "case_metrics.csv",
        CONTEXT_FIELDS + ["case_index", "num_mixed_cells", "num_merged_components"],
        cases,
    )
    with (diagnostics / "case_geometry.jsonl").open("w", encoding="utf-8") as stream:
        for context in (circle, zalesak):
            stream.write(json.dumps({**context, "case_index": 0, "geometry_type": context["experiment"]}) + "\n")

    cell_fields = CONTEXT_FIELDS + [
        "case_index",
        "cell_id",
        "merge_id",
        "is_merged",
        "orientation_status",
        "final_facet_class",
        "construction_path",
        "fallback_policy",
    ]
    cells = [
        {**circle, "case_index": "0", "cell_id": "1,1", "merge_id": "10", "is_merged": "1", "orientation_status": "oriented", "final_facet_class": "circular", "construction_path": "merged", "fallback_policy": ""},
        {**circle, "case_index": "0", "cell_id": "1,2", "merge_id": "10", "is_merged": "1", "orientation_status": "oriented", "final_facet_class": "circular", "construction_path": "merged", "fallback_policy": ""},
        {**zalesak, "case_index": "0", "cell_id": "2,1", "merge_id": "20", "is_merged": "1", "orientation_status": "unresolved_or_deadend", "final_facet_class": "linear", "construction_path": "plic_fallback", "fallback_policy": "LVIRA"},
        {**zalesak, "case_index": "0", "cell_id": "2,2", "merge_id": "20", "is_merged": "1", "orientation_status": "unresolved_or_deadend", "final_facet_class": "linear", "construction_path": "plic_fallback", "fallback_policy": "LVIRA"},
        {**zalesak, "case_index": "0", "cell_id": "3,1", "merge_id": "21", "is_merged": "0", "orientation_status": "oriented", "final_facet_class": "linear_corner", "construction_path": "direct_fit", "fallback_policy": ""},
        {**zalesak, "case_index": "0", "cell_id": "3,2", "merge_id": "22", "is_merged": "0", "orientation_status": "oriented", "final_facet_class": "curved_corner", "construction_path": "direct_fit", "fallback_policy": ""},
    ]
    _write_csv(diagnostics / "cell_metrics.csv", cell_fields, cells)

    event_fields = CONTEXT_FIELDS + [
        "case_index",
        "event_order",
        "merge_id",
        "member_cells_json",
        "stage",
        "event_kind",
        "fallback_policy",
        "fallback_reason",
        "facet_class",
        "facet_name",
    ]
    events = [
        {**zalesak, "case_index": "0", "event_order": "1", "merge_id": "20", "member_cells_json": "[[2,1],[2,2]]", "stage": "final_fallback", "event_kind": "plic_fallback", "fallback_policy": "LVIRA", "fallback_reason": "unresolved_orientation", "facet_class": "linear", "facet_name": "LVIRA"},
        {**zalesak, "case_index": "0", "event_order": "2", "merge_id": "21", "member_cells_json": "[[3,1]]", "stage": "linear_corner_rescues", "event_kind": "facet_assignment", "fallback_policy": "", "fallback_reason": "", "facet_class": "linear", "facet_name": "linear_support"},
        {**zalesak, "case_index": "0", "event_order": "3", "merge_id": "22", "member_cells_json": "[[3,2]]", "stage": "curved_corner_transition_rescue", "event_kind": "facet_assignment", "fallback_policy": "", "fallback_reason": "", "facet_class": "curved_corner", "facet_name": "corner"},
    ]
    _write_csv(diagnostics / "merge_events.csv", event_fields, events)

    _write_csv(
        diagnostics / "unresolved_plic_fallbacks.csv",
        CONTEXT_FIELDS + ["case_index", "merge_id", "policy"],
        [{**zalesak, "case_index": "0", "merge_id": "20", "policy": "LVIRA"}],
    )
    return diagnostics


def _overall_rows(result):
    return {
        (row["category"], row["subtype"]): row
        for row in result["payload"]["groups"]
        if row["scope"] == "overall"
    }


def test_summary_counts_cell_weighted_facets_merges_rescues_and_fallbacks(tmp_path):
    release = tmp_path / "release"
    _build_release(release)

    result = summarize_release(release)
    overall = _overall_rows(result)

    assert overall[("final_facet", "circular")]["mixed_cell_count"] == 2
    assert overall[("final_facet", "corner")]["mixed_cell_count"] == 1
    assert overall[("final_facet", "curved_corner")]["mixed_cell_count"] == 1
    assert overall[("merge", "merged")]["mixed_cell_count"] == 4
    assert overall[("merge", "merged")]["component_count"] == 2
    assert overall[("rescue", "exact_linear_support")]["mixed_cell_count"] == 1
    assert overall[("rescue", "curved_corner_transition")]["event_count"] == 1
    assert overall[("orientation", "unresolved_orientation")]["mixed_cell_count"] == 2
    assert overall[("plic_fallback", "LVIRA")]["mixed_cell_count"] == 2
    assert overall[("plic_fallback", "LVIRA")]["fraction_of_mixed_cells"] == pytest.approx(2 / 6)

    for path in result["paths"].values():
        assert path.is_file()
    markdown = result["paths"]["markdown"].read_text(encoding="utf-8")
    assert "2 / 2 runs" in markdown
    assert "exact_linear_support" in markdown
    assert "LVIRA" in markdown


def test_incomplete_release_is_refused_without_writing_outputs(tmp_path):
    release = tmp_path / "release"
    _build_release(release)
    manifest = json.loads((release / "sweep_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "running"
    (release / "sweep_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiagnosticSummaryError, match="incomplete release"):
        summarize_release(release)

    assert not (release / "diagnostic_summary").exists()


def test_missing_required_event_field_fails_clearly(tmp_path):
    release = tmp_path / "release"
    diagnostics = _build_release(release)
    path = diagnostics / "merge_events.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = [field for field in rows[0] if field != "fallback_reason"]
    for row in rows:
        row.pop("fallback_reason")
    _write_csv(path, fieldnames, rows)

    with pytest.raises(DiagnosticSummaryError, match="fallback_reason"):
        summarize_release(release)


def test_ambiguous_rescue_provenance_is_not_guessed(tmp_path):
    release = tmp_path / "release"
    diagnostics = _build_release(release)
    path = diagnostics / "merge_events.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[1]["facet_name"] = "corner_branch_linear"
    _write_csv(path, fieldnames, rows)

    with pytest.raises(DiagnosticSummaryError, match="cannot distinguish"):
        summarize_release(release)


def test_fallback_tables_must_agree(tmp_path):
    release = tmp_path / "release"
    diagnostics = _build_release(release)
    path = diagnostics / "unresolved_plic_fallbacks.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0]["policy"] = "Youngs"
    _write_csv(path, fieldnames, rows)

    with pytest.raises(DiagnosticSummaryError, match="disagrees across"):
        summarize_release(release)


def test_duplicate_cell_rows_are_rejected_even_when_case_count_matches(tmp_path):
    release = tmp_path / "release"
    diagnostics = _build_release(release)
    path = diagnostics / "cell_metrics.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[1]["cell_id"] = rows[0]["cell_id"]
    _write_csv(path, fieldnames, rows)

    with pytest.raises(DiagnosticSummaryError, match="Duplicate mixed-cell row"):
        summarize_release(release)
