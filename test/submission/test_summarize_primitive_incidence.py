import csv
import json
from pathlib import Path

import pytest

from submission.summarize_primitive_incidence import (
    PrimitiveIncidenceError,
    classify_primitive,
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


def _geometry(left, right):
    return json.dumps(
        {
            "left_branch": {"class": left},
            "right_branch": {"class": right},
        }
    )


def _build_release(root: Path):
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True)
    (root / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "planned_run_count": 1,
                "successful_run_count": 1,
                "failure_count": 0,
            }
        ),
        encoding="utf-8",
    )
    context = {
        "experiment": "zalesak",
        "algo": "circular+corner",
        "resolution": "0.64",
        "wiggle": "0.1",
        "seed": "0",
        "save_name": "run-z",
    }
    _write_csv(
        diagnostics / "case_metrics.csv",
        CONTEXT_FIELDS + ["case_index", "num_mixed_cells"],
        [{**context, "case_index": "0", "num_mixed_cells": "5"}],
    )
    cell_fields = CONTEXT_FIELDS + [
        "case_index",
        "cell_id",
        "final_facet_class",
        "construction_path",
        "fallback_policy",
        "facet_geometry_json",
    ]
    cells = [
        {**context, "case_index": "0", "cell_id": "1,1", "final_facet_class": "linear", "construction_path": "direct_fit", "fallback_policy": "", "facet_geometry_json": "{}"},
        {**context, "case_index": "0", "cell_id": "1,2", "final_facet_class": "circular", "construction_path": "direct_fit", "fallback_policy": "", "facet_geometry_json": "{}"},
        {**context, "case_index": "0", "cell_id": "1,3", "final_facet_class": "linear_corner", "construction_path": "direct_fit", "fallback_policy": "", "facet_geometry_json": "{}"},
        {**context, "case_index": "0", "cell_id": "1,4", "final_facet_class": "curved_corner", "construction_path": "direct_fit", "fallback_policy": "", "facet_geometry_json": _geometry("linear", "circular")},
        {**context, "case_index": "0", "cell_id": "1,5", "final_facet_class": "linear", "construction_path": "plic_fallback", "fallback_policy": "LVIRA", "facet_geometry_json": "{}"},
    ]
    _write_csv(diagnostics / "cell_metrics.csv", cell_fields, cells)
    event_fields = CONTEXT_FIELDS + [
        "case_index",
        "member_cells_json",
        "event_kind",
        "fallback_policy",
    ]
    _write_csv(
        diagnostics / "merge_events.csv",
        event_fields,
        [
            {**context, "case_index": "0", "member_cells_json": "[[1,1]]", "event_kind": "local_linear_fallback", "fallback_policy": ""},
            {**context, "case_index": "0", "member_cells_json": "[[1,5]]", "event_kind": "plic_fallback", "fallback_policy": "LVIRA"},
        ],
    )


def test_classifies_curved_corner_branch_pairs_without_order_dependence():
    row = {
        "final_facet_class": "curved_corner",
        "facet_geometry_json": _geometry("circular", "linear"),
    }
    assert classify_primitive(row) == "line_arc"
    row["facet_geometry_json"] = _geometry("circular", "circular")
    assert classify_primitive(row) == "arc_arc"


def test_summary_closes_primitives_and_overlays_fallbacks(tmp_path):
    release = tmp_path / "release"
    _build_release(release)
    result = summarize_release(
        release,
        methods={"zalesak": "circular+corner"},
    )
    row = result["payload"]["benchmark_rows"][0]
    assert row["mixed_cells"] == 5
    assert row["linear_cells"] == 2
    assert row["circular_cells"] == 1
    assert row["line_line_cells"] == 1
    assert row["line_arc_cells"] == 1
    assert row["arc_arc_cells"] == 0
    assert row["local_linear_fallback_cells"] == 1
    assert row["LVIRA_fallback_cells"] == 1
    assert result["paths"]["case_csv"].is_file()


def test_unsupported_curved_corner_pair_is_rejected():
    row = {
        "final_facet_class": "curved_corner",
        "facet_geometry_json": _geometry("linear", "linear"),
    }
    with pytest.raises(PrimitiveIncidenceError, match="Unsupported curved-corner"):
        classify_primitive(row)


def test_plic_cell_and_event_ledgers_must_agree(tmp_path):
    release = tmp_path / "release"
    _build_release(release)
    path = release / "diagnostics" / "merge_events.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    rows.pop()
    _write_csv(path, fields, rows)

    with pytest.raises(PrimitiveIncidenceError, match="disagree"):
        summarize_release(release, methods={"zalesak": "circular+corner"})
