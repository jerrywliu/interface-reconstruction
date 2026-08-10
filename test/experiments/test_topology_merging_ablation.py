import csv
from pathlib import Path
from types import SimpleNamespace

from experiments.submission.run_topology_merging_ablation import (
    DISPLAY_LABELS,
    _build_specs,
    _collect_cases,
    _summaries,
)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_submission_labels_are_scoped_and_explicit():
    assert DISPLAY_LABELS == {
        "safe_circle": "Ours (circular, per-cell)",
        "circular": "Ours (circular, graph-coordinated)",
    }


def test_build_specs_fixes_consistent_fallbacks_and_disables_c0(tmp_path):
    args = SimpleNamespace(
        wiggle=0.1,
        seed=0,
        num_cases=25,
        case_indices="0,6",
        corner_behavior_profile="pre_f8_corner",
        rescue_profile="exact_linear_support_only",
    )

    specs = _build_specs(args, [0.64], tmp_path / "ablation")

    assert [spec["algo"] for spec in specs] == ["safe_circle", "circular"]
    for spec in specs:
        command = spec["cmd"]
        assert command[command.index("--plic_fallback") + 1] == "LVIRA"
        assert command[command.index("--arc_failure_fallback") + 1] == "local_linear"
        assert command[command.index("--do_c0") + 1] == "0"


def test_collect_cases_measures_both_fallback_families(tmp_path):
    plot_dir = tmp_path / "plots" / "run"
    metrics_dir = plot_dir / "metrics"
    _write_csv(
        metrics_dir / "case_metrics.csv",
        [
            "case_index",
            "hausdorff",
            "facet_gap",
            "area_error",
            "num_mixed_cells",
            "num_merged_cells",
        ],
        [
            {
                "case_index": 3,
                "hausdorff": 0.2,
                "facet_gap": 0.01,
                "area_error": 0.001,
                "num_mixed_cells": 4,
                "num_merged_cells": 0,
            }
        ],
    )
    _write_csv(
        metrics_dir / "cell_metrics.csv",
        ["case_index", "fallback_policy", "final_facet_name"],
        [
            {"case_index": 3, "fallback_policy": "LVIRA", "final_facet_name": "LVIRA"},
            {"case_index": 3, "fallback_policy": "", "final_facet_name": "default_linear"},
            {"case_index": 3, "fallback_policy": "", "final_facet_name": "arc"},
            {"case_index": 3, "fallback_policy": "", "final_facet_name": "linear"},
        ],
    )
    _write_csv(
        metrics_dir / "merge_events.csv",
        ["case_index", "event_kind"],
        [{"case_index": 3, "event_kind": "local_linear_fallback"}],
    )
    spec = {
        "algo": "safe_circle",
        "display_label": DISPLAY_LABELS["safe_circle"],
        "resolution": 0.64,
        "cells_per_side": 64,
        "wiggle": 0.1,
        "seed": 0,
        "save_name": "run",
        "cmd": [],
        "plot_dir": str(plot_dir),
    }

    rows = _collect_cases(spec)
    summaries = _summaries(rows)

    assert rows[0]["plic_fallback_cells"] == 1
    assert rows[0]["local_linear_fallback_cells"] == 1
    assert rows[0]["local_linear_fallback_events"] == 1
    assert rows[0]["fraction_plic_fallback_cells"] == 0.25
    assert rows[0]["fraction_local_linear_fallback_cells"] == 0.25
    assert summaries[0]["hausdorff_median"] == 0.2
    assert summaries[0]["fraction_plic_fallback_cells"] == 0.25
