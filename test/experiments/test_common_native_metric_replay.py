import csv

from experiments.baselines.run_common_native_metric_replay import (
    _label,
    _supplemental_case_rows,
)


def test_supplemental_case_rows_fill_common_replay_provenance(tmp_path):
    path = tmp_path / "case_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("method", "variant", "source_run"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "Ours",
                "variant": "graph-coordinated circular",
                "source_run": "sealed-ellipse-run",
            }
        )

    rows = _supplemental_case_rows((path, tmp_path / "missing.csv"))

    assert len(rows) == 1
    assert rows[0]["geometry_file"] == "sealed-ellipse-run"
    assert rows[0]["metric_status"] == "complete"


def test_graph_circular_label_is_explicit():
    assert _label("Ours", "graph-coordinated circular") == "Ours (circular)"
