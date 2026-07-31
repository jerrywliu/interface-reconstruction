import csv

from experiments.static.run_perturbed_sweeps import _load_sweep_rows


def _write_sweep_csv(tmp_path, algorithms):
    csv_path = tmp_path / "sweep.csv"
    fieldnames = ["experiment", "algo", "metric_key", "metric_value"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm in algorithms:
            writer.writerow(
                {
                    "experiment": "lines",
                    "algo": algorithm,
                    "metric_key": "hausdorff_median",
                    "metric_value": "0.0",
                }
            )
    return csv_path


def test_load_sweep_rows_preserves_lvira_only_input(tmp_path):
    csv_path = _write_sweep_csv(tmp_path, ["LVIRA", "LVIRA"])

    rows = _load_sweep_rows(csv_path)

    assert [row["algo"] for row in rows] == ["LVIRA", "LVIRA"]


def test_load_sweep_rows_preserves_elvira_only_input(tmp_path):
    csv_path = _write_sweep_csv(tmp_path, ["ELVIRA", "ELVIRA"])

    rows = _load_sweep_rows(csv_path)

    assert [row["algo"] for row in rows] == ["ELVIRA", "ELVIRA"]


def test_load_sweep_rows_preserves_mixed_input(tmp_path):
    csv_path = _write_sweep_csv(tmp_path, ["ELVIRA", "LVIRA", "Youngs"])

    rows = _load_sweep_rows(csv_path)

    assert [row["algo"] for row in rows] == ["ELVIRA", "LVIRA", "Youngs"]


def test_load_sweep_rows_supports_explicit_legacy_interpretation(tmp_path):
    csv_path = _write_sweep_csv(tmp_path, ["LVIRA", "Youngs"])

    rows = _load_sweep_rows(csv_path, legacy_lvira_means_elvira=True)

    assert [row["algo"] for row in rows] == ["ELVIRA", "Youngs"]
