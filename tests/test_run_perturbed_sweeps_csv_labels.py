import csv
import sys

from experiments.static import run_perturbed_sweeps
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


def test_plot_from_csv_passes_legacy_keyword_only_when_requested(tmp_path, monkeypatch):
    csv_path = _write_sweep_csv(tmp_path, ["LVIRA"])
    calls = []

    def capture_summary_call(csv_value, output, **kwargs):
        calls.append((csv_value, output, kwargs))
        return {}

    monkeypatch.setattr(
        run_perturbed_sweeps,
        "_generate_summary_plots",
        capture_summary_call,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_perturbed_sweeps.py",
            "--plot_from_csv",
            str(csv_path),
            "--summary_dir",
            str(tmp_path / "plots"),
            "--legacy_lvira_means_elvira",
            "--no-notify",
        ],
    )

    run_perturbed_sweeps.main()

    assert calls == [
        (
            str(csv_path),
            (tmp_path / "plots").resolve(),
            {"legacy_lvira_means_elvira": True},
        )
    ]
