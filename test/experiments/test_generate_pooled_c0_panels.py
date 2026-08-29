import csv
import json

from experiments.static.generate_pooled_c0_panels import load_c0_case_index


def test_load_c0_case_index_combines_sealed_and_joint_rows(tmp_path):
    sealed = tmp_path / "case_metrics.csv"
    with sealed.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "experiment",
                "algo",
                "resolution",
                "wiggle",
                "hausdorff",
                "facet_gap",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment": "ellipses",
                "algo": "linear",
                "resolution": 0.5,
                "wiggle": 0.1,
                "hausdorff": 1.0,
                "facet_gap": 2.0,
            }
        )
    run = tmp_path / "appendix_b5_joint_c0_20260814_perturb_sweep_ellipses_x"
    (run / "metrics").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {"parameters": {"resolution": 0.5, "perturb_wiggle": 0.1}}
        )
    )
    with (run / "metrics/case_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["hausdorff", "facet_gap"]
        )
        writer.writeheader()
        writer.writerow({"hausdorff": 0.5, "facet_gap": 1e-12})

    data = load_c0_case_index(sealed, tmp_path)
    assert data["ellipses"]["linear"]["hausdorff"][0.5][0.1]["value"] == [1.0]
    assert data["ellipses"]["linear+C0"]["facet_gap"][0.5][0.1]["value"] == [1e-12]
