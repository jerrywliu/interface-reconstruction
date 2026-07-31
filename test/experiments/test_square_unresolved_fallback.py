import csv
import math

from experiments.static import squares


def test_square_area_metric_uses_active_lvira_fallback_cells(tmp_path):
    output_dir = tmp_path / "square_lvira_fallback"

    area_errors, facet_gaps, hausdorff_distances = squares.main(
        config_setting="static/square",
        resolution=0.5,
        facet_algo="linear",
        save_name=str(output_dir),
        num_squares=25,
        case_indices=[3],
        mesh_type="perturbed_quads",
        perturb_wiggle=0.2,
        perturb_seed=0,
        perturb_fix_boundary=True,
        plic_fallback="LVIRA",
        corner_behavior_profile="pre_f8_corner",
    )

    assert len(area_errors) == len(facet_gaps) == len(hausdorff_distances) == 1
    assert all(
        math.isfinite(value)
        for value in (area_errors[0], facet_gaps[0], hausdorff_distances[0])
    )

    metrics_dir = output_dir / "metrics"
    with (metrics_dir / "unresolved_plic_fallbacks.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        fallback_rows = list(csv.DictReader(stream))
    with (metrics_dir / "cell_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        cell_rows = list(csv.DictReader(stream))

    assert len(fallback_rows) == 2
    assert {row["policy"] for row in fallback_rows} == {"LVIRA"}
    assert len(cell_rows) == 30
    assert all(row["final_facet_class"] != "missing" for row in cell_rows)
