from __future__ import annotations

from pathlib import Path

import pytest

from experiments.static.analyze_extended_line_circle import _fit_order, _validate
from experiments.static.run_extended_convergence_smoke import (
    LINEAR_METHODS,
    _build_specs,
    _percentile,
    _summaries,
)


def test_build_specs_can_select_circle_family(tmp_path: Path) -> None:
    specs = _build_specs(tmp_path / "run", (256,), (0.0,), (0,), ("circles",))

    assert len(specs) == len(LINEAR_METHODS)
    assert {spec["experiment"] for spec in specs} == {"circles"}
    assert {spec["method"] for spec in specs} == set(LINEAR_METHODS)


def test_percentile_and_summary_include_iqr() -> None:
    rows = [
        {
            "experiment": "circles",
            "method": "linear",
            "cells_per_side": 256,
            "wiggle": 0.0,
            "hausdorff": value,
            "facet_gap": value,
            "global_relative_phase_area_error": value,
            "max_fitted_component_absolute_residual": value,
            "max_cell_area_relative_residual": value,
            "conservation_complete": True,
        }
        for value in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]

    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25) == 2.0
    summary = _summaries(rows)[0]
    assert summary["hausdorff_q1"] == 2.0
    assert summary["hausdorff_q3"] == 4.0
    assert summary["hausdorff_iqr"] == 2.0


def test_fit_order_recovers_second_order() -> None:
    rows = [
        {"cells_per_side": n, "hausdorff": 4.0 / n**2}
        for n in (256, 300, 512)
    ]

    assert _fit_order(rows, "hausdorff") == pytest.approx(2.0)


def test_validate_accepts_complete_expected_matrix() -> None:
    rows = []
    for method in LINEAR_METHODS:
        for n in (256, 300, 512):
            for wiggle in (0.0, 0.2):
                for case_index in range(5):
                    rows.append(
                        {
                            "experiment": "circles",
                            "method": method,
                            "cells_per_side": str(n),
                            "wiggle": str(wiggle),
                            "case_index": str(case_index),
                            "hausdorff": str(1.0 / n**2),
                            "facet_gap": str(1.0 / n**2),
                        }
                    )

    _validate(rows)
