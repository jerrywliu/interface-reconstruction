import matplotlib.pyplot as plt
import numpy as np

from experiments.static.generate_section6_maintext_figures import (
    RESOLUTION_QUANT_SPECS,
    _add_power_law_fit,
)


def test_circle_maintext_uses_facet_gap():
    assert RESOLUTION_QUANT_SPECS["circles"]["metrics"] == (
        "hausdorff",
        "facet_gap",
    )


def test_power_law_fit_reports_positive_order():
    curves = {
        "circular": {
            "x_values": np.asarray([0.5, 1.0, 1.5]),
            "median": np.asarray([8.0e-3, 1.0e-3, 8.0e-3 / 27.0]),
        }
    }
    fig, ax = plt.subplots()
    order = _add_power_law_fit(ax, curves, algo="circular")
    plt.close(fig)

    assert order is not None
    assert np.isclose(order, 3.0)
