import numpy as np

from experiments.static.generate_pooled_perturbed_panels import _pooled_curves


def test_pooled_curves_combine_all_cases_across_the_hidden_axis():
    data = {
        "linear": {
            "hausdorff": {
                0.32: {
                    0.0: {"value": [1.0, 2.0]},
                    0.1: {"value": [3.0, 4.0]},
                },
                0.64: {
                    0.0: {"value": [5.0, 6.0]},
                    0.1: {"value": [7.0, 8.0]},
                },
            }
        }
    }

    by_resolution = _pooled_curves(data, "hausdorff", "resolution")["linear"]
    assert np.allclose(by_resolution["median"], [2.5, 6.5])
    assert np.allclose(by_resolution["p25"], [1.75, 5.75])
    assert np.allclose(by_resolution["p75"], [3.25, 7.25])

    by_wiggle = _pooled_curves(data, "hausdorff", "wiggle")["linear"]
    assert np.allclose(by_wiggle["median"], [3.5, 5.5])
    assert np.allclose(by_wiggle["p25"], [1.75, 3.75])
    assert np.allclose(by_wiggle["p75"], [5.25, 7.25])
