import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from experiments.static.generate_staged_reconstruction_figure import add_fraction_key
from experiments.static.run_perturbed_sweeps import DISPLAY_LABELS


def test_publication_method_labels_use_approved_variant_names():
    assert DISPLAY_LABELS["safe_linear"] == "Ours (linear, per-cell)"
    assert DISPLAY_LABELS["linear"] == "Ours (linear, graph-coordinated)"
    assert DISPLAY_LABELS["safe_circle"] == "Ours (circular, per-cell)"
    assert DISPLAY_LABELS["circular"] == "Ours (circular, graph-coordinated)"
    assert "graph-coordinated" in DISPLAY_LABELS["linear+corner"]
    assert "graph-coordinated" in DISPLAY_LABELS["circular+corner"]


def test_staged_fraction_key_uses_vector_patches():
    figure, axis = plt.subplots()
    color_map = LinearSegmentedColormap.from_list(
        "test_fraction", ["#ffffff", "#000000"]
    )

    add_fraction_key(axis, color_map)
    key_axis = axis.child_axes[-1]

    assert len(key_axis.images) == 0
    assert len(key_axis.patches) == 64
    plt.close(figure)
