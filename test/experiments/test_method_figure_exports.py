import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from experiments.static.generate_staged_reconstruction_figure import add_fraction_key


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
