import numpy as np

from experiments.static.ellipses import sample_arc_points, sample_reconstructed_facet_points
from main.structs.facets.circular_facet import ArcFacet


def _hausdorff_points(points1, points2):
    d1 = max(np.min(np.linalg.norm(points2 - p, axis=1)) for p in points1)
    d2 = max(np.min(np.linalg.norm(points1 - p, axis=1)) for p in points2)
    return max(d1, d2)


def test_sample_reconstructed_facet_points_respects_signed_arc_orientation():
    facet = ArcFacet([0.0, 0.0], -10.0, [0.0, 10.0], [10.0, 0.0])

    correct = sample_reconstructed_facet_points(facet, 128)
    native = np.asarray(facet.sample(128))
    naive = sample_arc_points(
        facet.center,
        abs(facet.radius),
        facet.pLeft,
        facet.pRight,
        128,
    )

    assert _hausdorff_points(correct, native) < 1e-12
    assert _hausdorff_points(naive, native) > 1.0
