from types import SimpleNamespace

import pytest

from util.reconstruction import (
    IncompleteReconstructionError,
    _validate_active_reconstruction,
)


def _mesh(active_poly, stale_poly=None):
    merged_polys = {4: active_poly}
    if stale_poly is not None:
        merged_polys[3] = stale_poly
    return SimpleNamespace(
        coords_to_merge_id=[[None, 4], [4, None]],
        merged_polys=merged_polys,
    )


def test_complete_reconstruction_ignores_inactive_stale_merge_objects():
    active = object()
    stale = object()

    _validate_active_reconstruction(_mesh(active, stale), [active], [object()])


def test_complete_reconstruction_rejects_dropped_active_component():
    active = object()

    with pytest.raises(
        IncompleteReconstructionError, match=r"missing active merge ids=\[4\]"
    ):
        _validate_active_reconstruction(_mesh(active), [], [])


def test_complete_reconstruction_rejects_missing_facet():
    active = object()

    with pytest.raises(IncompleteReconstructionError, match="missing facets"):
        _validate_active_reconstruction(_mesh(active), [active], [None])
