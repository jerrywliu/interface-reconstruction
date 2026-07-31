"""Non-publishing helpers for final-figure trust-boundary tests."""

from pathlib import Path
from typing import Sequence

from submission.final_figure_orchestrator import (
    PUBLISHED_TREE_LEDGER,
    _copy_publication_tree,
    _published_tree_records,
    _rehash_before_publish,
    _verify_frozen_publication_tree,
    validate_published_logical_paths,
)
from submission.final_figure_provenance import (
    atomic_write_json,
    file_sha256,
    make_tree_read_only,
    snapshot_record,
)


def freeze_staging_for_test(
    *,
    staging: Path,
    destination: Path,
    manifest_path: Path,
    candidate_specs: Sequence[object] = (),
) -> str:
    """Copy and seal staging for inspection without reserving or publishing it."""

    if destination.exists():
        raise AssertionError(f"Test freeze destination already exists: {destination}")
    manifest_digest = file_sha256(manifest_path)
    accepted_records = [
        snapshot_record(path, staging, "accepted_artifact")
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    ]
    _rehash_before_publish(staging, manifest_path, manifest_digest, candidate_specs)
    destination.mkdir(parents=True, mode=0o700)
    _copy_publication_tree(staging, destination, accepted_records)
    tree_records = _published_tree_records(destination)
    ledger = destination / PUBLISHED_TREE_LEDGER
    atomic_write_json(ledger, {"schema_version": 1, "files": tree_records})
    ledger_sha256 = file_sha256(ledger)
    validate_published_logical_paths(destination)
    make_tree_read_only(destination)
    _verify_frozen_publication_tree(destination, ledger_sha256=ledger_sha256)
    return ledger_sha256
