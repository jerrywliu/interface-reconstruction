"""Materialize a conservation-smoke selection from a sealed final release.

The checked-in selector specification contains scientific coordinates only.
This helper resolves those coordinates through the release run inventory,
verifies every input consumed by the conservation analyzer against the
release-wide SHA-256 ledger, and writes a release-bound selection file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


DEFAULT_SPECIFICATION = Path(__file__).with_name("conservation_smoke_selection.json")
REQUIRED_RELEASE_INPUTS = (
    "submission_config.resolved.json",
    "diagnostics/run_inventory.csv",
)
REQUIRED_RUN_INPUTS = (
    "run_manifest.json",
    "vtk/mesh.vtk",
    "metrics/case_geometry.jsonl",
    "metrics/case_metrics.csv",
    "metrics/cell_metrics.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _decimal(value: Any, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid decimal {field}: {value!r}") from error


def _selector_key(row: Mapping[str, Any]) -> tuple[str, str, Decimal, Decimal, int]:
    return (
        str(row["experiment"]),
        str(row["algo"]),
        _decimal(row["resolution"], "resolution"),
        _decimal(row["wiggle"], "wiggle"),
        int(row["seed"]),
    )


def _read_sha256_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"Invalid SHA256SUMS line {line_number}")
        digest = line[:64].lower()
        relative = line[66:]
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid SHA-256 digest on line {line_number}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            raise ValueError(
                f"Unsafe SHA256SUMS path on line {line_number}: {relative!r}"
            )
        if relative in records:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        records[relative] = digest
    if not records:
        raise ValueError("SHA256SUMS is empty")
    return records


def _resolve_release_path(release_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"Unsafe release-relative path: {relative!r}")
    path = (release_root / Path(*pure.parts)).resolve()
    if not _is_within(path, release_root):
        raise ValueError(f"Release path escapes the release root: {relative!r}")
    return path


def _verify_ledger_file(
    release_root: Path,
    records: Mapping[str, str],
    relative: str,
) -> str:
    expected = records.get(relative)
    if expected is None:
        raise ValueError(f"Release input is absent from SHA256SUMS: {relative}")
    path = _resolve_release_path(release_root, relative)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Release input is missing or not a regular file: {relative}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for release input: {relative}")
    return actual


def _case_exists(path: Path, case_index: int) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        return any(
            int(row["case_index"]) == case_index for row in csv.DictReader(stream)
        )


def _geometry_exists(path: Path, case_index: int) -> bool:
    with path.open(encoding="utf-8") as stream:
        return any(int(json.loads(line)["case_index"]) == case_index for line in stream)


def materialize_selection(
    release_root: Path,
    specification_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    release_root = release_root.resolve(strict=True)
    if not release_root.is_dir():
        raise ValueError(f"Final release is not a directory: {release_root}")
    output_path = output_path.resolve()
    if _is_within(output_path, release_root):
        raise ValueError("Conservation selection output must be outside FINAL_ROOT")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Selection output already exists: {output_path}")

    manifest_path = release_root / "SHA256SUMS"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Sealed release SHA256SUMS is missing: {manifest_path}")
    manifest_records = _read_sha256_manifest(manifest_path)
    verified_inputs = {
        relative: _verify_ledger_file(release_root, manifest_records, relative)
        for relative in REQUIRED_RELEASE_INPUTS
    }

    resolved_config = json.loads(
        (release_root / "submission_config.resolved.json").read_text(encoding="utf-8")
    )
    source_commit = str(resolved_config["source"]["target_commit"])
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    selectors = specification.get("cases")
    if not isinstance(selectors, list) or not selectors:
        raise ValueError(
            "Conservation selector specification must contain nonempty cases"
        )

    with (release_root / "diagnostics/run_inventory.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        inventory_rows = list(csv.DictReader(stream))
    inventory: dict[tuple[str, str, Decimal, Decimal, int], list[dict[str, str]]] = {}
    for row in inventory_rows:
        inventory.setdefault(_selector_key(row), []).append(row)

    cases = []
    for selector in selectors:
        key = _selector_key(selector)
        candidates = inventory.get(key, [])
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one final-release run for {key}, found {len(candidates)}"
            )
        inventory_row = candidates[0]
        if inventory_row.get("source_commit") != source_commit:
            raise ValueError(f"Run source commit does not match final release: {key}")
        bundle_relative = inventory_row["run_bundle"]
        bundle = _resolve_release_path(release_root, bundle_relative)
        if not bundle.is_dir() or bundle.is_symlink():
            raise ValueError(f"Final-release run bundle is missing: {bundle_relative}")

        for suffix in REQUIRED_RUN_INPUTS:
            relative = f"{bundle_relative.rstrip('/')}/{suffix}"
            verified_inputs[relative] = _verify_ledger_file(
                release_root, manifest_records, relative
            )

        case_index = int(selector["case_index"])
        if case_index < 0:
            raise ValueError(f"case_index must be nonnegative: {case_index}")
        if not _case_exists(bundle / "metrics/case_metrics.csv", case_index):
            raise ValueError(
                f"case_index {case_index} is absent from {bundle_relative}"
            )
        if not _geometry_exists(bundle / "metrics/case_geometry.jsonl", case_index):
            raise ValueError(
                f"case geometry {case_index} is absent from {bundle_relative}"
            )

        cases.append(
            {
                "experiment": selector["experiment"],
                "algo": selector["algo"],
                "resolution": float(_decimal(selector["resolution"], "resolution")),
                "wiggle": float(_decimal(selector["wiggle"], "wiggle")),
                "seed": int(selector["seed"]),
                "case_index": case_index,
                "run_root": str(bundle),
                **({"stage": selector["stage"]} if "stage" in selector else {}),
            }
        )

    payload = {
        "schema_version": 2,
        "release_binding": {
            "release_name": release_root.name,
            "source_commit": source_commit,
            "sha256_manifest": "SHA256SUMS",
            "sha256_manifest_digest": _sha256(manifest_path),
            "verified_input_digests": dict(sorted(verified_inputs.items())),
        },
        "selector_specification": specification_path.name,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = materialize_selection(
        args.release_root,
        args.specification,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        f"Materialized {len(payload['cases'])} conservation cases at {args.output} "
        f"for SHA256SUMS {payload['release_binding']['sha256_manifest_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
