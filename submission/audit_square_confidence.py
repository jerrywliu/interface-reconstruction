#!/usr/bin/env python3
"""Audit the corrected square active-partition confidence sweep.

The audit is intentionally read-only with respect to both result roots. It writes a
case-level comparison CSV, a compact Markdown report, and a vector witness figure
to an explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


EXPECTED_RUNS = 30
EXPECTED_CASES = 750
EXPECTED_CASES_PER_RUN = 25
REQUIRED_METRICS = ("hausdorff", "facet_gap", "area_error")
OPTIONAL_METRICS = ("curvature_error", "tangent_error", "curvature_proxy_error")


class AuditError(RuntimeError):
    """Raised when a release invariant is violated."""


@dataclass(frozen=True, order=True)
class RunKey:
    experiment: str
    algo: str
    n: int
    wiggle: float
    seed: int


@dataclass(frozen=True)
class BeforeBundle:
    path: Path
    kind: str


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_key(row: Mapping[str, str]) -> RunKey:
    return RunKey(
        experiment=row["experiment"],
        algo=row["algo"],
        n=int(round(float(row["resolution"]) * 100.0)),
        wiggle=round(float(row["wiggle"]), 12),
        seed=int(row["seed"]),
    )


def finite_number(value: str, *, field: str, context: str) -> float:
    if value == "":
        raise AuditError(f"missing {field}: {context}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AuditError(f"nonfinite {field}={value!r}: {context}")
    return parsed


def load_json(path: Path) -> object:
    if not path.is_file():
        raise AuditError(f"missing JSON: {path}")
    return json.loads(path.read_text())


def group_by_case(rows: Iterable[Mapping[str, str]]) -> Dict[int, List[Mapping[str, str]]]:
    grouped: Dict[int, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["case_index"])].append(row)
    return dict(grouped)


def fallback_signature(rows: Iterable[Mapping[str, str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(
        sorted(
            (
                row["merge_id"],
                row["policy"],
                row["facet_name"],
                row["num_vertices"],
            )
            for row in rows
        )
    )


def active_facet_count(metadata_path: Path) -> int:
    payload = load_json(metadata_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("primitives"), list):
        raise AuditError(f"invalid facet metadata schema: {metadata_path}")
    primitives = payload["primitives"]
    facet_indices = {primitive.get("facet_index") for primitive in primitives}
    if None in facet_indices:
        raise AuditError(f"facet without facet_index: {metadata_path}")
    return len(facet_indices)


def discover_before_bundles(
    before_root: Path,
    before_plots_root: Path,
    expected_keys: Sequence[RunKey],
) -> Dict[RunKey, BeforeBundle]:
    expected = set(expected_keys)
    discovered: Dict[RunKey, BeforeBundle] = {}

    for row in read_csv(before_root / "diagnostics" / "run_inventory.csv"):
        key = run_key(row)
        if key not in expected:
            continue
        bundle = before_root / row["run_bundle"]
        discovered[key] = BeforeBundle(bundle, "archived_success")

    for row in read_csv(before_root / "failures.csv"):
        key = run_key(row)
        if key not in expected:
            continue
        bundle = before_plots_root / row["save_name"]
        if not bundle.is_dir():
            raise AuditError(f"missing retained failed-run bundle: {bundle}")
        if key in discovered:
            raise AuditError(f"duplicate before bundle for {key}")
        discovered[key] = BeforeBundle(bundle, "retained_failed_partial")

    missing = expected - set(discovered)
    extra = set(discovered) - expected
    if missing or extra:
        raise AuditError(f"before-bundle coverage mismatch: missing={missing}, extra={extra}")
    return discovered


def audit_manifest(after_root: Path) -> Mapping[str, object]:
    manifest = load_json(after_root / "sweep_manifest.json")
    if not isinstance(manifest, dict):
        raise AuditError("sweep manifest is not an object")
    expected = {
        "status": "completed",
        "planned_run_count": EXPECTED_RUNS,
        "successful_run_count": EXPECTED_RUNS,
        "failure_count": 0,
        "planned_case_count": EXPECTED_CASES,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise AuditError(
                f"manifest {field} mismatch: expected {value!r}, got {manifest.get(field)!r}"
            )
    if read_csv(after_root / "failures.csv"):
        raise AuditError("corrected confidence root has nonempty failures.csv")
    return manifest


def audit_and_compare(
    after_root: Path,
    before_root: Path,
    before_plots_root: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[RunKey, BeforeBundle]]:
    audit_manifest(after_root)
    inventory = read_csv(after_root / "diagnostics" / "run_inventory.csv")
    if len(inventory) != EXPECTED_RUNS:
        raise AuditError(f"expected {EXPECTED_RUNS} run inventory rows, got {len(inventory)}")

    after_by_key: Dict[RunKey, Mapping[str, str]] = {}
    for row in inventory:
        key = run_key(row)
        if key in after_by_key:
            raise AuditError(f"duplicate corrected run key: {key}")
        if key.experiment != "squares":
            raise AuditError(f"unexpected experiment in confidence root: {key.experiment}")
        if int(row["case_metrics_rows"]) != EXPECTED_CASES_PER_RUN:
            raise AuditError(f"unexpected case count for {key}: {row['case_metrics_rows']}")
        after_by_key[key] = row

    expected_methods = {"linear", "linear+corner", "circular"}
    expected_n = {50, 64, 100, 128, 150}
    expected_wiggles = {0.2, 0.3}
    if {key.algo for key in after_by_key} != expected_methods:
        raise AuditError("corrected run methods do not match the expected confidence grid")
    if {key.n for key in after_by_key} != expected_n:
        raise AuditError("corrected run resolutions do not match the expected confidence grid")
    if {key.wiggle for key in after_by_key} != expected_wiggles:
        raise AuditError("corrected run perturbations do not match the expected confidence grid")

    before_by_key = discover_before_bundles(
        before_root, before_plots_root, list(after_by_key)
    )

    comparison_rows: List[Dict[str, object]] = []
    source_commits = set()
    total_active_components = 0
    total_fallbacks = 0
    fallback_cases = set()
    raw_fallback_records = set()
    metric_diffs = {"hausdorff": [], "facet_gap": []}

    for key in sorted(after_by_key):
        inventory_row = after_by_key[key]
        after_bundle = after_root / inventory_row["run_bundle"]
        before_bundle = before_by_key[key]
        if not after_bundle.is_dir():
            raise AuditError(f"missing corrected raw bundle: {after_bundle}")

        run_manifest = load_json(after_bundle / "run_manifest.json")
        if not isinstance(run_manifest, dict):
            raise AuditError(f"invalid run manifest: {after_bundle}")
        source_commit = str(run_manifest.get("source_commit", ""))
        source_commits.add(source_commit)
        params = run_manifest.get("parameters", {})
        if not isinstance(params, dict) or params.get("plic_fallback") != "LVIRA":
            raise AuditError(f"non-LVIRA configured fallback: {after_bundle}")

        after_case_rows = read_csv(after_bundle / "metrics" / "case_metrics.csv")
        after_cell_rows = read_csv(after_bundle / "metrics" / "cell_metrics.csv")
        after_fallback_rows = read_csv(
            after_bundle / "metrics" / "unresolved_plic_fallbacks.csv"
        )
        after_event_rows = read_csv(after_bundle / "metrics" / "merge_events.csv")
        before_case_rows = read_csv(before_bundle.path / "metrics" / "case_metrics.csv")
        before_fallback_rows = read_csv(
            before_bundle.path / "metrics" / "unresolved_plic_fallbacks.csv"
        )

        if len(after_case_rows) != EXPECTED_CASES_PER_RUN:
            raise AuditError(f"expected 25 corrected cases for {key}, got {len(after_case_rows)}")
        after_cases = {int(row["case_index"]): row for row in after_case_rows}
        if set(after_cases) != set(range(EXPECTED_CASES_PER_RUN)):
            raise AuditError(f"corrected case indices are incomplete for {key}")
        before_cases = {int(row["case_index"]): row for row in before_case_rows}
        cells_by_case = group_by_case(after_cell_rows)
        fallbacks_by_case = group_by_case(after_fallback_rows)
        before_fallbacks_by_case = group_by_case(before_fallback_rows)
        events_by_case = group_by_case(after_event_rows)

        for case_index in range(EXPECTED_CASES_PER_RUN):
            context = f"{key.algo}, N={key.n}, w={key.wiggle}, case={case_index}"
            case_row = after_cases[case_index]
            for metric in REQUIRED_METRICS:
                finite_number(case_row[metric], field=metric, context=context)
            for metric in OPTIONAL_METRICS:
                if case_row.get(metric, ""):
                    finite_number(case_row[metric], field=metric, context=context)

            active_rows = cells_by_case.get(case_index, [])
            expected_mixed_cells = int(case_row["num_mixed_cells"])
            if len(active_rows) != expected_mixed_cells:
                raise AuditError(
                    f"mixed-cell count mismatch for {context}: "
                    f"case_metrics={expected_mixed_cells}, cell_metrics={len(active_rows)}"
                )
            rows_by_merge: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
            for row in active_rows:
                rows_by_merge[row["merge_id"]].append(row)
            expected_active_components = len(rows_by_merge)
            for merge_id, component_rows in rows_by_merge.items():
                expected_component_size = int(component_rows[0]["merge_component_size"])
                if len(component_rows) != expected_component_size:
                    raise AuditError(
                        f"active component size mismatch for {context}, merge {merge_id}: "
                        f"rows={len(component_rows)}, declared={expected_component_size}"
                    )
                facet_signatures = {
                    (
                        row["final_facet_class"],
                        row["final_facet_name"],
                        row["construction_path"],
                        row["fallback_policy"],
                        row["facet_geometry_json"],
                    )
                    for row in component_rows
                }
                if len(facet_signatures) != 1:
                    raise AuditError(
                        f"member cells disagree on facet/provenance for {context}, merge {merge_id}"
                    )
                row = component_rows[0]
                if row["final_facet_class"] in {"", "missing"}:
                    raise AuditError(f"missing active facet for {context}, merge {merge_id}")
                geometry = json.loads(row["facet_geometry_json"])
                if not isinstance(geometry, dict) or geometry.get("class") in {None, "missing"}:
                    raise AuditError(f"invalid active facet geometry for {context}")
            if int(case_row["num_final_missing_cells"]) != 0:
                raise AuditError(f"case reports missing final facets: {context}")

            after_meta = (
                after_bundle
                / "vtk"
                / "reconstructed"
                / "facets"
                / f"{case_index}.facet_metadata.json"
            )
            after_vtp = after_meta.with_name(f"{case_index}.vtp")
            facet_count = active_facet_count(after_meta)
            if facet_count != expected_active_components:
                raise AuditError(
                    f"facet metadata count mismatch for {context}: "
                    f"active={expected_active_components}, facets={facet_count}"
                )
            if not after_vtp.is_file():
                raise AuditError(f"missing corrected facet VTP: {after_vtp}")

            case_fallbacks = fallbacks_by_case.get(case_index, [])
            expected_fallbacks = int(case_row["num_plic_fallback_cells"])
            if len(case_fallbacks) != expected_fallbacks:
                raise AuditError(
                    f"fallback count mismatch for {context}: "
                    f"case_metrics={expected_fallbacks}, ledger={len(case_fallbacks)}"
                )
            cell_by_merge = {
                merge_id: component_rows[0]
                for merge_id, component_rows in rows_by_merge.items()
            }
            for fallback in case_fallbacks:
                if fallback["policy"] != "LVIRA" or fallback["facet_name"] != "LVIRA":
                    raise AuditError(f"non-LVIRA fallback provenance for {context}")
                cell = cell_by_merge.get(fallback["merge_id"])
                if cell is None:
                    raise AuditError(f"fallback references inactive merge for {context}")
                if (
                    cell["fallback_policy"] != "LVIRA"
                    or cell["final_facet_name"] != "LVIRA"
                    or cell["construction_path"] != "plic_fallback"
                ):
                    raise AuditError(f"fallback/cell provenance mismatch for {context}")
                matching_events = [
                    event
                    for event in events_by_case.get(case_index, [])
                    if event["merge_id"] == fallback["merge_id"]
                    and event["event_kind"] == "plic_fallback"
                    and event["fallback_policy"] == "LVIRA"
                    and event["facet_name"] == "LVIRA"
                ]
                if len(matching_events) != 1:
                    raise AuditError(f"fallback/merge-event mismatch for {context}")
                record = (
                    key,
                    case_index,
                    fallback["merge_id"],
                    fallback["policy"],
                    fallback["facet_name"],
                    fallback["num_vertices"],
                )
                if record in raw_fallback_records:
                    raise AuditError(f"duplicate raw fallback record for {context}")
                raw_fallback_records.add(record)

            total_active_components += expected_active_components
            total_fallbacks += len(case_fallbacks)
            if case_fallbacks:
                fallback_cases.add((key, case_index))

            before_meta = (
                before_bundle.path
                / "vtk"
                / "reconstructed"
                / "facets"
                / f"{case_index}.facet_metadata.json"
            )
            before_vtp = before_meta.with_name(f"{case_index}.vtp")
            metadata_available = before_meta.is_file()
            metadata_match: object = ""
            vtp_match: object = ""
            before_meta_hash = ""
            before_vtp_hash = ""
            if metadata_available:
                before_meta_hash = sha256(before_meta)
                metadata_match = before_meta_hash == sha256(after_meta)
                if not metadata_match:
                    raise AuditError(f"facet metadata changed for {context}")
                if not before_vtp.is_file():
                    raise AuditError(f"metadata exists without VTP for {context}")
                before_vtp_hash = sha256(before_vtp)
                vtp_match = before_vtp_hash == sha256(after_vtp)
                if not vtp_match:
                    raise AuditError(f"facet VTP changed for {context}")

            before_metric_row = before_cases.get(case_index)
            metric_available = before_metric_row is not None
            metric_values: MutableMapping[str, object] = {}
            for metric in ("hausdorff", "facet_gap"):
                before_value: object = ""
                difference: object = ""
                match: object = ""
                if before_metric_row is not None:
                    before_value = finite_number(
                        before_metric_row[metric], field=f"before_{metric}", context=context
                    )
                    after_value = float(case_row[metric])
                    difference = abs(float(before_value) - after_value)
                    match = difference == 0.0
                    metric_diffs[metric].append(float(difference))
                    if not match:
                        raise AuditError(f"{metric} changed for {context}: diff={difference}")
                metric_values[f"before_{metric}"] = before_value
                metric_values[f"after_{metric}"] = float(case_row[metric])
                metric_values[f"{metric}_abs_diff"] = difference
                metric_values[f"{metric}_match"] = match

            before_fallback_signature = fallback_signature(
                before_fallbacks_by_case.get(case_index, [])
            )
            after_fallback_signature = fallback_signature(case_fallbacks)
            fallback_match: object = ""
            if metadata_available:
                fallback_match = before_fallback_signature == after_fallback_signature
                if not fallback_match:
                    raise AuditError(f"fallback provenance changed for {context}")

            comparison_rows.append(
                {
                    "experiment": key.experiment,
                    "method": key.algo,
                    "N": key.n,
                    "resolution_parameter": key.n / 100.0,
                    "wiggle": key.wiggle,
                    "seed": key.seed,
                    "case_index": case_index,
                    "before_bundle_kind": before_bundle.kind,
                    "before_facet_metadata_available": metadata_available,
                    "before_saved_metrics_available": metric_available,
                    "mixed_cell_count": expected_mixed_cells,
                    "active_component_count": expected_active_components,
                    "nonnull_facet_count": facet_count,
                    "active_facet_invariant_pass": True,
                    "before_facet_metadata_sha256": before_meta_hash,
                    "after_facet_metadata_sha256": sha256(after_meta),
                    "facet_metadata_hash_match": metadata_match,
                    "before_facet_vtp_sha256": before_vtp_hash,
                    "after_facet_vtp_sha256": sha256(after_vtp),
                    "facet_vtp_hash_match": vtp_match,
                    "fallback_count": len(case_fallbacks),
                    "fallback_policy": "LVIRA" if case_fallbacks else "",
                    "fallback_provenance_match": fallback_match,
                    **metric_values,
                    "after_area_error": float(case_row["area_error"]),
                }
            )

    if len(source_commits) != 1 or "" in source_commits:
        raise AuditError(f"corrected raw bundles do not share one source commit: {source_commits}")
    if len(comparison_rows) != EXPECTED_CASES:
        raise AuditError(f"expected {EXPECTED_CASES} comparison rows, got {len(comparison_rows)}")

    consolidated_cases = read_csv(after_root / "diagnostics" / "case_metrics.csv")
    if len(consolidated_cases) != EXPECTED_CASES:
        raise AuditError(
            f"expected {EXPECTED_CASES} consolidated case rows, got {len(consolidated_cases)}"
        )
    consolidated_by_case = {}
    for row in consolidated_cases:
        key = (run_key(row), int(row["case_index"]))
        if key in consolidated_by_case:
            raise AuditError(f"duplicate consolidated case key: {key}")
        consolidated_by_case[key] = row
    if len(consolidated_by_case) != len(comparison_rows):
        raise AuditError("consolidated case-key coverage does not match raw bundles")
    for row in comparison_rows:
        key = RunKey(
            str(row["experiment"]),
            str(row["method"]),
            int(row["N"]),
            round(float(row["wiggle"]), 12),
            int(row["seed"]),
        )
        consolidated = consolidated_by_case.get((key, int(row["case_index"])))
        if consolidated is None:
            raise AuditError(f"missing consolidated case row for {key}, case {row['case_index']}")
        expected_values = {
            "hausdorff": float(row["after_hausdorff"]),
            "facet_gap": float(row["after_facet_gap"]),
            "area_error": float(row["after_area_error"]),
        }
        for metric, expected_value in expected_values.items():
            actual_value = finite_number(
                consolidated[metric],
                field=f"consolidated_{metric}",
                context=f"{key}, case {row['case_index']}",
            )
            if actual_value != expected_value:
                raise AuditError(
                    f"raw/consolidated {metric} mismatch for {key}, case {row['case_index']}"
                )

    consolidated_fallbacks = read_csv(
        after_root / "diagnostics" / "unresolved_plic_fallbacks.csv"
    )
    consolidated_fallback_records = {
        (
            run_key(row),
            int(row["case_index"]),
            row["merge_id"],
            row["policy"],
            row["facet_name"],
            row["num_vertices"],
        )
        for row in consolidated_fallbacks
    }
    if len(consolidated_fallback_records) != len(consolidated_fallbacks):
        raise AuditError("duplicate consolidated fallback records")
    if consolidated_fallback_records != raw_fallback_records:
        raise AuditError("raw/consolidated fallback provenance mismatch")

    metadata_rows = [row for row in comparison_rows if row["before_facet_metadata_available"]]
    metric_rows = [row for row in comparison_rows if row["before_saved_metrics_available"]]
    partial_metadata_rows = [
        row
        for row in metadata_rows
        if row["before_bundle_kind"] == "retained_failed_partial"
    ]
    archived_metadata_rows = [
        row for row in metadata_rows if row["before_bundle_kind"] == "archived_success"
    ]
    summary: Dict[str, object] = {
        "source_commit": next(iter(source_commits)),
        "run_count": len(after_by_key),
        "case_count": len(comparison_rows),
        "active_component_count": total_active_components,
        "active_facet_invariant_failures": 0,
        "finite_metric_case_count": len(comparison_rows),
        "raw_consolidated_metric_match_count": len(comparison_rows),
        "raw_consolidated_fallback_match_count": len(raw_fallback_records),
        "fallback_count": total_fallbacks,
        "fallback_case_count": len(fallback_cases),
        "fallback_policy": "LVIRA",
        "before_bundle_count": len(before_by_key),
        "before_archived_success_bundle_count": sum(
            bundle.kind == "archived_success" for bundle in before_by_key.values()
        ),
        "before_partial_bundle_count": sum(
            bundle.kind == "retained_failed_partial" for bundle in before_by_key.values()
        ),
        "before_metadata_case_count": len(metadata_rows),
        "before_archived_metadata_case_count": len(archived_metadata_rows),
        "before_partial_metadata_case_count": len(partial_metadata_rows),
        "metadata_hash_match_count": sum(
            row["facet_metadata_hash_match"] is True for row in metadata_rows
        ),
        "vtp_hash_match_count": sum(row["facet_vtp_hash_match"] is True for row in metadata_rows),
        "before_saved_metric_case_count": len(metric_rows),
        "hausdorff_match_count": sum(row["hausdorff_match"] is True for row in metric_rows),
        "facet_gap_match_count": sum(row["facet_gap_match"] is True for row in metric_rows),
        "max_hausdorff_abs_diff": max(metric_diffs["hausdorff"], default=0.0),
        "max_facet_gap_abs_diff": max(metric_diffs["facet_gap"], default=0.0),
        "fallback_provenance_match_count": sum(
            row["fallback_provenance_match"] is True for row in metadata_rows
        ),
    }
    return summary, comparison_rows, before_by_key


def write_comparison_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise AuditError("cannot write an empty comparison CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def witness_rows(
    comparison_rows: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    methods = ("linear", "linear+corner", "circular")
    indexed = {
        str(row["method"]): row
        for row in comparison_rows
        if row["N"] == 50 and row["wiggle"] == 0.2 and row["case_index"] == 3
    }
    if set(indexed) != set(methods):
        raise AuditError("witness rows are incomplete")
    return [indexed[method] for method in methods]


def read_vtp_polygons(path: Path) -> List[List[Tuple[float, float]]]:
    try:
        import vtk  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise AuditError("VTK is required to render the witness visual") from exc
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    polygons: List[List[Tuple[float, float]]] = []
    for cell_index in range(data.GetNumberOfCells()):
        cell = data.GetCell(cell_index)
        polygons.append(
            [
                tuple(data.GetPoint(cell.GetPointId(i))[:2])
                for i in range(cell.GetNumberOfPoints())
            ]
        )
    return polygons


def render_witness(
    output_pdf: Path,
    output_png: Path,
    after_root: Path,
    before_by_key: Mapping[RunKey, BeforeBundle],
    comparison_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )
    key = RunKey("squares", "linear", 50, 0.2, 0)
    before_bundle = before_by_key[key].path
    after_inventory = {
        run_key(row): row
        for row in read_csv(after_root / "diagnostics" / "run_inventory.csv")
    }
    after_bundle = after_root / after_inventory[key]["run_bundle"]
    case_index = 3

    before_cells = [
        row
        for row in read_csv(before_bundle / "metrics" / "cell_metrics.csv")
        if int(row["case_index"]) == case_index
    ]
    polygons = read_vtp_polygons(
        before_bundle / "vtk" / "reconstructed" / "mixed_cells" / f"{case_index}.vtp"
    )
    if len(before_cells) != 30 or len(polygons) != 30:
        raise AuditError("unexpected witness active-component count")
    merge_ids = [int(row["merge_id"]) for row in before_cells]
    stale_ids = sorted(set(range(max(merge_ids) + 1)) - set(merge_ids))
    child_ids = sorted(merge_id for merge_id in merge_ids if merge_id >= len(before_cells))
    if stale_ids != [17, 20] or child_ids != [30, 31]:
        raise AuditError(
            f"unexpected witness parent/child ids: stale={stale_ids}, children={child_ids}"
        )
    child_to_polygon = {
        int(row["merge_id"]): polygon for row, polygon in zip(before_cells, polygons)
    }

    metadata = load_json(
        before_bundle
        / "vtk"
        / "reconstructed"
        / "facets"
        / f"{case_index}.facet_metadata.json"
    )
    if not isinstance(metadata, dict):
        raise AuditError("invalid witness facet metadata")
    geometry_rows = [
        json.loads(line)
        for line in (before_bundle / "metrics" / "case_geometry.jsonl").read_text().splitlines()
        if line.strip()
    ]
    geometry = next(row for row in geometry_rows if row["case_index"] == case_index)
    true_vertices = [tuple(point) for point in geometry["vertices"]]

    x_values = [point[0] for polygon in polygons for point in polygon]
    y_values = [point[1] for polygon in polygons for point in polygon]
    x_pad = (max(x_values) - min(x_values)) * 0.08
    y_pad = (max(y_values) - min(y_values)) * 0.08

    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(4.3, 1.45))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]

    for axis in axes:
        for polygon in polygons:
            axis.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    facecolor="#f5f6f7",
                    edgecolor="#aeb5bc",
                    linewidth=0.65,
                    zorder=1,
                )
            )
        for child_id in child_ids:
            axis.add_patch(
                Polygon(
                    child_to_polygon[child_id],
                    closed=True,
                    facecolor="#9ecae1",
                    edgecolor="#17679a",
                    linewidth=1.6,
                    zorder=2,
                )
            )
        for primitive in metadata["primitives"]:
            if primitive["kind"] != "line":
                raise AuditError("linear witness unexpectedly contains a non-line primitive")
            p_left = primitive["p_left"]
            p_right = primitive["p_right"]
            axis.plot(
                [p_left[0], p_right[0]],
                [p_left[1], p_right[1]],
                color="#171717",
                linewidth=1.4,
                solid_capstyle="round",
                zorder=4,
            )
        closed_truth = true_vertices + [true_vertices[0]]
        axis.plot(
            [point[0] for point in closed_truth],
            [point[1] for point in closed_truth],
            color="#2f7d32",
            linewidth=1.35,
            linestyle=(0, (4, 2)),
            label="exact square",
            zorder=3,
        )
        axis.set_aspect("equal")
        axis.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad)
        axis.set_ylim(min(y_values) - y_pad, max(y_values) + y_pad)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(False)

    for stale_id, child_id in zip(stale_ids, child_ids):
        polygon = child_to_polygon[child_id]
        axes[0].add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="none",
                edgecolor="#c33d3d",
                linewidth=1.8,
                hatch="////",
                zorder=5,
            )
        )
        center = (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
        axes[0].annotate(
            f"retired {stale_id}\n(active {child_id})",
            xy=center,
            xytext=(center[0] - 0.5, center[1] - 2.5),
            ha="center",
            va="top",
            color="#a02020",
            fontsize=8,
            fontweight="bold",
            arrowprops={"arrowstyle": "-|>", "color": "#a02020", "lw": 0.9},
            zorder=6,
        )
        axes[1].annotate(
            f"active {child_id}\nLVIRA",
            xy=center,
            xytext=(center[0] - 0.5, center[1] - 2.5),
            ha="center",
            va="top",
            color="#075786",
            fontsize=8,
            fontweight="bold",
            arrowprops={"arrowstyle": "-|>", "color": "#075786", "lw": 0.9},
            zorder=6,
        )

    axes[0].set_title("Before: stale retained-parent accounting\n32 retained entries, 30 active pairs, 30 facets")
    axes[1].set_title("After: active-partition accounting\n30 active polygons, 30 facets")
    fig.suptitle(
        "Square active-partition witness: N=50, perturbation magnitude 0.2, case 3",
        fontsize=14,
        fontweight="bold",
    )

    table_axis = fig.add_subplot(grid[1, :])
    table_axis.axis("off")
    rows = witness_rows(comparison_rows)
    table_data = []
    for row in rows:
        table_data.append(
            [
                row["method"],
                "32 / 30 -> 30 / 30",
                str(row["fallback_count"]),
                "exact" if row["facet_metadata_hash_match"] else "changed",
                f"{float(row['after_hausdorff']):.3e}",
                f"{float(row['after_facet_gap']):.3e}",
                f"{float(row['after_area_error']):.3e}",
            ]
        )
    table = table_axis.table(
        cellText=table_data,
        colLabels=(
            "Method",
            "Retained / active -> polygon / facet",
            "LVIRA cells",
            "Pre/post facet JSON",
            "Hausdorff",
            "Facet gap",
            "Area error",
        ),
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=(0.11, 0.25, 0.09, 0.15, 0.11, 0.11, 0.11),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.35)
    for column in range(7):
        table[(0, column)].set_facecolor("#e5e8eb")
        table[(0, column)].set_text_props(fontweight="bold")
    table_axis.text(
        0.5,
        0.03,
        "Red hatching marks duplicate bookkeeping entries on the same blue active cells, not extra geometry. "
        "The pre/post facet JSON and VTP hashes are byte-identical; corrected metrics are shown above.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#333333",
        transform=table_axis.transAxes,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    summary: Mapping[str, object],
    after_root: Path,
    before_root: Path,
    before_plots_root: Path,
    comparison_csv: Path,
    witness_pdf: Path,
    witness_png: Path,
    comparison_rows: Sequence[Mapping[str, object]],
) -> None:
    witness = witness_rows(comparison_rows)
    witness_table = "\n".join(
        "| {method} | {active_component_count} | {fallback_count} | {after_hausdorff:.6e} | "
        "{after_facet_gap:.6e} | {after_area_error:.6e} |".format(**row)
        for row in witness
    )
    text = f"""# Square active-partition confidence audit

Audit date: 2026-07-31 PDT

Corrected source commit: `{summary['source_commit']}`

## Verdict

**PASS.** The corrected confidence sweep contains `{summary['run_count']}/{EXPECTED_RUNS}` successful runs and `{summary['case_count']}/{EXPECTED_CASES}` cases. Across `{summary['active_component_count']:,}` unique active mixed components, every component has one internally consistent non-null reconstructed facet, and each case's facet metadata contains exactly one facet index per active component. All required square metrics are finite.

The correction changes only the square area-metric partition supplied by the driver. It does not change reconstructed geometry: every available pre-fix facet JSON and VTP artifact is byte-identical to its corrected counterpart, and every saved pre-fix Hausdorff/facet-gap value is exactly unchanged.

## Coverage and invariants

| Check | Result |
|---|---:|
| Completed runs | `{summary['run_count']}/{EXPECTED_RUNS}` |
| Completed cases | `{summary['case_count']}/{EXPECTED_CASES}` |
| Active mixed components | `{summary['active_component_count']:,}` |
| Active-component/facet invariant failures | `{summary['active_facet_invariant_failures']}` |
| Cases with finite Hausdorff, facet gap, and area error | `{summary['finite_metric_case_count']}/{EXPECTED_CASES}` |
| Raw/consolidated case-metric matches | `{summary['raw_consolidated_metric_match_count']}/{EXPECTED_CASES}` |
| LVIRA fallback records | `{summary['fallback_count']}` in `{summary['fallback_case_count']}` cases |
| Raw/consolidated fallback-record matches | `{summary['raw_consolidated_fallback_match_count']}/{summary['fallback_count']}` |
| Non-LVIRA or inconsistent fallback records | `0` |

Each fallback was cross-checked in the run configuration, case summary, cell-level facet provenance, unresolved-fallback ledger, and merge-event ledger. Every record uses `LVIRA`, points to an active component, and ends in a non-null `LVIRA` line facet.

## Pre/post comparison

The invalid release supplied `{summary['before_archived_success_bundle_count']}` complete control bundles. Its `{summary['before_partial_bundle_count']}` failed settings retained partial run directories under `{before_plots_root}`. Together they provide pre-fix facet artifacts for `{summary['before_metadata_case_count']}` cases:

- `{summary['before_archived_metadata_case_count']}` cases from complete archived controls.
- `{summary['before_partial_metadata_case_count']}` cases from retained failed-run prefixes, including each failing witness.
- `{summary['metadata_hash_match_count']}/{summary['before_metadata_case_count']}` facet-metadata JSON hashes match exactly.
- `{summary['vtp_hash_match_count']}/{summary['before_metadata_case_count']}` facet VTP hashes match exactly.
- `{summary['fallback_provenance_match_count']}/{summary['before_metadata_case_count']}` available fallback signatures match exactly.
- `{summary['hausdorff_match_count']}/{summary['before_saved_metric_case_count']}` saved Hausdorff values match exactly; maximum absolute difference `{summary['max_hausdorff_abs_diff']:.1e}`.
- `{summary['facet_gap_match_count']}/{summary['before_saved_metric_case_count']}` saved facet-gap values match exactly; maximum absolute difference `{summary['max_facet_gap_abs_diff']:.1e}`.

The failed jobs raised during area evaluation, before writing the failing case's case-metric row. Their witness facet files and fallback ledgers were already present and match the corrected run byte for byte; subsequent cases were never attempted in those partial bundles. The comparison CSV distinguishes these expected absences from mismatches.

## N=50 witness

At `N=50`, perturbation magnitude `0.2`, case index `3`, the old driver supplied 32 retained dictionary entries to a 30-facet active reconstruction. Retired parents `17` and `20` had already been replaced by active single-cell children `30` and `31`, both reconstructed by LVIRA. The corrected driver supplies the returned 30-element active polygon list, so polygon/facet pairing is `30/30`.

| Method | Active pairs | LVIRA cells | Hausdorff | Facet gap | Corrected area error |
|---|---:|---:|---:|---:|---:|
{witness_table}

The [vector witness PDF]({witness_pdf.name}) and [300 DPI Slack preview]({witness_png.name}) show the stale parent overlays versus the corrected active-partition accounting. The reconstructed interface is identical in both panels.

## Artifacts and scope

- Case-level comparison: [`{comparison_csv.name}`]({comparison_csv.name})
- Witness PDF: [`{witness_pdf.name}`]({witness_pdf.name})
- Witness PNG: [`{witness_png.name}`]({witness_png.name})
- Corrected confidence root: `{after_root}`
- Invalid diagnostic root: `{before_root}`
- Retained failed-run prefixes: `{before_plots_root}`

This audit is scoped to the 30-run square confidence grid. It does not promote the invalid 970-run release and does not audit the still-running corrected authoritative release. Both source result roots were read only.
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("after_root", type=Path)
    parser.add_argument("before_root", type=Path)
    parser.add_argument(
        "--before-plots-root",
        type=Path,
        required=True,
        help="Top-level plots directory containing retained failed-run prefixes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    after_root = args.after_root.resolve()
    before_root = args.before_root.resolve()
    before_plots_root = args.before_plots_root.resolve()
    output_dir = args.output_dir.resolve()

    summary, comparison_rows, before_by_key = audit_and_compare(
        after_root, before_root, before_plots_root
    )
    comparison_csv = output_dir / "square_confidence_case_comparison.csv"
    witness_pdf = output_dir / "square_n50_w0p2_case3_before_after.pdf"
    witness_png = output_dir / "square_n50_w0p2_case3_before_after.png"
    report = output_dir / "README.md"
    write_comparison_csv(comparison_csv, comparison_rows)
    render_witness(witness_pdf, witness_png, after_root, before_by_key, comparison_rows)
    write_report(
        report,
        summary,
        after_root,
        before_root,
        before_plots_root,
        comparison_csv,
        witness_pdf,
        witness_png,
        comparison_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report: {report}")
    print(f"comparison: {comparison_csv}")
    print(f"witness: {witness_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
