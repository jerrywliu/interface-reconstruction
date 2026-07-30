#!/usr/bin/env python3
"""Summarize reconstruction-path incidence in a completed static release.

The report is intentionally cell weighted: every reported fraction uses the
number of mixed cells in the corresponding scope as its denominator. Event-based
diagnostics also retain component and raw-event counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = 1

CASE_KEY_FIELDS = (
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
    "case_index",
)

CELL_REQUIRED_FIELDS = CASE_KEY_FIELDS + (
    "cell_id",
    "merge_id",
    "is_merged",
    "orientation_status",
    "final_facet_class",
    "construction_path",
    "fallback_policy",
)

EVENT_REQUIRED_FIELDS = CASE_KEY_FIELDS + (
    "event_order",
    "merge_id",
    "member_cells_json",
    "stage",
    "event_kind",
    "fallback_policy",
    "fallback_reason",
    "facet_class",
    "facet_name",
)

FALLBACK_REQUIRED_FIELDS = CASE_KEY_FIELDS + (
    "merge_id",
    "policy",
)

CASE_REQUIRED_FIELDS = CASE_KEY_FIELDS + (
    "num_mixed_cells",
    "num_merged_components",
)

CANONICAL_METRICS = (
    ("final_facet", "circular"),
    ("final_facet", "corner"),
    ("final_facet", "curved_corner"),
    ("merge", "merged"),
    ("rescue", "exact_linear_support"),
    ("rescue", "corner_arc_corner_triplet"),
    ("rescue", "curved_corner_loop"),
    ("rescue", "curved_corner_transition"),
    ("orientation", "unresolved_orientation"),
    ("orientation", "unresolved_or_deadend_status"),
    ("plic_fallback", "Youngs"),
    ("plic_fallback", "ELVIRA"),
    ("plic_fallback", "LVIRA"),
)
CANONICAL_METRIC_SET = frozenset(CANONICAL_METRICS)

FINAL_FACET_SUBTYPES = {
    "circular": "circular",
    "linear_corner": "corner",
    "curved_corner": "curved_corner",
}


class DiagnosticSummaryError(RuntimeError):
    """Raised when a release cannot support an unambiguous diagnostic report."""


@dataclass(frozen=True, order=True)
class GroupKey:
    scope: str
    experiment: str = ""
    algo: str = ""


@dataclass
class Incidence:
    mixed_cell_count: int = 0
    component_count: int = 0
    event_count: int = 0


CaseKey = Tuple[str, ...]
ComponentKey = Tuple[CaseKey, str]
MetricKey = Tuple[str, str]


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        raise DiagnosticSummaryError(f"Missing required JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticSummaryError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticSummaryError(f"JSON root must be an object: {path}")
    return value


def _require_completed_release(release_root: Path) -> dict:
    manifest = _load_json_object(release_root / "sweep_manifest.json")
    status = manifest.get("status")
    if status != "completed":
        raise DiagnosticSummaryError(
            f"Refusing to summarize an incomplete release: sweep status is {status!r}, "
            "expected 'completed'."
        )

    planned = _parse_nonnegative_int(
        manifest.get("planned_run_count"), "sweep_manifest planned_run_count"
    )
    successful = _parse_nonnegative_int(
        manifest.get("successful_run_count"), "sweep_manifest successful_run_count"
    )
    failures = _parse_nonnegative_int(
        manifest.get("failure_count"), "sweep_manifest failure_count"
    )
    if planned <= 0 or successful != planned or failures != 0:
        raise DiagnosticSummaryError(
            "Refusing to summarize a non-final release: expected a positive planned "
            f"run count, all runs successful, and zero failures; found planned={planned}, "
            f"successful={successful}, failures={failures}."
        )
    return manifest


def _parse_nonnegative_int(value: object, label: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise DiagnosticSummaryError(f"{label} must be an integer, found {value!r}") from exc
    if parsed < 0 or str(value).strip() not in {str(parsed), f"+{parsed}"}:
        raise DiagnosticSummaryError(
            f"{label} must be a nonnegative integer, found {value!r}"
        )
    return parsed


def _parse_flag(value: object, label: str) -> bool:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise DiagnosticSummaryError(f"{label} must be 0 or 1, found {value!r}")
    return text == "1"


def _require_headers(path: Path, fieldnames: Optional[Sequence[str]], required: Iterable[str]) -> None:
    if fieldnames is None:
        raise DiagnosticSummaryError(f"CSV has no header: {path}")
    missing = sorted(set(required) - set(fieldnames))
    if missing:
        raise DiagnosticSummaryError(
            f"CSV {path} is missing required fields: {', '.join(missing)}"
        )


def _open_csv(path: Path, required: Iterable[str]):
    if not path.is_file():
        raise DiagnosticSummaryError(f"Missing required diagnostic CSV: {path}")
    try:
        stream = path.open(newline="", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DiagnosticSummaryError(f"Could not open diagnostic CSV {path}: {exc}") from exc
    reader = csv.DictReader(stream)
    try:
        _require_headers(path, reader.fieldnames, required)
    except Exception:
        stream.close()
        raise
    return stream, reader


def _case_key(row: Mapping[str, object], source: str) -> CaseKey:
    values = []
    missing = []
    for field in CASE_KEY_FIELDS:
        value = str(row.get(field, "")).strip()
        if not value:
            missing.append(field)
        values.append(value)
    if missing:
        raise DiagnosticSummaryError(
            f"{source} is missing case-key values: {', '.join(missing)}"
        )
    return tuple(values)


def _component_key(row: Mapping[str, object], source: str) -> ComponentKey:
    merge_id = str(row.get("merge_id", "")).strip()
    if not merge_id:
        raise DiagnosticSummaryError(f"{source} has an empty merge_id")
    return _case_key(row, source), merge_id


def _groups_for_case(case_key: CaseKey) -> Tuple[GroupKey, GroupKey, GroupKey]:
    experiment, algo = case_key[0], case_key[1]
    return (
        GroupKey("overall"),
        GroupKey("experiment", experiment=experiment),
        GroupKey("method", experiment=experiment, algo=algo),
    )


def _add_cell_incidence(
    incidences: MutableMapping[Tuple[GroupKey, MetricKey], Incidence],
    groups: Iterable[GroupKey],
    metric: MetricKey,
    count: int = 1,
) -> None:
    for group in groups:
        incidences[(group, metric)].mixed_cell_count += count


def _add_event_incidence(
    incidences: MutableMapping[Tuple[GroupKey, MetricKey], Incidence],
    groups: Iterable[GroupKey],
    metric: MetricKey,
    new_cell_count: int,
    first_component_event: bool,
) -> None:
    for group in groups:
        incidence = incidences[(group, metric)]
        incidence.event_count += 1
        if first_component_event:
            incidence.component_count += 1
        incidence.mixed_cell_count += new_cell_count


def _parse_member_cells(raw: object, source: str) -> Tuple[str, ...]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise DiagnosticSummaryError(
            f"{source} has invalid member_cells_json: {exc}"
        ) from exc
    if not isinstance(value, list) or not value:
        raise DiagnosticSummaryError(
            f"{source} member_cells_json must be a nonempty list"
        )
    normalized = []
    for index, cell in enumerate(value):
        if not isinstance(cell, list) or len(cell) != 2:
            raise DiagnosticSummaryError(
                f"{source} member_cells_json[{index}] must be a two-item list"
            )
        normalized.append(f"{cell[0]},{cell[1]}")
    if len(set(normalized)) != len(normalized):
        raise DiagnosticSummaryError(
            f"{source} member_cells_json contains duplicate cells"
        )
    return tuple(normalized)


def _classify_rescue(stage: str, facet_name: str, source: str) -> str:
    if stage == "curved_corner_loop_rescue":
        return "curved_corner_loop"
    if stage == "curved_corner_transition_rescue":
        return "curved_corner_transition"
    if stage == "linear_corner_rescues":
        if facet_name == "linear_support":
            return "exact_linear_support"
        if facet_name == "corner":
            return "corner_arc_corner_triplet"
        if facet_name == "corner_branch_linear":
            raise DiagnosticSummaryError(
                f"{source} records a corner_branch_linear assignment in the shared "
                "linear_corner_rescues stage. The archived provenance cannot distinguish "
                "the repeated-tiny-corner and bridge rescue paths; add an explicit "
                "rescue_type field before summarizing this release."
            )
    if stage == "post_fallback_rescues":
        raise DiagnosticSummaryError(
            f"{source} records an assignment in post_fallback_rescues, whose archived "
            "stage does not distinguish repeated-component and owner/intruder rescues; "
            "add an explicit rescue_type field before summarizing this release."
        )
    raise DiagnosticSummaryError(
        f"{source} uses unrecognized or ambiguous rescue provenance "
        f"stage={stage!r}, facet_name={facet_name!r}."
    )


def _read_case_metrics(
    path: Path,
) -> Tuple[Dict[CaseKey, int], Dict[GroupKey, int], Dict[GroupKey, int]]:
    case_mixed_cells: Dict[CaseKey, int] = {}
    case_counts: Counter = Counter()
    merged_component_counts: Counter = Counter()
    stream, reader = _open_csv(path, CASE_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            key = _case_key(row, source)
            if key in case_mixed_cells:
                raise DiagnosticSummaryError(f"Duplicate case key in {source}: {key}")
            mixed_cells = _parse_nonnegative_int(
                row["num_mixed_cells"], f"{source} num_mixed_cells"
            )
            merged_components = _parse_nonnegative_int(
                row["num_merged_components"], f"{source} num_merged_components"
            )
            case_mixed_cells[key] = mixed_cells
            for group in _groups_for_case(key):
                case_counts[group] += 1
                merged_component_counts[group] += merged_components
    except (csv.Error, UnicodeError) as exc:
        raise DiagnosticSummaryError(f"Could not parse {path}: {exc}") from exc
    finally:
        stream.close()
    if not case_mixed_cells:
        raise DiagnosticSummaryError(f"No case rows found in {path}")
    return case_mixed_cells, dict(case_counts), dict(merged_component_counts)


def _validate_case_geometry(path: Path, expected_cases: Set[CaseKey]) -> None:
    if not path.is_file():
        raise DiagnosticSummaryError(f"Missing required case geometry JSONL: {path}")
    observed: Set[CaseKey] = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                source = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DiagnosticSummaryError(f"Invalid JSON in {source}: {exc}") from exc
                if not isinstance(row, dict):
                    raise DiagnosticSummaryError(f"JSONL row must be an object: {source}")
                key = _case_key(row, source)
                if key in observed:
                    raise DiagnosticSummaryError(f"Duplicate case geometry key in {source}: {key}")
                observed.add(key)
    except (OSError, UnicodeError) as exc:
        raise DiagnosticSummaryError(f"Could not read {path}: {exc}") from exc
    if observed != expected_cases:
        missing = len(expected_cases - observed)
        extra = len(observed - expected_cases)
        raise DiagnosticSummaryError(
            f"Case geometry/case metrics key mismatch: missing={missing}, extra={extra}"
        )


def _read_cells(
    path: Path,
    case_mixed_cells: Mapping[CaseKey, int],
    incidences: MutableMapping[Tuple[GroupKey, MetricKey], Incidence],
) -> Tuple[Dict[GroupKey, int], Dict[ComponentKey, Tuple[str, int]], Dict[CaseKey, int]]:
    totals: Counter = Counter()
    observed_case_counts: Counter = Counter()
    plic_components: Dict[ComponentKey, Tuple[str, int]] = {}
    merged_components_seen: Set[ComponentKey] = set()
    completed_cases: Set[CaseKey] = set()
    current_case: Optional[CaseKey] = None
    current_cell_ids: Set[str] = set()
    stream, reader = _open_csv(path, CELL_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            case_key = _case_key(row, source)
            if case_key not in case_mixed_cells:
                raise DiagnosticSummaryError(f"Cell row has no matching case row: {source}")
            if case_key != current_case:
                if current_case is not None:
                    completed_cases.add(current_case)
                if case_key in completed_cases:
                    raise DiagnosticSummaryError(
                        f"Cell rows for case {case_key} are not contiguous in {path}"
                    )
                current_case = case_key
                current_cell_ids = set()
            cell_id = str(row["cell_id"]).strip()
            if not cell_id:
                raise DiagnosticSummaryError(f"{source} has an empty cell_id")
            if cell_id in current_cell_ids:
                raise DiagnosticSummaryError(
                    f"Duplicate mixed-cell row for case {case_key}, cell_id={cell_id!r}"
                )
            current_cell_ids.add(cell_id)
            groups = _groups_for_case(case_key)
            for group in groups:
                totals[group] += 1
            observed_case_counts[case_key] += 1

            final_class = str(row["final_facet_class"]).strip()
            if not final_class:
                raise DiagnosticSummaryError(f"{source} has an empty final_facet_class")
            subtype = FINAL_FACET_SUBTYPES.get(final_class)
            if subtype is not None:
                _add_cell_incidence(incidences, groups, ("final_facet", subtype))

            component_key = _component_key(row, source)
            if _parse_flag(row["is_merged"], f"{source} is_merged"):
                _add_cell_incidence(incidences, groups, ("merge", "merged"))
                if component_key not in merged_components_seen:
                    merged_components_seen.add(component_key)
                    for group in groups:
                        incidences[(group, ("merge", "merged"))].component_count += 1

            if str(row["orientation_status"]).strip() == "unresolved_or_deadend":
                _add_cell_incidence(
                    incidences,
                    groups,
                    ("orientation", "unresolved_or_deadend_status"),
                )

            construction_path = str(row["construction_path"]).strip()
            policy = str(row["fallback_policy"]).strip()
            if construction_path == "plic_fallback":
                if not policy:
                    raise DiagnosticSummaryError(
                        f"{source} is marked plic_fallback without fallback_policy"
                    )
                previous = plic_components.get(component_key)
                if previous is None:
                    plic_components[component_key] = (policy, 1)
                elif previous[0] != policy:
                    raise DiagnosticSummaryError(
                        f"Inconsistent fallback policies for component {component_key}: "
                        f"{previous[0]!r} and {policy!r}"
                    )
                else:
                    plic_components[component_key] = (policy, previous[1] + 1)
            elif policy:
                raise DiagnosticSummaryError(
                    f"{source} has fallback_policy={policy!r} but construction_path="
                    f"{construction_path!r}"
                )
    except (csv.Error, UnicodeError) as exc:
        raise DiagnosticSummaryError(f"Could not parse {path}: {exc}") from exc
    finally:
        stream.close()

    for case_key, expected in case_mixed_cells.items():
        observed = observed_case_counts.get(case_key, 0)
        if observed != expected:
            raise DiagnosticSummaryError(
                f"Mixed-cell count mismatch for case {case_key}: case_metrics={expected}, "
                f"cell_metrics={observed}"
            )
    return dict(totals), plic_components, dict(observed_case_counts)


def _read_events(
    path: Path,
    valid_cases: Set[CaseKey],
    incidences: MutableMapping[Tuple[GroupKey, MetricKey], Incidence],
) -> Dict[ComponentKey, Tuple[str, int]]:
    plic_components: Dict[ComponentKey, Tuple[str, int]] = {}
    incidence_components_seen: Set[Tuple[MetricKey, ComponentKey]] = set()
    incidence_cells_seen: Set[Tuple[MetricKey, CaseKey, str]] = set()
    stream, reader = _open_csv(path, EVENT_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            case_key = _case_key(row, source)
            if case_key not in valid_cases:
                raise DiagnosticSummaryError(f"Event row has no matching case row: {source}")
            component_key = _component_key(row, source)
            groups = _groups_for_case(case_key)
            stage = str(row["stage"]).strip()
            event_kind = str(row["event_kind"]).strip()
            facet_name = str(row["facet_name"]).strip()
            reason = str(row["fallback_reason"]).strip()
            plic_policy = ""

            metrics: List[MetricKey] = []
            if event_kind == "facet_assignment" and "rescue" in stage:
                metrics.append(("rescue", _classify_rescue(stage, facet_name, source)))
            elif "rescue" in stage and event_kind not in {"stage_snapshot"}:
                raise DiagnosticSummaryError(
                    f"{source} has unsupported rescue event_kind={event_kind!r}"
                )

            if event_kind == "local_linear_fallback":
                metrics.append(("fallback", f"local_linear:{reason or 'unspecified'}"))
            elif event_kind == "missing_fallback":
                metrics.append(("fallback", f"missing:{reason or 'unspecified'}"))
            elif event_kind == "plic_fallback":
                policy = str(row["fallback_policy"]).strip()
                if not policy or not reason:
                    raise DiagnosticSummaryError(
                        f"{source} PLIC fallback requires fallback_policy and fallback_reason"
                    )
                metrics.append(("plic_fallback", policy))
                previous = plic_components.get(component_key)
                if previous is not None and previous[0] != policy:
                    raise DiagnosticSummaryError(
                        f"Conflicting PLIC policies for event component {component_key}: "
                        f"{previous[0]!r} and {policy!r}"
                    )
                plic_policy = policy
                if reason == "unresolved_orientation":
                    metrics.append(("orientation", "unresolved_orientation"))

            if not metrics:
                continue
            member_cells = _parse_member_cells(row["member_cells_json"], source)
            if plic_policy:
                plic_components[component_key] = (plic_policy, len(member_cells))
            for metric in metrics:
                incidence_key = (metric, component_key)
                first = incidence_key not in incidence_components_seen
                if first:
                    incidence_components_seen.add(incidence_key)
                new_cell_count = 0
                for cell_id in member_cells:
                    cell_key = (metric, case_key, cell_id)
                    if cell_key not in incidence_cells_seen:
                        incidence_cells_seen.add(cell_key)
                        new_cell_count += 1
                _add_event_incidence(
                    incidences,
                    groups,
                    metric,
                    new_cell_count,
                    first_component_event=first,
                )
    except (csv.Error, UnicodeError) as exc:
        raise DiagnosticSummaryError(f"Could not parse {path}: {exc}") from exc
    finally:
        stream.close()
    return plic_components


def _read_fallback_table(path: Path, valid_cases: Set[CaseKey]) -> Dict[ComponentKey, str]:
    fallbacks: Dict[ComponentKey, str] = {}
    stream, reader = _open_csv(path, FALLBACK_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            case_key = _case_key(row, source)
            if case_key not in valid_cases:
                raise DiagnosticSummaryError(f"Fallback row has no matching case row: {source}")
            component_key = _component_key(row, source)
            policy = str(row["policy"]).strip()
            if not policy:
                raise DiagnosticSummaryError(f"{source} has an empty fallback policy")
            previous = fallbacks.get(component_key)
            if previous is not None:
                raise DiagnosticSummaryError(
                    f"Duplicate fallback component in {source}: {component_key}"
                )
            fallbacks[component_key] = policy
    except (csv.Error, UnicodeError) as exc:
        raise DiagnosticSummaryError(f"Could not parse {path}: {exc}") from exc
    finally:
        stream.close()
    return fallbacks


def _validate_fallback_sources(
    cell_fallbacks: Mapping[ComponentKey, Tuple[str, int]],
    event_fallbacks: Mapping[ComponentKey, Tuple[str, int]],
    table_fallbacks: Mapping[ComponentKey, str],
) -> None:
    cell_policies = {key: value[0] for key, value in cell_fallbacks.items()}
    event_policies = {key: value[0] for key, value in event_fallbacks.items()}
    if cell_policies != event_policies or event_policies != dict(table_fallbacks):
        keys = set(cell_policies) | set(event_policies) | set(table_fallbacks)
        mismatches = [
            key
            for key in keys
            if not (
                cell_policies.get(key)
                == event_policies.get(key)
                == table_fallbacks.get(key)
            )
        ]
        example = mismatches[0] if mismatches else "unknown"
        raise DiagnosticSummaryError(
            "PLIC fallback provenance disagrees across cell_metrics.csv, "
            "merge_events.csv, and unresolved_plic_fallbacks.csv: "
            f"{len(mismatches)} mismatched components; first={example}."
        )
    for key, (_, cell_count) in cell_fallbacks.items():
        event_member_count = event_fallbacks[key][1]
        if cell_count != event_member_count:
            raise DiagnosticSummaryError(
                f"PLIC fallback component {key} has {cell_count} cell rows but "
                f"{event_member_count} archived member cells."
            )


def _metric_rows(
    totals: Mapping[GroupKey, int],
    case_counts: Mapping[GroupKey, int],
    merged_component_counts: Mapping[GroupKey, int],
    incidences: Mapping[Tuple[GroupKey, MetricKey], Incidence],
) -> List[dict]:
    observed_metrics = {metric for _, metric in incidences}
    metrics = sorted(CANONICAL_METRIC_SET | observed_metrics)
    rows = []
    scope_order = {"overall": 0, "experiment": 1, "method": 2}
    ordered_groups = sorted(
        totals,
        key=lambda group: (
            scope_order[group.scope],
            group.experiment,
            group.algo,
        ),
    )
    for group in ordered_groups:
        denominator = totals[group]
        if denominator <= 0:
            raise DiagnosticSummaryError(f"Group has no mixed cells: {group}")
        for category, subtype in metrics:
            incidence = incidences.get((group, (category, subtype)), Incidence())
            component_count = incidence.component_count
            if (category, subtype) == ("merge", "merged"):
                expected = merged_component_counts.get(group, 0)
                if component_count != expected:
                    raise DiagnosticSummaryError(
                        f"Merged-component count mismatch for {group}: "
                        f"cell_metrics={component_count}, case_metrics={expected}"
                    )
            if incidence.mixed_cell_count > denominator:
                raise DiagnosticSummaryError(
                    f"Incidence exceeds mixed-cell denominator for {group}, "
                    f"{category}/{subtype}: {incidence.mixed_cell_count}>{denominator}"
                )
            rows.append(
                {
                    "scope": group.scope,
                    "experiment": group.experiment,
                    "algo": group.algo,
                    "case_count": case_counts.get(group, 0),
                    "mixed_cell_denominator": denominator,
                    "category": category,
                    "subtype": subtype,
                    "mixed_cell_count": incidence.mixed_cell_count,
                    "fraction_of_mixed_cells": incidence.mixed_cell_count / denominator,
                    "component_count": component_count,
                    "event_count": incidence.event_count,
                }
            )
    return rows


def _format_fraction(value: float) -> str:
    return f"{value:.6%}"


def _markdown_summary(release_name: str, manifest: Mapping[str, object], rows: Sequence[dict]) -> str:
    overall = {
        (row["category"], row["subtype"]): row
        for row in rows
        if row["scope"] == "overall"
    }
    overall_any = next(row for row in rows if row["scope"] == "overall")
    case_label = "case" if overall_any["case_count"] == 1 else "cases"
    lines = [
        f"# Submission Diagnostic Summary: `{release_name}`",
        "",
        (
            f"Validated completed sweep: **{manifest['successful_run_count']} / "
            f"{manifest['planned_run_count']} runs**, **{overall_any['case_count']} "
            f"{case_label}**, "
            f"and **{overall_any['mixed_cell_denominator']:,} mixed cells**."
        ),
        "",
        "All fractions below use mixed cells in the displayed scope as the denominator. "
        "Final facet classes describe the submitted reconstruction. Rescue and fallback "
        "cell counts are unique mixed cells belonging to an affected merge component; "
        "their event counts retain repeated assignments.",
        "",
        "## Overall Incidence",
        "",
        "| Diagnostic | Mixed cells | Fraction | Components | Events |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = (
        (("final_facet", "circular"), "Final circular facet"),
        (("final_facet", "corner"), "Final linear corner facet"),
        (("final_facet", "curved_corner"), "Final curved-corner facet"),
        (("merge", "merged"), "Cell in merged component"),
        (("orientation", "unresolved_orientation"), "Unresolved orientation"),
    )
    for key, label in labels:
        row = overall[key]
        lines.append(
            f"| {label} | {row['mixed_cell_count']:,} | "
            f"{_format_fraction(row['fraction_of_mixed_cells'])} | "
            f"{row['component_count']:,} | {row['event_count']:,} |"
        )

    rescue_rows = [
        row
        for key, row in overall.items()
        if key[0] == "rescue" and (
            row["mixed_cell_count"] or key in CANONICAL_METRIC_SET
        )
    ]
    lines.extend(
        [
            "",
            "## Rescue Breakdown",
            "",
            "| Rescue type | Mixed cells | Fraction | Components | Events |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rescue_rows, key=lambda item: item["subtype"]):
        lines.append(
            f"| `{row['subtype']}` | {row['mixed_cell_count']:,} | "
            f"{_format_fraction(row['fraction_of_mixed_cells'])} | "
            f"{row['component_count']:,} | {row['event_count']:,} |"
        )

    plic_rows = [
        row
        for key, row in overall.items()
        if key[0] == "plic_fallback" and (
            row["mixed_cell_count"] or key in CANONICAL_METRIC_SET
        )
    ]
    lines.extend(
        [
            "",
            "## PLIC Fallback Policies",
            "",
            "| Policy | Mixed cells | Fraction | Components | Events |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(plic_rows, key=lambda item: item["subtype"]):
        lines.append(
            f"| `{row['subtype']}` | {row['mixed_cell_count']:,} | "
            f"{_format_fraction(row['fraction_of_mixed_cells'])} | "
            f"{row['component_count']:,} | {row['event_count']:,} |"
        )

    by_method = defaultdict(dict)
    for row in rows:
        if row["scope"] == "method":
            by_method[(row["experiment"], row["algo"])][
                (row["category"], row["subtype"])
            ] = row
    lines.extend(
        [
            "",
            "## Experiment And Method",
            "",
            "| Experiment | Method | Mixed cells | Circular | Corner | Curved corner | Merged | Unresolved | PLIC | Rescue |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for (experiment, algo), metrics in sorted(by_method.items()):
        any_row = next(iter(metrics.values()))

        def fraction(category: str, subtype: str) -> float:
            return metrics[(category, subtype)]["fraction_of_mixed_cells"]

        plic_count = sum(
            row["mixed_cell_count"]
            for key, row in metrics.items()
            if key[0] == "plic_fallback"
        )
        # Rescue types may overlap, so this compact column is an upper-bound sum;
        # the exact non-overlapping type rows remain in CSV/JSON.
        rescue_sum = sum(
            row["mixed_cell_count"]
            for key, row in metrics.items()
            if key[0] == "rescue"
        )
        denominator = any_row["mixed_cell_denominator"]
        lines.append(
            f"| {experiment} | `{algo}` | {denominator:,} | "
            f"{_format_fraction(fraction('final_facet', 'circular'))} | "
            f"{_format_fraction(fraction('final_facet', 'corner'))} | "
            f"{_format_fraction(fraction('final_facet', 'curved_corner'))} | "
            f"{_format_fraction(fraction('merge', 'merged'))} | "
            f"{_format_fraction(fraction('orientation', 'unresolved_orientation'))} | "
            f"{_format_fraction(plic_count / denominator)} | "
            f"{_format_fraction(rescue_sum / denominator)} |"
        )

    fallback_rows = [
        row for key, row in overall.items() if key[0] == "fallback" and row["event_count"]
    ]
    if fallback_rows:
        lines.extend(
            [
                "",
                "## Other Fallbacks",
                "",
                "| Type | Mixed cells | Fraction | Components | Events |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(fallback_rows, key=lambda item: item["subtype"]):
            lines.append(
                f"| `{row['subtype']}` | {row['mixed_cell_count']:,} | "
                f"{_format_fraction(row['fraction_of_mixed_cells'])} | "
                f"{row['component_count']:,} | {row['event_count']:,} |"
            )

    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- `corner` means final facet class `linear_corner`; `curved_corner` is reported separately.",
            "- `unresolved_orientation` requires a structured PLIC fallback event whose reason is exactly `unresolved_orientation`; it does not conflate geometric dead ends.",
            "- `unresolved_or_deadend_status` is retained in CSV/JSON as the broader final cell-state diagnostic.",
            "- PLIC fallback rows are cross-checked across cell metrics, merge events, and the dedicated fallback ledger.",
            "- Rescue types are inferred only when the archived stage/facet pair is unique. Ambiguous rescue provenance is a hard error.",
            "- The compact method-table rescue percentage sums type-specific incidences and can double-count a cell touched by more than one rescue type; CSV/JSON preserve every type separately.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(output_dir: Path, payload: dict, rows: Sequence[dict], markdown: str, overwrite: bool) -> Dict[str, Path]:
    paths = {
        "csv": output_dir / "diagnostic_summary.csv",
        "json": output_dir / "diagnostic_summary.json",
        "markdown": output_dir / "README.md",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise DiagnosticSummaryError(
            "Refusing to overwrite existing diagnostic summary files: "
            + ", ".join(existing)
            + ". Pass --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "scope",
        "experiment",
        "algo",
        "case_count",
        "mixed_cell_denominator",
        "category",
        "subtype",
        "mixed_cell_count",
        "fraction_of_mixed_cells",
        "component_count",
        "event_count",
    ]
    csv_tmp = paths["csv"].with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["fraction_of_mixed_cells"] = format(
                row["fraction_of_mixed_cells"], ".17g"
            )
            writer.writerow(formatted)
    csv_tmp.replace(paths["csv"])

    json_tmp = paths["json"].with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    json_tmp.replace(paths["json"])

    markdown_tmp = paths["markdown"].with_suffix(".md.tmp")
    markdown_tmp.write_text(markdown, encoding="utf-8")
    markdown_tmp.replace(paths["markdown"])
    return paths


def summarize_release(
    release_root: Path,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    release_root = Path(release_root).resolve()
    if not release_root.is_dir():
        raise DiagnosticSummaryError(f"Release root does not exist: {release_root}")
    manifest = _require_completed_release(release_root)
    diagnostics = release_root / "diagnostics"

    case_mixed_cells, case_counts, merged_component_counts = _read_case_metrics(
        diagnostics / "case_metrics.csv"
    )
    _validate_case_geometry(
        diagnostics / "case_geometry.jsonl", set(case_mixed_cells)
    )

    incidences: MutableMapping[Tuple[GroupKey, MetricKey], Incidence] = defaultdict(
        Incidence
    )
    totals, cell_fallbacks, _ = _read_cells(
        diagnostics / "cell_metrics.csv", case_mixed_cells, incidences
    )
    event_fallbacks = _read_events(
        diagnostics / "merge_events.csv", set(case_mixed_cells), incidences
    )
    fallback_table = _read_fallback_table(
        diagnostics / "unresolved_plic_fallbacks.csv", set(case_mixed_cells)
    )
    _validate_fallback_sources(cell_fallbacks, event_fallbacks, fallback_table)

    rows = _metric_rows(
        totals, case_counts, merged_component_counts, incidences
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release": {
            "name": release_root.name,
            "planned_run_count": manifest["planned_run_count"],
            "successful_run_count": manifest["successful_run_count"],
            "failure_count": manifest["failure_count"],
        },
        "definitions": {
            "denominator": "unique mixed-cell rows within each scope",
            "corner": "final_facet_class == linear_corner",
            "curved_corner": "final_facet_class == curved_corner",
            "unresolved_orientation": (
                "event_kind == plic_fallback and fallback_reason == "
                "unresolved_orientation"
            ),
            "rescue_cell_count": (
                "unique mixed cells in components touched by a given rescue type"
            ),
        },
        "groups": rows,
    }
    markdown = _markdown_summary(release_root.name, manifest, rows)
    target = Path(output_dir).resolve() if output_dir else release_root / "diagnostic_summary"
    paths = _write_outputs(target, payload, rows, markdown, overwrite)
    return {"payload": payload, "paths": paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize final facet, merge, rescue, orientation, and PLIC fallback "
            "incidence from a completed static release."
        )
    )
    parser.add_argument("release_root", type=Path, help="completed release directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <release_root>/diagnostic_summary)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing summary in the output directory",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = summarize_release(
            args.release_root, output_dir=args.output_dir, overwrite=args.overwrite
        )
    except DiagnosticSummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for label, path in result["paths"].items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
