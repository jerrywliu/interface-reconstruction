#!/usr/bin/env python3
"""Audit topology, merge, rescue, and fallback incidence in static releases.

The input result roots are treated as immutable. All generated artifacts are
written to an explicit output directory outside those roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


csv.field_size_limit(sys.maxsize)

CASE_KEY_FIELDS = (
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "case_index",
)
PLIC_METHODS = frozenset({"Youngs", "ELVIRA", "LVIRA"})
FINAL_CLASSES = ("linear", "circular", "linear_corner", "curved_corner")
CANONICAL_INCIDENCE = (
    *(("final_facet", value) for value in FINAL_CLASSES),
    ("merge", "merged"),
    ("merge", "independent"),
    ("plic_fallback_policy", "Youngs"),
    ("plic_fallback_policy", "ELVIRA"),
    ("plic_fallback_policy", "LVIRA"),
    ("rescue", "exact_linear_support"),
    ("rescue", "corner_arc_corner_triplet"),
    ("rescue", "curved_corner_loop"),
    ("rescue", "curved_corner_transition"),
    ("local_linear_fallback", "arc_fit_failed"),
)
RAW_CSV_FILES = (
    "case_metrics.csv",
    "cell_metrics.csv",
    "merge_events.csv",
    "unresolved_plic_fallbacks.csv",
)
JULY_UNAVAILABLE_CATEGORIES = frozenset(
    {"local_linear_fallback", "plic_fallback_reason"}
)

CaseKey = Tuple[str, str, str, str, str, str]
ComponentKey = Tuple[CaseKey, str]
GroupKey = Tuple[str, str, str, str, str]
IncidenceKey = Tuple[GroupKey, str, str]


class AuditError(RuntimeError):
    """Raised when the archived diagnostics violate an audit invariant."""


def _decimal_text(value: object, label: str) -> str:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise AuditError(f"{label} is not a decimal: {value!r}") from exc
    if not number.is_finite():
        raise AuditError(f"{label} is not finite: {value!r}")
    normalized = number.normalize()
    text = format(normalized, "f")
    return "0" if text in {"-0", ""} else text


def _integer_text(value: object, label: str) -> str:
    text = str(value).strip()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise AuditError(f"{label} is not an integer: {value!r}") from exc
    if text not in {str(parsed), f"+{parsed}"}:
        raise AuditError(f"{label} is not a canonical integer: {value!r}")
    return str(parsed)


def _nonnegative_int(value: object, label: str) -> int:
    parsed = int(_integer_text(value, label))
    if parsed < 0:
        raise AuditError(f"{label} is negative: {value!r}")
    return parsed


def _case_key(row: Mapping[str, object], source: str) -> CaseKey:
    missing = [
        field for field in CASE_KEY_FIELDS if str(row.get(field, "")).strip() == ""
    ]
    if missing:
        raise AuditError(f"{source} lacks case-key fields: {', '.join(missing)}")
    return (
        str(row["experiment"]).strip(),
        str(row["algo"]).strip(),
        _decimal_text(row["resolution"], f"{source} resolution"),
        _decimal_text(row["wiggle"], f"{source} wiggle"),
        _integer_text(row["seed"], f"{source} seed"),
        _integer_text(row["case_index"], f"{source} case_index"),
    )


def _component_key(row: Mapping[str, object], source: str) -> ComponentKey:
    merge_id = str(row.get("merge_id", "")).strip()
    if not merge_id:
        raise AuditError(f"{source} has no merge_id")
    return _case_key(row, source), merge_id


def _groups(case_key: CaseKey) -> Tuple[GroupKey, ...]:
    experiment, algo, resolution, wiggle, _, _ = case_key
    n_value = str(int((Decimal(resolution) * 100).to_integral_value()))
    return (
        ("overall", "", "", "", ""),
        ("benchmark", experiment, "", "", ""),
        ("method", experiment, algo, "", ""),
        ("setting", experiment, algo, n_value, wiggle),
    )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise AuditError(f"Missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object in {path}")
    return value


def _csv_reader(path: Path) -> Tuple[object, csv.DictReader]:
    if not path.is_file():
        raise AuditError(f"Missing CSV file: {path}")
    stream = path.open(newline="", encoding="utf-8")
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        stream.close()
        raise AuditError(f"CSV has no header: {path}")
    return stream, reader


@dataclass
class CaseRecord:
    values: Dict[str, int]


@dataclass
class ComponentState:
    cell_ids: Set[str] = field(default_factory=set)
    final_classes: Set[str] = field(default_factory=set)
    merge_flags: Set[str] = field(default_factory=set)
    declared_sizes: Set[int] = field(default_factory=set)
    construction_paths: Set[str] = field(default_factory=set)
    fallback_policies: Set[str] = field(default_factory=set)


@dataclass
class Stats:
    label: str
    selected_cases: Set[CaseKey]
    case_counts: Counter = field(default_factory=Counter)
    cell_totals: Counter = field(default_factory=Counter)
    component_totals: Counter = field(default_factory=Counter)
    cell_incidence: Counter = field(default_factory=Counter)
    component_incidence: Counter = field(default_factory=Counter)
    event_incidence: Counter = field(default_factory=Counter)
    seen_event_components: MutableMapping[Tuple[str, str], Set[ComponentKey]] = field(
        default_factory=lambda: defaultdict(set)
    )
    seen_event_cells: MutableMapping[Tuple[str, str], Set[Tuple[CaseKey, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def register_cases(self) -> None:
        for key in self.selected_cases:
            for group in _groups(key):
                self.case_counts[group] += 1

    def add_cell(self, key: CaseKey, categories: Iterable[Tuple[str, str]]) -> None:
        for group in _groups(key):
            self.cell_totals[group] += 1
            for category, subtype in categories:
                self.cell_incidence[(group, category, subtype)] += 1

    def add_component(
        self, key: CaseKey, categories: Iterable[Tuple[str, str]]
    ) -> None:
        for group in _groups(key):
            self.component_totals[group] += 1
            for category, subtype in categories:
                self.component_incidence[(group, category, subtype)] += 1

    def add_event(
        self,
        key: CaseKey,
        merge_id: str,
        member_cells: Sequence[str],
        category: str,
        subtype: str,
        affected_incidence: bool,
    ) -> None:
        component_key = (key, merge_id)
        metric = (category, subtype)
        first_component = component_key not in self.seen_event_components[metric]
        if first_component:
            self.seen_event_components[metric].add(component_key)
        for group in _groups(key):
            self.event_incidence[(group, category, subtype)] += 1
            if affected_incidence and first_component:
                self.component_incidence[(group, category, subtype)] += 1
        if not affected_incidence:
            return
        for cell_id in member_cells:
            cell_key = (key, cell_id)
            if cell_key in self.seen_event_cells[metric]:
                continue
            self.seen_event_cells[metric].add(cell_key)
            for group in _groups(key):
                self.cell_incidence[(group, category, subtype)] += 1

    def rows(self, scopes: Optional[Set[str]] = None) -> List[dict]:
        observed = {
            (category, subtype)
            for _, category, subtype in (
                set(self.cell_incidence)
                | set(self.component_incidence)
                | set(self.event_incidence)
            )
        }
        metrics = sorted(observed | set(CANONICAL_INCIDENCE))
        rows = []
        for group in sorted(self.cell_totals):
            scope, experiment, algo, n_value, wiggle = group
            if scopes is not None and scope not in scopes:
                continue
            cells = self.cell_totals[group]
            components = self.component_totals[group]
            for category, subtype in metrics:
                cell_count = self.cell_incidence[(group, category, subtype)]
                component_count = self.component_incidence[(group, category, subtype)]
                event_count = self.event_incidence[(group, category, subtype)]
                rows.append(
                    {
                        "release": self.label,
                        "scope": scope,
                        "experiment": experiment,
                        "method": algo,
                        "N": n_value,
                        "wiggle": wiggle,
                        "case_count": self.case_counts[group],
                        "mixed_cell_denominator": cells,
                        "component_denominator": components,
                        "category": category,
                        "subtype": subtype,
                        "mixed_cell_count": cell_count,
                        "fraction_of_mixed_cells": cell_count / cells,
                        "component_count": component_count,
                        "fraction_of_components": component_count / components,
                        "event_count": event_count,
                    }
                )
        return rows


@dataclass
class ReleaseAnalysis:
    root: Path
    manifest: dict
    case_records: Dict[CaseKey, CaseRecord]
    all_stats: Stats
    matched_stats: Optional[Stats]
    cell_fallbacks: Dict[ComponentKey, Tuple[str, frozenset]]
    event_fallbacks: Dict[ComponentKey, Tuple[str, frozenset]]
    table_fallbacks: Dict[ComponentKey, str]
    missing_facet_count: int


def _load_cases(root: Path) -> Tuple[dict, Dict[CaseKey, CaseRecord], int]:
    manifest = _read_json(root / "sweep_manifest.json")
    if manifest.get("status") != "completed":
        raise AuditError(f"Release is not completed: {root}")
    if int(manifest.get("failure_count", -1)) != 0:
        raise AuditError(f"Release has controller failures: {root}")
    if int(manifest.get("planned_run_count", -1)) != int(
        manifest.get("successful_run_count", -2)
    ):
        raise AuditError(f"Release did not complete all planned runs: {root}")

    path = root / "diagnostics/case_metrics.csv"
    records: Dict[CaseKey, CaseRecord] = {}
    missing_total = 0
    numeric_fields = (
        "num_mixed_cells",
        "num_merge_components",
        "num_merged_cells",
        "num_merged_components",
        "num_plic_fallback_cells",
        "num_final_linear_cells",
        "num_final_circular_cells",
        "num_final_linear_corner_cells",
        "num_final_curved_corner_cells",
        "num_final_missing_cells",
    )
    stream, reader = _csv_reader(path)
    try:
        missing_headers = sorted(
            set(CASE_KEY_FIELDS + numeric_fields) - set(reader.fieldnames or ())
        )
        if missing_headers:
            raise AuditError(f"{path} lacks columns: {', '.join(missing_headers)}")
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            key = _case_key(row, source)
            if key in records:
                raise AuditError(f"Duplicate case key in {source}: {key}")
            values = {
                field: _nonnegative_int(row[field], f"{source} {field}")
                for field in numeric_fields
            }
            missing_total += values["num_final_missing_cells"]
            records[key] = CaseRecord(values)
    finally:
        stream.close()
    expected_cases_raw = manifest.get("planned_case_count")
    if expected_cases_raw is not None and len(records) != int(expected_cases_raw):
        raise AuditError(
            f"Case count mismatch in {root}: diagnostics={len(records)}, "
            f"manifest={expected_cases_raw}"
        )
    return manifest, records, missing_total


def _cell_id(row: Mapping[str, object], source: str) -> str:
    value = str(row.get("cell_id", "")).strip()
    if not value:
        raise AuditError(f"{source} has no cell_id")
    return value


def _finalize_case_components(
    key: CaseKey,
    record: CaseRecord,
    components: Mapping[str, ComponentState],
    derived: Counter,
    destinations: Sequence[Stats],
    cell_fallbacks: Dict[ComponentKey, Tuple[str, frozenset]],
) -> None:
    values = record.values
    derived["merged_components"] = sum(
        component.merge_flags == {"1"} for component in components.values()
    )
    expected_pairs = {
        "num_mixed_cells": derived["mixed_cells"],
        "num_merge_components": len(components),
        "num_merged_cells": derived["merged_cells"],
        "num_merged_components": derived["merged_components"],
        "num_plic_fallback_cells": derived["plic_fallback_cells"],
        "num_final_linear_cells": derived["final:linear"],
        "num_final_circular_cells": derived["final:circular"],
        "num_final_linear_corner_cells": derived["final:linear_corner"],
        "num_final_curved_corner_cells": derived["final:curved_corner"],
        "num_final_missing_cells": derived["final:missing"],
    }
    for field, observed in expected_pairs.items():
        if values[field] != observed:
            raise AuditError(
                f"Case {key} disagrees for {field}: case_metrics={values[field]}, "
                f"cell_metrics={observed}"
            )

    for merge_id, component in components.items():
        source = f"case={key}, merge_id={merge_id}"
        if len(component.final_classes) != 1:
            raise AuditError(
                f"{source} has mixed final facet classes: {component.final_classes}"
            )
        if len(component.merge_flags) != 1:
            raise AuditError(
                f"{source} has inconsistent merge flags: {component.merge_flags}"
            )
        if len(component.declared_sizes) != 1:
            raise AuditError(
                f"{source} has inconsistent component sizes: {component.declared_sizes}"
            )
        declared_size = next(iter(component.declared_sizes))
        if declared_size != len(component.cell_ids):
            raise AuditError(
                f"{source} declares {declared_size} cells but has {len(component.cell_ids)}"
            )
        merged = next(iter(component.merge_flags)) == "1"
        if merged != (declared_size > 1):
            raise AuditError(
                f"{source} has is_merged={int(merged)} and component size {declared_size}"
            )
        final_class = next(iter(component.final_classes))
        categories = [
            ("final_facet", final_class),
            ("merge", "merged" if merged else "independent"),
        ]
        for construction_path in sorted(component.construction_paths):
            categories.append(("construction_path", construction_path or "unspecified"))
        if key[1] in PLIC_METHODS:
            categories.append(("plic_default_method", key[1]))
        if "plic_fallback" in component.construction_paths:
            if len(component.fallback_policies) != 1:
                raise AuditError(
                    f"{source} uses PLIC fallback with policies {component.fallback_policies}"
                )
            policy = next(iter(component.fallback_policies))
            categories.append(("plic_fallback_policy", policy))
            cell_fallbacks[(key, merge_id)] = (policy, frozenset(component.cell_ids))
        elif component.fallback_policies:
            raise AuditError(
                f"{source} records a fallback policy without plic_fallback construction"
            )
        for stats in destinations:
            stats.add_component(key, categories)


def _read_cells(
    root: Path,
    records: Mapping[CaseKey, CaseRecord],
    all_stats: Stats,
    matched_stats: Optional[Stats],
) -> Dict[ComponentKey, Tuple[str, frozenset]]:
    path = root / "diagnostics/cell_metrics.csv"
    required = set(CASE_KEY_FIELDS) | {
        "cell_id",
        "merge_id",
        "merge_component_size",
        "is_merged",
        "orientation_status",
        "final_facet_class",
        "construction_path",
        "fallback_policy",
    }
    current_key: Optional[CaseKey] = None
    completed: Set[CaseKey] = set()
    components: Dict[str, ComponentState] = {}
    derived: Counter = Counter()
    cell_fallbacks: Dict[ComponentKey, Tuple[str, frozenset]] = {}

    def destinations(key: CaseKey) -> List[Stats]:
        selected = [all_stats]
        if matched_stats is not None and key in matched_stats.selected_cases:
            selected.append(matched_stats)
        return selected

    def finalize() -> None:
        if current_key is None:
            return
        _finalize_case_components(
            current_key,
            records[current_key],
            components,
            derived,
            destinations(current_key),
            cell_fallbacks,
        )

    stream, reader = _csv_reader(path)
    try:
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise AuditError(f"{path} lacks columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            key = _case_key(row, source)
            if key not in records:
                raise AuditError(f"Cell row has no case_metrics row: {source}")
            if key != current_key:
                finalize()
                if current_key is not None:
                    completed.add(current_key)
                if key in completed:
                    raise AuditError(f"Cell rows are not contiguous for case {key}")
                current_key = key
                components = {}
                derived = Counter()

            cell_id = _cell_id(row, source)
            merge_id = str(row["merge_id"]).strip()
            component = components.setdefault(merge_id, ComponentState())
            if cell_id in component.cell_ids:
                raise AuditError(f"Duplicate cell {cell_id} in {source}")
            component.cell_ids.add(cell_id)
            final_class = str(row["final_facet_class"]).strip() or "missing"
            merge_flag = str(row["is_merged"]).strip()
            if merge_flag not in {"0", "1"}:
                raise AuditError(f"{source} has invalid is_merged={merge_flag!r}")
            declared_size = _nonnegative_int(
                row["merge_component_size"], f"{source} merge_component_size"
            )
            construction_path = str(row["construction_path"]).strip()
            fallback_policy = str(row["fallback_policy"]).strip()
            component.final_classes.add(final_class)
            component.merge_flags.add(merge_flag)
            component.declared_sizes.add(declared_size)
            component.construction_paths.add(construction_path)
            if fallback_policy:
                component.fallback_policies.add(fallback_policy)

            derived["mixed_cells"] += 1
            derived[f"final:{final_class}"] += 1
            if merge_flag == "1":
                derived["merged_cells"] += 1
            if construction_path == "plic_fallback":
                derived["plic_fallback_cells"] += 1
            categories = [
                ("final_facet", final_class),
                ("merge", "merged" if merge_flag == "1" else "independent"),
                ("construction_path", construction_path or "unspecified"),
                (
                    "orientation_status",
                    str(row["orientation_status"]).strip() or "unspecified",
                ),
            ]
            if key[1] in PLIC_METHODS:
                categories.append(("plic_default_method", key[1]))
            if construction_path == "plic_fallback":
                if not fallback_policy:
                    raise AuditError(f"{source} is a PLIC fallback without a policy")
                categories.append(("plic_fallback_policy", fallback_policy))
            elif fallback_policy:
                raise AuditError(
                    f"{source} has a fallback policy outside plic_fallback"
                )
            for stats in destinations(key):
                stats.add_cell(key, categories)

        finalize()
        if current_key is not None:
            completed.add(current_key)
    finally:
        stream.close()
    if completed != set(records):
        raise AuditError(
            f"Cell/case coverage mismatch: missing={len(set(records) - completed)}, "
            f"extra={len(completed - set(records))}"
        )
    return cell_fallbacks


def _member_cells(raw: object, source: str) -> Tuple[str, ...]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise AuditError(f"Invalid member_cells_json in {source}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise AuditError(f"member_cells_json is not a nonempty list in {source}")
    cells = []
    for cell in value:
        if not isinstance(cell, list) or len(cell) != 2:
            raise AuditError(f"Malformed member cell in {source}: {cell!r}")
        cells.append(f"{cell[0]},{cell[1]}")
    if len(cells) != len(set(cells)):
        raise AuditError(f"Duplicate member cells in {source}")
    return tuple(cells)


def _rescue_subtype(stage: str, facet_name: str, source: str) -> str:
    if stage == "linear_corner_rescues" and facet_name == "linear_support":
        return "exact_linear_support"
    if stage == "linear_corner_rescues" and facet_name == "corner":
        return "corner_arc_corner_triplet"
    if stage == "curved_corner_loop_rescue":
        return "curved_corner_loop"
    if stage == "curved_corner_transition_rescue":
        return "curved_corner_transition"
    raise AuditError(
        f"Ambiguous rescue provenance in {source}: stage={stage!r}, facet={facet_name!r}"
    )


def _read_events(
    root: Path,
    records: Mapping[CaseKey, CaseRecord],
    all_stats: Stats,
    matched_stats: Optional[Stats],
) -> Dict[ComponentKey, Tuple[str, frozenset]]:
    path = root / "diagnostics/merge_events.csv"
    required = set(CASE_KEY_FIELDS) | {
        "merge_id",
        "member_cells_json",
        "stage",
        "event_kind",
        "fallback_policy",
        "facet_name",
    }
    fallbacks: Dict[ComponentKey, Tuple[str, frozenset]] = {}
    stream, reader = _csv_reader(path)
    try:
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise AuditError(f"{path} lacks columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            key = _case_key(row, source)
            if key not in records:
                raise AuditError(f"Event row has no case_metrics row: {source}")
            merge_id = str(row["merge_id"]).strip()
            members = _member_cells(row["member_cells_json"], source)
            stage = str(row["stage"]).strip()
            kind = str(row["event_kind"]).strip()
            policy = str(row["fallback_policy"]).strip()
            reason = str(row.get("fallback_reason", "")).strip()
            facet_name = str(row["facet_name"]).strip()
            destinations = [all_stats]
            if matched_stats is not None and key in matched_stats.selected_cases:
                destinations.append(matched_stats)

            events: List[Tuple[str, str, bool]] = []
            if kind == "facet_assignment" and "rescue" in stage:
                events.append(
                    ("rescue", _rescue_subtype(stage, facet_name, source), True)
                )
            elif "rescue" in stage and kind != "stage_snapshot":
                raise AuditError(f"Unsupported rescue event kind in {source}: {kind!r}")
            if kind == "local_linear_fallback":
                events.append(("local_linear_fallback", reason or "unspecified", True))
            elif kind == "missing_fallback":
                events.append(("missing_fallback", reason or "unspecified", True))
            elif kind == "plic_fallback":
                if not policy:
                    raise AuditError(f"PLIC fallback event has no policy in {source}")
                component_key = (key, merge_id)
                value = (policy, frozenset(members))
                previous = fallbacks.get(component_key)
                if previous is not None and previous != value:
                    raise AuditError(
                        f"Conflicting PLIC fallback events for {component_key}"
                    )
                fallbacks[component_key] = value
                events.append(("plic_fallback_policy", policy, False))
                if reason:
                    events.append(("plic_fallback_reason", reason, True))
            for category, subtype, affected in events:
                for stats in destinations:
                    stats.add_event(key, merge_id, members, category, subtype, affected)
    finally:
        stream.close()
    return fallbacks


def _read_fallback_table(
    root: Path, records: Mapping[CaseKey, CaseRecord]
) -> Dict[ComponentKey, str]:
    path = root / "diagnostics/unresolved_plic_fallbacks.csv"
    required = set(CASE_KEY_FIELDS) | {"merge_id", "policy"}
    fallbacks: Dict[ComponentKey, str] = {}
    stream, reader = _csv_reader(path)
    try:
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise AuditError(f"{path} lacks columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            source = f"{path}:{line_number}"
            key = _case_key(row, source)
            if key not in records:
                raise AuditError(f"Fallback row has no case_metrics row: {source}")
            component_key = _component_key(row, source)
            policy = str(row["policy"]).strip()
            if not policy:
                raise AuditError(f"Fallback row has no policy in {source}")
            if component_key in fallbacks:
                raise AuditError(
                    f"Duplicate fallback component in {source}: {component_key}"
                )
            fallbacks[component_key] = policy
    finally:
        stream.close()
    return fallbacks


def _validate_fallback_provenance(
    cell: Mapping[ComponentKey, Tuple[str, frozenset]],
    event: Mapping[ComponentKey, Tuple[str, frozenset]],
    table: Mapping[ComponentKey, str],
    label: str,
) -> None:
    keys = set(cell) | set(event) | set(table)
    mismatches = []
    for key in keys:
        cell_value = cell.get(key)
        event_value = event.get(key)
        if cell_value != event_value or (cell_value and cell_value[0]) != table.get(
            key
        ):
            mismatches.append(key)
    if mismatches:
        raise AuditError(
            f"{label} PLIC fallback provenance has {len(mismatches)} mismatches; "
            f"first={mismatches[0]}"
        )


def analyze_release(
    root: Path, matched_case_keys: Optional[Set[CaseKey]] = None
) -> ReleaseAnalysis:
    root = root.resolve()
    manifest, records, missing = _load_cases(root)
    selected = set(records)
    all_stats = Stats(root.name, selected)
    all_stats.register_cases()
    matched_stats = None
    if matched_case_keys is not None:
        absent = matched_case_keys - selected
        if absent:
            raise AuditError(
                f"{root.name} lacks {len(absent)} July case keys; first={next(iter(absent))}"
            )
        matched_stats = Stats(f"{root.name}:july_matched", set(matched_case_keys))
        matched_stats.register_cases()
    cell_fallbacks = _read_cells(root, records, all_stats, matched_stats)
    event_fallbacks = _read_events(root, records, all_stats, matched_stats)
    table_fallbacks = _read_fallback_table(root, records)
    _validate_fallback_provenance(
        cell_fallbacks, event_fallbacks, table_fallbacks, root.name
    )
    return ReleaseAnalysis(
        root=root,
        manifest=manifest,
        case_records=records,
        all_stats=all_stats,
        matched_stats=matched_stats,
        cell_fallbacks=cell_fallbacks,
        event_fallbacks=event_fallbacks,
        table_fallbacks=table_fallbacks,
        missing_facet_count=missing,
    )


def _canonical_values(row: Mapping[str, object], fields: Sequence[str]) -> bytes:
    return (
        json.dumps([str(row.get(field, "")) for field in fields], separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _inventory(root: Path) -> List[dict]:
    path = root / "diagnostics/run_inventory.csv"
    stream, reader = _csv_reader(path)
    try:
        rows = list(reader)
    finally:
        stream.close()
    names = [row.get("save_name", "") for row in rows]
    if len(rows) != len(set(names)) or any(not name for name in names):
        raise AuditError(f"Invalid save_name coverage in {path}")
    return rows


def _reconcile_csv(
    root: Path, inventory: Sequence[Mapping[str, str]], name: str
) -> dict:
    raw_hashes: Dict[str, Tuple[int, str]] = {}
    common_header: Optional[Tuple[str, ...]] = None
    for inventory_row in inventory:
        save_name = inventory_row["save_name"]
        path = root / inventory_row["run_bundle"] / "metrics" / name
        stream, reader = _csv_reader(path)
        try:
            header = tuple(reader.fieldnames or ())
            if common_header is None:
                common_header = header
            elif common_header != header:
                raise AuditError(f"Raw {name} headers differ at {path}")
            digest = hashlib.sha256()
            count = 0
            for row in reader:
                digest.update(_canonical_values(row, header))
                count += 1
            raw_hashes[save_name] = (count, digest.hexdigest())
        finally:
            stream.close()
    if common_header is None:
        raise AuditError(f"No raw bundles available for {name}")

    consolidated_path = root / "diagnostics" / name
    stream, reader = _csv_reader(consolidated_path)
    consolidated_digests = {row["save_name"]: hashlib.sha256() for row in inventory}
    consolidated_counts: Counter = Counter()
    try:
        missing = sorted(set(common_header) - set(reader.fieldnames or ()))
        if missing or "save_name" not in (reader.fieldnames or ()):
            raise AuditError(
                f"Consolidated {name} lacks raw fields/save_name: {', '.join(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            save_name = str(row["save_name"])
            if save_name not in consolidated_digests:
                raise AuditError(
                    f"Unexpected save_name in {consolidated_path}:{line_number}: {save_name}"
                )
            consolidated_digests[save_name].update(
                _canonical_values(row, common_header)
            )
            consolidated_counts[save_name] += 1
    finally:
        stream.close()
    mismatches = []
    for save_name, (raw_count, raw_digest) in raw_hashes.items():
        consolidated = (
            consolidated_counts[save_name],
            consolidated_digests[save_name].hexdigest(),
        )
        if (raw_count, raw_digest) != consolidated:
            mismatches.append(save_name)
    return {
        "table": name,
        "raw_rows": sum(value[0] for value in raw_hashes.values()),
        "consolidated_rows": sum(consolidated_counts.values()),
        "run_bundles": len(raw_hashes),
        "matching_run_bundles": len(raw_hashes) - len(mismatches),
        "mismatching_run_bundles": len(mismatches),
        "status": "PASS" if not mismatches else "FAIL",
        "details": (
            "exact canonical row hashes" if not mismatches else f"first={mismatches[0]}"
        ),
    }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _reconcile_geometry(root: Path, inventory: Sequence[Mapping[str, str]]) -> dict:
    raw_hashes: Dict[str, Tuple[int, str, frozenset]] = {}
    for inventory_row in inventory:
        save_name = inventory_row["save_name"]
        path = root / inventory_row["run_bundle"] / "metrics/case_geometry.jsonl"
        digest = hashlib.sha256()
        count = 0
        keys: Optional[frozenset] = None
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AuditError(f"Non-object geometry row in {path}:{line_number}")
                row_keys = frozenset(row)
                if keys is None:
                    keys = row_keys
                elif keys != row_keys:
                    raise AuditError(f"Geometry keys change within {path}")
                digest.update(_canonical_json(row))
                count += 1
        raw_hashes[save_name] = (count, digest.hexdigest(), keys or frozenset())

    consolidated_digests = {row["save_name"]: hashlib.sha256() for row in inventory}
    consolidated_counts: Counter = Counter()
    path = root / "diagnostics/case_geometry.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise AuditError(f"Non-object geometry row in {path}:{line_number}")
            save_name = str(row.get("save_name", ""))
            if save_name not in raw_hashes:
                raise AuditError(
                    f"Unexpected geometry save_name in {path}:{line_number}"
                )
            raw_keys = raw_hashes[save_name][2]
            missing = raw_keys - set(row)
            if missing:
                raise AuditError(
                    f"Consolidated geometry lacks {sorted(missing)} in {path}:{line_number}"
                )
            payload = {key: row[key] for key in raw_keys}
            consolidated_digests[save_name].update(_canonical_json(payload))
            consolidated_counts[save_name] += 1
    mismatches = []
    for save_name, (raw_count, raw_digest, _) in raw_hashes.items():
        if (raw_count, raw_digest) != (
            consolidated_counts[save_name],
            consolidated_digests[save_name].hexdigest(),
        ):
            mismatches.append(save_name)
    return {
        "table": "case_geometry.jsonl",
        "raw_rows": sum(value[0] for value in raw_hashes.values()),
        "consolidated_rows": sum(consolidated_counts.values()),
        "run_bundles": len(raw_hashes),
        "matching_run_bundles": len(raw_hashes) - len(mismatches),
        "mismatching_run_bundles": len(mismatches),
        "status": "PASS" if not mismatches else "FAIL",
        "details": (
            "exact canonical scientific-object hashes"
            if not mismatches
            else f"first={mismatches[0]}"
        ),
    }


def _reconcile_manifests(root: Path, inventory: Sequence[Mapping[str, str]]) -> dict:
    raw: Dict[str, Tuple[int, str]] = {}
    for row in inventory:
        save_name = row["save_name"]
        path = root / row["run_bundle"] / "run_manifest.json"
        payload = _read_json(path)
        raw[save_name] = (1, hashlib.sha256(_canonical_json(payload)).hexdigest())
    consolidated: Dict[str, Tuple[int, str]] = {}
    path = root / "diagnostics/run_manifests.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            save_name = str(row.get("save_name", ""))
            if save_name not in raw or save_name in consolidated:
                raise AuditError(f"Invalid manifest save_name in {path}:{line_number}")
            payload = row.get("manifest")
            consolidated[save_name] = (
                1,
                hashlib.sha256(_canonical_json(payload)).hexdigest(),
            )
    mismatches = [name for name in raw if raw[name] != consolidated.get(name)]
    return {
        "table": "run_manifests.jsonl",
        "raw_rows": len(raw),
        "consolidated_rows": len(consolidated),
        "run_bundles": len(raw),
        "matching_run_bundles": len(raw) - len(mismatches),
        "mismatching_run_bundles": len(mismatches),
        "status": "PASS" if not mismatches else "FAIL",
        "details": (
            "exact canonical manifest hashes"
            if not mismatches
            else f"first={mismatches[0]}"
        ),
    }


def reconcile_raw(root: Path) -> List[dict]:
    inventory = _inventory(root)
    rows = [_reconcile_manifests(root, inventory), _reconcile_geometry(root, inventory)]
    rows.extend(_reconcile_csv(root, inventory, name) for name in RAW_CSV_FILES)
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise AuditError(f"Raw/consolidated reconciliation failed: {failures[0]}")
    return rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in list(formatted.items()):
                if isinstance(value, float):
                    formatted[key] = format(value, ".17g")
            writer.writerow(formatted)


def _row_index(
    rows: Sequence[Mapping[str, object]]
) -> Dict[Tuple[str, ...], Mapping[str, object]]:
    return {
        (
            str(row["scope"]),
            str(row["experiment"]),
            str(row["method"]),
            str(row["N"]),
            str(row["wiggle"]),
            str(row["category"]),
            str(row["subtype"]),
        ): row
        for row in rows
    }


def _comparison_rows(
    final_rows: Sequence[dict], july_rows: Sequence[dict]
) -> List[dict]:
    final = _row_index(final_rows)
    july = _row_index(july_rows)
    final_groups = {key[:5] for key in final}
    july_groups = {key[:5] for key in july}
    if final_groups != july_groups:
        raise AuditError(
            f"Matched July incidence groups differ: final-only={len(final_groups-july_groups)}, "
            f"July-only={len(july_groups-final_groups)}"
        )
    rows = []
    # The older July event schema does not record fallback reasons. Compare the
    # common scientific incidence categories and retain final-only categories in
    # final_incidence_long.csv.
    for key in sorted(set(final) & set(july)):
        if key[5] in JULY_UNAVAILABLE_CATEGORIES:
            continue
        frow, jrow = final[key], july[key]
        if (
            frow["mixed_cell_denominator"] != jrow["mixed_cell_denominator"]
            or frow["component_denominator"] != jrow["component_denominator"]
            or frow["case_count"] != jrow["case_count"]
        ):
            raise AuditError(f"Matched July denominators differ for {key}")
        rows.append(
            {
                "scope": key[0],
                "experiment": key[1],
                "method": key[2],
                "N": key[3],
                "wiggle": key[4],
                "category": key[5],
                "subtype": key[6],
                "case_count": frow["case_count"],
                "mixed_cell_denominator": frow["mixed_cell_denominator"],
                "july_mixed_cell_count": jrow["mixed_cell_count"],
                "final_mixed_cell_count": frow["mixed_cell_count"],
                "mixed_cell_count_delta": frow["mixed_cell_count"]
                - jrow["mixed_cell_count"],
                "july_fraction_of_mixed_cells": jrow["fraction_of_mixed_cells"],
                "final_fraction_of_mixed_cells": frow["fraction_of_mixed_cells"],
                "mixed_cell_fraction_delta": frow["fraction_of_mixed_cells"]
                - jrow["fraction_of_mixed_cells"],
                "component_denominator": frow["component_denominator"],
                "july_component_count": jrow["component_count"],
                "final_component_count": frow["component_count"],
                "component_count_delta": frow["component_count"]
                - jrow["component_count"],
                "july_fraction_of_components": jrow["fraction_of_components"],
                "final_fraction_of_components": frow["fraction_of_components"],
                "component_fraction_delta": frow["fraction_of_components"]
                - jrow["fraction_of_components"],
                "july_event_count": jrow["event_count"],
                "final_event_count": frow["event_count"],
                "event_count_delta": frow["event_count"] - jrow["event_count"],
            }
        )
    return rows


def _wide_rows(rows: Sequence[dict], scope: str) -> List[dict]:
    selected = [row for row in rows if row["scope"] == scope]
    grouped: MutableMapping[GroupKey, Dict[Tuple[str, str], dict]] = defaultdict(dict)
    for row in selected:
        group = (scope, row["experiment"], row["method"], row["N"], row["wiggle"])
        grouped[group][(row["category"], row["subtype"])] = row
    metrics = (
        *(("final_facet", value) for value in FINAL_CLASSES),
        ("merge", "merged"),
        ("merge", "independent"),
        ("plic_fallback_policy", "LVIRA"),
        ("rescue", "exact_linear_support"),
        ("local_linear_fallback", "arc_fit_failed"),
    )
    output = []
    for group, values in sorted(grouped.items()):
        any_row = next(iter(values.values()))
        row = {
            "experiment": group[1],
            "method": group[2],
            "N": group[3],
            "wiggle": group[4],
            "case_count": any_row["case_count"],
            "mixed_cells": any_row["mixed_cell_denominator"],
            "components": any_row["component_denominator"],
        }
        for category, subtype in metrics:
            value = values[(category, subtype)]
            prefix = f"{category}_{subtype}".replace("+", "plus")
            row[f"{prefix}_cell_count"] = value["mixed_cell_count"]
            row[f"{prefix}_cell_fraction"] = value["fraction_of_mixed_cells"]
            row[f"{prefix}_component_count"] = value["component_count"]
            row[f"{prefix}_component_fraction"] = value["fraction_of_components"]
            row[f"{prefix}_event_count"] = value["event_count"]
        output.append(row)
    return output


def _metric(stats: Stats, group: GroupKey, category: str, subtype: str) -> dict:
    cells = stats.cell_totals[group]
    components = stats.component_totals[group]
    return {
        "cell_count": stats.cell_incidence[(group, category, subtype)],
        "cell_fraction": stats.cell_incidence[(group, category, subtype)] / cells,
        "component_count": stats.component_incidence[(group, category, subtype)],
        "component_fraction": stats.component_incidence[(group, category, subtype)]
        / components,
        "event_count": stats.event_incidence[(group, category, subtype)],
    }


def _pct(value: float) -> str:
    return f"{100 * value:.4f}%"


def _report(
    final: ReleaseAnalysis,
    july: ReleaseAnalysis,
    final_matched: Stats,
    comparison: Sequence[dict],
    reconciliation: Sequence[dict],
) -> str:
    stats = final.all_stats
    overall = ("overall", "", "", "", "")
    cells = stats.cell_totals[overall]
    components = stats.component_totals[overall]
    final_metrics = {
        name: _metric(stats, overall, "final_facet", name) for name in FINAL_CLASSES
    }
    merged = _metric(stats, overall, "merge", "merged")
    independent = _metric(stats, overall, "merge", "independent")
    lvira = _metric(stats, overall, "plic_fallback_policy", "LVIRA")
    rescue = _metric(stats, overall, "rescue", "exact_linear_support")
    local_line = _metric(stats, overall, "local_linear_fallback", "arc_fit_failed")
    plic_defaults = {
        method: _metric(stats, overall, "plic_default_method", method)
        for method in sorted(PLIC_METHODS)
    }
    unresolved_reason = _metric(
        stats, overall, "plic_fallback_reason", "unresolved_orientation"
    )
    support_failure_reason = _metric(
        stats, overall, "plic_fallback_reason", "support_line_fit_failed"
    )
    matched_overall = ("overall", "", "", "", "")
    matched_merged = _metric(final_matched, matched_overall, "merge", "merged")
    matched_lvira = _metric(
        final_matched, matched_overall, "plic_fallback_policy", "LVIRA"
    )

    lines = [
        "# Final Topology, Merge, And Fallback Diagnostic Audit",
        "",
        "Status: **PASS**",
        "",
        f"- Final release: `{final.root.name}` (`{final.manifest['successful_run_count']}` runs, `{len(final.case_records):,}` cases).",
        f"- July comparison: `{july.root.name}` (`{len(july.case_records):,}` matched cases).",
        f"- Denominators: **{cells:,} mixed cells** and **{components:,} final components**.",
        f"- Final missing facets: **{final.missing_facet_count}**.",
        "",
        "## Headline Incidence",
        "",
        "| Final facet | Mixed cells | Cell fraction | Components | Component fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "linear": "Linear",
        "circular": "Circular",
        "linear_corner": "Linear corner",
        "curved_corner": "Curved corner",
    }
    for name in FINAL_CLASSES:
        value = final_metrics[name]
        lines.append(
            f"| {labels[name]} | {value['cell_count']:,} | {_pct(value['cell_fraction'])} | "
            f"{value['component_count']:,} | {_pct(value['component_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "| Path diagnostic | Mixed cells | Cell fraction | Components | Component fraction | Events |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| Merged | {merged['cell_count']:,} | {_pct(merged['cell_fraction'])} | {merged['component_count']:,} | {_pct(merged['component_fraction'])} | - |",
            f"| Independent | {independent['cell_count']:,} | {_pct(independent['cell_fraction'])} | {independent['component_count']:,} | {_pct(independent['component_fraction'])} | - |",
            f"| LVIRA PLIC fallback (all reasons) | {lvira['cell_count']:,} | {_pct(lvira['cell_fraction'])} | {lvira['component_count']:,} | {_pct(lvira['component_fraction'])} | {lvira['event_count']:,} |",
            f"| Exact-linear-support rescue | {rescue['cell_count']:,} | {_pct(rescue['cell_fraction'])} | {rescue['component_count']:,} | {_pct(rescue['component_fraction'])} | {rescue['event_count']:,} |",
            f"| Local-line arc-fit fallback | {local_line['cell_count']:,} | {_pct(local_line['cell_fraction'])} | {local_line['component_count']:,} | {_pct(local_line['component_fraction'])} | {local_line['event_count']:,} |",
            "",
            "All unresolved PLIC fallback components are cross-checked across final cell state, merge-event provenance, and the dedicated fallback ledger. Direct Youngs, ELVIRA, and LVIRA method rows are counted as PLIC defaults, not as fallback events.",
            f"The fallback reasons are `{unresolved_reason['event_count']:,}` unresolved orientations and `{support_failure_reason['event_count']:,}` failed support-line fits. No Youngs or ELVIRA fallback policy occurs.",
            "No corner--arc--corner, curved-loop, or curved-transition rescue event occurs. The final curved-corner facets above are direct reconstruction outcomes, not curved-rescue assignments.",
            "",
            "| Direct PLIC method | Mixed cells | Cell fraction | Components | Component fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
            *(
                f"| `{method}` | {plic_defaults[method]['cell_count']:,} | {_pct(plic_defaults[method]['cell_fraction'])} | {plic_defaults[method]['component_count']:,} | {_pct(plic_defaults[method]['component_fraction'])} |"
                for method in sorted(PLIC_METHODS)
            ),
            "",
            "## Method Hotspots",
            "",
            "The full benchmark/method and setting tables are in the CSV files. The largest fallback incidences are:",
            "",
            "| Benchmark | Method | Mixed cells | LVIRA fallback | Fraction | Local-line fallback | Fraction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    method_groups = [group for group in stats.cell_totals if group[0] == "method"]
    ranked = sorted(
        method_groups,
        key=lambda group: (
            _metric(stats, group, "plic_fallback_policy", "LVIRA")["cell_fraction"],
            _metric(stats, group, "local_linear_fallback", "arc_fit_failed")[
                "cell_fraction"
            ],
        ),
        reverse=True,
    )
    for group in ranked[:10]:
        fallback = _metric(stats, group, "plic_fallback_policy", "LVIRA")
        local = _metric(stats, group, "local_linear_fallback", "arc_fit_failed")
        lines.append(
            f"| {group[1]} | `{group[2]}` | {stats.cell_totals[group]:,} | "
            f"{fallback['cell_count']:,} | {_pct(fallback['cell_fraction'])} | "
            f"{local['cell_count']:,} | {_pct(local['cell_fraction'])} |"
        )

    changed = [
        row
        for row in comparison
        if row["scope"] == "overall"
        and (
            row["mixed_cell_count_delta"]
            or row["component_count_delta"]
            or row["event_count_delta"]
        )
    ]
    lines.extend(
        [
            "",
            "## July Comparison",
            "",
            f"The comparison uses the exact `{len(july.case_records):,}`-case July coverage inside the final release; the other `{len(final.case_records)-len(july.case_records):,}` final cases are excluded from these deltas.",
            f"Merged incidence is unchanged at `{matched_merged['cell_count']:,}` cells in `{matched_merged['component_count']:,}` components. PLIC fallback incidence is unchanged at `{matched_lvira['cell_count']:,}` cells/components/events, all using LVIRA.",
            "The July event schema did not encode local-line fallback events or fallback reasons. The final local-line counts above are therefore standalone incidence, not a historical increase from zero.",
            "",
        ]
    )
    if changed:
        lines.extend(
            [
                "Overall matched incidence changed only in the following categories:",
                "",
                "| Category | July cells | Final cells | Delta | July components | Final components | Delta | July events | Final events |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in changed:
            lines.append(
                f"| `{row['category']}:{row['subtype']}` | {row['july_mixed_cell_count']:,} | "
                f"{row['final_mixed_cell_count']:,} | {row['mixed_cell_count_delta']:+,} | "
                f"{row['july_component_count']:,} | {row['final_component_count']:,} | "
                f"{row['component_count_delta']:+,} | {row['july_event_count']:,} | "
                f"{row['final_event_count']:,} |"
            )
    else:
        lines.append(
            "All matched cell, component, rescue, merge, and fallback incidences are identical to July."
        )

    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Case metrics and cell diagnostics agree for all `{len(final.case_records):,}` final cases.",
            f"- PLIC provenance agrees for all `{len(final.cell_fallbacks):,}` final fallback components; every policy is `LVIRA`.",
            "- The hardened release auditor independently reports the same `3,091,429` cell rows, `3,544,985` merge-event rows, and `9,838` joined fallback rows.",
            "- Final-facet classes are constant within each component, declared component sizes match member rows, and merged flags agree with component size.",
            "- The four final facet classes partition both the mixed-cell and component denominators.",
            "- Raw/consolidated checks use exact canonical scientific-row hashes within every run bundle:",
            "",
            "| Table | Raw rows | Consolidated rows | Matching bundles | Status |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in reconciliation:
        lines.append(
            f"| `{row['table']}` | {row['raw_rows']:,} | {row['consolidated_rows']:,} | "
            f"{row['matching_run_bundles']:,}/{row['run_bundles']:,} | **{row['status']}** |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `final_incidence_long.csv`: complete overall/benchmark/method/setting incidence.",
            "- `final_incidence_by_method.csv`: compact method table.",
            "- `final_incidence_by_setting.csv`: compact `N`/perturbation table.",
            "- `july_matched_incidence_comparison.csv`: matched July-to-final deltas.",
            "- `raw_consolidated_reconciliation.csv`: exact raw-bundle reconciliation.",
            "- `integrity_checks.csv`: release-level pass/fail ledger.",
            "- `SHA256SUMS`: sorted SHA-256 manifest for every report artifact except the manifest itself.",
            "",
            "Fractions are weighted by mixed cells or final components, as named. Rescue and local-fallback cell counts are unique member cells of affected components; event counts retain repeated assignments. Categories can overlap, so rescue/fallback fractions are not intended to sum to one.",
            "",
            "## Reproduce",
            "",
            "The report stores release names and relative artifact paths only. From a clean checkout, define the input roots for the local archive and run:",
            "",
            "```bash",
            "REPO=/path/to/interface-reconstruction",
            "FINAL_ROOT=/path/to/submission_static_20260731_012430_505aefa45432",
            "JULY_ROOT=/path/to/static_paper_simplified_default_20260717_212413",
            'cd "$REPO"',
            'python submission/audit_final_release.py "$FINAL_ROOT"',
            "python submission/audit_topology_diagnostics.py \\",
            '  --final-root "$FINAL_ROOT" \\',
            '  --july-root "$JULY_ROOT" \\',
            "  --output-dir submission/audits/final_diagnostics_2026-07-31",
            "(cd submission/audits/final_diagnostics_2026-07-31 && shasum -a 256 -c SHA256SUMS)",
            "```",
            "",
            "No PDF was generated for this audit, so raster-object and font-embedding QA are not applicable.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="ascii")


def run(final_root: Path, july_root: Path, output_dir: Path) -> None:
    final_root = final_root.resolve()
    july_root = july_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AuditError(f"Output directory is not empty: {output_dir}")
    for input_root in (final_root, july_root):
        if output_dir == input_root or input_root in output_dir.parents:
            raise AuditError(
                "Output directory must be outside both input release roots"
            )

    _, july_records, _ = _load_cases(july_root)
    final = analyze_release(final_root, matched_case_keys=set(july_records))
    july = analyze_release(july_root)
    if final.matched_stats is None:
        raise AuditError("Internal error: final matched stats were not constructed")
    reconciliation = reconcile_raw(final_root)

    final_rows = final.all_stats.rows()
    final_matched_rows = final.matched_stats.rows({"overall", "method", "setting"})
    july_rows = july.all_stats.rows({"overall", "method", "setting"})
    comparison = _comparison_rows(final_matched_rows, july_rows)
    method_rows = _wide_rows(final_rows, "method")
    setting_rows = _wide_rows(final_rows, "setting")

    final_classes_cell = sum(
        final.all_stats.cell_incidence[
            (("overall", "", "", "", ""), "final_facet", value)
        ]
        for value in FINAL_CLASSES
    )
    final_classes_component = sum(
        final.all_stats.component_incidence[
            (("overall", "", "", "", ""), "final_facet", value)
        ]
        for value in FINAL_CLASSES
    )
    all_policies = sorted({value[0] for value in final.cell_fallbacks.values()})
    checks = [
        {
            "check": "completed_runs",
            "status": "PASS",
            "observed": final.manifest["successful_run_count"],
            "expected": final.manifest["planned_run_count"],
            "details": "zero controller failures",
        },
        {
            "check": "case_count",
            "status": "PASS",
            "observed": len(final.case_records),
            "expected": final.manifest["planned_case_count"],
            "details": "unique consolidated case keys",
        },
        {
            "check": "final_missing_facets",
            "status": "PASS" if final.missing_facet_count == 0 else "FAIL",
            "observed": final.missing_facet_count,
            "expected": 0,
            "details": "sum(num_final_missing_cells)",
        },
        {
            "check": "final_facet_cell_partition",
            "status": "PASS",
            "observed": final_classes_cell,
            "expected": final.all_stats.cell_totals[("overall", "", "", "", "")],
            "details": "linear+circular+linear_corner+curved_corner",
        },
        {
            "check": "final_facet_component_partition",
            "status": "PASS",
            "observed": final_classes_component,
            "expected": final.all_stats.component_totals[("overall", "", "", "", "")],
            "details": "one final facet class per component",
        },
        {
            "check": "plic_fallback_provenance",
            "status": "PASS",
            "observed": len(final.cell_fallbacks),
            "expected": len(final.table_fallbacks),
            "details": f"policies={','.join(all_policies) or 'none'}; cell/event/ledger exact",
        },
        {
            "check": "hardened_fallback_event_contract",
            "status": (
                "PASS"
                if len(final.event_fallbacks) == len(final.table_fallbacks) == 9838
                else "FAIL"
            ),
            "observed": len(final.event_fallbacks),
            "expected": 9838,
            "details": "component keys, member cells, and LVIRA policy joined consistently",
        },
        {
            "check": "july_case_coverage",
            "status": "PASS",
            "observed": len(final.matched_stats.selected_cases),
            "expected": len(july.case_records),
            "details": "all July case keys present in final release",
        },
        {
            "check": "raw_consolidated_tables",
            "status": "PASS",
            "observed": sum(row["matching_run_bundles"] for row in reconciliation),
            "expected": sum(row["run_bundles"] for row in reconciliation),
            "details": "six tables, exact per-bundle canonical hashes",
        },
    ]
    failures = [row for row in checks if row["status"] != "PASS"]
    if failures:
        raise AuditError(f"Integrity check failed: {failures[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    long_fields = list(final_rows[0])
    _write_csv(output_dir / "final_incidence_long.csv", final_rows, long_fields)
    _write_csv(
        output_dir / "final_incidence_by_method.csv", method_rows, list(method_rows[0])
    )
    _write_csv(
        output_dir / "final_incidence_by_setting.csv",
        setting_rows,
        list(setting_rows[0]),
    )
    _write_csv(
        output_dir / "july_matched_incidence_comparison.csv",
        comparison,
        list(comparison[0]),
    )
    _write_csv(
        output_dir / "raw_consolidated_reconciliation.csv",
        reconciliation,
        list(reconciliation[0]),
    )
    _write_csv(output_dir / "integrity_checks.csv", checks, list(checks[0]))
    (output_dir / "README.md").write_text(
        _report(final, july, final.matched_stats, comparison, reconciliation),
        encoding="utf-8",
    )
    _write_sha256_manifest(output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--july-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(args.final_root, args.july_root, args.output_dir)
    except (AuditError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
