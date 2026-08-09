#!/usr/bin/env python3
"""Summarize final primitive and fallback incidence for submitted methods.

Final primitive classes are mutually exclusive and cell weighted. Fallbacks are
reported separately as construction provenance because both fallback paths end
in a straight facet and are therefore already included in the linear column.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Set, Tuple


SCHEMA_VERSION = 1

PAPER_METHODS = {
    "lines": "linear",
    "circles": "circular",
    "ellipses": "circular",
    "squares": "linear+corner",
    "zalesak": "circular+corner",
}

BENCHMARK_ORDER = tuple(PAPER_METHODS)
PRIMITIVES = ("linear", "circular", "line_line", "line_arc", "arc_arc")
FALLBACKS = ("local_linear", "LVIRA")

CASE_KEY_FIELDS = (
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
    "case_index",
)

CASE_REQUIRED_FIELDS = CASE_KEY_FIELDS + ("num_mixed_cells",)
CELL_REQUIRED_FIELDS = CASE_KEY_FIELDS + (
    "cell_id",
    "final_facet_class",
    "construction_path",
    "fallback_policy",
    "facet_geometry_json",
)
EVENT_REQUIRED_FIELDS = CASE_KEY_FIELDS + (
    "member_cells_json",
    "event_kind",
    "fallback_policy",
)

CaseKey = Tuple[str, ...]
CellKey = Tuple[CaseKey, str]


class PrimitiveIncidenceError(RuntimeError):
    """Raised when archived diagnostics do not support an exact incidence audit."""


@dataclass
class CaseCounts:
    primitive: Counter = field(default_factory=Counter)
    fallback_cells: MutableMapping[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @property
    def mixed_cells(self) -> int:
        return sum(self.primitive.values())


def _open_csv(path: Path, required_fields: Sequence[str]):
    if not path.is_file():
        raise PrimitiveIncidenceError(f"Missing required diagnostic: {path}")
    stream = path.open("r", newline="", encoding="utf-8")
    reader = csv.DictReader(stream)
    missing = [field for field in required_fields if field not in (reader.fieldnames or ())]
    if missing:
        stream.close()
        raise PrimitiveIncidenceError(
            f"{path} is missing required fields: {', '.join(missing)}"
        )
    return stream, reader


def _case_key(row: Mapping[str, str]) -> CaseKey:
    return tuple(str(row[field]).strip() for field in CASE_KEY_FIELDS)


def _selected(row: Mapping[str, str], methods: Mapping[str, str]) -> bool:
    return methods.get(str(row["experiment"]).strip()) == str(row["algo"]).strip()


def _parse_json_object(value: str, source: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PrimitiveIncidenceError(f"Invalid facet geometry at {source}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PrimitiveIncidenceError(f"Facet geometry is not an object at {source}")
    return parsed


def classify_primitive(row: Mapping[str, str], source: str = "cell row") -> str:
    """Map one final facet to a mutually exclusive paper-facing primitive."""
    facet_class = str(row["final_facet_class"]).strip()
    if facet_class == "linear":
        return "linear"
    if facet_class == "circular":
        return "circular"
    if facet_class == "linear_corner":
        return "line_line"
    if facet_class != "curved_corner":
        raise PrimitiveIncidenceError(
            f"Unsupported final_facet_class={facet_class!r} at {source}"
        )

    geometry = _parse_json_object(str(row["facet_geometry_json"]), source)
    try:
        branches = sorted(
            (
                str(geometry["left_branch"]["class"]).strip(),
                str(geometry["right_branch"]["class"]).strip(),
            )
        )
    except (KeyError, TypeError) as exc:
        raise PrimitiveIncidenceError(
            f"Curved corner lacks two branch classes at {source}"
        ) from exc
    if branches == ["circular", "linear"]:
        return "line_arc"
    if branches == ["circular", "circular"]:
        return "arc_arc"
    raise PrimitiveIncidenceError(
        f"Unsupported curved-corner branch classes {branches!r} at {source}"
    )


def _parse_member_cells(value: str, source: str) -> Set[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PrimitiveIncidenceError(f"Invalid member_cells_json at {source}: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise PrimitiveIncidenceError(f"Empty or invalid member cell list at {source}")
    member_cells = set()
    for cell in parsed:
        if not isinstance(cell, list) or len(cell) != 2:
            raise PrimitiveIncidenceError(f"Invalid member cell {cell!r} at {source}")
        member_cells.add(f"{cell[0]},{cell[1]}")
    return member_cells


def _load_manifest(release_root: Path) -> dict:
    path = release_root / "sweep_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimitiveIncidenceError(f"Could not read release manifest {path}: {exc}") from exc
    if manifest.get("status") != "completed" or manifest.get("failure_count") != 0:
        raise PrimitiveIncidenceError(f"Refusing to summarize incomplete release: {path}")
    if manifest.get("planned_run_count") != manifest.get("successful_run_count"):
        raise PrimitiveIncidenceError(f"Release run counts do not close: {path}")
    return manifest


def _read_cases(
    path: Path, methods: Mapping[str, str]
) -> Tuple[Dict[CaseKey, int], Dict[str, Set[CaseKey]]]:
    expected = {}
    by_benchmark = defaultdict(set)
    stream, reader = _open_csv(path, CASE_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            if not _selected(row, methods):
                continue
            key = _case_key(row)
            if key in expected:
                raise PrimitiveIncidenceError(f"Duplicate selected case at {path}:{line_number}")
            try:
                mixed_cells = int(row["num_mixed_cells"])
            except ValueError as exc:
                raise PrimitiveIncidenceError(
                    f"Invalid num_mixed_cells at {path}:{line_number}"
                ) from exc
            expected[key] = mixed_cells
            by_benchmark[key[0]].add(key)
    finally:
        stream.close()
    missing = sorted(set(methods) - set(by_benchmark))
    if missing:
        raise PrimitiveIncidenceError(
            "No selected cases for benchmark(s): " + ", ".join(missing)
        )
    return expected, dict(by_benchmark)


def _read_cells(
    path: Path,
    methods: Mapping[str, str],
    expected: Mapping[CaseKey, int],
    event_fallbacks: Mapping[str, Set[CellKey]],
) -> Tuple[Dict[CaseKey, CaseCounts], Dict[str, Set[CellKey]]]:
    counts = {key: CaseCounts() for key in expected}
    plic_by_policy = defaultdict(set)
    fallback_types_by_cell = defaultdict(set)
    for subtype, cell_keys in event_fallbacks.items():
        for cell_key in cell_keys:
            fallback_types_by_cell[cell_key].add(subtype)
    fallback_cells_seen = set()
    completed_cases = set()
    current_case = None
    current_cell_ids = set()
    stream, reader = _open_csv(path, CELL_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            if not _selected(row, methods):
                continue
            source = f"{path}:{line_number}"
            case_key = _case_key(row)
            if case_key not in expected:
                raise PrimitiveIncidenceError(f"Selected cell has no matching case at {source}")
            if case_key != current_case:
                if current_case is not None:
                    completed_cases.add(current_case)
                if case_key in completed_cases:
                    raise PrimitiveIncidenceError(
                        f"Selected cell rows for case {case_key} are not contiguous"
                    )
                current_case = case_key
                current_cell_ids = set()
            cell_id = str(row["cell_id"]).strip()
            cell_key = (case_key, cell_id)
            if not cell_id or cell_id in current_cell_ids:
                raise PrimitiveIncidenceError(f"Empty or duplicate cell_id at {source}")
            current_cell_ids.add(cell_id)
            primitive = classify_primitive(row, source)
            counts[case_key].primitive[primitive] += 1

            if cell_key in fallback_types_by_cell:
                if primitive != "linear":
                    raise PrimitiveIncidenceError(
                        f"Fallback event does not end in a straight facet at {source}"
                    )
                fallback_cells_seen.add(cell_key)

            construction_path = str(row["construction_path"]).strip()
            policy = str(row["fallback_policy"]).strip()
            if construction_path == "plic_fallback":
                if not policy:
                    raise PrimitiveIncidenceError(f"PLIC fallback lacks a policy at {source}")
                if primitive != "linear":
                    raise PrimitiveIncidenceError(f"PLIC fallback is not linear at {source}")
                plic_by_policy[policy].add(cell_key)
            elif policy:
                raise PrimitiveIncidenceError(
                    f"Fallback policy without plic_fallback construction path at {source}"
                )
    finally:
        stream.close()

    for key, expected_count in expected.items():
        observed = counts[key].mixed_cells
        if observed != expected_count:
            raise PrimitiveIncidenceError(
                f"Mixed-cell count mismatch for {key}: expected {expected_count}, observed {observed}"
            )
    unknown_event_cells = set(fallback_types_by_cell) - fallback_cells_seen
    if unknown_event_cells:
        raise PrimitiveIncidenceError(
            f"Fallback events reference {len(unknown_event_cells)} unknown selected cells"
        )
    return counts, dict(plic_by_policy)


def _read_fallback_events(
    path: Path,
    methods: Mapping[str, str],
    valid_cases: Mapping[CaseKey, int],
) -> Dict[str, Set[CellKey]]:
    fallback_cells = defaultdict(set)
    stream, reader = _open_csv(path, EVENT_REQUIRED_FIELDS)
    try:
        for line_number, row in enumerate(reader, start=2):
            if not _selected(row, methods):
                continue
            kind = str(row["event_kind"]).strip()
            if kind not in {"local_linear_fallback", "plic_fallback"}:
                continue
            source = f"{path}:{line_number}"
            case_key = _case_key(row)
            if case_key not in valid_cases:
                raise PrimitiveIncidenceError(
                    f"Selected fallback event has no matching case at {source}"
                )
            subtype = (
                "local_linear"
                if kind == "local_linear_fallback"
                else str(row["fallback_policy"]).strip()
            )
            if not subtype:
                raise PrimitiveIncidenceError(f"Fallback event lacks a policy at {source}")
            for cell_id in _parse_member_cells(row["member_cells_json"], source):
                cell_key = (case_key, cell_id)
                fallback_cells[subtype].add(cell_key)
    finally:
        stream.close()
    return dict(fallback_cells)


def _fraction(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def _aggregate_rows(cases: Mapping[CaseKey, CaseCounts]) -> Tuple[list, list, list]:
    case_rows = []
    for key, value in sorted(
        cases.items(),
        key=lambda item: (
            BENCHMARK_ORDER.index(item[0][0]) if item[0][0] in BENCHMARK_ORDER else 999,
            float(item[0][2]),
            float(item[0][3]),
            int(item[0][4]),
            int(item[0][6]),
        ),
    ):
        row = {
            "benchmark": key[0],
            "method": key[1],
            "resolution": key[2],
            "wiggle": key[3],
            "seed": key[4],
            "save_name": key[5],
            "case_index": key[6],
            "mixed_cells": value.mixed_cells,
        }
        for primitive in PRIMITIVES:
            count = value.primitive[primitive]
            row[f"{primitive}_cells"] = count
            row[f"{primitive}_fraction"] = _fraction(count, value.mixed_cells)
        for fallback in FALLBACKS:
            count = len(value.fallback_cells[fallback])
            row[f"{fallback}_fallback_cells"] = count
            row[f"{fallback}_fallback_fraction"] = _fraction(count, value.mixed_cells)
        case_rows.append(row)

    setting_groups = defaultdict(list)
    benchmark_groups = defaultdict(list)
    for row in case_rows:
        setting_groups[(row["benchmark"], row["method"], row["resolution"], row["wiggle"], row["seed"])].append(row)
        benchmark_groups[(row["benchmark"], row["method"])].append(row)

    def aggregate(group_rows: Iterable[dict], identity: dict) -> dict:
        group_rows = list(group_rows)
        denominator = sum(int(row["mixed_cells"]) for row in group_rows)
        result = {**identity, "instances": len(group_rows), "mixed_cells": denominator}
        for primitive in PRIMITIVES:
            count = sum(int(row[f"{primitive}_cells"]) for row in group_rows)
            result[f"{primitive}_cells"] = count
            result[f"{primitive}_fraction"] = _fraction(count, denominator)
            result[f"instances_with_{primitive}"] = sum(
                int(row[f"{primitive}_cells"]) > 0 for row in group_rows
            )
        for fallback in FALLBACKS:
            count = sum(int(row[f"{fallback}_fallback_cells"]) for row in group_rows)
            result[f"{fallback}_fallback_cells"] = count
            result[f"{fallback}_fallback_fraction"] = _fraction(count, denominator)
            result[f"instances_with_{fallback}_fallback"] = sum(
                int(row[f"{fallback}_fallback_cells"]) > 0 for row in group_rows
            )
        return result

    setting_rows = [
        aggregate(
            rows,
            {
                "benchmark": key[0],
                "method": key[1],
                "resolution": key[2],
                "wiggle": key[3],
                "seed": key[4],
            },
        )
        for key, rows in sorted(
            setting_groups.items(),
            key=lambda item: (
                BENCHMARK_ORDER.index(item[0][0]) if item[0][0] in BENCHMARK_ORDER else 999,
                float(item[0][2]),
                float(item[0][3]),
                int(item[0][4]),
            ),
        )
    ]
    benchmark_rows = [
        aggregate(rows, {"benchmark": key[0], "method": key[1]})
        for key, rows in sorted(
            benchmark_groups.items(),
            key=lambda item: BENCHMARK_ORDER.index(item[0][0])
            if item[0][0] in BENCHMARK_ORDER
            else 999,
        )
    ]
    benchmark_rows.append(
        aggregate(
            case_rows,
            {"benchmark": "All", "method": "benchmark-specific submitted method"},
        )
    )
    return case_rows, setting_rows, benchmark_rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise PrimitiveIncidenceError(f"Refusing to write empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def _markdown_report(release_root: Path, rows: Sequence[dict]) -> str:
    lines = [
        "# Final Primitive and Fallback Incidence",
        "",
        f"Source release: `{release_root.name}`.",
        "",
        "The audit uses one submitted method per benchmark: `linear` for lines, "
        "`circular` for circles and ellipses, `linear+corner` for squares, and "
        "`circular+corner` for Zalesak. Final primitive columns are mutually exclusive "
        "and sum to every original mixed cell. Fallback columns are an overlay: both "
        "fallback paths end in a straight facet and are included in `linear`.",
        "",
        "## Final Primitive Geometry",
        "",
        "| Benchmark | Instances | Mixed cells | Linear | Circular | Line-line | Line-arc | Arc-arc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['instances']:,} | {row['mixed_cells']:,} | "
            f"{_pct(row['linear_fraction'])} | {_pct(row['circular_fraction'])} | "
            f"{_pct(row['line_line_fraction'])} | {_pct(row['line_arc_fraction'])} | "
            f"{_pct(row['arc_arc_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Fallback Provenance",
            "",
            "| Benchmark | Local-line cells | Instances with local-line | LVIRA cells | Instances with LVIRA |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['local_linear_fallback_cells']:,} "
            f"({_pct(row['local_linear_fallback_fraction'])}) | "
            f"{row['instances_with_local_linear_fallback']:,} | "
            f"{row['LVIRA_fallback_cells']:,} ({_pct(row['LVIRA_fallback_fraction'])}) | "
            f"{row['instances_with_LVIRA_fallback']:,} |"
        )
    lines.extend(
        [
            "",
            "`Local-line` means a mass-matching line used after an oriented higher-order "
            "fit failed. `LVIRA` means the centered-stencil fallback used after orientation "
            "remained unresolved. Detailed setting- and case-indexed counts are stored beside "
            "this report.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_release(
    release_root: Path,
    output_dir: Path | None = None,
    methods: Mapping[str, str] = PAPER_METHODS,
) -> dict:
    release_root = Path(release_root).resolve()
    manifest = _load_manifest(release_root)
    diagnostics = release_root / "diagnostics"
    expected, _ = _read_cases(diagnostics / "case_metrics.csv", methods)
    event_fallbacks = _read_fallback_events(
        diagnostics / "merge_events.csv", methods, expected
    )
    cases, plic_cell_rows = _read_cells(
        diagnostics / "cell_metrics.csv", methods, expected, event_fallbacks
    )

    event_plic = {key: value for key, value in event_fallbacks.items() if key != "local_linear"}
    if event_plic != plic_cell_rows:
        raise PrimitiveIncidenceError(
            "PLIC fallback cells disagree between cell metrics and merge events"
        )
    unsupported_policies = sorted(set(event_plic) - {"LVIRA"})
    if unsupported_policies:
        raise PrimitiveIncidenceError(
            "Paper-facing methods used unexpected PLIC fallback policy/policies: "
            + ", ".join(unsupported_policies)
        )

    for subtype, cell_keys in event_fallbacks.items():
        for case_key, cell_id in cell_keys:
            cases[case_key].fallback_cells[subtype].add(cell_id)
    overlap = event_fallbacks.get("local_linear", set()) & event_fallbacks.get("LVIRA", set())
    if overlap:
        raise PrimitiveIncidenceError(
            f"Local-line and LVIRA fallback sets overlap for {len(overlap)} cells"
        )

    case_rows, setting_rows, benchmark_rows = _aggregate_rows(cases)
    all_row = benchmark_rows[-1]
    primitive_total = sum(int(all_row[f"{name}_cells"]) for name in PRIMITIVES)
    if primitive_total != int(all_row["mixed_cells"]):
        raise PrimitiveIncidenceError(
            f"Primitive classes do not close: {primitive_total} != {all_row['mixed_cells']}"
        )

    output_dir = Path(output_dir or release_root / "primitive_incidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_csv": output_dir / "primitive_incidence_by_benchmark.csv",
        "setting_csv": output_dir / "primitive_incidence_by_setting.csv",
        "case_csv": output_dir / "primitive_incidence_by_case.csv",
        "json": output_dir / "primitive_incidence_summary.json",
        "markdown": output_dir / "README.md",
    }
    _write_csv(paths["benchmark_csv"], benchmark_rows)
    _write_csv(paths["setting_csv"], setting_rows)
    _write_csv(paths["case_csv"], case_rows)
    source_state_path = diagnostics / "source_state.json"
    source_state = {}
    if source_state_path.is_file():
        try:
            source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PrimitiveIncidenceError(
                f"Could not read source state {source_state_path}: {exc}"
            ) from exc
    payload = {
        "schema_version": SCHEMA_VERSION,
        "release": release_root.name,
        "source_commit": source_state.get("source_commit", ""),
        "release_status": manifest.get("status"),
        "successful_run_count": manifest.get("successful_run_count"),
        "methods": dict(methods),
        "primitive_classes": list(PRIMITIVES),
        "fallback_classes": list(FALLBACKS),
        "benchmark_rows": benchmark_rows,
    }
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(
        _markdown_report(release_root, benchmark_rows), encoding="utf-8"
    )
    return {"payload": payload, "paths": paths}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = summarize_release(args.release_root, args.output_dir)
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
