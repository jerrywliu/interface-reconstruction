"""
Analyze getArcFacet call log files to extract challenging test cases.

Usage:
    python -m util.logging.analyze_log get_arc_facet_calls.log
    python -m util.logging.analyze_log get_arc_facet_calls.log --extract-problematic
    python -m util.logging.analyze_log get_arc_facet_calls.log --extract-slow --threshold 1.0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_OUTPUT = "test/geoms/getarcfacet/zalesak_harvest_cases.py"
TERMINAL_STATUSES = {"success", "failed", "exception"}
NON_SUCCESS_STATUSES = {"started_only", "failed", "exception"}


def load_log_entries(log_file: str) -> List[Dict[str, Any]]:
    """Load raw JSONL entries from a log file."""
    entries = []
    with open(log_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Warning: Could not parse line: {line[:100]}... Error: {error}")
    return entries


def consolidate_log_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse started/terminal events into one logical call entry per call_id."""
    buckets: Dict[Any, Dict[str, Optional[Dict[str, Any]]]] = {}
    for entry in entries:
        call_id = entry.get("call_id")
        bucket = buckets.setdefault(call_id, {"started": None, "terminal": None})
        status = entry.get("status")
        if status == "started":
            bucket["started"] = entry
        elif status in TERMINAL_STATUSES:
            bucket["terminal"] = entry
        else:
            bucket["terminal"] = entry

    consolidated = []
    for call_id in sorted(buckets.keys(), key=lambda value: (value is None, value)):
        started = buckets[call_id]["started"]
        terminal = buckets[call_id]["terminal"]
        if terminal is not None:
            merged = {}
            if started is not None:
                merged.update(started)
                merged["started_timestamp"] = started.get("timestamp")
            merged.update(terminal)
            merged.setdefault("inputs", (started or {}).get("inputs", {}))
            merged.setdefault("metadata", (started or {}).get("metadata", {}))
            consolidated.append(merged)
        elif started is not None:
            merged = dict(started)
            merged["started_timestamp"] = started.get("timestamp")
            merged["status"] = "started_only"
            merged["execution_time_seconds"] = None
            consolidated.append(merged)
    return consolidated


def _parse_filter_list(raw_value: Optional[str], cast=None) -> Optional[List[Any]]:
    if raw_value is None:
        return None
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(part) if cast else part)
    return values or None


def _metadata_value(entry: Dict[str, Any], key: str) -> Any:
    return entry.get("metadata", {}).get(key)


def filter_entries(
    entries: Sequence[Dict[str, Any]],
    experiment: Optional[Sequence[str]] = None,
    algo: Optional[Sequence[str]] = None,
    resolution: Optional[Sequence[float]] = None,
    source: Optional[Sequence[str]] = None,
    status: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    filtered = []
    for entry in entries:
        if experiment and _metadata_value(entry, "experiment") not in experiment:
            continue
        if algo and _metadata_value(entry, "algo") not in algo:
            continue
        if resolution is not None:
            entry_resolution = _metadata_value(entry, "resolution")
            if entry_resolution is None or float(entry_resolution) not in resolution:
                continue
        if source and _metadata_value(entry, "call_source") not in source:
            continue
        if status and entry.get("status") not in status:
            continue
        filtered.append(entry)
    return filtered


def print_summary(
    entries: Sequence[Dict[str, Any]],
    infinite_loop_threshold: float = 0.01,
    slow_threshold: float = 1.0,
):
    """Print summary statistics for consolidated logical calls."""
    total = len(entries)
    success = sum(1 for entry in entries if entry.get("status") == "success")
    failed = sum(1 for entry in entries if entry.get("status") == "failed")
    exceptions = sum(1 for entry in entries if entry.get("status") == "exception")
    started_only = sum(1 for entry in entries if entry.get("status") == "started_only")

    execution_times = [
        entry.get("execution_time_seconds", 0.0)
        for entry in entries
        if entry.get("execution_time_seconds") is not None
    ]
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
    max_time = max(execution_times) if execution_times else 0.0
    total_time = sum(execution_times)

    print("=" * 80)
    print("LOG FILE SUMMARY")
    print("=" * 80)
    print(f"Total logical calls: {total}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Exceptions: {exceptions}")
    print(f"  Started only: {started_only}")
    print()
    print("Execution time:")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Maximum: {max_time:.4f}s")
    print(f"  Total: {total_time:.2f}s")

    if failed > 0:
        instant_failures, slow_failures = categorize_failed_cases(
            entries, infinite_loop_threshold
        )
        print()
        print(f"Failed cases breakdown (threshold: {infinite_loop_threshold}s):")
        print(f"  Instant failures: {len(instant_failures)}")
        print(f"  Slow failures: {len(slow_failures)}")

    slow_successes = sum(
        1
        for entry in entries
        if entry.get("status") == "success"
        and (entry.get("execution_time_seconds") or 0.0) >= slow_threshold
    )
    if slow_successes > 0:
        print(f"  Slow successes (>={slow_threshold}s): {slow_successes}")
    print()


def extract_failed_cases(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in entries if entry.get("status") == "failed"]


def categorize_failed_cases(
    entries: Sequence[Dict[str, Any]],
    infinite_loop_threshold: float = 0.01,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    failed = extract_failed_cases(entries)
    instant_failures = []
    slow_failures = []
    for entry in failed:
        exec_time = entry.get("execution_time_seconds", 0.0) or 0.0
        if exec_time >= infinite_loop_threshold:
            slow_failures.append(entry)
        else:
            instant_failures.append(entry)
    return instant_failures, slow_failures


def extract_slow_cases(
    entries: Sequence[Dict[str, Any]], threshold: float = 1.0
) -> List[Dict[str, Any]]:
    return [
        entry
        for entry in entries
        if (entry.get("execution_time_seconds") or 0.0) >= threshold
    ]


def extract_problematic_cases(
    entries: Sequence[Dict[str, Any]],
    slow_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    result = []
    for entry in entries:
        status = entry.get("status")
        exec_time = entry.get("execution_time_seconds") or 0.0
        if status in NON_SUCCESS_STATUSES:
            result.append(entry)
        elif status == "success" and exec_time >= slow_threshold:
            result.append(entry)
    return result


def extract_extreme_area_fractions(
    entries: Sequence[Dict[str, Any]], min_frac: float = 0.0, max_frac: float = 1.0
) -> List[Dict[str, Any]]:
    result = []
    for entry in entries:
        inputs = entry.get("inputs", {})
        fractions = [inputs.get("a1", 0.5), inputs.get("a2", 0.5), inputs.get("a3", 0.5)]
        if any(frac < min_frac or frac > max_frac for frac in fractions):
            result.append(entry)
    return result


def _serialize_case_key(entry: Dict[str, Any]) -> str:
    inputs = entry.get("inputs", {})
    key = (
        inputs.get("poly1"),
        inputs.get("poly2"),
        inputs.get("poly3"),
        inputs.get("a1"),
        inputs.get("a2"),
        inputs.get("a3"),
        inputs.get("epsilon"),
    )
    return json.dumps(key, sort_keys=True)


def dedupe_extracted_cases(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()
    for entry in entries:
        key = _serialize_case_key(entry)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def limit_slow_success_cases(
    entries: Sequence[Dict[str, Any]], per_bucket_limit: int = 10
) -> List[Dict[str, Any]]:
    kept = []
    slow_success_buckets: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}

    for entry in entries:
        if entry.get("status") in NON_SUCCESS_STATUSES:
            kept.append(entry)
            continue
        bucket = (
            _metadata_value(entry, "algo"),
            _metadata_value(entry, "resolution"),
        )
        slow_success_buckets.setdefault(bucket, []).append(entry)

    for bucket_entries in slow_success_buckets.values():
        bucket_entries.sort(
            key=lambda entry: entry.get("execution_time_seconds", 0.0) or 0.0,
            reverse=True,
        )
        kept.extend(bucket_entries[:per_bucket_limit])

    return kept


def _format_resolution_tag(resolution: Any) -> str:
    if resolution is None:
        return "unknown"
    return f"{float(resolution):.2f}".replace(".", "p")


def _build_case_name(entry: Dict[str, Any]) -> str:
    metadata = entry.get("metadata", {})
    experiment = metadata.get("experiment", "case")
    algo = metadata.get("algo", "unknown").replace("+", "plus")
    resolution_tag = _format_resolution_tag(metadata.get("resolution"))
    call_id = entry.get("call_id", "unknown")
    return f"{experiment}_{algo}_r{resolution_tag}_call{call_id}"


def _build_case_description(entry: Dict[str, Any], infinite_loop_threshold: float) -> str:
    status = entry.get("status", "unknown")
    exec_time = entry.get("execution_time_seconds")
    metadata = entry.get("metadata", {})
    source = metadata.get("call_source")
    grid_coords = metadata.get("grid_coords")
    merge_id = metadata.get("merge_id")
    failure_type = ""
    if status == "failed":
        effective_time = exec_time or 0.0
        if effective_time >= infinite_loop_threshold:
            failure_type = ", slow failure"
        else:
            failure_type = ", instant failure"
    parts = [f"status={status}{failure_type}"]
    if exec_time is not None:
        parts.append(f"time={exec_time:.4f}s")
    if source is not None:
        parts.append(f"source={source}")
    if grid_coords is not None:
        parts.append(f"grid={grid_coords}")
    if merge_id is not None:
        parts.append(f"merge_id={merge_id}")
    return ", ".join(parts)


def format_test_case(
    entry: Dict[str, Any], infinite_loop_threshold: float = 0.01
) -> str:
    inputs = entry.get("inputs", {})
    metadata = entry.get("metadata", {})
    name = _build_case_name(entry)
    description = _build_case_description(entry, infinite_loop_threshold)
    return f"""    TestCase(
        name={name!r},
        poly1={inputs.get('poly1')!r},
        poly2={inputs.get('poly2')!r},
        poly3={inputs.get('poly3')!r},
        a1={inputs.get('a1')!r},
        a2={inputs.get('a2')!r},
        a3={inputs.get('a3')!r},
        epsilon={inputs.get('epsilon')!r},
        description={description!r},
        source_suite="zalesak_harvest",
        metadata={metadata!r},
    ),"""


def render_test_case_module(
    entries: Sequence[Dict[str, Any]], infinite_loop_threshold: float = 0.01
) -> str:
    lines = [
        '"""Auto-generated harvested getArcFacet cases from Zalesak runs."""',
        "",
        "from test.geoms.getarcfacet.case_harness import TestCase",
        "",
        "TEST_CASES = [",
    ]
    for entry in entries:
        lines.append(format_test_case(entry, infinite_loop_threshold))
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def write_test_case_module(
    output_path: str,
    entries: Sequence[Dict[str, Any]],
    infinite_loop_threshold: float = 0.01,
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_test_case_module(entries, infinite_loop_threshold),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze getArcFacet call logs")
    parser.add_argument("log_file", help="Path to log file")
    parser.add_argument("--extract-failed", action="store_true", help="Extract failed test cases")
    parser.add_argument("--extract-slow", action="store_true", help="Extract slow test cases")
    parser.add_argument(
        "--extract-problematic",
        action="store_true",
        help="Extract started_only, exception, failed, and slow-success cases",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Time threshold for slow cases (seconds)",
    )
    parser.add_argument(
        "--infinite-loop-threshold",
        type=float,
        default=0.01,
        help="Time threshold to distinguish instant and slow failures",
    )
    parser.add_argument(
        "--extract-instant-failures",
        action="store_true",
        help="Extract failures that completed quickly",
    )
    parser.add_argument(
        "--extract-infinite-loops",
        action="store_true",
        help="Extract failures above the slow-failure threshold",
    )
    parser.add_argument(
        "--extract-extreme",
        action="store_true",
        help="Extract cases with extreme area fractions",
    )
    parser.add_argument("--min-frac", type=float, default=0.0, help="Minimum area fraction")
    parser.add_argument("--max-frac", type=float, default=1.0, help="Maximum area fraction")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Comma-separated experiment filter applied to metadata.experiment",
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Comma-separated algorithm filter applied to metadata.algo",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Comma-separated resolution filter applied to metadata.resolution",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Comma-separated source filter applied to metadata.call_source",
    )
    parser.add_argument(
        "--status",
        type=str,
        help="Comma-separated logical status filter",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output file for extracted Python module (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    print(f"Loading log file: {args.log_file}")
    raw_entries = load_log_entries(args.log_file)
    entries = consolidate_log_entries(raw_entries)
    print(f"Loaded {len(raw_entries)} raw events")
    print(f"Consolidated to {len(entries)} logical calls\n")

    entries = filter_entries(
        entries,
        experiment=_parse_filter_list(args.experiment),
        algo=_parse_filter_list(args.algo),
        resolution=_parse_filter_list(args.resolution, float),
        source=_parse_filter_list(args.source),
        status=_parse_filter_list(args.status),
    )

    print_summary(entries, args.infinite_loop_threshold, args.threshold)

    extracted: List[Dict[str, Any]] = []

    if args.extract_failed:
        failed = extract_failed_cases(entries)
        print(f"Found {len(failed)} failed cases")
        extracted.extend(failed)

    if args.extract_instant_failures:
        instant_failures, _ = categorize_failed_cases(entries, args.infinite_loop_threshold)
        print(f"Found {len(instant_failures)} instant failures")
        extracted.extend(instant_failures)

    if args.extract_infinite_loops:
        _, slow_failures = categorize_failed_cases(entries, args.infinite_loop_threshold)
        print(f"Found {len(slow_failures)} slow failures")
        extracted.extend(slow_failures)

    if args.extract_slow:
        slow = extract_slow_cases(entries, args.threshold)
        print(f"Found {len(slow)} slow cases")
        extracted.extend(slow)

    if args.extract_problematic:
        problematic = extract_problematic_cases(entries, slow_threshold=args.threshold)
        print(f"Found {len(problematic)} problematic cases")
        extracted.extend(problematic)

    if args.extract_extreme:
        extreme = extract_extreme_area_fractions(entries, args.min_frac, args.max_frac)
        print(f"Found {len(extreme)} cases with extreme area fractions")
        extracted.extend(extreme)

    if extracted:
        extracted = dedupe_extracted_cases(extracted)
        extracted = limit_slow_success_cases(extracted, per_bucket_limit=10)
        print(f"\nTotal extracted cases after dedupe/cap: {len(extracted)}")
        write_test_case_module(args.output, extracted, args.infinite_loop_threshold)
        print(f"\nWrote harvested test module to {args.output}")


if __name__ == "__main__":
    main()
