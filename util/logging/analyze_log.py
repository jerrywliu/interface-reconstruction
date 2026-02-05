"""
Analyze getArcFacet call log files to extract challenging test cases.

Usage:
    python -m util.logging.analyze_log get_arc_facet_calls.log
    python -m util.logging.analyze_log get_arc_facet_calls.log --extract-failed
    python -m util.logging.analyze_log get_arc_facet_calls.log --extract-slow --threshold 1.0
    python -m util.logging.analyze_log get_arc_facet_calls.log --extract-problematic
    
Note: "Infinite loop" cases that hit maxTimestep quickly will show up as "failed"
      (not "slow") because they complete in <1 second. Use --extract-problematic
      to get both failed and slow cases.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_log_entries(log_file: str) -> List[Dict[str, Any]]:
    """Load all log entries from a log file."""
    entries = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:100]}... Error: {e}")
    return entries


def print_summary(entries: List[Dict[str, Any]], infinite_loop_threshold: float = 0.01):
    """Print summary statistics."""
    total = len(entries)
    success = sum(1 for e in entries if e.get("status") == "success")
    failed = sum(1 for e in entries if e.get("status") == "failed")
    exceptions = sum(1 for e in entries if e.get("status") == "exception")

    execution_times = [e.get("execution_time_seconds", 0) for e in entries]
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
    max_time = max(execution_times) if execution_times else 0
    total_time = sum(execution_times)

    print("=" * 80)
    print("LOG FILE SUMMARY")
    print("=" * 80)
    print(f"Total calls: {total}")
    print(f"  ✓ Success: {success} ({100*success/total:.1f}%)")
    print(f"  ✗ Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"  ⚠ Exceptions: {exceptions} ({100*exceptions/total:.1f}%)")
    print()
    print(f"Execution time:")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Maximum: {max_time:.4f}s")
    print(f"  Total: {total_time:.2f}s")
    
    # Categorize failed cases
    if failed > 0:
        instant_failures, infinite_loop_failures = categorize_failed_cases(
            entries, infinite_loop_threshold
        )
        print()
        print(f"Failed cases breakdown (threshold: {infinite_loop_threshold}s):")
        print(f"  ⚡ Instant failures (likely no valid solution): {len(instant_failures)} ({100*len(instant_failures)/failed:.1f}%)")
        print(f"  🔄 Infinite loop failures (hit maxTimestep): {len(infinite_loop_failures)} ({100*len(infinite_loop_failures)/failed:.1f}%)")
    
    # Count slow cases (>1s)
    slow_count = sum(1 for e in entries if e.get("execution_time_seconds", 0) > 1.0)
    if slow_count > 0:
        print(f"  Slow cases (>1.0s): {slow_count}")
    print()


def extract_failed_cases(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all failed cases."""
    return [e for e in entries if e.get("status") == "failed"]


def categorize_failed_cases(
    entries: List[Dict[str, Any]], 
    infinite_loop_threshold: float = 0.01
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Categorize failed cases into:
    1. Instant failures: Likely no valid solution exists (fail quickly)
    2. Infinite loop failures: Hit maxTimestep after many iterations (take longer)
    
    Args:
        entries: List of log entries
        infinite_loop_threshold: Time threshold (seconds) to distinguish infinite loops.
                                 Cases taking longer than this are considered "infinite loops".
                                 Default 0.01s (10ms) - infinite loops iterate many times.
    
    Returns:
        (instant_failures, infinite_loop_failures)
    """
    failed = extract_failed_cases(entries)
    
    instant_failures = []
    infinite_loop_failures = []
    
    for e in failed:
        exec_time = e.get("execution_time_seconds", 0)
        if exec_time >= infinite_loop_threshold:
            infinite_loop_failures.append(e)
        else:
            instant_failures.append(e)
    
    return instant_failures, infinite_loop_failures


def extract_slow_cases(
    entries: List[Dict[str, Any]], threshold: float = 1.0
) -> List[Dict[str, Any]]:
    """Extract cases that took longer than threshold seconds."""
    return [e for e in entries if e.get("execution_time_seconds", 0) > threshold]


def extract_problematic_cases(
    entries: List[Dict[str, Any]], 
    slow_threshold: float = 1.0,
    include_failed: bool = True,
    include_slow: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract problematic cases: slow cases, failed cases, or both.
    
    Args:
        entries: List of log entries
        slow_threshold: Time threshold for slow cases (seconds)
        include_failed: Include cases that failed to converge
        include_slow: Include cases that took longer than threshold
    
    Returns:
        List of problematic entries
    """
    result = []
    for e in entries:
        status = e.get("status", "unknown")
        exec_time = e.get("execution_time_seconds", 0)
        
        is_failed = (status == "failed") and include_failed
        is_slow = (exec_time > slow_threshold) and include_slow
        
        if is_failed or is_slow:
            result.append(e)
    
    return result


def extract_extreme_area_fractions(
    entries: List[Dict[str, Any]], min_frac: float = 0.0, max_frac: float = 1.0
) -> List[Dict[str, Any]]:
    """Extract cases with extreme area fractions."""
    result = []
    for e in entries:
        inputs = e.get("inputs", {})
        a1 = inputs.get("a1", 0.5)
        a2 = inputs.get("a2", 0.5)
        a3 = inputs.get("a3", 0.5)

        if (
            a1 < min_frac
            or a1 > max_frac
            or a2 < min_frac
            or a2 > max_frac
            or a3 < min_frac
            or a3 > max_frac
        ):
            result.append(e)
    return result


def format_test_case(entry: Dict[str, Any], infinite_loop_threshold: float = 0.01) -> str:
    """Format a log entry as a test case."""
    inputs = entry.get("inputs", {})
    call_id = entry.get("call_id", "unknown")
    status = entry.get("status", "unknown")
    exec_time = entry.get("execution_time_seconds", 0)
    
    # Determine failure type for failed cases
    failure_type = ""
    if status == "failed":
        if exec_time >= infinite_loop_threshold:
            failure_type = " (infinite loop - hit maxTimestep)"
        else:
            failure_type = " (instant failure - likely no valid solution)"

    return f"""TestCase(
    name="case_from_log_{call_id}",
    poly1={inputs.get('poly1')},
    poly2={inputs.get('poly2')},
    poly3={inputs.get('poly3')},
    a1={inputs.get('a1')},
    a2={inputs.get('a2')},
    a3={inputs.get('a3')},
    epsilon={inputs.get('epsilon')},
    description="From log: status={status}{failure_type}, time={exec_time:.4f}s"
),"""


def main():
    parser = argparse.ArgumentParser(description="Analyze getArcFacet call logs")
    parser.add_argument("log_file", help="Path to log file")
    parser.add_argument(
        "--extract-failed", action="store_true", help="Extract failed test cases"
    )
    parser.add_argument(
        "--extract-slow", action="store_true", help="Extract slow test cases"
    )
    parser.add_argument(
        "--extract-problematic",
        action="store_true",
        help="Extract problematic cases (failed OR slow). This is useful for finding "
             "'infinite loop' cases that hit maxTimestep quickly (failed) or take a long time (slow).",
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
        help="Time threshold to distinguish infinite loop failures from instant failures (seconds). "
             "Failed cases taking longer than this are considered 'infinite loops'. Default: 0.01s",
    )
    parser.add_argument(
        "--extract-instant-failures",
        action="store_true",
        help="Extract instant failures (likely no valid solution exists, fail quickly)",
    )
    parser.add_argument(
        "--extract-infinite-loops",
        action="store_true",
        help="Extract infinite loop failures (hit maxTimestep after many iterations)",
    )
    parser.add_argument(
        "--extract-extreme",
        action="store_true",
        help="Extract cases with extreme area fractions",
    )
    parser.add_argument(
        "--min-frac",
        type=float,
        default=0.0,
        help="Minimum area fraction for extreme cases",
    )
    parser.add_argument(
        "--max-frac",
        type=float,
        default=1.0,
        help="Maximum area fraction for extreme cases",
    )
    parser.add_argument(
        "--output", type=str, help="Output file for extracted test cases"
    )

    args = parser.parse_args()

    # Load entries
    print(f"Loading log file: {args.log_file}")
    entries = load_log_entries(args.log_file)
    print(f"Loaded {len(entries)} entries\n")

    # Print summary
    print_summary(entries, args.infinite_loop_threshold)

    # Extract cases based on flags
    extracted = []

    if args.extract_failed:
        failed = extract_failed_cases(entries)
        print(f"Found {len(failed)} failed cases")
        extracted.extend(failed)

    if args.extract_instant_failures:
        instant_failures, _ = categorize_failed_cases(entries, args.infinite_loop_threshold)
        print(f"Found {len(instant_failures)} instant failures (<{args.infinite_loop_threshold}s)")
        extracted.extend(instant_failures)

    if args.extract_infinite_loops:
        _, infinite_loop_failures = categorize_failed_cases(entries, args.infinite_loop_threshold)
        print(f"Found {len(infinite_loop_failures)} infinite loop failures (>={args.infinite_loop_threshold}s)")
        extracted.extend(infinite_loop_failures)

    if args.extract_slow:
        slow = extract_slow_cases(entries, args.threshold)
        print(f"Found {len(slow)} slow cases (>{args.threshold}s)")
        extracted.extend(slow)

    if args.extract_problematic:
        problematic = extract_problematic_cases(
            entries, 
            slow_threshold=args.threshold,
            include_failed=True,
            include_slow=True,
        )
        print(f"Found {len(problematic)} problematic cases (failed OR slow >{args.threshold}s)")
        extracted.extend(problematic)

    if args.extract_extreme:
        extreme = extract_extreme_area_fractions(entries, args.min_frac, args.max_frac)
        print(f"Found {len(extreme)} cases with extreme area fractions")
        extracted.extend(extreme)

    # Remove duplicates (by call_id)
    seen_ids = set()
    unique_extracted = []
    for e in extracted:
        call_id = e.get("call_id")
        if call_id not in seen_ids:
            seen_ids.add(call_id)
            unique_extracted.append(e)

    if unique_extracted:
        print(f"\nTotal unique extracted cases: {len(unique_extracted)}")

        # Format as test cases
        test_cases = [format_test_case(e, args.infinite_loop_threshold) for e in unique_extracted]

        if args.output:
            with open(args.output, "w") as f:
                f.write("# Extracted test cases from log file\n")
                f.write(
                    "# Format: TestCase objects for test_get_arc_facet_error.py\n\n"
                )
                for tc in test_cases:
                    f.write(tc + "\n\n")
            print(f"\n✓ Test cases written to {args.output}")
        else:
            print("\n" + "=" * 80)
            print("EXTRACTED TEST CASES")
            print("=" * 80)
            for tc in test_cases:
                print(tc)
                print()


if __name__ == "__main__":
    main()


