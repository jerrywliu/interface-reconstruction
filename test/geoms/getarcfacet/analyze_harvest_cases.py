"""
Cluster harvested Zalesak getArcFacet cases and select a compact priority subset.

Usage:
    python -m test.geoms.getarcfacet.analyze_harvest_cases
    python -m test.geoms.getarcfacet.analyze_harvest_cases --write-priority-module
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

from test.geoms.getarcfacet.case_harness import TestCase
from test.geoms.getarcfacet.zalesak_harvest_cases import TEST_CASES as HARVEST_CASES


DEFAULT_PRIORITY_OUTPUT = (
    Path(__file__).resolve().parent / "zalesak_priority_cases.py"
)

_TIME_PATTERN = re.compile(r"time=([0-9.]+)s")

TOPOLOGY_LABELS = {
    ((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)): "vertical_chain",
    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)): "elbow_lower_right",
    ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0)): "elbow_upper_right",
}


def _case_center(poly: Sequence[Sequence[float]]) -> Tuple[float, float]:
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def parse_case_status(test_case: TestCase) -> str:
    description = test_case.description or ""
    if "exception" in description:
        return "exception"
    if "failed" in description:
        return "failed"
    return "other"


def parse_execution_time(test_case: TestCase) -> float:
    description = test_case.description or ""
    match = _TIME_PATTERN.search(description)
    return float(match.group(1)) if match else 0.0


def mean_fraction(test_case: TestCase) -> float:
    return (test_case.a1 + test_case.a2 + test_case.a3) / 3.0


def topology_signature(test_case: TestCase) -> Tuple[Tuple[float, float], ...]:
    centers = [
        _case_center(test_case.poly1),
        _case_center(test_case.poly2),
        _case_center(test_case.poly3),
    ]
    min_x = min(x for x, _ in centers)
    min_y = min(y for _, y in centers)

    x_values = sorted({point[0] for point in test_case.poly1})
    y_values = sorted({point[1] for point in test_case.poly1})
    dx = x_values[-1] - x_values[0]
    dy = y_values[-1] - y_values[0]

    return tuple(
        sorted(
            (round((x - min_x) / dx, 3), round((y - min_y) / dy, 3))
            for x, y in centers
        )
    )


def topology_label(test_case: TestCase) -> str:
    return TOPOLOGY_LABELS.get(topology_signature(test_case), "unknown")


def summarize_cases(cases: Sequence[TestCase]) -> str:
    resolution_counts = Counter(case.metadata.get("resolution") for case in cases)
    topology_counts = Counter(topology_label(case) for case in cases)
    group_counts = Counter((topology_label(case), parse_case_status(case)) for case in cases)

    lines = [
        f"Total harvested cases: {len(cases)}",
        f"By resolution: {dict(sorted(resolution_counts.items()))}",
        f"By topology: {dict(sorted(topology_counts.items()))}",
        "By topology/status:",
    ]
    for key, count in sorted(group_counts.items()):
        lines.append(f"  {key}: {count}")
    return "\n".join(lines)


def select_priority_cases(cases: Sequence[TestCase]) -> List[TestCase]:
    selected: List[TestCase] = []
    selected_names = set()

    def add_case(test_case: TestCase):
        if test_case.name not in selected_names:
            selected.append(test_case)
            selected_names.add(test_case.name)

    # 1. Keep the slowest exception from each unique hotspot first.
    slow_exceptions = [
        case
        for case in cases
        if parse_case_status(case) == "exception" and parse_execution_time(case) >= 0.1
    ]
    seen_hotspots = set()
    for case in sorted(
        slow_exceptions, key=lambda item: (-parse_execution_time(item), item.name)
    ):
        hotspot = tuple(map(tuple, case.metadata.get("merge_coords", [])))
        if hotspot in seen_hotspots:
            continue
        seen_hotspots.add(hotspot)
        add_case(case)
        if len(selected) >= 4:
            break

    # 2. Add vertical-chain instant failures at both low- and high-fraction extremes.
    vertical_failed = [
        case
        for case in cases
        if parse_case_status(case) == "failed"
        and topology_label(case) == "vertical_chain"
    ]
    if vertical_failed:
        add_case(min(vertical_failed, key=mean_fraction))
        add_case(max(vertical_failed, key=mean_fraction))

    # 3. Add one instant-failure elbow representative per elbow topology.
    for label in ("elbow_lower_right", "elbow_upper_right"):
        elbow_failed = [
            case
            for case in cases
            if parse_case_status(case) == "failed" and topology_label(case) == label
        ]
        if elbow_failed:
            add_case(
                max(
                    elbow_failed,
                    key=lambda item: (parse_execution_time(item), -mean_fraction(item)),
                )
            )

    # 4. Preserve one coarse-resolution case explicitly, even if it is not in the slow set.
    coarse_cases = [case for case in cases if case.metadata.get("resolution") == 0.5]
    if coarse_cases:
        add_case(max(coarse_cases, key=parse_execution_time))

    return selected


def render_priority_module(priority_cases: Sequence[TestCase]) -> str:
    names = [case.name for case in priority_cases]
    names_literal = ",\n    ".join(repr(name) for name in names)
    return f'''"""Curated high-signal subset of harvested Zalesak getArcFacet cases."""\n\nfrom test.geoms.getarcfacet.zalesak_harvest_cases import TEST_CASES as HARVEST_CASES\n\nPRIORITY_CASE_NAMES = [\n    {names_literal}\n]\n\n_CASES_BY_NAME = {{case.name: case for case in HARVEST_CASES}}\nTEST_CASES = [_CASES_BY_NAME[name] for name in PRIORITY_CASE_NAMES]\n'''


def write_priority_module(
    output_path: Path = DEFAULT_PRIORITY_OUTPUT,
    cases: Sequence[TestCase] = HARVEST_CASES,
) -> List[TestCase]:
    priority_cases = select_priority_cases(cases)
    output_path.write_text(render_priority_module(priority_cases), encoding="utf-8")
    return priority_cases


def _format_priority_table(cases: Iterable[TestCase]) -> str:
    lines = []
    for case in cases:
        lines.append(
            "  "
            f"{case.name}: "
            f"{topology_label(case)}, "
            f"{parse_case_status(case)}, "
            f"t={parse_execution_time(case):.4f}s, "
            f"merge={case.metadata.get('merge_coords')}, "
            f"fractions=({case.a1:.6f}, {case.a2:.6f}, {case.a3:.6f})"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize harvested Zalesak getArcFacet cases"
    )
    parser.add_argument(
        "--write-priority-module",
        action="store_true",
        help="Write the curated priority subset module",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRIORITY_OUTPUT,
        help=f"Priority module output path (default: {DEFAULT_PRIORITY_OUTPUT})",
    )
    args = parser.parse_args()

    print(summarize_cases(HARVEST_CASES))
    priority_cases = select_priority_cases(HARVEST_CASES)
    print("\nPriority cases:")
    print(_format_priority_table(priority_cases))

    if args.write_priority_module:
        write_priority_module(args.output, HARVEST_CASES)
        print(f"\nWrote priority module to {args.output}")


if __name__ == "__main__":
    main()
