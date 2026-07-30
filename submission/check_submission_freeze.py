#!/usr/bin/env python3
"""Check structural consistency of the prospective submission freeze."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
PAPER = WORKSPACE / "overleaf" / "interface-reconstruction-paper"
CONFIG_PATH = REPO / "submission" / "submission_config.json"
PROVENANCE_PATH = REPO / "submission" / "figure_provenance.csv"
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{[^}]*?/([^}/]+\.pdf)\}")
GENERATED_WORKTREE_ROOTS = frozenset(
    {"logs", "output", "plots", "results", "tmp"}
)


def active_paper_figures() -> set[str]:
    figures: set[str] = set()
    for tex_path in PAPER.rglob("*.tex"):
        for raw_line in tex_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("%", 1)[0]
            figures.update(INCLUDE_RE.findall(line))
    return figures


def expected_run_count(config: dict) -> int:
    wiggles = len(config["benchmark_grid"]["wiggles"])
    full = len(config["benchmark_grid"]["full_resolutions"])
    short = len(config["benchmark_grid"]["short_resolutions"])
    total = 0
    for benchmark in config["benchmarks"].values():
        resolutions = full if benchmark["resolutions"] == "full_resolutions" else short
        runs = len(benchmark["methods"]) * resolutions * wiggles
        if runs != benchmark["planned_runs"]:
            raise AssertionError(f"planned run mismatch for {benchmark['driver']}: {runs}")
        total += runs
    return total


def uncommitted_source_paths(
    repo: Path = REPO, allowed_generated_paths: tuple[Path, ...] = ()
) -> list[str]:
    """Return changed tracked or untracked paths outside known artifact roots."""
    commands = (
        ["git", "diff", "--name-only", "-z"],
        ["git", "diff", "--cached", "--name-only", "-z"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    changed: set[str] = set()
    for command in commands:
        output = subprocess.check_output(command, cwd=repo)
        changed.update(
            raw_path.decode("utf-8")
            for raw_path in output.split(b"\0")
            if raw_path
        )

    repo = repo.resolve()
    allowed_parts = []
    for allowed_path in allowed_generated_paths:
        resolved = (
            allowed_path.resolve()
            if allowed_path.is_absolute()
            else (repo / allowed_path).resolve()
        )
        try:
            allowed_parts.append(resolved.relative_to(repo).parts)
        except ValueError:
            continue

    def is_generated(path: str) -> bool:
        parts = Path(path).parts
        if parts[0] in GENERATED_WORKTREE_ROOTS:
            return True
        return any(parts[: len(prefix)] == prefix for prefix in allowed_parts)

    return sorted(path for path in changed if not is_generated(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="check only that source/config/test worktree paths are committed",
    )
    parser.add_argument(
        "--allow-generated-path",
        action="append",
        type=Path,
        default=[],
        help="additional generated worktree path to exclude from the source audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_changes = uncommitted_source_paths(
        allowed_generated_paths=tuple(args.allow_generated_path)
    )
    if args.source_only:
        if source_changes:
            print("SOURCE NOT CLEAN")
            for path in source_changes:
                print(f"- {path}")
            return 1
        print("SOURCE CLEAN")
        return 0

    config = json.loads(args.config.read_text(encoding="utf-8"))
    with PROVENANCE_PATH.open(newline="", encoding="utf-8") as stream:
        provenance = list(csv.DictReader(stream))

    errors: list[str] = []
    if source_changes:
        errors.append(
            "uncommitted source/config/test paths: " + ", ".join(source_changes)
        )
    runs = expected_run_count(config)
    if runs != config["planned_totals"]["runs"]:
        errors.append(f"computed {runs} runs, configured {config['planned_totals']['runs']}")
    cases = runs * config["benchmark_grid"]["trials_per_setting"]
    if cases != config["planned_totals"]["cases"]:
        errors.append(
            f"computed {cases} cases, configured {config['planned_totals']['cases']}"
        )

    active = active_paper_figures()
    recorded = {row["paper_file"] for row in provenance}
    if active != recorded:
        errors.append(f"missing provenance: {sorted(active - recorded)}")
        errors.append(f"inactive provenance rows: {sorted(recorded - active)}")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    target = config["source"]["target_commit"]
    if not target:
        errors.append("final source.target_commit is unset")
    elif target != head:
        errors.append(f"target commit {target} does not match current HEAD {head}")

    if config["status"] != "frozen":
        errors.append(f"configuration status is {config['status']!r}, not 'frozen'")
    if config.get("launch_approved") is not True:
        errors.append("launch_approved is not true")

    print(f"Active paper figures: {len(active)}")
    print(f"Provenance rows: {len(provenance)}")
    print(f"Planned runs/cases: {runs}/{cases}")
    print(f"Configuration: {args.config}")
    if errors:
        print("NOT FROZEN")
        for error in errors:
            print(f"- {error}")
        return 1
    print("FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
