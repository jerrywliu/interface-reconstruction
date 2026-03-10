#!/usr/bin/env python3
"""
Send camera-ready static release artifacts to Slack.
"""

import argparse
from pathlib import Path

from util.io.slack import load_slack_env, send_results_to_slack


CANONICAL_FIGURES = [
    "line_reconstruction_perturbed_all_methods_2x2.png",
    "circle_reconstruction_perturbed_all_methods_5x2_axes.png",
    "ellipse_reconstruction_perturbed_all_methods_5x2_axes.png",
    "square_reconstruction_perturbed_all_methods_2x2.png",
    "zalesak_reconstruction_perturbed_all_methods_2x2.png",
]


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def collect_files(release_dir):
    release = Path(release_dir).resolve()
    paper_dir = release / "paper_figs"
    files = []

    for name in CANONICAL_FIGURES:
        candidate = paper_dir / name
        if candidate.exists():
            files.append(str(candidate))

    if not files and paper_dir.exists():
        for candidate in sorted(paper_dir.glob("*all_methods*.png")):
            files.append(str(candidate))

    manifest = release / "manifest.md"
    if manifest.exists():
        files.append(str(manifest))

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Send static camera-ready release artifacts to Slack."
    )
    parser.add_argument(
        "--release_dir",
        type=str,
        required=True,
        help="Path to results/static/camera_ready/<release_id>",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Optional base message.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="Max files per Slack message.",
    )
    args = parser.parse_args()

    load_slack_env()

    files = collect_files(args.release_dir)
    release_name = Path(args.release_dir).resolve().name
    base = args.message or f"Static camera-ready release complete: {release_name}"

    if not files:
        send_results_to_slack(f"{base} (no files found to attach)", [])
        return

    chunks = list(chunked(files, max(1, args.chunk_size)))
    for index, file_chunk in enumerate(chunks, start=1):
        suffix = f" ({index}/{len(chunks)})" if len(chunks) > 1 else ""
        send_results_to_slack(f"{base}{suffix}", file_chunk)


if __name__ == "__main__":
    main()
