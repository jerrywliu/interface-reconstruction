#!/usr/bin/env python3
"""Fail closed when submission PDFs contain raster images or unembedded fonts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.trusted_figure_runtime import (
    TrustedFigureRuntime,
    TrustedFigureRuntimeError,
    prepare_trusted_figure_runtime,
    run_attested_tool,
)


class PdfQaError(RuntimeError):
    """Raised when a PDF cannot be inspected reliably."""


@dataclass(frozen=True)
class FontRecord:
    name: str
    embedded: bool
    subset: bool
    font_type: str


@dataclass(frozen=True)
class PdfQaReport:
    path: str
    image_objects: int
    fonts: tuple[FontRecord, ...]
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def parse_pdfimages_list(output: str) -> int:
    """Count image XObjects reported by ``pdfimages -list``."""
    row_pattern = re.compile(r"^\s*\d+\s+\d+\s+")
    return sum(bool(row_pattern.match(line)) for line in output.splitlines())


def parse_pdffonts(output: str) -> tuple[FontRecord, ...]:
    """Parse the stable trailing columns emitted by Poppler's ``pdffonts``."""
    fonts = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[-1].isdigit() or not fields[-2].isdigit():
            continue
        embedded, subset = fields[-5], fields[-4]
        if embedded not in {"yes", "no"} or subset not in {"yes", "no"}:
            continue
        fonts.append(
            FontRecord(
                name=fields[0],
                embedded=embedded == "yes",
                subset=subset == "yes",
                font_type=" ".join(fields[1:-6]),
            )
        )
    return tuple(fonts)


@contextmanager
def _runtime_or_temporary(
    runtime: Optional[TrustedFigureRuntime],
) -> Iterator[TrustedFigureRuntime]:
    if runtime is not None:
        yield runtime
        return
    with tempfile.TemporaryDirectory(prefix="pdf-vector-qa-runtime-") as raw:
        try:
            yield prepare_trusted_figure_runtime(Path(raw) / "runtime")
        except TrustedFigureRuntimeError as exc:
            raise PdfQaError(str(exc)) from exc


def inspect_pdf(
    path: Path,
    *,
    require_fonts: bool = True,
    runner: Optional[Callable[[Sequence[str]], str]] = None,
    runtime: Optional[TrustedFigureRuntime] = None,
) -> PdfQaReport:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise PdfQaError(f"Not a PDF file: {path}")

    if runner is not None:
        image_output = runner(("pdfimages", "-list", str(path)))
        font_output = runner(("pdffonts", str(path)))
    else:
        with _runtime_or_temporary(runtime) as active_runtime:
            try:
                image_output = run_attested_tool(
                    active_runtime, "pdfimages", ("-list", str(path)), timeout=30
                )
                font_output = run_attested_tool(
                    active_runtime, "pdffonts", (str(path),), timeout=30
                )
            except TrustedFigureRuntimeError as exc:
                raise PdfQaError(str(exc)) from exc
    image_count = parse_pdfimages_list(image_output)
    fonts = parse_pdffonts(font_output)
    issues = []
    if image_count:
        issues.append(f"contains {image_count} raster image object(s)")
    unembedded = sorted(font.name for font in fonts if not font.embedded)
    if unembedded:
        issues.append("unembedded fonts: " + ", ".join(unembedded))
    if require_fonts and not fonts:
        issues.append("no fonts reported")

    return PdfQaReport(
        path=str(path),
        image_objects=image_count,
        fonts=fonts,
        issues=tuple(issues),
    )


def discover_pdfs(inputs: Sequence[Path]) -> tuple[Path, ...]:
    pdfs = set()
    for raw_path in inputs:
        path = Path(raw_path).resolve()
        if not path.exists():
            raise PdfQaError(f"Input does not exist: {path}")
        if path.is_dir():
            pdfs.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() == ".pdf"
            )
        elif path.suffix.lower() == ".pdf":
            pdfs.add(path)
        else:
            raise PdfQaError(f"Input is neither a PDF nor a directory: {path}")
    if not pdfs:
        raise PdfQaError("No PDF files found")
    return tuple(sorted(pdfs))


def _json_payload(reports: Sequence[PdfQaReport]) -> dict:
    return {
        "schema_version": 1,
        "passed": all(report.passed for report in reports),
        "pdf_count": len(reports),
        "reports": [asdict(report) | {"passed": report.passed} for report in reports],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files or directories")
    parser.add_argument(
        "--allow-no-fonts",
        action="store_true",
        help="allow vector-only PDFs that contain no font resources",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the complete QA report as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reports = [
            inspect_pdf(path, require_fonts=not args.allow_no_fonts)
            for path in discover_pdfs(args.paths)
        ]
    except PdfQaError as exc:
        print(f"PDF QA ERROR: {exc}", file=sys.stderr)
        return 2

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        font_summary = f"{len(report.fonts)} font(s)"
        print(
            f"{status} {report.path}: {report.image_objects} image object(s), "
            f"{font_summary}"
        )
        for issue in report.issues:
            print(f"  - {issue}")

    payload = _json_payload(reports)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    passed_count = sum(report.passed for report in reports)
    print(f"PDF QA: {passed_count}/{len(reports)} passed")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
