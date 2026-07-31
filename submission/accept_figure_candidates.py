#!/usr/bin/env python3
"""Fail-closed acceptance gate for the 38 final paper-figure candidates.

The command validates the explicit candidate allowlist, matching 300-DPI PNG
previews, vector PDF properties, and provenance before creating an indexed
review packet. The packet concatenates source PDF pages directly, preserving
their vector content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.pdf_vector_qa import PdfQaError, PdfQaReport, inspect_pdf


DEFAULT_ALLOWLIST = Path(__file__).with_name("final_figure_candidates.json")
EXPECTED_COUNTS = {
    "candidate_pdfs": 38,
    "unpaired_candidates": 14,
    "paired_slots": 12,
    "paired_candidates": 24,
}
ROOT_KEYS = ("figure_root", "c0_root")
VARIANTS = ("unpaired", "with_endpoints", "clean")
INDEX_ROWS_PER_PAGE = 14
SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

SOURCE_MAP_JSON = "figure_candidate_source_map.json"
SOURCE_MAP_CSV = "figure_candidate_source_map.csv"
VECTOR_QA_JSON = "figure_candidate_vector_qa.json"
REVIEW_PDF = "figure_candidate_review.pdf"


class FigureAcceptanceError(RuntimeError):
    """Raised when the candidate set cannot be accepted unambiguously."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    slot_id: str
    section: str
    title: str
    variant: str
    root: str
    pdf: str


@dataclass(frozen=True)
class PreviewRecord:
    path: Path
    sha256: str
    width_px: int
    height_px: int
    dpi_x: float
    dpi_y: float


@dataclass(frozen=True)
class AcceptedCandidate:
    order: int
    spec: CandidateSpec
    pdf_path: Path
    pdf_sha256: str
    pdf_page_count: int
    preview: PreviewRecord
    pdf_qa: PdfQaReport
    review_page_start: int = 0
    review_page_end: int = 0


@dataclass(frozen=True)
class AcceptanceOutputs:
    review_pdf: Path
    source_map_json: Path
    source_map_csv: Path
    vector_qa_json: Path


def _safe_relative_path(raw: object, *, field: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise FigureAcceptanceError(f"{field} must be a nonempty POSIX path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise FigureAcceptanceError(f"{field} must be a safe relative path: {raw}")
    return path.as_posix()


def _require_nonempty_string(raw: object, *, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise FigureAcceptanceError(f"{field} must be a nonempty string")
    return raw.strip()


def load_candidate_allowlist(
    path: Path = DEFAULT_ALLOWLIST,
) -> Tuple[CandidateSpec, ...]:
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FigureAcceptanceError(
            f"Candidate allowlist does not exist: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FigureAcceptanceError(f"Invalid candidate allowlist JSON: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise FigureAcceptanceError("Candidate allowlist schema_version must be 1")
    if payload.get("expected_counts") != EXPECTED_COUNTS:
        raise FigureAcceptanceError(
            f"Candidate allowlist counts must be exactly {EXPECTED_COUNTS}"
        )
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise FigureAcceptanceError(
            "Candidate allowlist must contain a candidates list"
        )

    specs: List[CandidateSpec] = []
    for index, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, dict):
            raise FigureAcceptanceError(f"Candidate {index} must be an object")
        expected_fields = {
            "candidate_id",
            "slot_id",
            "section",
            "title",
            "variant",
            "root",
            "pdf",
        }
        missing_fields = expected_fields - set(raw)
        if missing_fields:
            raise FigureAcceptanceError(
                f"Candidate {index} is missing fields: {', '.join(sorted(missing_fields))}"
            )
        unknown_fields = set(raw) - expected_fields
        if unknown_fields:
            raise FigureAcceptanceError(
                f"Candidate {index} has unknown fields: {', '.join(sorted(unknown_fields))}"
            )
        spec = CandidateSpec(
            candidate_id=_require_nonempty_string(
                raw["candidate_id"], field=f"candidate {index} candidate_id"
            ),
            slot_id=_require_nonempty_string(
                raw["slot_id"], field=f"candidate {index} slot_id"
            ),
            section=_require_nonempty_string(
                raw["section"], field=f"candidate {index} section"
            ),
            title=_require_nonempty_string(
                raw["title"], field=f"candidate {index} title"
            ),
            variant=_require_nonempty_string(
                raw["variant"], field=f"candidate {index} variant"
            ),
            root=_require_nonempty_string(raw["root"], field=f"candidate {index} root"),
            pdf=_safe_relative_path(raw["pdf"], field=f"candidate {index} pdf"),
        )
        if spec.variant not in VARIANTS:
            raise FigureAcceptanceError(
                f"Candidate {spec.candidate_id} has invalid variant {spec.variant}"
            )
        if spec.root not in ROOT_KEYS:
            raise FigureAcceptanceError(
                f"Candidate {spec.candidate_id} has invalid root {spec.root}"
            )
        if PurePosixPath(spec.pdf).suffix.lower() != ".pdf":
            raise FigureAcceptanceError(
                f"Candidate {spec.candidate_id} is not a PDF path: {spec.pdf}"
            )
        specs.append(spec)

    _validate_allowlist_contract(specs)
    return tuple(specs)


def _validate_allowlist_contract(specs: Sequence[CandidateSpec]) -> None:
    if len(specs) != EXPECTED_COUNTS["candidate_pdfs"]:
        raise FigureAcceptanceError(
            f"Expected 38 candidate PDFs, found {len(specs)} in the allowlist"
        )
    candidate_ids = [spec.candidate_id for spec in specs]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise FigureAcceptanceError("Candidate IDs must be unique")
    rooted_paths = [(spec.root, spec.pdf) for spec in specs]
    if len(set(rooted_paths)) != len(rooted_paths):
        raise FigureAcceptanceError("Candidate root/PDF paths must be unique")

    unpaired = [spec for spec in specs if spec.variant == "unpaired"]
    paired = [spec for spec in specs if spec.variant != "unpaired"]
    if len(unpaired) != EXPECTED_COUNTS["unpaired_candidates"]:
        raise FigureAcceptanceError(
            f"Expected 14 unpaired candidates, found {len(unpaired)}"
        )
    if len(paired) != EXPECTED_COUNTS["paired_candidates"]:
        raise FigureAcceptanceError(
            f"Expected 24 paired candidates, found {len(paired)}"
        )

    paired_slots: Dict[str, set] = {}
    for spec in paired:
        paired_slots.setdefault(spec.slot_id, set()).add(spec.variant)
    if len(paired_slots) != EXPECTED_COUNTS["paired_slots"]:
        raise FigureAcceptanceError(
            f"Expected 12 paired slots, found {len(paired_slots)}"
        )
    invalid_slots = {
        slot: sorted(variants)
        for slot, variants in paired_slots.items()
        if variants != {"with_endpoints", "clean"}
    }
    if invalid_slots:
        raise FigureAcceptanceError(
            "Every paired slot must contain exactly with_endpoints and clean: "
            + json.dumps(invalid_slots, sort_keys=True)
        )
    unpaired_slots = {spec.slot_id for spec in unpaired}
    if len(unpaired_slots) != EXPECTED_COUNTS["unpaired_candidates"]:
        raise FigureAcceptanceError(
            f"Expected 14 distinct unpaired slots, found {len(unpaired_slots)}"
        )
    if unpaired_slots & set(paired_slots):
        raise FigureAcceptanceError("A slot cannot be both paired and unpaired")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_pdfs(root: Path) -> set:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    }


def verify_candidate_inventory(
    specs: Sequence[CandidateSpec], roots: Mapping[str, Path]
) -> None:
    errors: List[str] = []
    for key in ROOT_KEYS:
        root = roots[key]
        expected = {spec.pdf for spec in specs if spec.root == key}
        actual = _inventory_pdfs(root)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            errors.append(
                f"Missing candidate PDFs under {key} ({root}):\n  - "
                + "\n  - ".join(missing)
            )
        if unexpected:
            errors.append(
                f"Unexpected candidate PDFs under {key} ({root}):\n  - "
                + "\n  - ".join(unexpected)
            )
    if errors:
        raise FigureAcceptanceError("\n".join(errors))


def inspect_png_preview(path: Path, *, required_dpi: float = 300.0) -> PreviewRecord:
    path = Path(path).resolve()
    if not path.is_file():
        raise FigureAcceptanceError(f"Missing matching PNG preview: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            image_format = image.format
            width, height = image.size
            dpi = image.info.get("dpi")
    except (OSError, UnidentifiedImageError) as exc:
        raise FigureAcceptanceError(f"Unreadable PNG preview {path}: {exc}") from exc
    if image_format != "PNG":
        raise FigureAcceptanceError(f"Preview is not encoded as PNG: {path}")
    if width <= 0 or height <= 0:
        raise FigureAcceptanceError(f"Preview has invalid dimensions: {path}")
    if not isinstance(dpi, (tuple, list)) or len(dpi) < 2:
        raise FigureAcceptanceError(f"Preview does not record DPI metadata: {path}")
    dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
    if abs(dpi_x - required_dpi) > 0.5 or abs(dpi_y - required_dpi) > 0.5:
        raise FigureAcceptanceError(
            f"Preview is not 300 DPI: {path} records {dpi_x:.3f} x {dpi_y:.3f}"
        )
    return PreviewRecord(
        path=path,
        sha256=_sha256(path),
        width_px=width,
        height_px=height,
        dpi_x=dpi_x,
        dpi_y=dpi_y,
    )


def _run_text_tool(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise FigureAcceptanceError(
            f"Required tool is unavailable: {command[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        raise FigureAcceptanceError(f"{' '.join(command)} failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FigureAcceptanceError(f"{' '.join(command)} timed out") from exc
    return completed.stdout


def pdf_page_count(path: Path) -> int:
    output = _run_text_tool(("pdfinfo", str(Path(path).resolve())))
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    if not match:
        raise FigureAcceptanceError(f"pdfinfo did not report a page count for {path}")
    pages = int(match.group(1))
    if pages < 1:
        raise FigureAcceptanceError(f"PDF has no pages: {path}")
    return pages


def _truncate(text: str, *, font: str, size: float, max_width: float) -> str:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    if stringWidth(text, font, size) <= max_width:
        return text
    suffix = "..."
    while text and stringWidth(text + suffix, font, size) > max_width:
        text = text[:-1]
    return text + suffix


def _write_vector_index(
    path: Path,
    records: Sequence[AcceptedCandidate],
    *,
    source_commit: str,
) -> int:
    try:
        import reportlab
        from reportlab.lib.colors import HexColor
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError as exc:
        raise FigureAcceptanceError(
            "reportlab is required; install requirements-figures.txt"
        ) from exc

    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    regular_font = "FigureReviewVera"
    bold_font = "FigureReviewVeraBold"
    if regular_font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(regular_font, str(font_dir / "Vera.ttf")))
    if bold_font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(bold_font, str(font_dir / "VeraBd.ttf")))

    index_pages = max(1, math.ceil(len(records) / INDEX_ROWS_PER_PAGE))
    width, height = landscape(letter)
    document = canvas.Canvas(
        str(path),
        pagesize=(width, height),
        pageCompression=1,
        initialFontName=regular_font,
        initialFontSize=8,
    )
    document.setTitle("Final Figure Candidate Review Index")
    document.setAuthor("Interface Reconstruction collaborators")

    for page_index in range(index_pages):
        start = page_index * INDEX_ROWS_PER_PAGE
        page_records = records[start : start + INDEX_ROWS_PER_PAGE]
        document.setFillColor(HexColor("#111827"))
        document.setFont(bold_font, 17)
        document.drawString(38, height - 42, "Final Figure Candidate Review")
        document.setFillColor(HexColor("#2563eb"))
        document.rect(38, height - 53, 170, 3, stroke=0, fill=1)
        document.setFillColor(HexColor("#4b5563"))
        document.setFont(regular_font, 7.8)
        document.drawString(
            38,
            height - 69,
            f"38 candidates | source {source_commit} | index {page_index + 1}/{index_pages}",
        )

        y = height - 94
        for record in page_records:
            page_label = (
                str(record.review_page_start)
                if record.review_page_start == record.review_page_end
                else f"{record.review_page_start}-{record.review_page_end}"
            )
            document.setFillColor(HexColor("#111827"))
            document.setFont(bold_font, 8.5)
            document.drawString(38, y, f"p. {page_label}")
            document.drawString(88, y, record.spec.section)
            title = f"{record.spec.title} [{record.spec.variant}]"
            document.drawString(
                158,
                y,
                _truncate(title, font=bold_font, size=8.5, max_width=590),
            )
            document.setFillColor(HexColor("#6b7280"))
            document.setFont(regular_font, 6.8)
            source = f"{record.spec.root}/{record.spec.pdf}"
            source = _truncate(source, font=regular_font, size=6.8, max_width=640)
            document.drawString(88, y - 12, source)
            document.drawRightString(width - 38, y - 12, record.pdf_sha256[:12])
            document.setStrokeColor(HexColor("#e5e7eb"))
            document.line(38, y - 20, width - 38, y - 20)
            y -= 34

        document.setFillColor(HexColor("#6b7280"))
        document.setFont(regular_font, 6.8)
        document.drawString(
            38, 20, "PDF pages are concatenated directly; no figure is rasterized."
        )
        document.drawRightString(width - 38, 20, str(page_index + 1))
        document.showPage()
    document.save()
    return index_pages


def build_vector_review_pdf(
    records: Sequence[AcceptedCandidate],
    output: Path,
    *,
    source_commit: str,
) -> None:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="figure-review-", dir=str(output.parent)
    ) as raw:
        temporary = Path(raw)
        index_pdf = temporary / "index.pdf"
        merged_pdf = temporary / "review.pdf"
        _write_vector_index(index_pdf, records, source_commit=source_commit)
        command = ["pdfunite", str(index_pdf)]
        command.extend(str(record.pdf_path) for record in records)
        command.append(str(merged_pdf))
        _run_text_tool(command)
        os.replace(merged_pdf, output)


def _qa_dict(report: PdfQaReport) -> dict:
    return asdict(report) | {"passed": report.passed}


def _root_contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_invocation(
    roots: Mapping[str, Path], output_dir: Path, source_commit: str
) -> None:
    if not SOURCE_COMMIT_PATTERN.fullmatch(source_commit):
        raise FigureAcceptanceError("source_commit must be a full 40-character SHA-1")
    if roots["figure_root"] == roots["c0_root"]:
        raise FigureAcceptanceError("figure_root and c0_root must be distinct")
    for key, root in roots.items():
        if not root.is_dir():
            raise FigureAcceptanceError(f"{key} is not a directory: {root}")
        if _root_contains(root, output_dir):
            raise FigureAcceptanceError(
                f"output_dir must be outside {key} so review outputs cannot contaminate the candidate inventory"
            )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FigureAcceptanceError(f"output_dir must be new or empty: {output_dir}")


def _write_csv(
    path: Path, records: Sequence[AcceptedCandidate], source_commit: str
) -> None:
    fieldnames = (
        "order",
        "candidate_id",
        "slot_id",
        "section",
        "title",
        "variant",
        "root",
        "source_commit",
        "pdf_relative_path",
        "pdf_sha256",
        "pdf_page_count",
        "png_relative_path",
        "png_sha256",
        "png_width_px",
        "png_height_px",
        "png_dpi_x",
        "png_dpi_y",
        "pdf_image_objects",
        "pdf_font_count",
        "review_page_start",
        "review_page_end",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "order": record.order,
                    "candidate_id": record.spec.candidate_id,
                    "slot_id": record.spec.slot_id,
                    "section": record.spec.section,
                    "title": record.spec.title,
                    "variant": record.spec.variant,
                    "root": record.spec.root,
                    "source_commit": source_commit,
                    "pdf_relative_path": record.spec.pdf,
                    "pdf_sha256": record.pdf_sha256,
                    "pdf_page_count": record.pdf_page_count,
                    "png_relative_path": PurePosixPath(record.spec.pdf)
                    .with_suffix(".png")
                    .as_posix(),
                    "png_sha256": record.preview.sha256,
                    "png_width_px": record.preview.width_px,
                    "png_height_px": record.preview.height_px,
                    "png_dpi_x": f"{record.preview.dpi_x:.3f}",
                    "png_dpi_y": f"{record.preview.dpi_y:.3f}",
                    "pdf_image_objects": record.pdf_qa.image_objects,
                    "pdf_font_count": len(record.pdf_qa.fonts),
                    "review_page_start": record.review_page_start,
                    "review_page_end": record.review_page_end,
                }
            )


def accept_figure_candidates(
    *,
    figure_root: Path,
    c0_root: Path,
    output_dir: Path,
    source_commit: str,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    pdf_inspector: Callable[..., PdfQaReport] = inspect_pdf,
    page_counter: Callable[[Path], int] = pdf_page_count,
    review_builder: Callable[..., None] = build_vector_review_pdf,
) -> AcceptanceOutputs:
    roots = {
        "figure_root": Path(figure_root).resolve(),
        "c0_root": Path(c0_root).resolve(),
    }
    output_dir = Path(output_dir).resolve()
    source_commit = source_commit.lower()
    _validate_invocation(roots, output_dir, source_commit)
    specs = load_candidate_allowlist(allowlist_path)
    verify_candidate_inventory(specs, roots)

    records: List[AcceptedCandidate] = []
    errors: List[str] = []
    for order, spec in enumerate(specs, start=1):
        pdf_path = roots[spec.root] / spec.pdf
        preview_path = pdf_path.with_suffix(".png")
        try:
            preview = inspect_png_preview(preview_path)
        except FigureAcceptanceError as exc:
            errors.append(str(exc))
            continue
        try:
            report = pdf_inspector(pdf_path, require_fonts=True)
        except PdfQaError as exc:
            errors.append(f"PDF QA could not inspect {pdf_path}: {exc}")
            continue
        if not report.passed:
            errors.append(f"PDF QA failed for {pdf_path}: " + "; ".join(report.issues))
            continue
        try:
            pages = page_counter(pdf_path)
        except FigureAcceptanceError as exc:
            errors.append(str(exc))
            continue
        if pages < 1:
            errors.append(f"PDF has no pages: {pdf_path}")
            continue
        records.append(
            AcceptedCandidate(
                order=order,
                spec=spec,
                pdf_path=pdf_path,
                pdf_sha256=_sha256(pdf_path),
                pdf_page_count=pages,
                preview=preview,
                pdf_qa=report,
            )
        )
    if errors:
        raise FigureAcceptanceError("\n".join(errors))
    if len(records) != EXPECTED_COUNTS["candidate_pdfs"]:
        raise FigureAcceptanceError(
            f"Internal error: accepted {len(records)} of 38 candidate records"
        )

    index_pages = max(1, math.ceil(len(records) / INDEX_ROWS_PER_PAGE))
    review_page = index_pages + 1
    numbered_records: List[AcceptedCandidate] = []
    for record in records:
        numbered_records.append(
            replace(
                record,
                review_page_start=review_page,
                review_page_end=review_page + record.pdf_page_count - 1,
            )
        )
        review_page += record.pdf_page_count

    output_dir.mkdir(parents=True, exist_ok=True)
    review_pdf = output_dir / REVIEW_PDF
    try:
        review_builder(numbered_records, review_pdf, source_commit=source_commit)
        review_report = pdf_inspector(review_pdf, require_fonts=True)
    except (FigureAcceptanceError, PdfQaError):
        review_pdf.unlink(missing_ok=True)
        raise
    if not review_report.passed:
        review_pdf.unlink(missing_ok=True)
        raise FigureAcceptanceError(
            "Review PDF QA failed: " + "; ".join(review_report.issues)
        )

    qa_path = output_dir / VECTOR_QA_JSON
    qa_payload = {
        "schema_version": 1,
        "passed": True,
        "candidate_pdf_count": len(numbered_records),
        "candidate_reports": [_qa_dict(record.pdf_qa) for record in numbered_records],
        "review_report": _qa_dict(review_report),
    }
    qa_path.write_text(json.dumps(qa_payload, indent=2) + "\n", encoding="utf-8")

    source_map_json = output_dir / SOURCE_MAP_JSON
    source_map_csv = output_dir / SOURCE_MAP_CSV
    source_payload = {
        "schema_version": 1,
        "passed": True,
        "source_commit": source_commit,
        "allowlist": {
            "path": str(Path(allowlist_path).resolve()),
            "sha256": _sha256(Path(allowlist_path).resolve()),
            "expected_counts": EXPECTED_COUNTS,
        },
        "roots": {key: str(path) for key, path in roots.items()},
        "review": {
            "path": str(review_pdf),
            "sha256": _sha256(review_pdf),
            "index_pages": index_pages,
            "page_count": review_page - 1,
        },
        "vector_qa": {
            "path": str(qa_path),
            "sha256": _sha256(qa_path),
        },
        "candidates": [
            {
                "order": record.order,
                **asdict(record.spec),
                "pdf_path": str(record.pdf_path),
                "pdf_sha256": record.pdf_sha256,
                "pdf_page_count": record.pdf_page_count,
                "png_path": str(record.preview.path),
                "png_sha256": record.preview.sha256,
                "png_width_px": record.preview.width_px,
                "png_height_px": record.preview.height_px,
                "png_dpi_x": record.preview.dpi_x,
                "png_dpi_y": record.preview.dpi_y,
                "review_page_start": record.review_page_start,
                "review_page_end": record.review_page_end,
            }
            for record in numbered_records
        ],
    }
    source_map_json.write_text(
        json.dumps(source_payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(source_map_csv, numbered_records, source_commit)
    return AcceptanceOutputs(
        review_pdf=review_pdf,
        source_map_json=source_map_json,
        source_map_csv=source_map_csv,
        vector_qa_json=qa_path,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--c0-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty directory outside both candidate roots",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="full 40-character source commit shared by all candidate generators",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="explicit 38-candidate allowlist JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        outputs = accept_figure_candidates(
            figure_root=args.figure_root,
            c0_root=args.c0_root,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
            allowlist_path=args.allowlist,
        )
    except FigureAcceptanceError as exc:
        print(f"FIGURE ACCEPTANCE ERROR: {exc}", file=sys.stderr)
        return 2
    print("FIGURE CANDIDATES ACCEPTED: 38/38 PDFs and 38/38 PNG previews")
    print(f"Review PDF: {outputs.review_pdf}")
    print(f"Source map JSON: {outputs.source_map_json}")
    print(f"Source map CSV: {outputs.source_map_csv}")
    print(f"Vector QA JSON: {outputs.vector_qa_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
