#!/usr/bin/env python3
"""Internal review builder for orchestrated final paper-figure candidates.

This module deliberately has no standalone acceptance command. The final-figure
orchestrator passes an in-memory state derived from its own trusted generation
run. Caller-authored provenance manifests are never an acceptance input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.final_figure_provenance import file_sha256
from submission.pdf_vector_qa import PdfQaError, PdfQaReport, inspect_pdf
from submission.trusted_figure_runtime import (
    TrustedFigureRuntime,
    TrustedFigureRuntimeError,
    prepare_trusted_figure_runtime,
    run_attested_tool,
)


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
class PdfPageInfo:
    page_count: int
    width_points: float
    height_points: float


@dataclass(frozen=True)
class ProvenanceEvidence:
    manifest_path: Path
    manifest_sha256: str
    generator: str
    generator_source_commit: str


_ORCHESTRATION_AUTHORITY = object()


@dataclass(frozen=True)
class _OrchestratedAcceptanceState:
    authority: object
    figure_root: Path
    c0_root: Path
    snapshot_root: Path
    release_anchor: Mapping[str, object]
    generator_source_commit: str
    orchestration_record: Path
    orchestration_record_sha256: str
    allowlist_path: Path
    allowlist_sha256: str
    candidate_records: Tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class AcceptedCandidate:
    order: int
    spec: CandidateSpec
    pdf_path: Path
    pdf_sha256: str
    pdf_page_count: int
    pdf_width_points: float
    pdf_height_points: float
    preview: PreviewRecord
    pdf_qa: PdfQaReport
    provenance: ProvenanceEvidence
    review_page_start: int = 0
    review_page_end: int = 0


@dataclass(frozen=True)
class AcceptanceOutputs:
    review_pdf: Path
    source_map_json: Path
    source_map_csv: Path
    vector_qa_json: Path


def _create_orchestrated_acceptance_state(
    *,
    figure_root: Path,
    c0_root: Path,
    snapshot_root: Path,
    release_anchor: Mapping[str, object],
    generator_source_commit: str,
    orchestration_record: Path,
    allowlist_path: Path,
    candidate_records: Sequence[Mapping[str, object]],
) -> _OrchestratedAcceptanceState:
    """Create the process-local capability consumed by internal acceptance."""

    snapshot_root = Path(snapshot_root).resolve()
    figure_root = Path(figure_root).resolve()
    c0_root = Path(c0_root).resolve()
    orchestration_record = Path(orchestration_record).resolve()
    allowlist_path = Path(allowlist_path).resolve()
    expected_roots = {
        "figure_root": (snapshot_root / "candidates" / "figure_root").resolve(),
        "c0_root": (snapshot_root / "candidates" / "c0_root").resolve(),
    }
    if (
        figure_root != expected_roots["figure_root"]
        or c0_root != expected_roots["c0_root"]
    ):
        raise FigureAcceptanceError(
            "Acceptance roots are not the orchestrator's private candidate roots"
        )
    if not orchestration_record.is_file() or not _root_contains(
        snapshot_root, orchestration_record
    ):
        raise FigureAcceptanceError("Orchestration record is outside private staging")
    if not allowlist_path.is_file() or not _root_contains(
        snapshot_root, allowlist_path
    ):
        raise FigureAcceptanceError("Approved allowlist is outside private staging")
    if not isinstance(release_anchor.get("source_commit"), str):
        raise FigureAcceptanceError("Orchestration state lacks a release source commit")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", generator_source_commit or ""):
        raise FigureAcceptanceError(
            "Orchestration state lacks an approved generator commit"
        )
    return _OrchestratedAcceptanceState(
        authority=_ORCHESTRATION_AUTHORITY,
        figure_root=figure_root,
        c0_root=c0_root,
        snapshot_root=snapshot_root,
        release_anchor=dict(release_anchor),
        generator_source_commit=generator_source_commit,
        orchestration_record=orchestration_record,
        orchestration_record_sha256=file_sha256(orchestration_record),
        allowlist_path=allowlist_path,
        allowlist_sha256=file_sha256(allowlist_path),
        candidate_records=tuple(dict(record) for record in candidate_records),
    )


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


def _candidate_provenance_from_state(
    specs: Sequence[CandidateSpec],
    roots: Mapping[str, Path],
    state: _OrchestratedAcceptanceState,
) -> Dict[str, ProvenanceEvidence]:
    expected = {spec.candidate_id: spec for spec in specs}
    if len(state.candidate_records) != len(expected):
        raise FigureAcceptanceError("Orchestrated candidate set is incomplete")
    evidence: Dict[str, ProvenanceEvidence] = {}
    for raw in state.candidate_records:
        candidate_id = raw.get("candidate_id")
        if candidate_id not in expected or candidate_id in evidence:
            raise FigureAcceptanceError(
                "Orchestrated candidate is unknown or duplicated"
            )
        spec = expected[candidate_id]
        expected_path = _candidate_pdf_logical_path(spec).as_posix()
        if raw.get("path") != expected_path:
            raise FigureAcceptanceError(
                f"Orchestrated candidate path mismatch for {candidate_id}"
            )
        candidate_path = roots[spec.root] / spec.pdf
        if raw.get("sha256") != _sha256(candidate_path):
            raise FigureAcceptanceError(
                f"Orchestrated candidate checksum mismatch for {candidate_id}"
            )
        generator = raw.get("generator")
        if not isinstance(generator, str) or not generator:
            raise FigureAcceptanceError(
                f"Orchestrated candidate lacks generator for {candidate_id}"
            )
        evidence[candidate_id] = ProvenanceEvidence(
            manifest_path=state.orchestration_record,
            manifest_sha256=state.orchestration_record_sha256,
            generator=generator,
            generator_source_commit=state.generator_source_commit,
        )
    if set(evidence) != set(expected):
        raise FigureAcceptanceError(
            "Orchestration does not cover exactly 38 candidates"
        )
    return evidence


@contextmanager
def _runtime_or_temporary(
    runtime: Optional[TrustedFigureRuntime],
) -> Iterator[TrustedFigureRuntime]:
    if runtime is not None:
        yield runtime
        return
    with tempfile.TemporaryDirectory(prefix="figure-acceptance-runtime-") as raw:
        try:
            yield prepare_trusted_figure_runtime(Path(raw) / "runtime")
        except TrustedFigureRuntimeError as exc:
            raise FigureAcceptanceError(str(exc)) from exc


def _run_text_tool(
    runtime: TrustedFigureRuntime, name: str, arguments: Sequence[str]
) -> str:
    try:
        return run_attested_tool(runtime, name, arguments)
    except TrustedFigureRuntimeError as exc:
        raise FigureAcceptanceError(str(exc)) from exc


def pdf_page_info(
    path: Path, *, runtime: Optional[TrustedFigureRuntime] = None
) -> PdfPageInfo:
    with _runtime_or_temporary(runtime) as active_runtime:
        output = _run_text_tool(active_runtime, "pdfinfo", (str(Path(path).resolve()),))
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", output, flags=re.MULTILINE
    )
    if not page_match or not size_match:
        raise FigureAcceptanceError(f"pdfinfo did not report pages and size for {path}")
    pages = int(page_match.group(1))
    if pages < 1:
        raise FigureAcceptanceError(f"PDF has no pages: {path}")
    return PdfPageInfo(
        page_count=pages,
        width_points=float(size_match.group(1)),
        height_points=float(size_match.group(2)),
    )


def render_pdf_preview(
    pdf_path: Path,
    output_path: Path,
    *,
    dpi: int = 300,
    page: int = 1,
    runtime: Optional[TrustedFigureRuntime] = None,
) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_path.with_suffix("")
    with _runtime_or_temporary(runtime) as active_runtime:
        _run_text_tool(
            active_runtime,
            "pdftocairo",
            (
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                "-f",
                str(page),
                "-l",
                str(page),
                str(Path(pdf_path).resolve()),
                str(prefix),
            ),
        )
    if not output_path.is_file():
        raise FigureAcceptanceError(f"PDF renderer did not produce {output_path}")


def inspect_generated_preview(
    path: Path,
    page_info: PdfPageInfo,
    *,
    required_dpi: float = 300.0,
    logical_path: Optional[Path] = None,
) -> PreviewRecord:
    path = Path(path).resolve()
    if not path.is_file():
        raise FigureAcceptanceError(f"Missing generated PNG preview: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            image_format = image.format
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise FigureAcceptanceError(f"Unreadable generated PNG {path}: {exc}") from exc
    if image_format != "PNG":
        raise FigureAcceptanceError(f"Generated preview is not PNG: {path}")
    expected_width = page_info.width_points * required_dpi / 72.0
    expected_height = page_info.height_points * required_dpi / 72.0
    if min(width, height) < 250:
        raise FigureAcceptanceError(f"Generated preview is implausibly tiny: {path}")
    if abs(width - expected_width) > 2.0 or abs(height - expected_height) > 2.0:
        raise FigureAcceptanceError(
            f"Generated preview dimensions do not match PDF page: {path} is "
            f"{width}x{height}, expected about {expected_width:.1f}x{expected_height:.1f}"
        )
    if (
        abs((width / height) - (page_info.width_points / page_info.height_points))
        > 0.002
    ):
        raise FigureAcceptanceError(f"Generated preview aspect ratio is stale: {path}")
    return PreviewRecord(
        path=Path(logical_path) if logical_path is not None else path,
        sha256=_sha256(path),
        width_px=width,
        height_px=height,
        dpi_x=required_dpi,
        dpi_y=required_dpi,
    )


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
    runtime: Optional[TrustedFigureRuntime] = None,
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
        arguments = [str(index_pdf)]
        arguments.extend(str(record.pdf_path) for record in records)
        arguments.append(str(merged_pdf))
        with _runtime_or_temporary(runtime) as active_runtime:
            _run_text_tool(active_runtime, "pdfunite", arguments)
        os.replace(merged_pdf, output)


def _png_pixels_equal(first: Path, second: Path) -> bool:
    with Image.open(first) as left, Image.open(second) as right:
        left.load()
        right.load()
        return (
            left.mode == right.mode
            and left.size == right.size
            and left.tobytes() == right.tobytes()
        )


def verify_review_page_map(
    records: Sequence[AcceptedCandidate],
    review_pdf: Path,
    candidate_previews: Mapping[str, Path],
    *,
    renderer: Callable[..., None] = render_pdf_preview,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="review-page-map-", dir=str(Path(review_pdf).resolve().parent)
    ) as raw:
        scratch = Path(raw)
        for record in records:
            rendered = scratch / f"{record.spec.candidate_id}.png"
            renderer(
                review_pdf,
                rendered,
                dpi=300,
                page=record.review_page_start,
            )
            source_preview = candidate_previews[record.spec.candidate_id]
            if not _png_pixels_equal(source_preview, rendered):
                raise FigureAcceptanceError(
                    f"Review page map mismatch for {record.spec.candidate_id} at "
                    f"page {record.review_page_start}"
                )


def _candidate_pdf_logical_path(spec: CandidateSpec) -> Path:
    return Path("candidates") / spec.root / spec.pdf


def _candidate_preview_logical_path(spec: CandidateSpec) -> Path:
    return Path("review") / "previews" / f"{spec.candidate_id}.png"


def _qa_dict(report: PdfQaReport, *, logical_path: Path) -> dict:
    return asdict(report) | {
        "path": logical_path.as_posix(),
        "passed": report.passed,
    }


def _root_contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_invocation(roots: Mapping[str, Path], output_dir: Path) -> None:
    if roots["figure_root"] == roots["c0_root"]:
        raise FigureAcceptanceError("figure_root and c0_root must be distinct")
    for key, root in roots.items():
        if not root.is_dir():
            raise FigureAcceptanceError(f"{key} is not a directory: {root}")
        if _root_contains(root, output_dir):
            raise FigureAcceptanceError(
                f"output_dir must be outside {key} so review outputs cannot contaminate the candidate inventory"
            )
    if output_dir.exists():
        raise FigureAcceptanceError(f"output_dir must not exist: {output_dir}")


def _write_csv(
    path: Path,
    records: Sequence[AcceptedCandidate],
    source_commit: str,
    snapshot_root: Path,
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
        "pdf_width_points",
        "pdf_height_points",
        "png_relative_path",
        "png_sha256",
        "png_width_px",
        "png_height_px",
        "png_dpi_x",
        "png_dpi_y",
        "pdf_image_objects",
        "pdf_font_count",
        "provenance_manifest",
        "provenance_manifest_sha256",
        "provenance_generator",
        "generator_source_commit",
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
                    "pdf_relative_path": _candidate_pdf_logical_path(
                        record.spec
                    ).as_posix(),
                    "pdf_sha256": record.pdf_sha256,
                    "pdf_page_count": record.pdf_page_count,
                    "pdf_width_points": f"{record.pdf_width_points:.3f}",
                    "pdf_height_points": f"{record.pdf_height_points:.3f}",
                    "png_relative_path": _candidate_preview_logical_path(
                        record.spec
                    ).as_posix(),
                    "png_sha256": record.preview.sha256,
                    "png_width_px": record.preview.width_px,
                    "png_height_px": record.preview.height_px,
                    "png_dpi_x": f"{record.preview.dpi_x:.3f}",
                    "png_dpi_y": f"{record.preview.dpi_y:.3f}",
                    "pdf_image_objects": record.pdf_qa.image_objects,
                    "pdf_font_count": len(record.pdf_qa.fonts),
                    "provenance_manifest": record.provenance.manifest_path.relative_to(
                        snapshot_root
                    ).as_posix(),
                    "provenance_manifest_sha256": record.provenance.manifest_sha256,
                    "provenance_generator": record.provenance.generator,
                    "generator_source_commit": (
                        record.provenance.generator_source_commit
                    ),
                    "review_page_start": record.review_page_start,
                    "review_page_end": record.review_page_end,
                }
            )


def _accept_orchestrated_candidates(
    *,
    orchestration_state: _OrchestratedAcceptanceState,
    output_dir: Path,
    pdf_inspector: Callable[..., PdfQaReport] = inspect_pdf,
    page_inspector: Callable[[Path], PdfPageInfo] = pdf_page_info,
    preview_renderer: Callable[..., None] = render_pdf_preview,
    review_builder: Callable[..., None] = build_vector_review_pdf,
    review_map_verifier: Callable[..., None] = verify_review_page_map,
) -> AcceptanceOutputs:
    if orchestration_state.authority is not _ORCHESTRATION_AUTHORITY:
        raise FigureAcceptanceError(
            "Acceptance requires process-local state created by the final-figure orchestrator"
        )
    roots = {
        "figure_root": Path(orchestration_state.figure_root).resolve(),
        "c0_root": Path(orchestration_state.c0_root).resolve(),
    }
    output_dir = Path(output_dir).resolve()
    _validate_invocation(roots, output_dir)
    allowlist_path = orchestration_state.allowlist_path
    if file_sha256(allowlist_path) != orchestration_state.allowlist_sha256:
        raise FigureAcceptanceError(
            "Private approved allowlist mutated before acceptance"
        )
    specs = load_candidate_allowlist(allowlist_path)
    verify_candidate_inventory(specs, roots)
    provenance = _candidate_provenance_from_state(specs, roots, orchestration_state)
    anchor = dict(orchestration_state.release_anchor)
    source_commit = anchor["source_commit"]
    snapshot_root = Path(orchestration_state.snapshot_root).resolve()

    candidate_audits = []
    errors: List[str] = []
    for order, spec in enumerate(specs, start=1):
        pdf_path = roots[spec.root] / spec.pdf
        try:
            report = pdf_inspector(pdf_path, require_fonts=True)
        except PdfQaError as exc:
            errors.append(f"PDF QA could not inspect {pdf_path}: {exc}")
            continue
        if not report.passed:
            errors.append(f"PDF QA failed for {pdf_path}: " + "; ".join(report.issues))
            continue
        try:
            page_info = page_inspector(pdf_path)
        except FigureAcceptanceError as exc:
            errors.append(str(exc))
            continue
        if page_info.page_count != 1:
            errors.append(
                f"Candidate PDF must contain exactly one page: {pdf_path} has "
                f"{page_info.page_count}"
            )
            continue
        candidate_audits.append(
            (order, spec, pdf_path, _sha256(pdf_path), page_info, report)
        )
    if errors:
        raise FigureAcceptanceError("\n".join(errors))
    if len(candidate_audits) != EXPECTED_COUNTS["candidate_pdfs"]:
        raise FigureAcceptanceError(
            f"Internal error: audited {len(candidate_audits)} of 38 candidates"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        preview_dir = staging_dir / "previews"
        preview_dir.mkdir()
        records: List[AcceptedCandidate] = []
        actual_preview_paths = {}
        for order, spec, pdf_path, digest, page_info, report in candidate_audits:
            staged_preview = preview_dir / f"{spec.candidate_id}.png"
            preview_renderer(pdf_path, staged_preview, dpi=300, page=1)
            logical_preview = _candidate_preview_logical_path(spec)
            preview = inspect_generated_preview(
                staged_preview,
                page_info,
                required_dpi=300.0,
                logical_path=logical_preview,
            )
            actual_preview_paths[spec.candidate_id] = staged_preview
            records.append(
                AcceptedCandidate(
                    order=order,
                    spec=spec,
                    pdf_path=pdf_path,
                    pdf_sha256=digest,
                    pdf_page_count=1,
                    pdf_width_points=page_info.width_points,
                    pdf_height_points=page_info.height_points,
                    preview=preview,
                    pdf_qa=report,
                    provenance=provenance[spec.candidate_id],
                )
            )

        index_pages = max(1, math.ceil(len(records) / INDEX_ROWS_PER_PAGE))
        numbered_records = [
            replace(
                record,
                review_page_start=index_pages + record.order,
                review_page_end=index_pages + record.order,
            )
            for record in records
        ]
        staged_review_pdf = staging_dir / REVIEW_PDF
        review_builder(numbered_records, staged_review_pdf, source_commit=source_commit)
        review_report = pdf_inspector(staged_review_pdf, require_fonts=True)
        if not review_report.passed:
            raise FigureAcceptanceError(
                "Review PDF QA failed: " + "; ".join(review_report.issues)
            )
        review_info = page_inspector(staged_review_pdf)
        expected_review_pages = index_pages + len(numbered_records)
        if review_info.page_count != expected_review_pages:
            raise FigureAcceptanceError(
                f"Merged review page count is {review_info.page_count}; expected "
                f"{expected_review_pages}"
            )
        review_map_verifier(
            numbered_records,
            staged_review_pdf,
            actual_preview_paths,
            renderer=preview_renderer,
        )

        staged_qa = staging_dir / VECTOR_QA_JSON
        logical_qa = Path("review") / VECTOR_QA_JSON
        qa_payload = {
            "schema_version": 2,
            "passed": True,
            "candidate_pdf_count": len(numbered_records),
            "candidate_reports": [
                _qa_dict(
                    record.pdf_qa,
                    logical_path=_candidate_pdf_logical_path(record.spec),
                )
                for record in numbered_records
            ],
            "review_report": _qa_dict(
                review_report, logical_path=Path("review") / REVIEW_PDF
            ),
            "measured_review_page_count": review_info.page_count,
            "review_page_map_verified": True,
        }
        staged_qa.write_text(json.dumps(qa_payload, indent=2) + "\n", encoding="utf-8")

        staged_source_map_json = staging_dir / SOURCE_MAP_JSON
        staged_source_map_csv = staging_dir / SOURCE_MAP_CSV
        source_payload = {
            "schema_version": 2,
            "passed": True,
            "source_commit": source_commit,
            "release_source_commit": source_commit,
            "release": anchor,
            "allowlist": {
                "path": allowlist_path.relative_to(snapshot_root).as_posix(),
                "sha256": _sha256(Path(allowlist_path).resolve()),
                "expected_counts": EXPECTED_COUNTS,
            },
            "roots": {
                key: path.relative_to(snapshot_root).as_posix()
                for key, path in roots.items()
            },
            "review": {
                "path": f"review/{REVIEW_PDF}",
                "sha256": _sha256(staged_review_pdf),
                "index_pages": index_pages,
                "page_count": review_info.page_count,
                "page_map_verified": True,
            },
            "vector_qa": {
                "path": str(logical_qa),
                "sha256": _sha256(staged_qa),
            },
            "candidates": [
                {
                    "order": record.order,
                    "candidate_id": record.spec.candidate_id,
                    "slot_id": record.spec.slot_id,
                    "section": record.spec.section,
                    "title": record.spec.title,
                    "variant": record.spec.variant,
                    "root": record.spec.root,
                    "pdf_path": _candidate_pdf_logical_path(record.spec).as_posix(),
                    "pdf_sha256": record.pdf_sha256,
                    "pdf_page_count": record.pdf_page_count,
                    "pdf_width_points": record.pdf_width_points,
                    "pdf_height_points": record.pdf_height_points,
                    "png_path": record.preview.path.as_posix(),
                    "png_sha256": record.preview.sha256,
                    "png_width_px": record.preview.width_px,
                    "png_height_px": record.preview.height_px,
                    "png_dpi_x": record.preview.dpi_x,
                    "png_dpi_y": record.preview.dpi_y,
                    "provenance_manifest": record.provenance.manifest_path.relative_to(
                        snapshot_root
                    ).as_posix(),
                    "provenance_manifest_sha256": (record.provenance.manifest_sha256),
                    "provenance_generator": record.provenance.generator,
                    "generator_source_commit": (
                        record.provenance.generator_source_commit
                    ),
                    "review_page_start": record.review_page_start,
                    "review_page_end": record.review_page_end,
                }
                for record in numbered_records
            ],
        }
        staged_source_map_json.write_text(
            json.dumps(source_payload, indent=2) + "\n", encoding="utf-8"
        )
        _write_csv(
            staged_source_map_csv,
            numbered_records,
            source_commit,
            snapshot_root,
        )

        expected_files = {
            REVIEW_PDF,
            SOURCE_MAP_JSON,
            SOURCE_MAP_CSV,
            VECTOR_QA_JSON,
            *{
                f"previews/{record.spec.candidate_id}.png"
                for record in numbered_records
            },
        }
        actual_files = {
            path.relative_to(staging_dir).as_posix()
            for path in staging_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise FigureAcceptanceError(
                "Staged acceptance artifact inventory is incomplete or contaminated"
            )
        os.replace(staging_dir, output_dir)
    except Exception as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if isinstance(exc, FigureAcceptanceError):
            raise
        if isinstance(exc, PdfQaError):
            raise FigureAcceptanceError(str(exc)) from exc
        raise FigureAcceptanceError(f"Acceptance staging failed: {exc}") from exc

    return AcceptanceOutputs(
        review_pdf=output_dir / REVIEW_PDF,
        source_map_json=output_dir / SOURCE_MAP_JSON,
        source_map_csv=output_dir / SOURCE_MAP_CSV,
        vector_qa_json=output_dir / VECTOR_QA_JSON,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    print(
        "FIGURE ACCEPTANCE ERROR: this module is internal-only; run "
        "submission/final_figure_orchestrator.py",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
