import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from submission.accept_figure_candidates import (
    DEFAULT_ALLOWLIST,
    EXPECTED_COUNTS,
    AcceptedCandidate,
    CandidateSpec,
    FigureAcceptanceError,
    PreviewRecord,
    accept_figure_candidates,
    build_vector_review_pdf,
    load_candidate_allowlist,
)
from submission.pdf_vector_qa import FontRecord, PdfQaReport, inspect_pdf


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "5" * 40


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_report(path, *, require_fonts=True):
    assert require_fonts
    return PdfQaReport(
        path=str(path),
        image_objects=0,
        fonts=(FontRecord("Embedded", True, True, "TrueType"),),
        issues=(),
    )


def _populate_candidate_roots(tmp_path):
    roots = {
        "figure_root": tmp_path / "figures",
        "c0_root": tmp_path / "c0",
    }
    for root in roots.values():
        root.mkdir()
    for spec in load_candidate_allowlist():
        pdf = roots[spec.root] / spec.pdf
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"%PDF-1.4\n{spec.candidate_id}\n%%EOF\n".encode())
        preview = pdf.with_suffix(".png")
        Image.new("RGB", (12, 8), "white").save(preview, dpi=(300, 300))
    return roots


def _fake_review_builder(records, output, *, source_commit):
    assert len(records) == 38
    assert source_commit == SOURCE_COMMIT
    output.write_bytes(b"%PDF-1.4\nsynthetic vector review\n%%EOF\n")


def test_explicit_allowlist_has_exact_38_candidate_contract():
    specs = load_candidate_allowlist()
    assert len(specs) == EXPECTED_COUNTS["candidate_pdfs"] == 38
    assert sum(spec.variant == "unpaired" for spec in specs) == 14
    assert len({spec.slot_id for spec in specs if spec.variant == "unpaired"}) == 14
    paired = [spec for spec in specs if spec.variant != "unpaired"]
    assert len(paired) == 24
    slots = {}
    for spec in paired:
        slots.setdefault(spec.slot_id, set()).add(spec.variant)
    assert len(slots) == 12
    assert all(variants == {"with_endpoints", "clean"} for variants in slots.values())
    assert len({(spec.root, spec.pdf) for spec in specs}) == 38


def test_acceptance_fails_on_missing_and_unexpected_candidate_pdfs(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    missing = roots["figure_root"] / load_candidate_allowlist()[0].pdf
    missing.unlink()
    unexpected = roots["c0_root"] / "summary_plots/unexpected.pdf"
    unexpected.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(FigureAcceptanceError) as caught:
        accept_figure_candidates(
            figure_root=roots["figure_root"],
            c0_root=roots["c0_root"],
            output_dir=tmp_path / "review",
            source_commit=SOURCE_COMMIT,
            pdf_inspector=_passing_report,
            page_counter=lambda path: 1,
            review_builder=_fake_review_builder,
        )
    message = str(caught.value)
    assert "Missing candidate PDFs" in message
    assert "Unexpected candidate PDFs" in message
    assert "perfect_reconstruction_plic_stencil.pdf" in message
    assert "unexpected.pdf" in message


def test_acceptance_requires_matching_300_dpi_png_preview(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    first = load_candidate_allowlist()[0]
    preview = (roots[first.root] / first.pdf).with_suffix(".png")
    Image.new("RGB", (12, 8), "white").save(preview, dpi=(72, 72))

    with pytest.raises(FigureAcceptanceError, match="not 300 DPI"):
        accept_figure_candidates(
            figure_root=roots["figure_root"],
            c0_root=roots["c0_root"],
            output_dir=tmp_path / "review",
            source_commit=SOURCE_COMMIT,
            pdf_inspector=_passing_report,
            page_counter=lambda path: 1,
            review_builder=_fake_review_builder,
        )


def test_acceptance_rejects_missing_matching_png_preview(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    first = load_candidate_allowlist()[0]
    (roots[first.root] / first.pdf).with_suffix(".png").unlink()

    with pytest.raises(FigureAcceptanceError, match="Missing matching PNG preview"):
        accept_figure_candidates(
            figure_root=roots["figure_root"],
            c0_root=roots["c0_root"],
            output_dir=tmp_path / "review",
            source_commit=SOURCE_COMMIT,
            pdf_inspector=_passing_report,
            page_counter=lambda path: 1,
            review_builder=_fake_review_builder,
        )


def test_acceptance_reuses_pdf_qa_and_fails_closed(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    failed_name = load_candidate_allowlist()[7].candidate_id

    def inspector(path, *, require_fonts=True):
        if failed_name in path.read_text(encoding="utf-8"):
            return PdfQaReport(str(path), 1, (), ("contains 1 raster image object(s)",))
        return _passing_report(path, require_fonts=require_fonts)

    with pytest.raises(FigureAcceptanceError, match="contains 1 raster image"):
        accept_figure_candidates(
            figure_root=roots["figure_root"],
            c0_root=roots["c0_root"],
            output_dir=tmp_path / "review",
            source_commit=SOURCE_COMMIT,
            pdf_inspector=inspector,
            page_counter=lambda path: 1,
            review_builder=_fake_review_builder,
        )


def test_success_writes_hash_source_maps_and_review_page_index(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    inspected = []

    def inspector(path, *, require_fonts=True):
        inspected.append(Path(path))
        return _passing_report(path, require_fonts=require_fonts)

    outputs = accept_figure_candidates(
        figure_root=roots["figure_root"],
        c0_root=roots["c0_root"],
        output_dir=tmp_path / "review",
        source_commit=SOURCE_COMMIT,
        pdf_inspector=inspector,
        page_counter=lambda path: 1,
        review_builder=_fake_review_builder,
    )

    assert len(inspected) == 39
    payload = json.loads(outputs.source_map_json.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["allowlist"]["sha256"] == _sha256(DEFAULT_ALLOWLIST)
    assert payload["review"]["index_pages"] == 3
    assert payload["review"]["page_count"] == 41
    assert payload["review"]["sha256"] == _sha256(outputs.review_pdf)
    assert len(payload["candidates"]) == 38
    assert payload["candidates"][0]["review_page_start"] == 4
    assert payload["candidates"][-1]["review_page_end"] == 41
    assert all(len(row["pdf_sha256"]) == 64 for row in payload["candidates"])
    assert all(len(row["png_sha256"]) == 64 for row in payload["candidates"])

    with outputs.source_map_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 38
    assert rows[0]["review_page_start"] == "4"
    assert rows[-1]["review_page_end"] == "41"
    qa = json.loads(outputs.vector_qa_json.read_text(encoding="utf-8"))
    assert qa["passed"]
    assert qa["candidate_pdf_count"] == 38
    assert len(qa["candidate_reports"]) == 38


@pytest.mark.skipif(
    shutil.which("pdfunite") is None or shutil.which("pdfinfo") is None,
    reason="Poppler PDF composition tools are unavailable",
)
def test_vector_review_builder_concatenates_pdf_pages_without_rasterizing(tmp_path):
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
    font_name = "AcceptanceTestVera"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))

    records = []
    for index in range(2):
        pdf = tmp_path / f"candidate_{index}.pdf"
        drawing = canvas.Canvas(
            str(pdf),
            pagesize=(180, 120),
            initialFontName=font_name,
            initialFontSize=10,
        )
        drawing.setFont(font_name, 10)
        drawing.drawString(20, 60, f"Vector candidate {index}")
        drawing.line(20, 50, 160, 50)
        drawing.save()
        preview = tmp_path / f"candidate_{index}.png"
        Image.new("RGB", (12, 8), "white").save(preview, dpi=(300, 300))
        qa = inspect_pdf(pdf)
        assert qa.passed
        records.append(
            AcceptedCandidate(
                order=index + 1,
                spec=CandidateSpec(
                    candidate_id=f"candidate_{index}",
                    slot_id=f"slot_{index}",
                    section="Test",
                    title=f"Candidate {index}",
                    variant="unpaired",
                    root="figure_root",
                    pdf=pdf.name,
                ),
                pdf_path=pdf,
                pdf_sha256=_sha256(pdf),
                pdf_page_count=1,
                preview=PreviewRecord(
                    path=preview,
                    sha256=_sha256(preview),
                    width_px=12,
                    height_px=8,
                    dpi_x=300.0,
                    dpi_y=300.0,
                ),
                pdf_qa=qa,
                review_page_start=index + 2,
                review_page_end=index + 2,
            )
        )

    output = tmp_path / "review.pdf"
    build_vector_review_pdf(records, output, source_commit=SOURCE_COMMIT)
    info = subprocess.run(
        ["pdfinfo", str(output)], check=True, capture_output=True, text=True
    ).stdout
    assert "Pages:           3" in info
    assert inspect_pdf(output).passed


def test_documented_direct_cli_entry_point_runs():
    completed = subprocess.run(
        [sys.executable, "submission/accept_figure_candidates.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Fail-closed acceptance gate" in completed.stdout
