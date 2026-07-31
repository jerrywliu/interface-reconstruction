import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from submission.accept_figure_candidates import (
    EXPECTED_COUNTS,
    AcceptedCandidate,
    CandidateSpec,
    FigureAcceptanceError,
    PdfPageInfo,
    PreviewRecord,
    ProvenanceEvidence,
    build_vector_review_pdf,
    inspect_generated_preview,
    load_candidate_allowlist,
    pdf_page_info,
    render_pdf_preview,
    verify_candidate_inventory,
    verify_review_page_map,
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
    snapshot = tmp_path / "snapshot"
    roots = {
        "figure_root": snapshot / "candidates" / "figure_root",
        "c0_root": snapshot / "candidates" / "c0_root",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    for spec in load_candidate_allowlist():
        pdf = roots[spec.root] / spec.pdf
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"%PDF-1.4\n{spec.candidate_id}\n%%EOF\n".encode())
    return roots


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


def test_candidate_inventory_fails_on_missing_and_unexpected_pdfs(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    missing = roots["figure_root"] / load_candidate_allowlist()[0].pdf
    missing.unlink()
    unexpected = roots["c0_root"] / "summary_plots/unexpected.pdf"
    unexpected.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(FigureAcceptanceError) as caught:
        verify_candidate_inventory(load_candidate_allowlist(), roots)
    message = str(caught.value)
    assert "Missing candidate PDFs" in message
    assert "Unexpected candidate PDFs" in message
    assert "perfect_reconstruction_plic_stencil.pdf" in message
    assert "unexpected.pdf" in message


def test_forged_standalone_manifest_cannot_invoke_acceptance(tmp_path):
    forged = tmp_path / "forged.json"
    forged.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "review"
    completed = subprocess.run(
        [
            sys.executable,
            "submission/accept_figure_candidates.py",
            "--orchestration-manifest",
            str(forged),
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "no acceptance command" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("size", [(1, 1), (300, 150)])
def test_tiny_or_stale_rendered_preview_fails(tmp_path, size):
    preview = tmp_path / "preview.png"
    Image.new("RGB", size, "white").save(preview, dpi=(300, 300))

    with pytest.raises(FigureAcceptanceError, match="tiny|dimensions do not match"):
        inspect_generated_preview(
            preview,
            PdfPageInfo(1, 72.0, 72.0),
            required_dpi=300.0,
            logical_path=Path("review/previews/test.png"),
        )


@pytest.mark.skipif(
    shutil.which("pdfinfo") is None,
    reason="Poppler PDF tools are unavailable",
)
def test_multi_page_candidate_fails(tmp_path):
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "two-pages.pdf"
    drawing = canvas.Canvas(str(pdf), pagesize=(72, 72))
    drawing.drawString(10, 36, "one")
    drawing.showPage()
    drawing.drawString(10, 36, "two")
    drawing.save()

    assert pdf_page_info(pdf).page_count == 2


@pytest.mark.skipif(
    shutil.which("pdfunite") is None
    or shutil.which("pdfinfo") is None
    or shutil.which("pdftocairo") is None,
    reason="Poppler PDF tools are unavailable",
)
def test_wrong_review_page_map_fails(tmp_path):
    from reportlab.pdfgen import canvas

    records = []
    for index, color in enumerate(((1, 0, 0), (0, 0, 1))):
        pdf = tmp_path / f"candidate_{index}.pdf"
        drawing = canvas.Canvas(str(pdf), pagesize=(72, 72))
        drawing.setFillColorRGB(*color)
        drawing.rect(0, 0, 72, 72, stroke=0, fill=1)
        drawing.save()
        preview = tmp_path / f"candidate_{index}.png"
        render_pdf_preview(pdf, preview, dpi=300, page=1)
        records.append(_review_record(index, pdf, preview))

    review = tmp_path / "wrong_order.pdf"
    build_vector_review_pdf(
        list(reversed(records)), review, source_commit=SOURCE_COMMIT
    )
    with pytest.raises(FigureAcceptanceError, match="Review page map mismatch"):
        verify_review_page_map(
            records,
            review,
            {record.spec.candidate_id: record.preview.path for record in records},
        )


def _review_record(index, pdf, preview):
    return AcceptedCandidate(
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
        pdf_width_points=72.0,
        pdf_height_points=72.0,
        preview=PreviewRecord(
            path=preview,
            sha256=_sha256(preview),
            width_px=300,
            height_px=300,
            dpi_x=300.0,
            dpi_y=300.0,
        ),
        pdf_qa=_passing_report(pdf),
        provenance=ProvenanceEvidence(
            manifest_path=pdf.with_suffix(".json"),
            manifest_sha256="a" * 64,
            generator="test",
            generator_source_commit=SOURCE_COMMIT,
        ),
        review_page_start=index + 2,
        review_page_end=index + 2,
    )


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
        Image.new("RGB", (750, 500), "white").save(preview)
        qa = inspect_pdf(pdf)
        assert qa.passed
        record = _review_record(index, pdf, preview)
        records.append(
            AcceptedCandidate(
                **{
                    **record.__dict__,
                    "pdf_qa": qa,
                    "pdf_width_points": 180.0,
                    "pdf_height_points": 120.0,
                }
            )
        )

    output = tmp_path / "review.pdf"
    build_vector_review_pdf(records, output, source_commit=SOURCE_COMMIT)
    info = subprocess.run(
        ["pdfinfo", str(output)], check=True, capture_output=True, text=True
    ).stdout
    assert "Pages:           3" in info
    assert inspect_pdf(output).passed


def test_standalone_acceptance_cli_is_deliberately_disabled():
    completed = subprocess.run(
        [sys.executable, "submission/accept_figure_candidates.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "no acceptance command" in completed.stderr
    assert "run_final_figure_orchestrator" in completed.stderr
