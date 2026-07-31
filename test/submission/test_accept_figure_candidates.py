import csv
import hashlib
import importlib
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
    PdfPageInfo,
    PreviewRecord,
    ProvenanceEvidence,
    _accept_orchestrated_candidates,
    _create_orchestrated_acceptance_state,
    build_vector_review_pdf,
    load_candidate_allowlist,
    render_pdf_preview,
    verify_review_page_map,
)
from submission.pdf_vector_qa import FontRecord, PdfQaReport, inspect_pdf
from submission.final_figure_orchestrator import (
    _reserve_publication,
    finalize_publication,
    validate_published_logical_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "5" * 40
APPROVED_COMMIT = "6" * 40
PROFILE = {
    "plic_fallback": "LVIRA",
    "corner_behavior_profile": "pre_f8_corner",
    "rescue_profile": "exact_linear_support_only",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _orchestrated_state(roots):
    snapshot = roots["figure_root"].parents[1]
    private_allowlist = snapshot / "provenance" / "approved_candidate_allowlist.json"
    private_allowlist.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_ALLOWLIST, private_allowlist)
    candidates = []
    for spec in load_candidate_allowlist():
        output = roots[spec.root] / spec.pdf
        candidates.append(
            {
                "candidate_id": spec.candidate_id,
                "path": (Path("candidates") / spec.root / spec.pdf).as_posix(),
                "sha256": _sha256(output),
                "generator": "test_generator",
            }
        )
    manifest_path = snapshot / "provenance" / "final_figure_orchestration.json"
    _write_json(manifest_path, {"record": "created by orchestrator test fixture"})
    return _create_orchestrated_acceptance_state(
        figure_root=roots["figure_root"],
        c0_root=roots["c0_root"],
        snapshot_root=snapshot,
        release_anchor={
            "name": "release",
            "source_commit": SOURCE_COMMIT,
            "reconstruction_profile": PROFILE,
            "artifacts": {},
        },
        generator_source_commit=APPROVED_COMMIT,
        orchestration_record=manifest_path,
        allowlist_path=private_allowlist,
        candidate_records=candidates,
    )


def _acceptance_fixture(tmp_path):
    roots = _populate_candidate_roots(tmp_path)
    state = _orchestrated_state(roots)
    return roots, state


def _fake_page_inspector(path):
    path = Path(path)
    if path.name == "figure_candidate_review.pdf":
        return PdfPageInfo(41, 792.0, 612.0)
    return PdfPageInfo(1, 72.0, 72.0)


def _fake_preview_renderer(pdf_path, output_path, *, dpi, page):
    assert dpi == 300
    assert page >= 1
    Image.new("RGB", (300, 300), "white").save(output_path)


def _fake_review_builder(records, output, *, source_commit):
    assert len(records) == 38
    assert source_commit == SOURCE_COMMIT
    output.write_bytes(b"%PDF-1.4\nsynthetic vector review\n%%EOF\n")


def _accept(tmp_path, state, **overrides):
    options = {
        "orchestration_state": state,
        "output_dir": tmp_path / "review",
        "pdf_inspector": _passing_report,
        "page_inspector": _fake_page_inspector,
        "preview_renderer": _fake_preview_renderer,
        "review_builder": _fake_review_builder,
        "review_map_verifier": lambda *args, **kwargs: None,
    }
    options.update(overrides)
    return _accept_orchestrated_candidates(**options)


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
    roots, state = _acceptance_fixture(tmp_path)
    missing = roots["figure_root"] / load_candidate_allowlist()[0].pdf
    missing.unlink()
    unexpected = roots["c0_root"] / "summary_plots/unexpected.pdf"
    unexpected.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(FigureAcceptanceError) as caught:
        _accept(tmp_path, state)
    message = str(caught.value)
    assert "Missing candidate PDFs" in message
    assert "Unexpected candidate PDFs" in message
    assert "perfect_reconstruction_plic_stencil.pdf" in message
    assert "unexpected.pdf" in message


def test_historical_file_masquerade_fails_provenance_checksum(tmp_path):
    roots, state = _acceptance_fixture(tmp_path)
    spec = load_candidate_allowlist()[0]
    candidate = roots[spec.root] / spec.pdf
    candidate.write_bytes(b"%PDF-1.4\nhistorical candidate with same name\n%%EOF\n")

    with pytest.raises(FigureAcceptanceError, match="candidate checksum mismatch"):
        _accept(tmp_path, state)


def test_private_allowlist_mutation_fails_before_acceptance(tmp_path):
    _roots, state = _acceptance_fixture(tmp_path)
    state.allowlist_path.write_bytes(state.allowlist_path.read_bytes() + b"\n")

    with pytest.raises(FigureAcceptanceError, match="allowlist mutated"):
        _accept(tmp_path, state)


def test_forged_standalone_manifest_cannot_invoke_acceptance(tmp_path):
    forged = tmp_path / "forged.json"
    _write_json(forged, {"status": "completed", "scientific_contracts": {}})
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
    assert "internal-only" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("size", [(1, 1), (300, 150)])
def test_tiny_or_stale_rendered_preview_fails(tmp_path, size):
    _roots, state = _acceptance_fixture(tmp_path)

    def bad_renderer(pdf_path, output_path, *, dpi, page):
        Image.new("RGB", size, "white").save(output_path)

    with pytest.raises(FigureAcceptanceError, match="tiny|dimensions do not match"):
        _accept(
            tmp_path,
            state,
            preview_renderer=bad_renderer,
        )
    assert not (tmp_path / "review").exists()


def test_multi_page_candidate_fails(tmp_path):
    _roots, state = _acceptance_fixture(tmp_path)
    failed_name = load_candidate_allowlist()[3].candidate_id

    def page_inspector(path):
        if failed_name in Path(path).read_text(encoding="utf-8"):
            return PdfPageInfo(2, 72.0, 72.0)
        return _fake_page_inspector(path)

    with pytest.raises(FigureAcceptanceError, match="exactly one page"):
        _accept(tmp_path, state, page_inspector=page_inspector)


def test_wrong_merged_review_page_count_fails(tmp_path):
    _roots, state = _acceptance_fixture(tmp_path)

    def page_inspector(path):
        if Path(path).name == "figure_candidate_review.pdf":
            return PdfPageInfo(40, 792.0, 612.0)
        return PdfPageInfo(1, 72.0, 72.0)

    with pytest.raises(FigureAcceptanceError, match="Merged review page count"):
        _accept(tmp_path, state, page_inspector=page_inspector)
    assert not (tmp_path / "review").exists()


def test_acceptance_reuses_pdf_qa_and_fails_closed(tmp_path):
    _roots, state = _acceptance_fixture(tmp_path)
    failed_name = load_candidate_allowlist()[7].candidate_id

    def inspector(path, *, require_fonts=True):
        if failed_name in Path(path).read_text(encoding="utf-8"):
            return PdfQaReport(str(path), 1, (), ("contains 1 raster image object(s)",))
        return _passing_report(path, require_fonts=require_fonts)

    with pytest.raises(FigureAcceptanceError, match="contains 1 raster image"):
        _accept(tmp_path, state, pdf_inspector=inspector)


def test_success_writes_hash_source_maps_and_generated_previews(tmp_path):
    _roots, state = _acceptance_fixture(tmp_path)
    inspected = []

    def inspector(path, *, require_fonts=True):
        inspected.append(Path(path))
        return _passing_report(path, require_fonts=require_fonts)

    outputs = _accept(tmp_path, state, pdf_inspector=inspector)

    assert len(inspected) == 39
    assert len(list((tmp_path / "review" / "previews").glob("*.png"))) == 38
    payload = json.loads(outputs.source_map_json.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["release"]["source_commit"] == SOURCE_COMMIT
    assert payload["allowlist"]["sha256"] == _sha256(DEFAULT_ALLOWLIST)
    assert (
        payload["allowlist"]["path"] == "provenance/approved_candidate_allowlist.json"
    )
    assert payload["review"]["index_pages"] == 3
    assert payload["review"]["page_count"] == 41
    assert payload["review"]["sha256"] == _sha256(outputs.review_pdf)
    assert len(payload["candidates"]) == 38
    assert payload["candidates"][0]["review_page_start"] == 4
    assert payload["candidates"][-1]["review_page_end"] == 41
    assert all(len(row["pdf_sha256"]) == 64 for row in payload["candidates"])
    assert all(len(row["png_sha256"]) == 64 for row in payload["candidates"])
    assert all(row["png_width_px"] == 300 for row in payload["candidates"])
    assert all(
        len(row["provenance_manifest_sha256"]) == 64 for row in payload["candidates"]
    )

    with outputs.source_map_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 38
    assert rows[0]["review_page_start"] == "4"
    assert rows[-1]["review_page_end"] == "41"
    qa = json.loads(outputs.vector_qa_json.read_text(encoding="utf-8"))
    assert qa["passed"]
    assert qa["candidate_pdf_count"] == 38
    assert qa["measured_review_page_count"] == 41
    assert qa["review_page_map_verified"]


def test_published_artifact_paths_are_final_root_relative_and_exist(tmp_path):
    _roots, state = _acceptance_fixture(tmp_path)
    staging = state.snapshot_root
    output = tmp_path / "published-review"
    reservation = _reserve_publication(output)
    finalize_publication(
        staging=staging,
        output_root=output,
        reservation=reservation,
        manifest_path=state.orchestration_record,
        acceptance_runner=_accept_orchestrated_candidates,
        acceptance_kwargs={
            "orchestration_state": state,
            "output_dir": staging / "review",
            "pdf_inspector": _passing_report,
            "page_inspector": _fake_page_inspector,
            "preview_renderer": _fake_preview_renderer,
            "review_builder": _fake_review_builder,
            "review_map_verifier": lambda *args, **kwargs: None,
        },
        candidate_specs=load_candidate_allowlist(),
    )

    checked = validate_published_logical_paths(output)
    assert len(checked) >= 38 * 7
    for relative in checked:
        assert not Path(relative).is_absolute()
        target = output / relative
        target.resolve().relative_to(output.resolve())
        assert target.exists()
    for artifact in output.rglob("*"):
        if artifact.suffix in {".json", ".csv"}:
            assert str(staging) not in artifact.read_text(encoding="utf-8")


def test_late_write_failure_removes_staging_and_publishes_nothing(
    tmp_path, monkeypatch
):
    _roots, state = _acceptance_fixture(tmp_path)
    module = importlib.import_module("submission.accept_figure_candidates")

    def fail_late(*args, **kwargs):
        raise OSError("simulated late write failure")

    monkeypatch.setattr(module, "_write_csv", fail_late)
    with pytest.raises(FigureAcceptanceError, match="simulated late write failure"):
        _accept(tmp_path, state)
    assert not (tmp_path / "review").exists()
    assert not list(tmp_path.glob(".review.staging-*"))


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
    assert "internal-only" in completed.stderr
    assert "final_figure_orchestrator.py" in completed.stderr
