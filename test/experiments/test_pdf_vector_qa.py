from pathlib import Path

import pytest

from submission.pdf_vector_qa import (
    PdfQaError,
    discover_pdfs,
    inspect_pdf,
    parse_pdffonts,
    parse_pdfimages_list,
)


PDFIMAGES_EMPTY = """\
page   num  type   width height color comp bpc  enc interp  object ID x-ppi y-ppi size ratio
--------------------------------------------------------------------------------------------
"""

PDFIMAGES_ONE = PDFIMAGES_EMPTY + """\
   1     0 image     640   480  rgb     3   8  image  no        17  0   144   144 32.0K 3.6%
"""

PDFFONTS_EMBEDDED = """\
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
BMQQDV+DejaVuSans                    CID TrueType      Identity-H       yes yes yes     15  0
FJAHUB+CMR10                         Type 1            Builtin          yes yes no      21  0
"""


def test_poppler_parsers_detect_images_and_embedded_fonts():
    assert parse_pdfimages_list(PDFIMAGES_EMPTY) == 0
    assert parse_pdfimages_list(PDFIMAGES_ONE) == 1

    fonts = parse_pdffonts(PDFFONTS_EMBEDDED)
    assert [font.name for font in fonts] == ["BMQQDV+DejaVuSans", "FJAHUB+CMR10"]
    assert all(font.embedded for font in fonts)
    assert fonts[0].font_type == "CID TrueType"


def test_inspect_pdf_passes_vector_pdf_with_embedded_fonts(tmp_path):
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    def runner(command):
        return PDFIMAGES_EMPTY if command[0] == "pdfimages" else PDFFONTS_EMBEDDED

    report = inspect_pdf(path, runner=runner)
    assert report.passed
    assert report.image_objects == 0
    assert len(report.fonts) == 2


def test_inspect_pdf_fails_for_raster_or_unembedded_font(tmp_path):
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    fonts = PDFFONTS_EMBEDDED.replace("yes yes yes", "no  no  yes", 1)

    def runner(command):
        return PDFIMAGES_ONE if command[0] == "pdfimages" else fonts

    report = inspect_pdf(path, runner=runner)
    assert not report.passed
    assert report.issues == (
        "contains 1 raster image object(s)",
        "unembedded fonts: BMQQDV+DejaVuSans",
    )


def test_inspect_pdf_requires_a_font_by_default(tmp_path):
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    runner = lambda command: PDFIMAGES_EMPTY

    assert inspect_pdf(path, runner=runner).issues == ("no fonts reported",)
    assert inspect_pdf(path, require_fonts=False, runner=runner).passed


def test_discover_pdfs_recurses_and_rejects_empty_inputs(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    first = tmp_path / "a.pdf"
    second = nested / "b.PDF"
    first.touch()
    second.touch()

    assert discover_pdfs([tmp_path]) == tuple(sorted((first.resolve(), second.resolve())))

    with pytest.raises(PdfQaError, match="No PDF files found"):
        discover_pdfs([empty])
