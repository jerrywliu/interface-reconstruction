#!/usr/bin/env python3
"""Assemble regenerated Section 6 figures into a page-addressable review PDF."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


EXPERIMENTS = ("lines", "squares", "circles", "ellipses", "zalesak")


@dataclass(frozen=True)
class FigureItem:
    title: str
    path: Path
    subtitle: str = ""


def _paired_items(
    *,
    title: str,
    directory: Path,
    stem: str,
    clean_subtitle: str = (
        "Clean main panels; endpoint labels remain in spyglasses and corner diamonds "
        "remain everywhere."
    ),
) -> list[FigureItem]:
    return [
        FigureItem(
            f"{title} - facet endpoints",
            directory / f"{stem}_with_endpoints.png",
            "Open circles label every reconstructed facet endpoint.",
        ),
        FigureItem(
            f"{title} - clean main panels",
            directory / f"{stem}_clean.png",
            clean_subtitle,
        ),
    ]


def _paper_items(
    run_root: Path,
    *,
    endpoint_pairs: bool,
) -> list[tuple[str, list[FigureItem]]]:
    review = run_root / "figure_review"
    paper = review / (
        "paired_paper_figures" if endpoint_pairs else "current_run_paper_figures"
    )
    summaries = review / "all_method_summary_plots"

    if endpoint_pairs:
        representative_items = [
            item
            for exp in EXPERIMENTS
            for item in _paired_items(
                title=f"{exp.title()}: representative reconstruction",
                directory=paper / "representative_cases",
                stem=f"{exp}_maintext_representative",
            )
        ]
        resolution_items = [
            item
            for exp in EXPERIMENTS
            for item in _paired_items(
                title=f"{exp.title()}: best method by resolution",
                directory=paper / "appendix_cases",
                stem=f"{exp}_best_by_resolution",
                clean_subtitle=(
                    "Clean panels hide circular endpoint markers; semantic corner "
                    "diamonds remain."
                ),
            )
        ]
        cartesian_items = [
            item
            for exp in EXPERIMENTS
            for item in _paired_items(
                title=f"{exp.title()}: Cartesian representative",
                directory=paper / "appendix_cases",
                stem=f"{exp}_cartesian_representative",
            )
        ]
    else:
        representative_items = [
            FigureItem(
                f"{exp.title()}: representative reconstruction",
                paper / "representative_cases" / f"{exp}_maintext_representative.png",
            )
            for exp in EXPERIMENTS
        ]
        resolution_items = [
            FigureItem(
                f"{exp.title()}: best method by resolution",
                paper / "appendix_cases" / f"{exp}_best_by_resolution.png",
            )
            for exp in EXPERIMENTS
        ]
        cartesian_items = [
            FigureItem(
                f"{exp.title()}: Cartesian representative",
                paper / "appendix_cases" / f"{exp}_cartesian_representative.png",
            )
            for exp in EXPERIMENTS
        ]

    sections = [
        (
            "Main-text quantitative panels",
            [
                FigureItem(
                    f"{exp.title()}: quantitative comparison",
                    paper / "summary_plots" / f"{exp}_maintext_metrics.png",
                    "Merged baseline comparison with newly rerun affected methods.",
                )
                for exp in EXPERIMENTS
            ],
        ),
        (
            "Main-text representative reconstructions",
            representative_items,
        ),
        (
            "Appendix all-method summaries",
            [
                FigureItem(
                    f"{exp.title()}: all-method summary",
                    summaries
                    / (
                        f"{exp}_all_methods_5x2_axes.png"
                        if exp in {"circles", "ellipses"}
                        else f"{exp}_all_methods_2x2.png"
                    ),
                    "Generated from the review merged CSV, including frozen baselines.",
                )
                for exp in EXPERIMENTS
            ],
        ),
        (
            "Appendix resolution strips",
            resolution_items,
        ),
        (
            "Appendix Cartesian representatives",
            cartesian_items,
        ),
    ]
    return sections


def _diagnostic_items(
    run_root: Path,
    *,
    endpoint_pairs: bool,
) -> tuple[str, list[FigureItem]]:
    manifest_path = run_root / "figure_review" / (
        "diagnostic_pair_manifest.json"
        if endpoint_pairs
        else "diagnostic_manifest.json"
    )
    specs = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = []
    for spec in specs:
        metric_subtitle = (
            f"{spec['purpose']} Ours ({spec['algo']}): "
            f"Hausdorff={spec['hausdorff']}, facet gap={spec['facet_gap']}"
        )
        if endpoint_pairs:
            items.extend(
                [
                    FigureItem(
                        f"{spec['title']} - facet endpoints",
                        Path(spec["sources"]["with_endpoints"]),
                        metric_subtitle,
                    ),
                    FigureItem(
                        f"{spec['title']} - clean main panels",
                        Path(spec["sources"]["clean"]),
                        (
                            f"{metric_subtitle} Clean main panels retain spyglass "
                            "endpoint labels and all semantic corner diamonds."
                        ),
                    ),
                ]
            )
        else:
            source = spec.get("source") or next(iter(spec["sources"].values()))
            items.append(FigureItem(spec["title"], Path(source), metric_subtitle))
    return "Focused tail diagnostics", items


def _draw_wrapped_text(pdf, text: str, x: float, y: float, max_width: float, size: float):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and stringWidth(candidate, "Helvetica", size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= size + 2
    return y


def _draw_cover(pdf, run_root: Path, figure_count: int, *, endpoint_pairs: bool):
    width, height = landscape(letter)
    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawString(54, height - 110, "Section 6 Figure Review")
    pdf.setFillColor(HexColor("#2563eb"))
    pdf.rect(54, height - 132, 140, 4, stroke=0, fill=1)
    pdf.setFillColor(HexColor("#374151"))
    pdf.setFont("Helvetica", 13)
    pdf.drawString(54, height - 170, f"{figure_count} figures, one per page")
    pdf.drawString(54, height - 194, "July 17 affected-method rerun + frozen comparison baselines")
    if endpoint_pairs:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(
            54,
            height - 218,
            "Qualitative figures are paired: facet endpoints, then clean main panels.",
        )
    pdf.setFont("Helvetica", 9)
    _draw_wrapped_text(pdf, str(run_root), 54, 92, width - 108, 9)
    pdf.setFillColor(HexColor("#6b7280"))
    pdf.drawRightString(width - 36, 24, "Review artifact; manuscript files unchanged")


def _draw_index_page(pdf, section_rows, page_numbers, title: str):
    width, height = landscape(letter)
    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(42, height - 48, title)
    y = height - 82
    for section, items in section_rows:
        pdf.setFillColor(HexColor("#1d4ed8"))
        pdf.setFont("Helvetica-Bold", 10.5)
        pdf.drawString(42, y, section)
        y -= 17
        pdf.setFillColor(HexColor("#374151"))
        pdf.setFont("Helvetica", 9.2)
        for item in items:
            page = page_numbers[str(item.path)]
            pdf.drawString(54, y, item.title)
            pdf.drawRightString(width - 48, y, str(page))
            y -= 15
        y -= 8


def _draw_figure_page(pdf, item: FigureItem, section: str, page_number: int, run_root: Path):
    width, height = landscape(letter)
    pdf.setFillColor(HexColor("#2563eb"))
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(38, height - 26, section.upper())
    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(38, height - 48, item.title)
    image_top = height - 72
    if item.subtitle:
        pdf.setFillColor(HexColor("#4b5563"))
        pdf.setFont("Helvetica", 9.5)
        subtitle_bottom = _draw_wrapped_text(
            pdf,
            item.subtitle,
            38,
            height - 64,
            width - 76,
            9.5,
        )
        image_top = min(image_top, subtitle_bottom - 2)

    with Image.open(item.path) as image:
        image_width, image_height = image.size
    max_width = width - 64
    max_height = image_top - 38
    scale = min(max_width / image_width, max_height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    x = (width - draw_width) / 2
    y = 38 + (max_height - draw_height) / 2
    pdf.drawImage(
        str(item.path),
        x,
        y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )

    relative_path = item.path.relative_to(run_root)
    pdf.setFillColor(HexColor("#6b7280"))
    pdf.setFont("Helvetica", 7.2)
    pdf.drawString(38, 20, str(relative_path))
    pdf.drawRightString(width - 38, 20, str(page_number))


def build_pdf(run_root: Path, output: Path, *, endpoint_pairs: bool = False):
    sections = _paper_items(run_root, endpoint_pairs=endpoint_pairs)
    sections.append(_diagnostic_items(run_root, endpoint_pairs=endpoint_pairs))
    missing = [item.path for _, items in sections for item in items if not item.path.exists()]
    if missing:
        raise FileNotFoundError("Missing review figures:\n" + "\n".join(map(str, missing)))

    figure_count = sum(len(items) for _, items in sections)
    if endpoint_pairs:
        index_sections = [sections[:3], sections[3:5], sections[5:]]
    else:
        index_sections = [sections[:3], sections[3:]]
    index_page_count = len(index_sections)
    first_figure_page = 2 + index_page_count
    page_numbers = {}
    page = first_figure_page
    for _, items in sections:
        for item in items:
            page_numbers[str(item.path)] = page
            page += 1

    approval_path = run_root / "figure_review" / (
        "FIGURE_ENDPOINT_PAIR_APPROVAL.md"
        if endpoint_pairs
        else "FIGURE_APPROVAL.md"
    )
    if approval_path.exists():
        approval_text = approval_path.read_text(encoding="utf-8")
        diagnostic_specs = json.loads(
            (
                run_root
                / "figure_review"
                / (
                    "diagnostic_pair_manifest.json"
                    if endpoint_pairs
                    else "diagnostic_manifest.json"
                )
            ).read_text(
                encoding="utf-8"
            )
        )
        for spec in diagnostic_specs:
            if endpoint_pairs:
                pending = f"- PDF pages pending: {spec['title']}."
                annotated_page = page_numbers[
                    str(Path(spec["sources"]["with_endpoints"]))
                ]
                clean_page = page_numbers[str(Path(spec["sources"]["clean"]))]
                numbered = (
                    f"- PDF pages {annotated_page}-{clean_page}: {spec['title']}."
                )
            else:
                source = spec.get("source") or next(iter(spec["sources"].values()))
                pending = f"- PDF page pending: {spec['title']}."
                numbered = (
                    f"- PDF page {page_numbers[str(Path(source))]}: {spec['title']}."
                )
            approval_text = approval_text.replace(pending, numbered)
        approval_path.write_text(approval_text, encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=landscape(letter), pageCompression=1)
    pdf.setTitle("Section 6 Figure Review")
    pdf.setAuthor("Interface Reconstruction collaborators")

    _draw_cover(pdf, run_root, figure_count, endpoint_pairs=endpoint_pairs)
    pdf.showPage()
    for index, index_rows in enumerate(index_sections, start=1):
        _draw_index_page(
            pdf,
            index_rows,
            page_numbers,
            f"Figure Index ({index} of {index_page_count})",
        )
        pdf.showPage()

    manifest = []
    page = first_figure_page
    for section, items in sections:
        for item in items:
            _draw_figure_page(pdf, item, section, page, run_root)
            manifest.append(
                {
                    "page": page,
                    "section": section,
                    "title": item.title,
                    "subtitle": item.subtitle,
                    "source": str(item.path),
                }
            )
            pdf.showPage()
            page += 1
    pdf.save()

    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "pdf": str(output),
                "figure_count": figure_count,
                "page_count": page - 1,
                "endpoint_pairs": endpoint_pairs,
                "figures": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint_pairs", action="store_true")
    args = parser.parse_args()
    manifest = build_pdf(
        args.run_root.resolve(),
        args.output.resolve(),
        endpoint_pairs=args.endpoint_pairs,
    )
    print(args.output.resolve())
    print(manifest)


if __name__ == "__main__":
    main()
