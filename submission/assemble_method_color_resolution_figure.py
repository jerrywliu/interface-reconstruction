#!/usr/bin/env python3
"""Assemble the approved 5x2 resolution montage from recolored vector panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.submission.generate_revision_layout_prototypes import (
    _compile_standalone,
    _trim_for_box,
)
from submission.pdf_vector_qa import inspect_pdf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT / "results/submission/paper_method_style_20260903/resolution"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/submission/paper_method_style_20260903/resolution_montage"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    names = ("lines", "squares", "circles", "ellipses", "zalesak")
    display = {
        "lines": "Lines",
        "squares": "Squares",
        "circles": "Circles",
        "ellipses": "Ellipses",
        "zalesak": "Zalesak",
    }
    paths = {
        name: source
        / ("zalesak_case6" if name == "zalesak" else name)
        / "summary_plots"
        / f"{name}_resolution_cartesian_vs_perturbed.pdf"
        for name in names
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source panels:\n" + "\n".join(missing))

    row_boxes = [
        (0.51, 0.10, 0.995, 0.345),
        (0.51, 0.41, 0.995, 0.665),
        (0.51, 0.72, 0.995, 0.995),
    ]
    spyglass_boxes = [
        (0.53, 0.08, 0.88, 0.32),
        (0.53, 0.41, 0.88, 0.65),
        (0.53, 0.72, 0.88, 0.98),
    ]
    tex_path = output / "benchmark_resolution_5x2.tex"
    tex_lines = [
        r"\documentclass[border=2pt]{standalone}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{>{\raggedleft\arraybackslash}p{0.70in}cc}",
        r" & \textbf{$N=32$} & \textbf{$N=64$}\\",
    ]
    for name in names:
        boxes = spyglass_boxes if name in {"squares", "zalesak"} else row_boxes
        cells = []
        for row in (1, 2):
            trim = _trim_for_box(paths[name], boxes[row])
            cells.append(
                rf"\includegraphics[width=2.52in,trim={{{trim}}},clip]"
                rf"{{{paths[name]}}}"
            )
        tex_lines.append(
            rf"\textbf{{{display[name]}}} & "
            + " & ".join(cells)
            + r"\\[-1pt]"
        )
    tex_lines.extend((r"\end{tabular}", r"\end{document}"))
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    pdf_path = _compile_standalone(tex_path)
    report = inspect_pdf(pdf_path, require_fonts=True)
    if not report.passed:
        raise RuntimeError(f"vector QA failed: {report.issues}")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "paper Figure B.24 method-color refresh",
                "scientific_data_changed": False,
                "sources": {name: str(path) for name, path in paths.items()},
                "output": str(pdf_path),
                "vector_qa": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pdf_path)


if __name__ == "__main__":
    main()
