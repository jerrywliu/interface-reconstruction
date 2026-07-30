#!/usr/bin/env python3
"""Generate candid case diagnostics and an approval sheet for Section 6 figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.static import generate_section6_maintext_figures as figures


DIAGNOSTIC_SPECS = (
    {
        "experiment": "squares",
        "resolution": 0.50,
        "wiggle": 0.10,
        "case_index": 24,
        "purpose": "Main-text square choice; perfect-reconstruction validation.",
    },
    {
        "experiment": "squares",
        "resolution": 0.64,
        "wiggle": 0.10,
        "case_index": 22,
        "purpose": "Proposed square resolution-strip choice.",
    },
    {
        "experiment": "zalesak",
        "resolution": 1.00,
        "wiggle": 0.10,
        "case_index": 12,
        "purpose": "Main-text Zalesak choice; perfect-reconstruction validation.",
    },
    {
        "experiment": "zalesak",
        "resolution": 0.64,
        "wiggle": 0.10,
        "case_index": 20,
        "purpose": "Proposed Zalesak resolution-strip choice.",
    },
    {
        "experiment": "zalesak",
        "resolution": 0.50,
        "wiggle": 0.10,
        "case_index": 12,
        "purpose": "Displaced low-resolution choice; genuine missed-corner tail.",
    },
    {
        "experiment": "zalesak",
        "resolution": 1.50,
        "wiggle": 0.20,
        "case_index": 23,
        "purpose": "Largest current Zalesak Hausdorff tail.",
    },
    {
        "experiment": "ellipses",
        "resolution": 0.50,
        "wiggle": 0.30,
        "case_index": 20,
        "purpose": (
            "Largest current smooth-interface Hausdorff tail; the global geometry "
            "still looks normal, so the metric discrepancy remains diagnostic."
        ),
    },
    {
        "experiment": "circles",
        "resolution": 1.00,
        "wiggle": 0.05,
        "case_index": 0,
        "purpose": (
            "Largest current circle Hausdorff tail with a fallback event; the global "
            "geometry still looks normal, so the metric discrepancy remains diagnostic."
        ),
    },
)

OURS_ALGO = {
    "lines": "linear",
    "squares": "linear+corner",
    "circles": "circular",
    "ellipses": "circular",
    "zalesak": "circular+corner",
}


def _load_case_metrics(path: Path) -> dict[tuple, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (
                row["experiment"],
                row["algo"],
                float(row["resolution"]),
                float(row["wiggle"]),
                int(row["case_index"]),
            ): row
            for row in csv.DictReader(handle)
        }


def _number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _case_filename(spec: dict, suffix: str = "") -> str:
    return (
        f"{spec['experiment']}_case{spec['case_index']:02d}_"
        f"N{round(spec['resolution'] * 100):03d}_w{_number_tag(spec['wiggle'])}"
        f"{suffix}.png"
    )


def _metric_value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in {"", "nan", "NaN"} else float("nan")


def _format_metric(value: float) -> str:
    return f"{value:.4g}"


def _window_label(spec: dict) -> str:
    inset = spec.get("inset")
    if not inset:
        return "full geometry"
    if "zoom" in inset:
        return f"{inset['kind']} spyglass, {inset['zoom']:g}x"
    return f"{inset['kind']} spyglass, half-span {inset['half_span']:g}"


def _review_rows(case_metrics: dict[tuple, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    groups = (
        ("Main representative", figures.REPRESENTATIVE_CASES, False),
        ("Appendix Cartesian", figures.APPENDIX_CARTESIAN_CASES, False),
        ("Appendix resolution strip", figures.APPENDIX_BEST_METHODS, True),
    )
    for group, specs, is_strip in groups:
        for experiment, spec in specs.items():
            algo = OURS_ALGO[experiment]
            resolutions = spec["resolutions"] if is_strip else [spec["resolution"]]
            metric_rows = [
                case_metrics[
                    (
                        experiment,
                        algo,
                        float(resolution),
                        float(spec["wiggle"]),
                        int(spec["case_index"]),
                    )
                ]
                for resolution in resolutions
            ]
            hausdorff = max(_metric_value(row, "hausdorff") for row in metric_rows)
            facet_gap = max(_metric_value(row, "facet_gap") for row in metric_rows)
            fallback = max(
                _metric_value(row, "fraction_plic_fallback_cells")
                for row in metric_rows
            )
            rows.append(
                {
                    "group": group,
                    "experiment": experiment,
                    "case_index": str(spec["case_index"]),
                    "resolution": ", ".join(
                        str(round(resolution * 100)) for resolution in resolutions
                    ),
                    "wiggle": f"{spec['wiggle']:g}",
                    "window": "full geometry" if is_strip else _window_label(spec),
                    "max_hausdorff": _format_metric(hausdorff),
                    "max_facet_gap": _format_metric(facet_gap),
                    "max_plic_fallback_fraction": _format_metric(fallback),
                    "status": "review" if hausdorff >= 1.0 else "proposed",
                }
            )
    return rows


def _write_approval_sheet(
    *,
    review_root: Path,
    rows: list[dict[str, str]],
    diagnostic_manifest: list[dict],
    endpoint_variants: str,
) -> tuple[Path, Path]:
    paired = endpoint_variants == "paired"
    csv_path = review_root / (
        "figure_endpoint_pair_approval.csv" if paired else "figure_approval.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = review_root / (
        "FIGURE_ENDPOINT_PAIR_APPROVAL.md" if paired else "FIGURE_APPROVAL.md"
    )
    table_lines = [
        "| Group | Experiment | Case | N | w | Window | Max Hausdorff | Max facet gap | Status |",
        "|---|---|---:|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        table_lines.append(
            "| {group} | {experiment} | {case_index} | {resolution} | {wiggle} | "
            "{window} | {max_hausdorff} | {max_facet_gap} | {status} |".format(**row)
        )

    diagnostic_lines = []
    for item in diagnostic_manifest:
        page_label = "PDF pages pending" if paired else "PDF page pending"
        diagnostic_lines.append(
            f"- {page_label}: {item['title']}. {item['purpose']} "
            f"Hausdorff `{item['hausdorff']}`, facet gap `{item['facet_gap']}`."
        )

    markdown_path.write_text(
        "\n".join(
            [
                "# Section 6 Figure Approval",
                "",
                "Proposed paper-facing cases and plotting windows from the July 17 canonical sweep.",
                "The quantitative panels use all cases; only reconstruction illustrations select individual cases.",
                (
                    "Every reconstruction illustration is exported both with facet endpoints "
                    "and with clean main panels. Clean variants retain endpoint labels in "
                    "spyglasses and retain semantic corner diamonds everywhere."
                    if paired
                    else ""
                ),
                "",
                "## Proposed Figures",
                "",
                *table_lines,
                "",
                "## Candid Tail Checks",
                "",
                *diagnostic_lines,
                "",
                "## Approval Decisions",
                "",
                "- [ ] Approve the five main representative cases and their spyglass windows.",
                "- [ ] Approve the proposed appendix resolution cases: lines 0, squares 22, circles 12, ellipses 12, Zalesak 20.",
                "- [ ] Approve the five Cartesian representative cases.",
                "- [ ] Approve the quantitative and all-method panel formatting.",
                (
                    "- [ ] Choose the annotated or clean-main-panel variant for each "
                    "paper-facing reconstruction figure."
                    if paired
                    else ""
                ),
                "- [ ] Promote approved figures to Overleaf; no manuscript assets have been changed yet.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return markdown_path, csv_path


def generate(
    *,
    run_root: Path,
    plots_root: Path,
    endpoint_variants: str = "annotated",
) -> Path:
    review_root = run_root / "figure_review"
    paired = endpoint_variants == "paired"
    output_dir = review_root / (
        "diagnostic_cases_paired" if paired else "diagnostic_cases"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    case_metrics = _load_case_metrics(run_root / "diagnostics" / "case_metrics.csv")
    figures.PLOTS_ROOT = plots_root

    manifest = []
    for requested in DIAGNOSTIC_SPECS:
        experiment = requested["experiment"]
        spec = dict(figures.REPRESENTATIVE_CASES[experiment])
        spec.update(
            resolution=requested["resolution"],
            wiggle=requested["wiggle"],
            case_index=requested["case_index"],
        )
        sources = {}
        for variant_name, suffix, show_main_endpoints in figures._endpoint_variant_specs(
            endpoint_variants
        ):
            output = output_dir / _case_filename(requested, suffix)
            figures._generate_representative_figure(
                experiment,
                figures._endpoint_visibility_spec(
                    spec,
                    show_main_endpoints=show_main_endpoints,
                ),
                output,
            )
            sources[variant_name] = str(output)

        algo = OURS_ALGO[experiment]
        row = case_metrics[
            (
                experiment,
                algo,
                float(requested["resolution"]),
                float(requested["wiggle"]),
                int(requested["case_index"]),
            )
        ]
        hausdorff = _format_metric(_metric_value(row, "hausdorff"))
        facet_gap = _format_metric(_metric_value(row, "facet_gap"))
        entry = {
            **requested,
            "algo": algo,
            "title": (
                f"{experiment.title()}: case {requested['case_index']}, "
                f"N={round(requested['resolution'] * 100)}, "
                f"w={requested['wiggle']:g}"
            ),
            "purpose": requested["purpose"],
            "hausdorff": hausdorff,
            "facet_gap": facet_gap,
            "sources": sources,
        }
        if not paired:
            entry["source"] = next(iter(sources.values()))
        manifest.append(entry)

    manifest_path = review_root / (
        "diagnostic_pair_manifest.json" if paired else "diagnostic_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = _review_rows(case_metrics)
    markdown_path, csv_path = _write_approval_sheet(
        review_root=review_root,
        rows=rows,
        diagnostic_manifest=manifest,
        endpoint_variants=endpoint_variants,
    )
    print(manifest_path)
    print(markdown_path)
    print(csv_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--plots_root", type=Path, required=True)
    parser.add_argument(
        "--endpoint_variants",
        choices=sorted(figures.ENDPOINT_VARIANT_MODES),
        default="annotated",
    )
    args = parser.parse_args()
    generate(
        run_root=args.run_root.resolve(),
        plots_root=args.plots_root.resolve(),
        endpoint_variants=args.endpoint_variants,
    )


if __name__ == "__main__":
    main()
