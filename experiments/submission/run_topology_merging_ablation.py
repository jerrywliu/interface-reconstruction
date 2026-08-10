#!/usr/bin/env python3
"""Run the submission topology/orientation-and-merging ablation on Zalesak."""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ("safe_circle", "circular")
DISPLAY_LABELS = {
    "safe_circle": "Ours (circular, per-cell)",
    "circular": "Ours (circular, graph-coordinated)",
}
DEFAULT_RESOLUTIONS = (0.50, 0.64, 1.00, 1.28, 1.50)
PLIC_NAMES = {"Youngs", "ELVIRA", "LVIRA"}
PLIC_FALLBACK = "LVIRA"
ARC_FAILURE_FALLBACK = "local_linear"


def _parse_resolutions(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _source_state():
    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--short"),
    }


def _run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_specs(args, resolutions, output_dir):
    specs = []
    run_tag = output_dir.name
    for resolution in resolutions:
        for algo in ALGORITHMS:
            res_tag = str(resolution).replace(".", "p")
            save_name = f"{run_tag}_{algo}_r{res_tag}_w{str(args.wiggle).replace('.', 'p')}_s{args.seed}"
            cmd = [
                sys.executable,
                "-m",
                "experiments.static.zalesak",
                "--config",
                "static/zalesak",
                "--resolution",
                str(resolution),
                "--facet_algo",
                algo,
                "--save_name",
                save_name,
                "--num_cases",
                str(args.num_cases),
                "--mesh_type",
                "perturbed_quads",
                "--perturb_wiggle",
                str(args.wiggle),
                "--perturb_seed",
                str(args.seed),
                "--perturb_fix_boundary",
                "1",
                "--do_c0",
                "0",
                "--plic_fallback",
                PLIC_FALLBACK,
                "--arc_failure_fallback",
                ARC_FAILURE_FALLBACK,
                "--corner_behavior_profile",
                args.corner_behavior_profile,
                "--rescue_profile",
                args.rescue_profile,
            ]
            if args.case_indices:
                cmd.extend(["--case_indices", args.case_indices])
            specs.append(
                {
                    "algo": algo,
                    "display_label": DISPLAY_LABELS[algo],
                    "resolution": resolution,
                    "cells_per_side": int(round(100 * resolution)),
                    "wiggle": args.wiggle,
                    "seed": args.seed,
                    "save_name": save_name,
                    "cmd": cmd,
                    "plot_dir": str(REPO_ROOT / "plots" / save_name),
                }
            )
    return specs


def _execute(spec, logs_dir):
    log_path = logs_dir / f"{spec['save_name']}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            spec["cmd"],
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode, log_path


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row, key, default=float("nan")):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _int(row, key, default=0):
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def _collect_cases(spec):
    metrics_dir = Path(spec["plot_dir"]) / "metrics"
    case_rows = _read_csv(metrics_dir / "case_metrics.csv")
    cell_rows = _read_csv(metrics_dir / "cell_metrics.csv")
    event_rows = _read_csv(metrics_dir / "merge_events.csv")

    cells_by_case = defaultdict(list)
    for row in cell_rows:
        cells_by_case[_int(row, "case_index", -1)].append(row)
    events_by_case = defaultdict(list)
    for row in event_rows:
        events_by_case[_int(row, "case_index", -1)].append(row)

    output = []
    for row in case_rows:
        case_index = _int(row, "case_index", -1)
        cells = cells_by_case[case_index]
        events = events_by_case[case_index]
        num_mixed = _int(row, "num_mixed_cells", len(cells))
        plic_cells = sum(
            bool(cell.get("fallback_policy", ""))
            or cell.get("final_facet_name", "") in PLIC_NAMES
            for cell in cells
        )
        local_line_cells = sum(
            cell.get("final_facet_name", "") == "default_linear" for cell in cells
        )
        local_line_events = sum(
            event.get("event_kind", "") == "local_linear_fallback"
            for event in events
        )
        output.append(
            {
                **{key: value for key, value in spec.items() if key not in {"cmd", "plot_dir"}},
                "case_index": case_index,
                "hausdorff": _float(row, "hausdorff"),
                "facet_gap": _float(row, "facet_gap"),
                "area_error": _float(row, "area_error"),
                "num_mixed_cells": num_mixed,
                "num_merged_cells": _int(row, "num_merged_cells"),
                "plic_fallback_cells": plic_cells,
                "local_linear_fallback_cells": local_line_cells,
                "local_linear_fallback_events": local_line_events,
                "fraction_merged_cells": _int(row, "num_merged_cells") / num_mixed
                if num_mixed
                else 0.0,
                "fraction_plic_fallback_cells": plic_cells / num_mixed
                if num_mixed
                else 0.0,
                "fraction_local_linear_fallback_cells": local_line_cells / num_mixed
                if num_mixed
                else 0.0,
            }
        )
    return output


def _median(values):
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return float("nan")
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def _mean(values):
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def _summaries(case_rows):
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[(row["algo"], row["resolution"])].append(row)

    summaries = []
    for (algo, resolution), rows in sorted(grouped.items()):
        mixed_cells = sum(row["num_mixed_cells"] for row in rows)
        summary = {
            "algo": algo,
            "display_label": DISPLAY_LABELS[algo],
            "resolution": resolution,
            "cells_per_side": int(round(100 * resolution)),
            "num_cases": len(rows),
            "hausdorff_median": _median([row["hausdorff"] for row in rows]),
            "hausdorff_mean": _mean([row["hausdorff"] for row in rows]),
            "facet_gap_median": _median([row["facet_gap"] for row in rows]),
            "area_error_median": _median([row["area_error"] for row in rows]),
            "hausdorff_gt_1_cases": sum(row["hausdorff"] > 1 for row in rows),
            "fraction_merged_cells": sum(row["num_merged_cells"] for row in rows)
            / mixed_cells
            if mixed_cells
            else 0.0,
            "fraction_plic_fallback_cells": sum(
                row["plic_fallback_cells"] for row in rows
            )
            / mixed_cells
            if mixed_cells
            else 0.0,
            "fraction_local_linear_fallback_cells": sum(
                row["local_linear_fallback_cells"] for row in rows
            )
            / mixed_cells
            if mixed_cells
            else 0.0,
        }
        summaries.append(summary)
    return summaries


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summaries, output_dir):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    metrics = [
        ("hausdorff_median", "Median Hausdorff", True),
        ("facet_gap_median", "Median facet gap", True),
        ("fraction_plic_fallback_cells", "PLIC fallback fraction", False),
        ("fraction_local_linear_fallback_cells", "Local-line fallback fraction", False),
        ("fraction_merged_cells", "Merged mixed-cell fraction", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)
    colors = {"safe_circle": "#d97706", "circular": "#0f766e"}
    styles = {"safe_circle": "--", "circular": "-"}
    for ax, (metric, title, log_scale) in zip(axes.flat, metrics):
        for algo in ALGORITHMS:
            rows = sorted(
                (row for row in summaries if row["algo"] == algo),
                key=lambda row: row["cells_per_side"],
            )
            ax.plot(
                [row["cells_per_side"] for row in rows],
                [max(row[metric], 1e-14) if log_scale else row[metric] for row in rows],
                marker="o",
                color=colors[algo],
                linestyle=styles[algo],
                linewidth=2.0,
                label=DISPLAY_LABELS[algo],
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Cells per side, N")
        ax.grid(True, alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(output_dir / f"topology_merging_ablation.{suffix}", **kwargs)
    plt.close(fig)


def _write_readme(output_dir, args, summaries, failures):
    lines = [
        "# Zalesak topology/merging ablation",
        "",
        "Comparison: independent-cell circular reconstruction versus circular reconstruction with the topology/orientation-and-merging scaffold.",
        "",
        f"- PLIC fallback: `{PLIC_FALLBACK}`",
        f"- Oriented arc-fit fallback: `{ARC_FAILURE_FALLBACK}`",
        "- Optional C0 pass: disabled for both methods",
        f"- Perturbation magnitude: `{args.wiggle}`",
        f"- Failed jobs: `{len(failures)}`",
        "",
        "| Method | N | Cases | Median Hausdorff | Median gap | PLIC fallback fraction | Local-line fallback fraction | Merged fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {label} | {n} | {cases} | {h:.6g} | {gap:.6g} | {plic:.4f} | {line:.4f} | {merged:.4f} |".format(
                label=row["display_label"],
                n=row["cells_per_side"],
                cases=row["num_cases"],
                h=row["hausdorff_median"],
                gap=row["facet_gap_median"],
                plic=row["fraction_plic_fallback_cells"],
                line=row["fraction_local_linear_fallback_cells"],
                merged=row["fraction_merged_cells"],
            )
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="0.50,0.64,1.00,1.28,1.50")
    parser.add_argument("--wiggle", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=25)
    parser.add_argument("--case-indices", default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--corner-behavior-profile", default="pre_f8_corner")
    parser.add_argument("--rescue-profile", default="exact_linear_support_only")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resolutions = _parse_resolutions(args.resolutions)
    if args.smoke:
        resolutions = [0.64, 1.00]
        args.case_indices = "0,6"
        args.num_cases = max(args.num_cases, 7)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    output_dir = Path(
        args.output_dir
        or REPO_ROOT
        / "results"
        / "static"
        / f"submission_topology_merging_ablation_{_run_id()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    specs = _build_specs(args, resolutions, output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if args.dry_run else "running",
        "comparison": {
            algo: DISPLAY_LABELS[algo] for algo in ALGORITHMS
        },
        "matched_fallback_hierarchy": {
            "unresolved_orientation_or_support": PLIC_FALLBACK,
            "resolved_supports_after_arc_failure": ARC_FAILURE_FALLBACK,
        },
        "parameters": vars(args),
        "source": _source_state(),
        "jobs": specs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Output: {output_dir}")
    for spec in specs:
        print(" ".join(spec["cmd"]))
    if args.dry_run:
        return 0

    failures = []
    completed = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_execute, spec, logs_dir): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            code, log_path = future.result()
            if code:
                failures.append({**spec, "returncode": code, "log_path": str(log_path)})
                print(f"FAILED {spec['save_name']}: {log_path}")
            else:
                completed.append(spec)
                print(f"Completed {spec['save_name']}")

    case_rows = []
    for spec in completed:
        case_rows.extend(_collect_cases(spec))
    summaries = _summaries(case_rows)
    _write_csv(output_dir / "case_metrics.csv", case_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    if failures:
        _write_csv(output_dir / "failures.csv", failures)
    if summaries:
        _plot(summaries, output_dir)
    _write_readme(output_dir, args, summaries, failures)

    manifest["status"] = "complete" if not failures else "failed"
    manifest["completed_jobs"] = len(completed)
    manifest["failed_jobs"] = len(failures)
    manifest["case_rows"] = len(case_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
