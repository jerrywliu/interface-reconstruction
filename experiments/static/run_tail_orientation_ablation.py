#!/usr/bin/env python3
"""Run and analyze targeted orientation-tail ablations.

The target set is the union of cases that used unresolved PLIC fallback in the
baseline sweep and cases whose Hausdorff error newly crossed 1 relative to the
prior paper sweep. Runs are grouped by setting so each driver invocation can
replay only the relevant deterministic case indices.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.static.run_perturbed_sweeps import EXPERIMENTS
from experiments.static.sweep_diagnostics import (
    consolidate_run_diagnostics,
    prepare_diagnostic_bundle,
)


DEFAULT_BASELINE = Path(
    "results/static/static_paper_simplified_default_20260717_212413"
)
DEFAULT_PRIOR = Path(
    "results/static/static_paper_affected_diagnostics_20260714_102206"
)
SETTING_FIELDS = ("experiment", "algo", "resolution", "wiggle", "seed")
PROFILE_CHOICES = (
    "pre_f8_corner",
    "pre_f8_corner_late_hint",
    "pre_f8_corner_greedy_retry",
    "pre_f8_corner_greedy_continue",
    "pre_f8_corner_late_hint_retry",
)
COLORS = {
    "circles": "#0072B2",
    "ellipses": "#009E73",
    "lines": "#CC79A7",
    "squares": "#E69F00",
    "zalesak": "#D55E00",
}


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _setting_key(row):
    return (
        row["experiment"],
        row["algo"],
        float(row["resolution"]),
        float(row["wiggle"]),
        int(row["seed"]),
    )


def _case_key(row):
    return (*_setting_key(row), int(row["case_index"]))


def _safe_name(value):
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")


def _target_cases(baseline_root, extra_case_paths=()):
    targets = defaultdict(set)
    fallback_path = baseline_root / "diagnostics" / "unresolved_plic_fallbacks.csv"
    for row in _read_csv(fallback_path):
        targets[_case_key(row)].add("baseline_plic_fallback")

    comparison_path = (
        baseline_root
        / "comparison"
        / "static_paper_affected_diagnostics_20260714_102206"
        / "case_comparison.csv"
    )
    for row in _read_csv(comparison_path):
        if float(row["old_hausdorff"]) <= 1.0 < float(row["new_hausdorff"]):
            targets[_case_key(row)].add("introduced_hausdorff_above_one")
    for extra_path in extra_case_paths:
        for row in _read_csv(extra_path):
            if row.get("hausdorff_outcome") not in (None, "", "stable"):
                targets[_case_key(row)].add(
                    f"guardrail_{row['hausdorff_outcome']}"
                )
    return targets


def _group_targets(targets):
    grouped = defaultdict(list)
    for case_key in targets:
        grouped[case_key[:-1]].append(case_key[-1])
    return {key: sorted(values) for key, values in grouped.items()}


def _build_spec(setting, case_indices, profile, stamp, output_root):
    experiment, algo, resolution, wiggle, seed = setting
    experiment_spec = next(item for item in EXPERIMENTS if item["name"] == experiment)
    n_value = int(round(100 * resolution))
    save_name = "_".join(
        [
            "tail",
            stamp,
            experiment,
            _safe_name(algo),
            f"n{n_value}",
            f"w{_safe_name(wiggle)}",
            f"s{seed}",
            profile,
        ]
    )
    cmd = [
        sys.executable,
        "-m",
        experiment_spec["module"],
        "--config",
        experiment_spec["config"],
        "--resolution",
        str(resolution),
        "--facet_algo",
        algo,
        "--save_name",
        save_name,
        "--mesh_type",
        "perturbed_quads",
        "--perturb_wiggle",
        str(wiggle),
        "--perturb_seed",
        str(seed),
        "--perturb_fix_boundary",
        "1",
        experiment_spec["num_arg"],
        "25",
        "--case_indices",
        ",".join(str(index) for index in case_indices),
        "--plic_fallback",
        "LVIRA",
        "--corner_behavior_profile",
        profile,
    ]
    rescue_profile = ""
    if experiment == "zalesak":
        rescue_profile = "exact_linear_support_only"
        cmd += ["--rescue_profile", rescue_profile]
    return {
        "experiment": experiment,
        "algo": algo,
        "resolution": resolution,
        "wiggle": wiggle,
        "seed": seed,
        "case_indices": case_indices,
        "corner_behavior_profile": profile,
        "plic_fallback": "LVIRA",
        "rescue_profile": rescue_profile,
        "save_name": save_name,
        "run_dir": Path("plots") / save_name,
        "log_path": output_root / "logs" / f"{save_name}.log",
        "cmd": cmd,
    }


def _execute(spec):
    spec["log_path"].parent.mkdir(parents=True, exist_ok=True)
    with spec["log_path"].open("w", encoding="utf-8") as log:
        result = subprocess.run(
            spec["cmd"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def _run_specs(specs, output_root, max_workers):
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_execute, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            code = future.result()
            completed += 1
            print(
                f"[{completed}/{len(specs)}] {spec['save_name']} "
                f"cases={len(spec['case_indices'])} exit={code}",
                flush=True,
            )
            if code:
                failures.append(
                    {
                        **{field: spec[field] for field in SETTING_FIELDS},
                        "corner_behavior_profile": spec["corner_behavior_profile"],
                        "case_indices": ",".join(
                            str(index) for index in spec["case_indices"]
                        ),
                        "exit_code": code,
                        "log_path": str(spec["log_path"]),
                    }
                )
    _write_csv(output_root / "failures.csv", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} targeted runs failed")


def _consolidate(specs, output_root, profiles):
    for profile in profiles:
        diagnostics_dir = prepare_diagnostic_bundle(
            output_root / profile / "diagnostics"
        )
        for spec in specs:
            if spec["corner_behavior_profile"] != profile:
                continue
            consolidate_run_diagnostics(
                spec["run_dir"],
                diagnostics_dir,
                {field: spec[field] for field in (*SETTING_FIELDS, "save_name", "corner_behavior_profile")},
            )


def _index_cases(root):
    return {
        _case_key(row): row
        for row in _read_csv(Path(root) / "diagnostics" / "case_metrics.csv")
    }


def _float(row, field, default=0.0):
    value = row.get(field, "")
    return default if value in (None, "") else float(value)


def _int(row, field, default=0):
    value = row.get(field, "")
    return default if value in (None, "") else int(float(value))


def _outcome(baseline, candidate):
    threshold = max(1e-10, 0.01 * max(abs(baseline), abs(candidate)))
    if candidate < baseline - threshold:
        return "improved"
    if candidate > baseline + threshold:
        return "worsened"
    return "stable"


def _comparison_rows(targets, baseline_root, prior_root, output_root, profiles):
    baseline = _index_cases(baseline_root)
    prior = _index_cases(prior_root)
    comparisons = []
    for profile in profiles:
        candidate = _index_cases(output_root / profile)
        if candidate.keys() != targets.keys():
            missing = sorted(targets.keys() - candidate.keys())
            extra = sorted(candidate.keys() - targets.keys())
            raise ValueError(
                f"Profile {profile} case mismatch: missing={missing[:5]} extra={extra[:5]}"
            )
        for key in sorted(targets):
            baseline_row = baseline[key]
            prior_row = prior[key]
            candidate_row = candidate[key]
            baseline_h = float(baseline_row["hausdorff"])
            candidate_h = float(candidate_row["hausdorff"])
            comparisons.append(
                {
                    "corner_behavior_profile": profile,
                    "experiment": key[0],
                    "algo": key[1],
                    "resolution": key[2],
                    "cells_per_side": int(round(100 * key[2])),
                    "wiggle": key[3],
                    "seed": key[4],
                    "case_index": key[5],
                    "target_reasons": ";".join(sorted(targets[key])),
                    "prior_hausdorff": float(prior_row["hausdorff"]),
                    "baseline_hausdorff": baseline_h,
                    "candidate_hausdorff": candidate_h,
                    "candidate_minus_baseline": candidate_h - baseline_h,
                    "hausdorff_outcome": _outcome(baseline_h, candidate_h),
                    "prior_fallback_cells": _int(prior_row, "num_plic_fallback_cells"),
                    "baseline_fallback_cells": _int(
                        baseline_row, "num_plic_fallback_cells"
                    ),
                    "candidate_fallback_cells": _int(
                        candidate_row, "num_plic_fallback_cells"
                    ),
                    "candidate_early_orientation_hints": _int(
                        candidate_row, "num_early_orientation_hints"
                    ),
                    "candidate_late_orientation_hints": _int(
                        candidate_row, "num_late_orientation_hints"
                    ),
                    "candidate_orientation_retry_passes": _int(
                        candidate_row, "num_orientation_retry_passes"
                    ),
                    "baseline_joint_floor": int(
                        baseline_h < 1e-6
                        and _float(baseline_row, "facet_gap") < 1e-6
                    ),
                    "candidate_joint_floor": int(
                        candidate_h < 1e-6
                        and _float(candidate_row, "facet_gap") < 1e-6
                    ),
                }
            )
    return comparisons


def _summary(rows, subset_name):
    if not rows:
        return {}
    baseline = np.asarray([row["baseline_hausdorff"] for row in rows])
    candidate = np.asarray([row["candidate_hausdorff"] for row in rows])
    return {
        "corner_behavior_profile": rows[0]["corner_behavior_profile"],
        "subset": subset_name,
        "case_count": len(rows),
        "material_improved_cases": sum(
            row["hausdorff_outcome"] == "improved" for row in rows
        ),
        "material_worsened_cases": sum(
            row["hausdorff_outcome"] == "worsened" for row in rows
        ),
        "material_stable_cases": sum(
            row["hausdorff_outcome"] == "stable" for row in rows
        ),
        "baseline_hausdorff_median": float(np.median(baseline)),
        "candidate_hausdorff_median": float(np.median(candidate)),
        "baseline_hausdorff_p95": float(np.quantile(baseline, 0.95)),
        "candidate_hausdorff_p95": float(np.quantile(candidate, 0.95)),
        "baseline_hausdorff_max": float(np.max(baseline)),
        "candidate_hausdorff_max": float(np.max(candidate)),
        "baseline_hausdorff_above_one": int(np.count_nonzero(baseline > 1.0)),
        "candidate_hausdorff_above_one": int(np.count_nonzero(candidate > 1.0)),
        "fixed_hausdorff_above_one": int(
            np.count_nonzero((baseline > 1.0) & (candidate <= 1.0))
        ),
        "introduced_hausdorff_above_one": int(
            np.count_nonzero((baseline <= 1.0) & (candidate > 1.0))
        ),
        "baseline_joint_floor_cases": sum(row["baseline_joint_floor"] for row in rows),
        "candidate_joint_floor_cases": sum(row["candidate_joint_floor"] for row in rows),
        "baseline_fallback_cells": sum(row["baseline_fallback_cells"] for row in rows),
        "candidate_fallback_cells": sum(row["candidate_fallback_cells"] for row in rows),
        "early_orientation_hints": sum(
            row["candidate_early_orientation_hints"] for row in rows
        ),
        "late_orientation_hints": sum(
            row["candidate_late_orientation_hints"] for row in rows
        ),
        "orientation_retry_passes": sum(
            row["candidate_orientation_retry_passes"] for row in rows
        ),
    }


def _summaries(comparisons, profiles):
    summaries = []
    for profile in profiles:
        profile_rows = [
            row for row in comparisons if row["corner_behavior_profile"] == profile
        ]
        subsets = {
            "all_targets": profile_rows,
            "baseline_plic_fallback": [
                row
                for row in profile_rows
                if "baseline_plic_fallback" in row["target_reasons"]
            ],
            "introduced_hausdorff_above_one": [
                row
                for row in profile_rows
                if "introduced_hausdorff_above_one" in row["target_reasons"]
            ],
        }
        summaries.extend(
            _summary(rows, name) for name, rows in subsets.items() if rows
        )
        for experiment in sorted({row["experiment"] for row in profile_rows}):
            rows = [row for row in profile_rows if row["experiment"] == experiment]
            summaries.append(_summary(rows, f"experiment:{experiment}"))
    return summaries


def _plot(comparisons, summaries, profiles, output_root):
    fig, axes = plt.subplots(
        len(profiles),
        3,
        figsize=(12.5, 4.1 * len(profiles)),
        squeeze=False,
    )
    for row_index, profile in enumerate(profiles):
        rows = [
            row for row in comparisons if row["corner_behavior_profile"] == profile
        ]
        ax = axes[row_index, 0]
        for experiment in sorted({row["experiment"] for row in rows}):
            exp_rows = [row for row in rows if row["experiment"] == experiment]
            ax.scatter(
                [max(row["baseline_hausdorff"], 1e-12) for row in exp_rows],
                [max(row["candidate_hausdorff"], 1e-12) for row in exp_rows],
                s=24,
                alpha=0.75,
                color=COLORS[experiment],
                label=experiment,
                edgecolors="none",
            )
        bounds = [1e-12, 30]
        ax.plot(bounds, bounds, color="#555555", linewidth=1, linestyle="--")
        ax.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds)
        ax.set_xlabel("Simplified baseline Hausdorff")
        ax.set_ylabel("Candidate Hausdorff")
        ax.set_title(profile.replace("pre_f8_corner_", "").replace("_", " "))
        if row_index == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)

        summary = next(
            item
            for item in summaries
            if item["corner_behavior_profile"] == profile
            and item["subset"] == "all_targets"
        )
        labels = ["H > 1", "PLIC fallback\ncells", "joint floor\ncases"]
        baseline_values = [
            summary["baseline_hausdorff_above_one"],
            summary["baseline_fallback_cells"],
            summary["baseline_joint_floor_cases"],
        ]
        candidate_values = [
            summary["candidate_hausdorff_above_one"],
            summary["candidate_fallback_cells"],
            summary["candidate_joint_floor_cases"],
        ]
        x = np.arange(len(labels))
        width = 0.36
        ax = axes[row_index, 1]
        ax.bar(x - width / 2, baseline_values, width, label="simplified", color="#999999")
        ax.bar(x + width / 2, candidate_values, width, label="candidate", color="#0072B2")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Count")
        ax.set_title("Tail counts")
        if row_index == 0:
            ax.legend(frameon=False)

        experiments = sorted({row["experiment"] for row in rows})
        baseline_severe = [
            sum(
                row["baseline_hausdorff"] > 1
                for row in rows
                if row["experiment"] == experiment
            )
            for experiment in experiments
        ]
        candidate_severe = [
            sum(
                row["candidate_hausdorff"] > 1
                for row in rows
                if row["experiment"] == experiment
            )
            for experiment in experiments
        ]
        ax = axes[row_index, 2]
        x = np.arange(len(experiments))
        ax.bar(x - width / 2, baseline_severe, width, color="#999999")
        ax.bar(x + width / 2, candidate_severe, width, color="#0072B2")
        ax.set_xticks(x, experiments, rotation=25, ha="right")
        ax.set_ylabel("Cases with Hausdorff > 1")
        ax.set_title("Severe tail by problem")

    fig.suptitle("Targeted orientation-tail ablation", fontsize=14)
    fig.tight_layout()
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png_path = plot_dir / "orientation_tail_all_methods_comparison.png"
    pdf_path = plot_dir / "orientation_tail_all_methods_comparison.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _write_readme(output_root, targets, summaries, profiles, plot_paths):
    lines = [
        "# Targeted orientation-tail ablation",
        "",
        "Target set: every case with an unresolved PLIC fallback in the July 17",
        "simplified sweep, plus every case whose Hausdorff error newly crossed 1",
        "relative to the July 14 paper sweep.",
        "",
        f"- Unique target cases: {len(targets)}",
        f"- Profiles: {', '.join(profiles)}",
        f"- Plot: `{plot_paths[0].relative_to(output_root)}`",
        "",
        "## All-target summary",
        "",
        "| Profile | H>1 baseline -> candidate | fallback cells baseline -> candidate | material improved / worsened | late hints | retry passes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        summary = next(
            item
            for item in summaries
            if item["corner_behavior_profile"] == profile
            and item["subset"] == "all_targets"
        )
        lines.append(
            "| {profile} | {bh} -> {ch} | {bf} -> {cf} | {improved} / {worsened} | {hints} | {retries} |".format(
                profile=profile,
                bh=summary["baseline_hausdorff_above_one"],
                ch=summary["candidate_hausdorff_above_one"],
                bf=summary["baseline_fallback_cells"],
                cf=summary["candidate_fallback_cells"],
                improved=summary["material_improved_cases"],
                worsened=summary["material_worsened_cases"],
                hints=summary["late_orientation_hints"],
                retries=summary["orientation_retry_passes"],
            )
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `target_manifest.csv`: exact deterministic case selection and reasons",
        "- `case_comparison.csv`: prior, simplified baseline, and candidate metrics",
        "- `profile_summary.csv`: aggregate and per-problem summaries",
        "- `<profile>/diagnostics/`: consolidated case/cell/merge diagnostics",
        "- `logs/`: one log per grouped driver invocation",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--profiles",
        default="pre_f8_corner_late_hint,pre_f8_corner_greedy_retry",
    )
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--extra-cases",
        type=Path,
        action="append",
        default=[],
        help="comparison CSV whose non-stable cases should be added to the target set",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = sorted(set(profiles) - set(PROFILE_CHOICES))
    if unknown:
        parser.error(f"unknown profiles: {','.join(unknown)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or Path(
        f"results/static/tail_orientation_ablation_{stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    targets = _target_cases(args.baseline_root, args.extra_cases)
    grouped_targets = _group_targets(targets)
    specs = [
        _build_spec(setting, case_indices, profile, stamp, output_root)
        for profile in profiles
        for setting, case_indices in sorted(grouped_targets.items())
    ]

    manifest_rows = []
    for case_key, reasons in sorted(targets.items()):
        manifest_rows.append(
            {
                "experiment": case_key[0],
                "algo": case_key[1],
                "resolution": case_key[2],
                "cells_per_side": int(round(100 * case_key[2])),
                "wiggle": case_key[3],
                "seed": case_key[4],
                "case_index": case_key[5],
                "target_reasons": ";".join(sorted(reasons)),
            }
        )
    _write_csv(output_root / "target_manifest.csv", manifest_rows)
    (output_root / "run_plan.json").write_text(
        json.dumps(
            {
                "baseline_root": str(args.baseline_root),
                "prior_root": str(args.prior_root),
                "profiles": profiles,
                "unique_target_cases": len(targets),
                "grouped_settings": len(grouped_targets),
                "driver_invocations": len(specs),
                "commands": [spec["cmd"] for spec in specs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Targets={len(targets)} settings={len(grouped_targets)} "
        f"profiles={len(profiles)} invocations={len(specs)} output={output_root}",
        flush=True,
    )
    if args.dry_run:
        return

    _run_specs(specs, output_root, args.max_workers)
    _consolidate(specs, output_root, profiles)
    comparisons = _comparison_rows(
        targets, args.baseline_root, args.prior_root, output_root, profiles
    )
    summaries = _summaries(comparisons, profiles)
    _write_csv(output_root / "case_comparison.csv", comparisons)
    _write_csv(output_root / "profile_summary.csv", summaries)
    plot_paths = _plot(comparisons, summaries, profiles, output_root)
    _write_readme(output_root, targets, summaries, profiles, plot_paths)
    print(f"Complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
