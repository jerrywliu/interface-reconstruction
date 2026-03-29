#!/usr/bin/env python3
"""
Run and plot the appendix C0 comparison study for selected static benchmarks.

Current study:
- ellipses:
  - Ours (linear)
  - Ours (linear, C0)
  - Ours (circular)
- zalesak:
  - Ours (circular)
  - Ours (circular, C0)
  - Ours (circular+corner)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import generate_section6_maintext_figures as maintext_figs
from experiments.static import run_perturbed_sweeps as sweeps


APPENDIX_EXPERIMENTS = [
    {
        "name": "ellipses",
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "num_arg": "--num_ellipses",
        "num_default": 25,
        "resolutions": sweeps.DEFAULT_RESOLUTIONS,
        "wiggles": sweeps.DEFAULT_WIGGLES,
        "metrics": ("hausdorff", "facet_gap"),
        "variants": [
            {
                "label": "linear",
                "display": "Ours (linear)",
                "facet_algo": "linear",
                "do_c0": False,
            },
            {
                "label": "linear+C0",
                "display": "Ours (linear, C0)",
                "facet_algo": "linear",
                "do_c0": True,
            },
            {
                "label": "circular",
                "display": "Ours (circular)",
                "facet_algo": "circular",
                "do_c0": False,
            },
        ],
        "representative": {
            "resolution": 0.32,
            "wiggle": 0.10,
            "seed": 0,
            "case_index": 12,
            "methods": [
                ("linear", "Ours (linear)"),
                ("linear+C0", "Ours (linear, C0)"),
                ("circular", "Ours (circular)"),
            ],
            "min_span": 66.0,
            "margin_frac": 0.12,
            "inset": None,
        },
    },
    {
        "name": "zalesak",
        "module": "experiments.static.zalesak",
        "config": "static/zalesak",
        "num_arg": "--num_cases",
        "num_default": 25,
        "resolutions": sweeps.DEFAULT_RESOLUTIONS_SHORT,
        "wiggles": sweeps.DEFAULT_WIGGLES,
        "metrics": ("hausdorff", "facet_gap"),
        "variants": [
            {
                "label": "circular",
                "display": "Ours (circular)",
                "facet_algo": "circular",
                "do_c0": False,
            },
            {
                "label": "circular+C0",
                "display": "Ours (circular, C0)",
                "facet_algo": "circular",
                "do_c0": True,
            },
            {
                "label": "circular+corner",
                "display": "Ours (circular+corner)",
                "facet_algo": "circular+corner",
                "do_c0": False,
            },
        ],
        "representative": {
            "resolution": 1.00,
            "wiggle": 0.10,
            "seed": 0,
            "case_index": 12,
            "methods": [
                ("circular", "Ours (circular)"),
                ("circular+C0", "Ours (circular, C0)"),
                ("circular+corner", "Ours (circular+corner)"),
            ],
            "min_span": 42.0,
            "margin_frac": 0.12,
            "inset": {"kind": "zalesak_corner", "zoom": 3.0},
        },
    },
]


def _parse_list(raw, cast=float):
    if raw is None:
        return []
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_str_list(raw):
    if raw is None:
        return []
    return [part.strip().lower() for part in str(raw).split(",") if part.strip()]


def _variant_by_label(exp_spec, label):
    for variant in exp_spec["variants"]:
        if variant["label"].lower() == label.lower():
            return variant
    return None


def _selected_experiments(raw_only):
    only = set(_parse_str_list(raw_only))
    if not only:
        return APPENDIX_EXPERIMENTS
    return [exp for exp in APPENDIX_EXPERIMENTS if exp["name"] in only]


def _selected_variants(exp_spec, raw_algos):
    only = set(_parse_str_list(raw_algos))
    if not only:
        return exp_spec["variants"]
    selected = []
    for variant in exp_spec["variants"]:
        if variant["label"].lower() in only or variant["facet_algo"].lower() in only:
            selected.append(variant)
    return selected


def _variant_save_name(exp_name, label, resolution, wiggle, seed):
    base = sweeps._make_save_name(exp_name, label, resolution, wiggle, seed)
    return f"appendix_c0_{base}"


def _build_rows_index(rows):
    data = {}
    for row in rows:
        exp = row["experiment"]
        data.setdefault(exp, []).append(row)
    return data


def _load_rows(csv_path):
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def _generate_plots(csv_path: Path, out_dir: Path):
    rows = _load_rows(csv_path)
    data = sweeps._build_metric_index(rows)

    summary_dir = out_dir / "summary_plots"
    representative_dir = out_dir / "representative_cases"
    summary_dir.mkdir(parents=True, exist_ok=True)
    representative_dir.mkdir(parents=True, exist_ok=True)

    outputs = {"summary": {}, "representative": {}}
    original_make_save_name = maintext_figs._make_save_name
    maintext_figs._make_save_name = _variant_save_name
    try:
        for exp_spec in APPENDIX_EXPERIMENTS:
            exp_name = exp_spec["name"]
            exp_data = data.get(exp_name, {})
            if not exp_data:
                continue
            methods = [variant["label"] for variant in exp_spec["variants"] if variant["label"] in exp_data]
            if not methods:
                continue
            metric_out = summary_dir / f"{exp_name}_appendix_c0_2x2.png"
            maintext_figs._generate_quantitative_panel(
                exp_name=exp_name,
                exp_data=exp_data,
                methods=methods,
                metrics=exp_spec["metrics"],
                out_path=metric_out,
            )
            outputs["summary"][exp_name] = str(metric_out)

            rep_out = representative_dir / f"{exp_name}_appendix_c0_representative.png"
            rep_spec = exp_spec["representative"]
            try:
                maintext_figs._generate_representative_figure(
                    exp_name=exp_name,
                    spec=rep_spec,
                    out_path=rep_out,
                )
                outputs["representative"][exp_name] = str(rep_out)
            except FileNotFoundError:
                print(
                    f"[WARN] skipping representative figure for {exp_name}: "
                    f"representative case {rep_spec['case_index']} artifacts not present in this run set"
                )
    finally:
        maintext_figs._make_save_name = original_make_save_name

    return outputs


def _run_subprocess(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run appendix C0 static comparison sweeps.")
    parser.add_argument("--only", type=str, default=None, help="comma-separated experiments to run")
    parser.add_argument("--algos", type=str, default=None, help="comma-separated variant labels to run")
    parser.add_argument("--resolutions", type=str, default=None, help="comma-separated resolutions override")
    parser.add_argument("--wiggles", type=str, default=None, help="comma-separated wiggles override")
    parser.add_argument("--seeds", type=str, default="0", help="comma-separated seeds")
    parser.add_argument("--ellipses", type=int, default=25, help="number of ellipse cases")
    parser.add_argument("--zalesak", type=int, default=25, help="number of Zalesak cases")
    parser.add_argument("--out_csv", type=str, default=None, help="output CSV path")
    parser.add_argument("--out_dir", type=str, default=None, help="output artifact directory")
    parser.add_argument("--log_dir", type=str, default=None, help="log directory")
    parser.add_argument("--collect_existing", action="store_true", help="collect existing runs only")
    parser.add_argument("--plot_from_csv", type=str, default=None, help="generate plots only from an existing CSV")
    parser.add_argument("--dry_run", action="store_true", help="print commands without executing")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir
        or REPO_ROOT
        / "results"
        / "static"
        / "camera_ready"
        / f"static_appendix_c0_{stamp}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_from_csv:
        csv_path = Path(args.plot_from_csv).resolve()
        outputs = _generate_plots(csv_path, out_dir)
        print(f"Generated appendix C0 plots from {csv_path}")
        for bucket, paths in outputs.items():
            for name, path in paths.items():
                print(f"[{bucket}] {name}: {path}")
        return

    out_csv = Path(args.out_csv or out_dir / "csv" / "appendix_c0_sweep.csv").resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir or out_dir / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    resolutions_override = _parse_list(args.resolutions, float)
    wiggles_override = _parse_list(args.wiggles, float)
    seeds = _parse_list(args.seeds, int) or [0]

    fieldnames = [
        "experiment",
        "algo",
        "facet_algo",
        "do_c0",
        "resolution",
        "wiggle",
        "seed",
        "metric_key",
        "metric_value",
        "save_name",
    ]

    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for exp_spec in _selected_experiments(args.only):
            variants = _selected_variants(exp_spec, args.algos)
            if not variants:
                continue
            resolutions = resolutions_override or exp_spec["resolutions"]
            wiggles = wiggles_override or exp_spec["wiggles"]
            num_value = getattr(args, exp_spec["name"])

            for resolution in resolutions:
                for wiggle in wiggles:
                    for seed in seeds:
                        for variant in variants:
                            save_name = _variant_save_name(
                                exp_spec["name"],
                                variant["label"],
                                resolution,
                                wiggle,
                                seed,
                            )
                            cmd = [
                                sys.executable,
                                "-m",
                                exp_spec["module"],
                                "--config",
                                exp_spec["config"],
                                "--resolution",
                                str(resolution),
                                "--facet_algo",
                                variant["facet_algo"],
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
                                "--do_c0",
                                "1" if variant["do_c0"] else "0",
                                exp_spec["num_arg"],
                                str(num_value),
                            ]

                            if args.dry_run:
                                print(" ".join(cmd))
                                continue

                            if args.collect_existing:
                                metrics = sweeps._collect_metrics(exp_spec["name"], save_name)
                                if not metrics:
                                    print(
                                        f"[SKIP] missing existing metrics for {exp_spec['name']} {variant['label']} r={resolution} w={wiggle} s={seed}"
                                    )
                                    continue
                            else:
                                log_path = log_dir / f"{save_name}.log"
                                code = _run_subprocess(cmd, log_path)
                                if code != 0:
                                    raise RuntimeError(
                                        f"{exp_spec['name']} {variant['label']} r={resolution} w={wiggle} s={seed} failed; see {log_path}"
                                    )
                                metrics = sweeps._collect_metrics(exp_spec["name"], save_name)

                            for key, value in metrics.items():
                                writer.writerow(
                                    {
                                        "experiment": exp_spec["name"],
                                        "algo": variant["label"],
                                        "facet_algo": variant["facet_algo"],
                                        "do_c0": int(variant["do_c0"]),
                                        "resolution": resolution,
                                        "wiggle": wiggle,
                                        "seed": seed,
                                        "metric_key": key,
                                        "metric_value": value,
                                        "save_name": save_name,
                                    }
                                )

    outputs = _generate_plots(out_csv, out_dir)
    print(f"Appendix C0 sweep CSV: {out_csv}")
    for bucket, paths in outputs.items():
        for name, path in paths.items():
            print(f"[{bucket}] {name}: {path}")


if __name__ == "__main__":
    main()
