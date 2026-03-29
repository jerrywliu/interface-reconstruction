#!/usr/bin/env python3
"""
Run deterministic single-case resolution-study reconstructions for the appendix.

This script runs one representative case per experiment, method, resolution, and
mesh perturbation setting, then assembles qualitative comparison figures with:
  - Cartesian grid on the left
  - Perturbed grid on the right
  - rows for N = 32, 50, 64

The best-method comparison used for each geometry is:
  - lines: linear
  - squares: linear+corner
  - circles: circular
  - ellipses: circular
  - zalesak: circular+corner
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import generate_section6_maintext_figures as maintext_figs
from experiments.static import run_perturbed_sweeps as sweeps


PLOTS_ROOT = REPO_ROOT / "plots"

APPENDIX_RESOLUTION_EXPERIMENTS = [
    {
        "name": "lines",
        "module": "experiments.static.lines",
        "config": "static/line",
        "num_arg": "--num_lines",
        "num_default": 25,
        "case_index": 12,
        "algo": "linear",
        "display": "Ours (linear)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 100.0,
        "margin_frac": 0.00,
        "inset": None,
    },
    {
        "name": "squares",
        "module": "experiments.static.squares",
        "config": "static/square",
        "num_arg": "--num_squares",
        "num_default": 25,
        "case_index": 12,
        "algo": "linear+corner",
        "display": "Ours (linear+corner)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 42.0,
        "margin_frac": 0.10,
        "inset": {"kind": "square_corner", "zoom": 2.8},
    },
    {
        "name": "circles",
        "module": "experiments.static.circles",
        "config": "static/circle",
        "num_arg": "--num_circles",
        "num_default": 25,
        "case_index": 12,
        "algo": "circular",
        "display": "Ours (circular)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 26.0,
        "margin_frac": 0.14,
        "inset": None,
    },
    {
        "name": "ellipses",
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "num_arg": "--num_ellipses",
        "num_default": 25,
        "case_index": 12,
        "algo": "circular",
        "display": "Ours (circular)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 66.0,
        "margin_frac": 0.12,
        "inset": None,
    },
    {
        "name": "zalesak",
        "module": "experiments.static.zalesak",
        "config": "static/zalesak",
        "num_arg": "--num_cases",
        "num_default": 25,
        "case_index": 12,
        "algo": "circular+corner",
        "display": "Ours (circular+corner)",
        "resolutions": [0.32, 0.50, 0.64],
        "wiggles": [0.0, 0.1],
        "seed": 0,
        "min_span": 42.0,
        "margin_frac": 0.12,
        "inset": {"kind": "zalesak_corner", "zoom": 3.0},
    },
]


def _parse_str_list(raw):
    if raw is None:
        return []
    return [part.strip().lower() for part in str(raw).split(",") if part.strip()]


def _selected_experiments(raw_only):
    only = set(_parse_str_list(raw_only))
    if not only:
        return APPENDIX_RESOLUTION_EXPERIMENTS
    return [exp for exp in APPENDIX_RESOLUTION_EXPERIMENTS if exp["name"] in only]


def _save_name(exp_name: str, algo: str, resolution: float, wiggle: float, seed: int) -> str:
    base = sweeps._make_save_name(exp_name, algo, resolution, wiggle, seed)
    return f"appendix_resolution_{base}"


def _required_paths(save_name: str, case_index: int) -> tuple[Path, Path]:
    mesh_path = PLOTS_ROOT / save_name / "vtk" / "mesh.vtk"
    facet_path = (
        PLOTS_ROOT / save_name / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    )
    return mesh_path, facet_path


def _run_exists(save_name: str, case_index: int) -> bool:
    mesh_path, facet_path = _required_paths(save_name, case_index)
    return mesh_path.exists() and facet_path.exists()


def _run_subprocess(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def _load_true_segments(exp_spec, base_save_name: str):
    exp_name = exp_spec["name"]
    case_index = exp_spec["case_index"]
    if exp_name == "lines":
        mesh_segments = maintext_figs._mesh_segments(PLOTS_ROOT / base_save_name / "vtk" / "mesh.vtk")
        bounds = maintext_figs._segments_bounds(mesh_segments)
        true_segments = maintext_figs._line_true_segments(case_index, bounds)
    else:
        true_segments = maintext_figs._load_true_segments(exp_name, base_save_name, case_index)
    return true_segments


def _figure_bounds(exp_spec, true_segments: np.ndarray):
    if exp_spec["name"] == "lines":
        # For lines, the true segments are already built against the domain bounds.
        return maintext_figs._segments_bounds(true_segments)
    return maintext_figs._compute_view_bounds(
        true_segments,
        min_span=exp_spec["min_span"],
        margin_frac=exp_spec["margin_frac"],
    )


def _generate_figure(exp_spec: dict, out_path: Path):
    nrows = len(exp_spec["resolutions"])
    ncols = len(exp_spec["wiggles"])
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 3.1 * nrows))
    axes = np.atleast_2d(axes)

    base_save_name = _save_name(
        exp_spec["name"],
        exp_spec["algo"],
        exp_spec["resolutions"][0],
        exp_spec["wiggles"][0],
        exp_spec["seed"],
    )
    true_segments = _load_true_segments(exp_spec, base_save_name)
    bounds = _figure_bounds(exp_spec, true_segments)

    for row, resolution in enumerate(exp_spec["resolutions"]):
        for col, wiggle in enumerate(exp_spec["wiggles"]):
            ax = axes[row, col]
            save_name = _save_name(
                exp_spec["name"],
                exp_spec["algo"],
                resolution,
                wiggle,
                exp_spec["seed"],
            )
            mesh_segments = maintext_figs._mesh_segments(PLOTS_ROOT / save_name / "vtk" / "mesh.vtk")
            recon_segments, endpoint_points = maintext_figs._load_reconstructed_segments_and_endpoints(
                save_name, exp_spec["case_index"]
            )
            condition = "Cartesian" if wiggle == 0.0 else "Perturbed"
            title = f"{condition}, N={int(round(resolution * 100))}"
            maintext_figs._plot_panel(
                ax,
                exp_name=exp_spec["name"],
                spec={"case_index": exp_spec["case_index"], "inset": exp_spec["inset"]},
                algo=exp_spec["algo"],
                mesh_segments=mesh_segments,
                true_segments=true_segments,
                recon_segments=recon_segments,
                endpoint_points=endpoint_points,
                title=title,
                bounds=bounds,
            )

    fig.suptitle(f"{exp_spec['display']} on {exp_spec['name'].title()}", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic appendix resolution-study reconstruction visuals."
    )
    parser.add_argument("--only", type=str, default=None, help="comma-separated experiments to run")
    parser.add_argument("--out_dir", type=Path, default=None, help="artifact output directory")
    parser.add_argument("--log_dir", type=Path, default=None, help="log directory override")
    parser.add_argument("--skip_existing", action="store_true", help="skip runs whose required artifacts already exist")
    parser.add_argument("--plot_only", action="store_true", help="generate figures only from existing run outputs")
    parser.add_argument("--dry_run", action="store_true", help="print commands without executing")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir
        or REPO_ROOT
        / "results"
        / "static"
        / "camera_ready"
        / f"static_appendix_resolution_visuals_{stamp}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir or out_dir / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "out_dir": str(out_dir),
        "summary_plots": {},
        "runs": [],
    }

    for exp_spec in _selected_experiments(args.only):
        for resolution in exp_spec["resolutions"]:
            for wiggle in exp_spec["wiggles"]:
                save_name = _save_name(
                    exp_spec["name"],
                    exp_spec["algo"],
                    resolution,
                    wiggle,
                    exp_spec["seed"],
                )
                run_record = {
                    "experiment": exp_spec["name"],
                    "algo": exp_spec["algo"],
                    "resolution": resolution,
                    "wiggle": wiggle,
                    "seed": exp_spec["seed"],
                    "case_index": exp_spec["case_index"],
                    "save_name": save_name,
                }
                manifest["runs"].append(run_record)

                if args.plot_only:
                    continue

                cmd = [
                    sys.executable,
                    "-m",
                    exp_spec["module"],
                    "--config",
                    exp_spec["config"],
                    "--resolution",
                    str(resolution),
                    "--facet_algo",
                    exp_spec["algo"],
                    "--save_name",
                    save_name,
                    "--mesh_type",
                    "perturbed_quads",
                    "--perturb_wiggle",
                    str(wiggle),
                    "--perturb_seed",
                    str(exp_spec["seed"]),
                    "--perturb_fix_boundary",
                    "1",
                    "--case_indices",
                    str(exp_spec["case_index"]),
                    exp_spec["num_arg"],
                    str(exp_spec["num_default"]),
                ]

                if args.dry_run:
                    print(" ".join(cmd))
                    continue

                if args.skip_existing and _run_exists(save_name, exp_spec["case_index"]):
                    print(
                        f"[SKIP] {exp_spec['name']} {exp_spec['algo']} "
                        f"N={int(round(resolution * 100))} wiggle={wiggle}"
                    )
                    continue

                log_path = log_dir / f"{save_name}.log"
                code = _run_subprocess(cmd, log_path)
                if code != 0:
                    raise RuntimeError(
                        f"{exp_spec['name']} {exp_spec['algo']} N={resolution} w={wiggle} failed; see {log_path}"
                    )

        if args.dry_run:
            continue

        out_path = summary_dir / f"{exp_spec['name']}_resolution_cartesian_vs_perturbed.png"
        _generate_figure(exp_spec, out_path)
        manifest["summary_plots"][exp_spec["name"]] = str(out_path)
        print(f"[summary] {exp_spec['name']}: {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
