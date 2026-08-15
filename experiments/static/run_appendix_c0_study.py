#!/usr/bin/env python3
"""
Run and plot the appendix C0 comparison study for selected static benchmarks.

Current study:
- ellipses:
  - Graph-coordinated linear
  - Graph-coordinated linear + joint C0
  - Graph-coordinated circular
- zalesak:
  - Graph-coordinated circular
  - Graph-coordinated circular + joint C0
  - Graph-coordinated circular+corner
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import mpmath as mp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import generate_section6_maintext_figures as maintext_figs
from experiments.static.figure_generation_provenance import (
    frozen_reconstruction_profile,
    generation_provenance,
    reconstruction_cli_args,
    vector_figure_artifacts,
)
from experiments.static import run_perturbed_sweeps as sweeps
from experiments.submission.conservation_analyzer import load_run_grid
from main.geoms.geoms import getArea
from util.metrics.area_metrics import facet_area_in_polygon


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
                "display": "Graph-coordinated linear",
                "facet_algo": "linear",
                "do_c0": False,
            },
            {
                "label": "linear+C0",
                "display": "Graph-coordinated linear + joint C0",
                "facet_algo": "linear",
                "do_c0": True,
            },
            {
                "label": "circular",
                "display": "Graph-coordinated circular",
                "facet_algo": "circular",
                "do_c0": False,
            },
        ],
        "representative": {
            "resolution": 0.32,
            "wiggle": 0.10,
            "seed": 0,
            "case_index": 9,
            "methods": [
                ("linear", "Graph-coordinated linear"),
                ("linear+C0", "Graph-coordinated linear + joint C0"),
                ("circular", "Graph-coordinated circular"),
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
                "display": "Graph-coordinated circular",
                "facet_algo": "circular",
                "do_c0": False,
            },
            {
                "label": "circular+C0",
                "display": "Graph-coordinated circular + joint C0",
                "facet_algo": "circular",
                "do_c0": True,
            },
            {
                "label": "circular+corner",
                "display": "Graph-coordinated circular+corner",
                "facet_algo": "circular+corner",
                "do_c0": False,
            },
        ],
        "representative": {
            "resolution": 1.00,
            "wiggle": 0.10,
            "seed": 0,
            "case_index": 22,
            "methods": [
                ("circular", "Graph-coordinated circular"),
                ("circular+C0", "Graph-coordinated circular + joint C0"),
                ("circular+corner", "Graph-coordinated circular+corner"),
            ],
            "min_span": 42.0,
            "margin_frac": 0.12,
            "inset": {"kind": "zalesak_corner", "zoom": 3.0},
        },
    },
]


DEFAULT_AUTHORITATIVE_GUARDED_ROOT = REPO_ROOT / (
    "results/submission/final_figures_87c40309d16c_20260803_final/"
    "provenance/guarded_c0"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "results/static/camera_ready/appendix_b5_joint_c0_20260814"
)
PUBLICATION_LABELS = {
    "linear": "Graph-coordinated linear",
    "linear+C0": "Graph-coordinated linear + {c0_mode} C0",
    "circular": "Graph-coordinated circular",
    "circular+C0": "Graph-coordinated circular + {c0_mode} C0",
    "circular+corner": "Graph-coordinated circular+corner",
}
REPRESENTATIVE_BASELINE_SLOTS = {
    "ellipses": {"linear": "00", "circular": "02"},
    "zalesak": {"circular": "00", "circular+corner": "02"},
}
PRIMARY_METRICS = ("hausdorff_median", "facet_gap_median")


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


def _publication_label(label: str, c0_mode: str) -> str:
    return PUBLICATION_LABELS.get(label, label).format(c0_mode=c0_mode)


def _representative_spec(exp_spec: Mapping[str, Any], c0_mode: str) -> dict:
    spec = dict(exp_spec["representative"])
    spec["methods"] = [
        (label, _publication_label(label, c0_mode))
        for label, _old_display in spec["methods"]
    ]
    return spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _float_or_nan(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer columns for empty CSV {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _selected_grid(
    exp_spec: Mapping[str, Any],
    resolutions_override: Sequence[float],
    wiggles_override: Sequence[float],
    seeds: Sequence[int],
) -> set[tuple[float, float, int]]:
    resolutions = resolutions_override or exp_spec["resolutions"]
    wiggles = wiggles_override or exp_spec["wiggles"]
    return {
        (float(resolution), float(wiggle), int(seed))
        for resolution in resolutions
        for wiggle in wiggles
        for seed in seeds
    }


def _load_authoritative_rows(
    exp_spec: Mapping[str, Any],
    root: Path,
    *,
    variants: Sequence[Mapping[str, Any]],
    grid: set[tuple[float, float, int]],
    c0_mode: str,
    include_c0: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = root / exp_spec["name"] / "metrics.csv"
    manifest_path = root / exp_spec["name"] / "manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"authoritative {exp_spec['name']} inputs missing below {root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {
        variant["label"] for variant in variants if include_c0 or not variant["do_c0"]
    }
    source_commit = manifest["generation_provenance"]["source_commit"]
    source_hash = _sha256_file(csv_path)
    rows = []
    for raw in _read_csv(csv_path):
        key = (float(raw["resolution"]), float(raw["wiggle"]), int(raw["seed"]))
        if raw["algo"] not in allowed or key not in grid:
            continue
        is_c0 = bool(int(raw["do_c0"]))
        rows.append(
            {
                **raw,
                "c0_mode": c0_mode if is_c0 else "none",
                "data_source": "authoritative_guarded_reuse",
                "source_path": str(csv_path.resolve()),
                "source_sha256": source_hash,
                "source_commit": source_commit,
            }
        )
    return rows, {
        "metrics": _file_record(csv_path),
        "manifest": _file_record(manifest_path),
        "source_commit": source_commit,
    }


def _variant_save_name(exp_name, label, resolution, wiggle, seed, prefix="appendix_c0"):
    base = sweeps._make_save_name(exp_name, label, resolution, wiggle, seed)
    return f"{prefix}_{base}"


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


def _generate_plots(
    csv_path: Path,
    out_dir: Path,
    save_prefix: str = "appendix_c0",
    endpoint_variants: str = "annotated",
    c0_mode: str = "joint",
    generate_representatives: bool = True,
):
    rows = _load_rows(csv_path)
    data = sweeps._build_metric_index(rows)

    summary_dir = out_dir / "summary_plots"
    representative_dir = out_dir / "representative_cases"
    summary_dir.mkdir(parents=True, exist_ok=True)
    representative_dir.mkdir(parents=True, exist_ok=True)

    outputs = {"summary": {}, "representative": {}}
    original_make_save_name = maintext_figs._make_save_name
    original_display_labels = dict(sweeps.DISPLAY_LABELS)
    sweeps.DISPLAY_LABELS.update(
        {label: _publication_label(label, c0_mode) for label in PUBLICATION_LABELS}
    )
    maintext_figs._make_save_name = (
        lambda exp_name, label, resolution, wiggle, seed: _variant_save_name(
            exp_name, label, resolution, wiggle, seed, prefix=save_prefix
        )
    )
    try:
        for exp_spec in APPENDIX_EXPERIMENTS:
            exp_name = exp_spec["name"]
            exp_data = data.get(exp_name, {})
            if not exp_data:
                continue
            methods = [
                variant["label"]
                for variant in exp_spec["variants"]
                if variant["label"] in exp_data
            ]
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
            outputs["summary"][exp_name] = vector_figure_artifacts(metric_out)

            if not generate_representatives:
                continue
            rep_spec = _representative_spec(exp_spec, c0_mode)
            variant_outputs = {}
            for (
                variant_name,
                suffix,
                show_main_endpoints,
            ) in maintext_figs._endpoint_variant_specs(endpoint_variants):
                rep_out = (
                    representative_dir
                    / f"{exp_name}_appendix_c0_representative{suffix}.png"
                )
                try:
                    maintext_figs._generate_representative_figure(
                        exp_name=exp_name,
                        spec=maintext_figs._endpoint_visibility_spec(
                            rep_spec,
                            show_main_endpoints=show_main_endpoints,
                        ),
                        out_path=rep_out,
                    )
                    variant_outputs[variant_name] = vector_figure_artifacts(rep_out)
                except FileNotFoundError:
                    print(
                        f"[WARN] skipping representative figure for {exp_name}: "
                        f"representative case {rep_spec['case_index']} artifacts "
                        "not present in this run set"
                    )
                    variant_outputs = {}
                    break
            if variant_outputs:
                outputs["representative"][exp_name] = variant_outputs
    finally:
        maintext_figs._make_save_name = original_make_save_name
        sweeps.DISPLAY_LABELS.clear()
        sweeps.DISPLAY_LABELS.update(original_display_labels)

    return outputs


def _print_outputs(outputs):
    for exp_name, artifacts in outputs["summary"].items():
        print(f"[summary] {exp_name}: {artifacts['pdf']}")
    for exp_name, variants in outputs["representative"].items():
        for variant_name, artifacts in variants.items():
            print(f"[representative:{variant_name}] {exp_name}: {artifacts['pdf']}")


def _write_manifest(path: Path, manifest: dict):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _run_subprocess(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def _build_command(
    exp_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    resolution: float,
    wiggle: float,
    seed: int,
    num_cases: int,
    save_name: str,
    c0_mode: str,
    case_indices: str | None,
    reconstruction_profile: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        str(exp_spec["module"]),
        "--config",
        str(exp_spec["config"]),
        "--resolution",
        str(resolution),
        "--facet_algo",
        str(variant["facet_algo"]),
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
        "--c0_mode",
        c0_mode,
        str(exp_spec["num_arg"]),
        str(num_cases),
    ]
    command.extend(
        reconstruction_cli_args(str(exp_spec["name"]), reconstruction_profile)
    )
    if case_indices is not None:
        command += ["--case_indices", case_indices]
    return command


def _manifest_case_indices(raw: Any) -> tuple[int, ...]:
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        return tuple(int(value) for value in raw.split(",") if value.strip())
    return tuple(int(value) for value in raw)


def _validate_run_manifest(
    run_dir: Path,
    job: Mapping[str, Any],
    *,
    case_indices: str | None,
    c0_mode: str,
) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    expected = {
        "facet_algo": job["facet_algo"],
        "do_c0": bool(job["do_c0"]),
        "mesh_type": "perturbed_quads",
        "perturb_wiggle": float(job["wiggle"]),
        "perturb_seed": int(job["seed"]),
        "perturb_fix_boundary": 1,
        "c0_mode": c0_mode,
        "plic_fallback": "LVIRA",
        "corner_behavior_profile": "pre_f8_corner",
    }
    if job["experiment"] == "zalesak":
        expected["rescue_profile"] = "exact_linear_support_only"
    for field, expected_value in expected.items():
        observed = parameters.get(field)
        if isinstance(expected_value, float):
            if not math.isclose(float(observed), expected_value, abs_tol=1.0e-15):
                raise ValueError(
                    f"{run_dir.name}: {field}={observed!r}, expected {expected_value!r}"
                )
        elif observed != expected_value:
            raise ValueError(
                f"{run_dir.name}: {field}={observed!r}, expected {expected_value!r}"
            )
    if not math.isclose(
        float(parameters["resolution"]),
        float(job["resolution"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{run_dir.name}: resolution mismatch")
    expected_indices = _manifest_case_indices(case_indices)
    if (
        expected_indices
        and _manifest_case_indices(parameters.get("case_indices")) != expected_indices
    ):
        raise ValueError(f"{run_dir.name}: case-index subset mismatch")
    case_metrics_path = run_dir / "metrics" / "case_metrics.csv"
    if not case_metrics_path.is_file():
        raise ValueError(f"{run_dir.name}: case_metrics.csv is missing")
    observed_indices = tuple(
        sorted(int(row["case_index"]) for row in _read_csv(case_metrics_path))
    )
    expected_output_indices = (
        expected_indices if expected_indices else tuple(range(int(job["num_cases"])))
    )
    if observed_indices != expected_output_indices:
        raise ValueError(
            f"{run_dir.name}: incomplete case support {observed_indices}; "
            f"expected {expected_output_indices}"
        )
    return payload


def _run_job(
    job: Mapping[str, Any],
    *,
    log_dir: Path,
    reuse_existing: bool,
    case_indices: str | None,
    c0_mode: str,
) -> dict[str, Any]:
    run_dir = REPO_ROOT / "plots" / str(job["save_name"])
    log_path = log_dir / f"{job['save_name']}.log"
    if reuse_existing and (run_dir / "run_manifest.json").is_file():
        try:
            _validate_run_manifest(
                run_dir, job, case_indices=case_indices, c0_mode=c0_mode
            )
        except (FileNotFoundError, ValueError):
            shutil.rmtree(run_dir)
        else:
            return {
                **job,
                "status": "reused_generated_run",
                "run_dir": str(run_dir),
            }
    code = _run_subprocess(list(job["command"]), log_path)
    if code != 0:
        raise RuntimeError(f"{job['save_name']} failed; inspect {log_path}")
    _validate_run_manifest(run_dir, job, case_indices=case_indices, c0_mode=c0_mode)
    return {
        **job,
        "status": "completed",
        "run_dir": str(run_dir),
        "log": str(log_path),
    }


def _run_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    log_dir: Path,
    workers: int,
    reuse_existing: bool,
    case_indices: str | None,
    c0_mode: str,
) -> list[dict[str, Any]]:
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                log_dir=log_dir,
                reuse_existing=reuse_existing,
                case_indices=case_indices,
                c0_mode=c0_mode,
            ): job
            for job in jobs
        }
        for count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            completed.append(record)
            print(
                f"C0 reconstruction {count}/{len(jobs)} complete: "
                f"{record['save_name']}",
                flush=True,
            )
    return sorted(
        completed,
        key=lambda row: (
            row["experiment"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
        ),
    )


def _group_csv_rows(path: Path, key: str) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in _read_csv(path):
        grouped.setdefault(int(row[key]), []).append(row)
    return grouped


def _disk_polygon_intersection_area(
    polygon: Sequence[Sequence[float]], center: Sequence[float], radius: float
) -> float:
    """Integrate the exact disk area cut by a simple polygon edge by edge."""

    def edge_contribution(first: Sequence[float], second: Sequence[float]) -> float:
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        quadratic = dx * dx + dy * dy
        linear = 2.0 * (first[0] * dx + first[1] * dy)
        constant = first[0] ** 2 + first[1] ** 2 - radius**2
        samples = [0.0, 1.0]
        discriminant = linear * linear - 4.0 * quadratic * constant
        if quadratic > 0.0 and discriminant > 0.0:
            root = math.sqrt(discriminant)
            samples.extend(
                value
                for value in (
                    (-linear - root) / (2.0 * quadratic),
                    (-linear + root) / (2.0 * quadratic),
                )
                if 0.0 < value < 1.0
            )
        samples.sort()

        contribution = 0.0
        for left, right in zip(samples, samples[1:]):
            p = (first[0] + left * dx, first[1] + left * dy)
            q = (first[0] + right * dx, first[1] + right * dy)
            midpoint = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
            cross = p[0] * q[1] - p[1] * q[0]
            if midpoint[0] ** 2 + midpoint[1] ** 2 <= radius**2 * (1.0 + 1.0e-14):
                contribution += cross / 2.0
            else:
                dot = p[0] * q[0] + p[1] * q[1]
                contribution += radius**2 * math.atan2(cross, dot) / 2.0
        return contribution

    shifted = [
        (float(point[0]) - center[0], float(point[1]) - center[1]) for point in polygon
    ]
    return abs(
        math.fsum(
            edge_contribution(shifted[index], shifted[(index + 1) % len(shifted)])
            for index in range(len(shifted))
        )
    )


def _disk_polygon_intersection_area_high_precision(
    polygon: Sequence[Sequence[float]], center: Sequence[float], radius: float
) -> float:
    """High-precision counterpart for nearly linear, very-large-radius arcs."""

    with mp.workdps(60):
        radius_mp = mp.mpf(str(radius))
        center_mp = tuple(mp.mpf(str(value)) for value in center)
        shifted = [
            (
                mp.mpf(str(point[0])) - center_mp[0],
                mp.mpf(str(point[1])) - center_mp[1],
            )
            for point in polygon
        ]

        def edge_contribution(first, second):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            quadratic = dx * dx + dy * dy
            linear = 2 * (first[0] * dx + first[1] * dy)
            constant = first[0] ** 2 + first[1] ** 2 - radius_mp**2
            samples = [mp.mpf("0"), mp.mpf("1")]
            discriminant = linear * linear - 4 * quadratic * constant
            if quadratic > 0 and discriminant > 0:
                root = mp.sqrt(discriminant)
                samples.extend(
                    value
                    for value in (
                        (-linear - root) / (2 * quadratic),
                        (-linear + root) / (2 * quadratic),
                    )
                    if 0 < value < 1
                )
            samples.sort()

            contribution = mp.mpf("0")
            for left, right in zip(samples, samples[1:]):
                p = (first[0] + left * dx, first[1] + left * dy)
                q = (first[0] + right * dx, first[1] + right * dy)
                midpoint = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                cross = p[0] * q[1] - p[1] * q[0]
                if midpoint[0] ** 2 + midpoint[1] ** 2 <= radius_mp**2:
                    contribution += cross / 2
                else:
                    dot = p[0] * q[0] + p[1] * q[1]
                    contribution += radius_mp**2 * mp.atan2(cross, dot) / 2
            return contribution

        area = mp.fsum(
            edge_contribution(shifted[index], shifted[(index + 1) % len(shifted)])
            for index in range(len(shifted))
        )
        return float(abs(area))


def _robust_facet_area(
    polygon: Sequence[Sequence[float]], geometry: Mapping[str, Any]
) -> float:
    if geometry.get("class") != "circular":
        return facet_area_in_polygon(polygon, geometry)
    polygon_area = abs(getArea(polygon))
    radius = abs(float(geometry["radius"]))
    if radius > 1.0e4 * math.sqrt(polygon_area):
        disk_area = _disk_polygon_intersection_area_high_precision(
            polygon, geometry["center"], radius
        )
    else:
        disk_area = _disk_polygon_intersection_area(polygon, geometry["center"], radius)
    area = disk_area if float(geometry["radius"]) > 0.0 else polygon_area - disk_area
    tolerance = 1.0e-10 * max(1.0, polygon_area)
    if area < -tolerance or area > polygon_area + tolerance:
        raise ValueError(
            f"analytic circular area {area} lies outside [0, {polygon_area}]"
        )
    return min(max(area, 0.0), polygon_area)


def _fallback_normalized_conservation(
    grid: Any, rows: Sequence[Mapping[str, Any]]
) -> tuple[float, float]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["merge_id"]), []).append(row)

    relative = []
    signed_global = []
    total_cell_area = 0.0
    for merge_id, members in groups.items():
        geometries = {
            json.dumps(json.loads(str(row["facet_geometry_json"])), sort_keys=True)
            for row in members
        }
        if len(geometries) != 1:
            raise ValueError(f"merge_id={merge_id}: inconsistent final facets")
        geometry = json.loads(next(iter(geometries)))
        if geometry is None:
            raise ValueError(f"merge_id={merge_id}: missing final facet")

        component_area = 0.0
        component_residuals = []
        for row in members:
            x = int(row.get("cell_x", str(row["cell_id"]).split(",")[0]))
            y = int(row.get("cell_y", str(row["cell_id"]).split(",")[1]))
            polygon = grid.cell_polygon(x, y)
            cell_area = abs(getArea(polygon))
            target = float(row["cell_fraction"]) * cell_area
            component_residuals.append(_robust_facet_area(polygon, geometry) - target)
            component_area += cell_area
        residual = math.fsum(component_residuals)
        relative.append(abs(residual) / component_area)
        signed_global.append(residual)
        total_cell_area += component_area
    return max(relative, default=0.0), abs(math.fsum(signed_global)) / total_cell_area


def _normalized_conservation(
    grid: Any, rows: Sequence[Mapping[str, Any]], *, stage: str
) -> tuple[float, float, str]:
    del stage
    conservation, global_conservation = _fallback_normalized_conservation(grid, rows)
    return conservation, global_conservation, "partition_invariant_analytic"


def _collect_generated_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    c0_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    qa_rows = []
    for record in records:
        run_dir = Path(str(record["run_dir"]))
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        metrics = sweeps._collect_metrics(
            str(record["experiment"]), str(record["save_name"])
        )
        if not metrics:
            raise ValueError(f"no aggregate metrics found for {record['save_name']}")
        for key, value in sorted(metrics.items()):
            metric_rows.append(
                {
                    "experiment": record["experiment"],
                    "algo": record["variant"],
                    "facet_algo": record["facet_algo"],
                    "do_c0": int(bool(record["do_c0"])),
                    "c0_mode": c0_mode if record["do_c0"] else "none",
                    "resolution": record["resolution"],
                    "wiggle": record["wiggle"],
                    "seed": record["seed"],
                    "metric_key": key,
                    "metric_value": value,
                    "save_name": record["save_name"],
                    "data_source": "generated",
                    "source_path": str(run_dir.resolve()),
                    "source_sha256": _sha256_file(run_dir / "run_manifest.json"),
                    "source_commit": manifest["source_commit"],
                }
            )

        if not record["do_c0"]:
            continue
        case_metrics = {
            int(row["case_index"]): row
            for row in _read_csv(run_dir / "metrics" / "case_metrics.csv")
        }
        cell_rows = _group_csv_rows(
            run_dir / "metrics" / "cell_metrics.csv", "case_index"
        )
        grid = load_run_grid(run_dir, repo_root=REPO_ROOT)
        if set(case_metrics) != set(cell_rows):
            raise ValueError(
                f"{record['save_name']}: case/cell diagnostic support differs"
            )
        for case_index in sorted(case_metrics):
            case = case_metrics[case_index]
            conservation, global_conservation, conservation_evaluator = (
                _normalized_conservation(grid, cell_rows[case_index], stage="after_c0")
            )
            qa_rows.append(
                {
                    "experiment": record["experiment"],
                    "variant": record["variant"],
                    "c0_mode": case.get("c0_mode") or c0_mode,
                    "resolution": record["resolution"],
                    "wiggle": record["wiggle"],
                    "seed": record["seed"],
                    "case_index": case_index,
                    "source_run": record["save_name"],
                    "num_mixed_cells": int(case["num_mixed_cells"]),
                    "num_final_missing_cells": int(case["num_final_missing_cells"]),
                    "num_c0_eligible_joins": int(
                        case.get("num_c0_eligible_joins") or 0
                    ),
                    "num_c0_bad_joins_before_joint": int(
                        case.get("num_c0_bad_joins_before_joint") or 0
                    ),
                    "num_c0_bad_joins_after_joint": int(
                        case.get("num_c0_bad_joins_after_joint") or 0
                    ),
                    "num_c0_joint_components": int(
                        case.get("num_c0_joint_components") or 0
                    ),
                    "num_c0_joint_components_solved": int(
                        case.get("num_c0_joint_components_solved") or 0
                    ),
                    "num_c0_joint_components_failed": int(
                        case.get("num_c0_joint_components_failed") or 0
                    ),
                    "num_c0_exact_c1_components": int(
                        case.get("num_c0_exact_c1_components") or 0
                    ),
                    "num_c0_conservative_fallback_components": int(
                        case.get("num_c0_conservative_fallback_components") or 0
                    ),
                    "max_c0_relative_area_residual": _float_or_nan(
                        case.get("max_c0_relative_area_residual")
                    ),
                    "max_c0_tangent_angle_radians": _float_or_nan(
                        case.get("max_c0_tangent_angle_radians")
                    ),
                    "independent_max_relative_conservation_residual": conservation,
                    "independent_global_relative_conservation_residual": global_conservation,
                    "independent_conservation_evaluator": conservation_evaluator,
                    "hausdorff": _float_or_nan(case.get("hausdorff")),
                    "facet_gap": _float_or_nan(case.get("facet_gap")),
                }
            )
    return metric_rows, qa_rows


def _compare_guarded_joint(
    joint_rows: Sequence[Mapping[str, Any]],
    guarded_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def key(row):
        return (
            row["experiment"],
            row["algo"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
            row["metric_key"],
        )

    guarded = {key(row): row for row in guarded_rows if int(row["do_c0"])}
    comparison = []
    for row in joint_rows:
        if not int(row["do_c0"]):
            continue
        match = guarded.get(key(row))
        if match is None:
            raise ValueError(f"missing guarded comparison for {key(row)}")
        joint_value = float(row["metric_value"])
        guarded_value = float(match["metric_value"])
        ratio = (
            joint_value / guarded_value
            if guarded_value != 0.0
            else 1.0 if joint_value == 0.0 else math.inf
        )
        comparison.append(
            {
                "experiment": row["experiment"],
                "algo": row["algo"],
                "resolution": row["resolution"],
                "wiggle": row["wiggle"],
                "seed": row["seed"],
                "metric_key": row["metric_key"],
                "guarded_value": guarded_value,
                "joint_value": joint_value,
                "joint_minus_guarded": joint_value - guarded_value,
                "joint_over_guarded": ratio,
            }
        )

    summary = []
    for experiment in sorted({row["experiment"] for row in comparison}):
        for metric_key in PRIMARY_METRICS:
            selected = [
                row
                for row in comparison
                if row["experiment"] == experiment and row["metric_key"] == metric_key
            ]
            if not selected:
                continue
            guarded_values = np.asarray([row["guarded_value"] for row in selected])
            joint_values = np.asarray([row["joint_value"] for row in selected])
            tolerance = 1.0e-14 * np.maximum(
                np.maximum(np.abs(guarded_values), np.abs(joint_values)), 1.0
            )
            summary.append(
                {
                    "experiment": experiment,
                    "metric_key": metric_key,
                    "matched_settings": len(selected),
                    "guarded_median_of_subrun_medians": float(
                        np.median(guarded_values)
                    ),
                    "joint_median_of_subrun_medians": float(np.median(joint_values)),
                    "joint_better_settings": int(
                        np.sum(joint_values < guarded_values - tolerance)
                    ),
                    "equal_settings": int(
                        np.sum(np.abs(joint_values - guarded_values) <= tolerance)
                    ),
                    "joint_worse_settings": int(
                        np.sum(joint_values > guarded_values + tolerance)
                    ),
                }
            )
    return comparison, summary


def _qa_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("joint C0 QA table is empty")

    def finite_values(field: str) -> list[float]:
        return [float(row[field]) for row in rows if math.isfinite(float(row[field]))]

    summary = {
        "case_rows": len(rows),
        "experiments": sorted({str(row["experiment"]) for row in rows}),
        "c0_modes": sorted({str(row["c0_mode"]) for row in rows}),
        "independent_conservation_evaluators": sorted(
            {
                str(row["independent_conservation_evaluator"])
                for row in rows
                if row.get("independent_conservation_evaluator")
            }
        ),
        "joint_components": sum(int(row["num_c0_joint_components"]) for row in rows),
        "joint_components_solved": sum(
            int(row["num_c0_joint_components_solved"]) for row in rows
        ),
        "joint_components_failed": sum(
            int(row["num_c0_joint_components_failed"]) for row in rows
        ),
        "exact_c1_components": sum(
            int(row["num_c0_exact_c1_components"]) for row in rows
        ),
        "conservative_fallback_components": sum(
            int(row["num_c0_conservative_fallback_components"]) for row in rows
        ),
        "bad_joins_before_joint": sum(
            int(row["num_c0_bad_joins_before_joint"]) for row in rows
        ),
        "bad_joins_after_joint": sum(
            int(row["num_c0_bad_joins_after_joint"]) for row in rows
        ),
        "final_missing_cells": sum(int(row["num_final_missing_cells"]) for row in rows),
        "cases_with_remaining_bad_joins": sum(
            int(row["num_c0_bad_joins_after_joint"]) > 0 for row in rows
        ),
        "cases_with_failed_components": sum(
            int(row["num_c0_joint_components_failed"]) > 0 for row in rows
        ),
        "max_reported_c0_relative_area_residual": max(
            finite_values("max_c0_relative_area_residual"), default=0.0
        ),
        "max_independent_relative_conservation_residual": max(
            finite_values("independent_max_relative_conservation_residual"), default=0.0
        ),
        "max_independent_global_conservation_residual": max(
            finite_values("independent_global_relative_conservation_residual"),
            default=0.0,
        ),
        "max_facet_gap": max(finite_values("facet_gap"), default=0.0),
        "nonfinite_hausdorff_rows": sum(
            not math.isfinite(float(row["hausdorff"])) for row in rows
        ),
        "nonfinite_facet_gap_rows": sum(
            not math.isfinite(float(row["facet_gap"])) for row in rows
        ),
    }
    if summary["c0_modes"] != ["joint"]:
        raise ValueError(f"unexpected C0 mode in QA rows: {summary['c0_modes']}")
    if summary["nonfinite_hausdorff_rows"] or summary["nonfinite_facet_gap_rows"]:
        raise ValueError("non-finite primary metrics in joint C0 QA table")
    return summary


def _copy_representative_input(source: Path, destination: Path, case_index: int):
    if destination.exists():
        shutil.rmtree(destination)
    files = (
        Path("vtk/mesh.vtk"),
        Path("metrics/case_geometry.jsonl"),
        Path(f"vtk/reconstructed/facets/{case_index}.vtp"),
        Path(f"vtk/reconstructed/facets/{case_index}.facet_metadata.json"),
    )
    for relative in files:
        source_file = source / relative
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def _stage_representative_inputs(
    out_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    authoritative_root: Path,
    save_prefix: str,
) -> Path:
    staging = out_dir / "representative_inputs"
    generated_by_key = {
        (
            row["experiment"],
            row["variant"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
        ): Path(str(row["run_dir"]))
        for row in records
    }
    for exp_spec in APPENDIX_EXPERIMENTS:
        representative = exp_spec["representative"]
        for variant in exp_spec["variants"]:
            label = variant["label"]
            save_name = _variant_save_name(
                exp_spec["name"],
                label,
                representative["resolution"],
                representative["wiggle"],
                representative["seed"],
                prefix=save_prefix,
            )
            if variant["do_c0"]:
                source = generated_by_key[
                    (
                        exp_spec["name"],
                        label,
                        float(representative["resolution"]),
                        float(representative["wiggle"]),
                        int(representative["seed"]),
                    )
                ]
            else:
                slot = REPRESENTATIVE_BASELINE_SLOTS[exp_spec["name"]][label]
                source = (
                    authoritative_root
                    / exp_spec["name"]
                    / "representative_inputs"
                    / slot
                )
            _copy_representative_input(
                source, staging / save_name, int(representative["case_index"])
            )
    return staging


def _write_sha256sums(root: Path) -> Path:
    path = root / "SHA256SUMS"
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name not in {"SHA256SUMS", "manifest.json"}
    )
    path.write_text(
        "".join(
            f"{_sha256_file(candidate)}  {candidate.relative_to(root)}\n"
            for candidate in files
        ),
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run appendix C0 static comparison sweeps."
    )
    parser.add_argument(
        "--only", type=str, default=None, help="comma-separated experiments to run"
    )
    parser.add_argument(
        "--algos", type=str, default=None, help="comma-separated variant labels to run"
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="comma-separated resolutions override",
    )
    parser.add_argument(
        "--wiggles", type=str, default=None, help="comma-separated wiggles override"
    )
    parser.add_argument("--seeds", type=str, default="0", help="comma-separated seeds")
    parser.add_argument(
        "--ellipses", type=int, default=25, help="number of ellipse cases"
    )
    parser.add_argument(
        "--zalesak", type=int, default=25, help="number of Zalesak cases"
    )
    parser.add_argument("--out_csv", type=str, default=None, help="output CSV path")
    parser.add_argument(
        "--out_dir", type=str, default=None, help="output artifact directory"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="log directory")
    parser.add_argument(
        "--c0_mode",
        choices=("joint", "guarded"),
        default="joint",
        help="C0 implementation used by generated C0 variants",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="maximum concurrently running reconstruction subprocesses",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="reuse exact generated runs after manifest validation",
    )
    parser.add_argument(
        "--recompute_no_c0",
        action="store_true",
        help="recompute unchanged no-C0 variants instead of importing sealed rows",
    )
    parser.add_argument(
        "--authoritative_guarded_root",
        type=Path,
        default=DEFAULT_AUTHORITATIVE_GUARDED_ROOT,
        help="sealed guarded-C0 package used for no-C0 rows and comparison",
    )
    parser.add_argument(
        "--plot_from_csv",
        type=str,
        default=None,
        help="generate plots only from an existing CSV",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="appendix_b5_joint_c0_20260814",
        help="prefix for reconstruction directories",
    )
    parser.add_argument(
        "--endpoint_variants",
        choices=sorted(maintext_figs.ENDPOINT_VARIANT_MODES),
        default="annotated",
        help=(
            "Qualitative endpoint-marker exports: annotated, clean main panels, "
            "or paired. Spyglass endpoint labels are always retained."
        ),
    )
    parser.add_argument(
        "--plots_root",
        type=Path,
        default=maintext_figs.PLOTS_ROOT,
        help="root containing saved per-run plot artifacts",
    )
    parser.add_argument(
        "--case_indices",
        type=str,
        default=None,
        help="comma-separated deterministic case indices to run",
    )
    parser.add_argument(
        "--skip_representatives",
        action="store_true",
        help="skip staging and rendering qualitative representative panels",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="print commands without executing"
    )
    args = parser.parse_args()
    started = time.monotonic()
    out_dir = Path(args.out_dir or DEFAULT_OUTPUT_ROOT).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_profile = frozen_reconstruction_profile()
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "status": "planned" if args.dry_run else "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_provenance": generation_provenance(
            profile=reconstruction_profile,
            profile_application=(
                "explicitly_applied_to_generated_C0_runs; unchanged no-C0 rows "
                "imported from a hash-pinned authoritative package"
            ),
        ),
        "parameters": {
            "only": args.only,
            "algos": args.algos,
            "resolutions": args.resolutions,
            "wiggles": args.wiggles,
            "seeds": args.seeds,
            "case_indices": args.case_indices,
            "save_prefix": args.save_prefix,
            "endpoint_variants": args.endpoint_variants,
            "c0_mode": args.c0_mode,
            "workers": args.workers,
            "reuse_existing": args.reuse_existing,
            "recompute_no_c0": args.recompute_no_c0,
            "authoritative_guarded_root": str(
                args.authoritative_guarded_root.resolve()
            ),
        },
        "inputs": {},
        "runs": [],
        "outputs": {},
    }
    _write_manifest(manifest_path, manifest)

    if args.plot_from_csv:
        csv_path = Path(args.plot_from_csv).resolve()
        outputs = _generate_plots(
            csv_path,
            out_dir,
            save_prefix=args.save_prefix,
            endpoint_variants=args.endpoint_variants,
            c0_mode=args.c0_mode,
            generate_representatives=not args.skip_representatives,
        )
        manifest["status"] = "completed"
        manifest["inputs"]["aggregate_metrics"] = _file_record(csv_path)
        manifest["outputs"] = outputs
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        print(f"Generated appendix C0 plots from {csv_path}")
        _print_outputs(outputs)
        print(f"Manifest: {manifest_path}")
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
        "c0_mode",
        "resolution",
        "wiggle",
        "seed",
        "metric_key",
        "metric_value",
        "save_name",
        "data_source",
        "source_path",
        "source_sha256",
        "source_commit",
    ]
    jobs = []
    reused_rows = []
    guarded_rows = []
    authoritative_inputs = {}
    selected_experiments = _selected_experiments(args.only)
    for exp_spec in selected_experiments:
        variants = _selected_variants(exp_spec, args.algos)
        if not variants:
            continue
        grid = _selected_grid(exp_spec, resolutions_override, wiggles_override, seeds)
        historical, historical_inputs = _load_authoritative_rows(
            exp_spec,
            args.authoritative_guarded_root.resolve(),
            variants=variants,
            grid=grid,
            c0_mode="guarded",
            include_c0=True,
        )
        guarded_rows.extend(historical)
        authoritative_inputs[exp_spec["name"]] = historical_inputs
        if not args.recompute_no_c0:
            reused_rows.extend(row for row in historical if not int(row["do_c0"]))

        for resolution, wiggle, seed in sorted(grid):
            for variant in variants:
                if not args.recompute_no_c0 and not variant["do_c0"]:
                    continue
                save_name = _variant_save_name(
                    exp_spec["name"],
                    variant["label"],
                    resolution,
                    wiggle,
                    seed,
                    prefix=args.save_prefix,
                )
                command = _build_command(
                    exp_spec,
                    variant,
                    resolution=resolution,
                    wiggle=wiggle,
                    seed=seed,
                    num_cases=getattr(args, exp_spec["name"]),
                    save_name=save_name,
                    c0_mode=args.c0_mode,
                    case_indices=args.case_indices,
                    reconstruction_profile=reconstruction_profile,
                )
                job = {
                    "experiment": exp_spec["name"],
                    "variant": variant["label"],
                    "facet_algo": variant["facet_algo"],
                    "do_c0": bool(variant["do_c0"]),
                    "c0_mode": args.c0_mode,
                    "resolution": resolution,
                    "wiggle": wiggle,
                    "seed": seed,
                    "num_cases": getattr(args, exp_spec["name"]),
                    "save_name": save_name,
                    "command": command,
                    "status": "planned",
                }
                jobs.append(job)
                manifest["runs"].append(job)

    manifest["inputs"]["authoritative_guarded"] = authoritative_inputs
    _write_manifest(manifest_path, manifest)
    if args.dry_run:
        for job in jobs:
            print(" ".join(job["command"]))
        print(
            f"Planned {len(jobs)} generated runs and {len(reused_rows)} reused metric rows"
        )
        return

    try:
        completed_records = _run_jobs(
            jobs,
            log_dir=log_dir,
            workers=args.workers,
            reuse_existing=args.reuse_existing,
            case_indices=args.case_indices,
            c0_mode=args.c0_mode,
        )
        manifest["runs"] = completed_records
        generated_rows, qa_rows = _collect_generated_rows(
            completed_records, c0_mode=args.c0_mode
        )
        primary_rows = sorted(
            [*reused_rows, *generated_rows],
            key=lambda row: (
                row["experiment"],
                row["algo"],
                float(row["resolution"]),
                float(row["wiggle"]),
                int(row["seed"]),
                row["metric_key"],
            ),
        )
        _write_csv(out_csv, primary_rows, fieldnames=fieldnames)
        qa_path = out_dir / "csv" / "joint_c0_case_qa.csv"
        _write_csv(qa_path, qa_rows)
        qa_summary = _qa_summary(qa_rows) if args.c0_mode == "joint" else {}
        qa_summary_path = out_dir / "qa_summary.json"
        qa_summary_path.write_text(
            json.dumps(qa_summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        comparison_rows, comparison_summary = _compare_guarded_joint(
            generated_rows, guarded_rows
        )
        comparison_path = out_dir / "csv" / "joint_vs_guarded.csv"
        comparison_summary_path = out_dir / "csv" / "joint_vs_guarded_summary.csv"
        _write_csv(comparison_path, comparison_rows)
        _write_csv(comparison_summary_path, comparison_summary)

        if not args.skip_representatives and not args.recompute_no_c0:
            staged_root = _stage_representative_inputs(
                out_dir,
                records=completed_records,
                authoritative_root=args.authoritative_guarded_root.resolve(),
                save_prefix=args.save_prefix,
            )
            maintext_figs.PLOTS_ROOT = staged_root
        else:
            maintext_figs.PLOTS_ROOT = args.plots_root.resolve()

        outputs = _generate_plots(
            out_csv,
            out_dir,
            save_prefix=args.save_prefix,
            endpoint_variants=args.endpoint_variants,
            c0_mode=args.c0_mode,
            generate_representatives=not args.skip_representatives,
        )
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["qa"] = qa_summary
        manifest["outputs"] = {
            **outputs,
            "aggregate_metrics": _file_record(out_csv),
            "case_qa": _file_record(qa_path),
            "qa_summary": _file_record(qa_summary_path),
            "joint_vs_guarded": _file_record(comparison_path),
            "joint_vs_guarded_summary": _file_record(comparison_summary_path),
        }
        _write_manifest(manifest_path, manifest)
        sums_path = _write_sha256sums(out_dir)
        manifest["outputs"]["sha256sums"] = _file_record(sums_path)
        _write_manifest(manifest_path, manifest)
    except Exception:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["runtime_seconds"] = time.monotonic() - started
        _write_manifest(manifest_path, manifest)
        raise

    print(f"Appendix C0 sweep CSV: {out_csv}")
    _print_outputs(outputs)
    print(f"Joint/guarded comparison: {comparison_summary_path}")
    print(f"Case QA: {qa_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
