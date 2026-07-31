#!/usr/bin/env python3
"""Audit square and Zalesak perfect-reconstruction evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CRITICAL_METHODS = {
    "squares": "linear+corner",
    "zalesak": "circular+corner",
}
KEY_COLUMNS = [
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "case_index",
]
METRICS = ["hausdorff", "facet_gap", "area_error"]
THRESHOLDS = [1e-6, 1e-8, 1e-10]
EXPECTED_RESOLUTIONS = [50, 64, 100, 128, 150]
EXPECTED_WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-release", type=Path, required=True)
    parser.add_argument("--july-simplified", type=Path, required=True)
    parser.add_argument("--july-complex", type=Path, required=True)
    parser.add_argument("--invalid-release", type=Path, required=True)
    parser.add_argument("--keep-all-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Return a checkout-relative provenance path when a known root is present."""
    resolved = path.resolve()
    for marker in ("results", "submission"):
        if marker in resolved.parts:
            index = resolved.parts.index(marker)
            return Path(*resolved.parts[index:]).as_posix()
    return path.name


def read_cases(root: Path) -> pd.DataFrame:
    path = root / "diagnostics" / "case_metrics.csv"
    data = pd.read_csv(path, low_memory=False)
    data["N"] = np.rint(data["resolution"].astype(float) * 100).astype(int)
    return data


def critical_rows(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for experiment, algo in CRITICAL_METHODS.items():
        pieces.append(
            data.loc[data["experiment"].eq(experiment) & data["algo"].eq(algo)].copy()
        )
    return pd.concat(pieces, ignore_index=True)


def assert_final_contract(final_root: Path, data: pd.DataFrame) -> None:
    manifest = json.loads((final_root / "sweep_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["planned_run_count"] == 970
    assert manifest["successful_run_count"] == 970
    assert manifest["failure_count"] == 0
    assert manifest["planned_case_count"] == 24250
    assert len(data) == 24250

    rows = critical_rows(data)
    for experiment, algo in CRITICAL_METHODS.items():
        selected = rows.loc[rows["experiment"].eq(experiment) & rows["algo"].eq(algo)]
        assert len(selected) == 625
        assert sorted(selected["N"].unique()) == EXPECTED_RESOLUTIONS
        assert sorted(selected["wiggle"].unique()) == EXPECTED_WIGGLES
        assert selected["case_index"].nunique() == 25
        assert np.isfinite(selected[METRICS].to_numpy(dtype=float)).all()
        assert int(selected["num_final_missing_cells"].sum()) == 0


def summarize_group(group: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {"case_count": len(group)}
    for metric in METRICS:
        values = group[metric].to_numpy(dtype=float)
        result[f"finite_{metric}_rows"] = int(np.isfinite(values).sum())
        result[f"{metric}_median"] = float(np.median(values))
        result[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        result[f"{metric}_max"] = float(np.max(values))

    for threshold in THRESHOLDS:
        label = f"{threshold:.0e}"
        h_floor = group["hausdorff"].le(threshold)
        hg_floor = h_floor & group["facet_gap"].le(threshold)
        hga_floor = hg_floor & group["area_error"].le(threshold)
        result[f"hausdorff_at_most_{label}"] = int(h_floor.sum())
        result[f"joint_hg_at_most_{label}"] = int(hg_floor.sum())
        result[f"joint_hga_at_most_{label}"] = int(hga_floor.sum())

    result["hausdorff_above_1e-02"] = int(group["hausdorff"].gt(1e-2).sum())
    result["hausdorff_above_1"] = int(group["hausdorff"].gt(1.0).sum())
    result["plic_fallback_cases"] = int(group["num_plic_fallback_cells"].gt(0).sum())
    result["plic_fallback_cells"] = int(group["num_plic_fallback_cells"].sum())
    result["final_missing_cells"] = int(group["num_final_missing_cells"].sum())
    return result


def make_setting_summary(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = final.groupby(["experiment", "algo", "N", "wiggle"], sort=True)
    for keys, group in grouped:
        row = dict(zip(["experiment", "algo", "N", "wiggle"], keys))
        row.update(summarize_group(group))
        rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 50
    return result


def make_resolution_summary(
    final: pd.DataFrame, setting_summary: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = final.groupby(["experiment", "algo", "N"], sort=True)
    for keys, group in grouped:
        experiment, algo, n = keys
        row = {"experiment": experiment, "algo": algo, "N": n}
        row.update(summarize_group(group))
        settings = setting_summary.loc[
            setting_summary["experiment"].eq(experiment)
            & setting_summary["algo"].eq(algo)
            & setting_summary["N"].eq(n)
        ]
        row["setting_count"] = len(settings)
        for threshold in THRESHOLDS:
            label = f"{threshold:.0e}"
            row[f"setting_medians_at_most_{label}"] = int(
                settings["hausdorff_median"].le(threshold).sum()
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 10
    return result


def make_threshold_sensitivity(
    final: pd.DataFrame, setting_summary: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for experiment, algo in CRITICAL_METHODS.items():
        method_rows = final.loc[
            final["experiment"].eq(experiment) & final["algo"].eq(algo)
        ]
        method_settings = setting_summary.loc[
            setting_summary["experiment"].eq(experiment)
            & setting_summary["algo"].eq(algo)
        ]
        local_specs = [("overall", method_rows, "all")]
        local_specs.extend(
            ("resolution", group.copy(), int(n))
            for n, group in method_rows.groupby("N", sort=True)
        )
        for scope, group, n in local_specs:
            settings = (
                method_settings
                if scope == "overall"
                else method_settings.loc[method_settings["N"].eq(n)]
            )
            for threshold in THRESHOLDS:
                h_floor = group["hausdorff"].le(threshold)
                hg_floor = h_floor & group["facet_gap"].le(threshold)
                rows.append(
                    {
                        "experiment": experiment,
                        "algo": algo,
                        "scope": scope,
                        "N": n,
                        "threshold": threshold,
                        "case_count": len(group),
                        "hausdorff_floor_cases": int(h_floor.sum()),
                        "hausdorff_floor_fraction": float(h_floor.mean()),
                        "joint_hg_floor_cases": int(hg_floor.sum()),
                        "joint_hg_floor_fraction": float(hg_floor.mean()),
                        "setting_count": len(settings),
                        "setting_medians_at_floor": int(
                            settings["hausdorff_median"].le(threshold).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def material_change(old: pd.Series, new: pd.Series) -> pd.Series:
    difference = new - old
    scale = np.maximum(np.maximum(old.abs(), new.abs()), np.finfo(float).tiny)
    return difference.abs().gt(1e-10) & difference.abs().div(scale).gt(0.01)


def make_paired_comparison(
    final: pd.DataFrame, old: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_columns = (
        KEY_COLUMNS
        + METRICS
        + [
            "num_plic_fallback_cells",
            "num_final_missing_cells",
        ]
    )
    new_columns = old_columns + ["N"]
    paired = old[old_columns].merge(
        final[new_columns],
        on=KEY_COLUMNS,
        suffixes=("_complex", "_final"),
        validate="one_to_one",
    )
    assert len(paired) == 1250
    for metric in METRICS:
        paired[f"{metric}_delta_final_minus_complex"] = (
            paired[f"{metric}_final"] - paired[f"{metric}_complex"]
        )
    changed = material_change(paired["hausdorff_complex"], paired["hausdorff_final"])
    paired["hausdorff_material_change"] = np.select(
        [changed & paired["hausdorff_delta_final_minus_complex"].lt(0), changed],
        ["improved", "worsened"],
        default="stable",
    )

    rows = []
    for experiment, algo in CRITICAL_METHODS.items():
        method = paired.loc[
            paired["experiment"].eq(experiment) & paired["algo"].eq(algo)
        ]
        groups = [("overall", "all", method)]
        groups.extend(("resolution", int(n), group) for n, group in method.groupby("N"))
        for scope, n, group in groups:
            row: dict[str, float | int | str] = {
                "experiment": experiment,
                "algo": algo,
                "scope": scope,
                "N": n,
                "case_count": len(group),
            }
            for version in ["complex", "final"]:
                values = group[f"hausdorff_{version}"]
                row[f"{version}_hausdorff_median"] = float(values.median())
                row[f"{version}_hausdorff_p95"] = float(values.quantile(0.95))
                row[f"{version}_hausdorff_max"] = float(values.max())
                row[f"{version}_hausdorff_at_most_1e-06"] = int(values.le(1e-6).sum())
                row[f"{version}_hausdorff_above_1"] = int(values.gt(1.0).sum())
            counts = group["hausdorff_material_change"].value_counts()
            row["material_improved_cases"] = int(counts.get("improved", 0))
            row["material_worsened_cases"] = int(counts.get("worsened", 0))
            row["material_stable_cases"] = int(counts.get("stable", 0))
            rows.append(row)
    return paired.sort_values(KEY_COLUMNS), pd.DataFrame(rows)


def make_simplified_crosscheck(
    final: pd.DataFrame, simplified: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = simplified[KEY_COLUMNS + METRICS].merge(
        final[KEY_COLUMNS + METRICS],
        on=KEY_COLUMNS,
        suffixes=("_july17", "_final"),
        validate="one_to_one",
    )
    rows = []
    for experiment, algo in CRITICAL_METHODS.items():
        group = paired.loc[
            paired["experiment"].eq(experiment) & paired["algo"].eq(algo)
        ]
        row: dict[str, float | int | str] = {
            "experiment": experiment,
            "algo": algo,
            "case_count": len(group),
        }
        for metric in METRICS:
            delta = group[f"{metric}_final"] - group[f"{metric}_july17"]
            row[f"exact_{metric}_matches"] = int(delta.eq(0).sum())
            row[f"max_abs_{metric}_difference"] = float(delta.abs().max())
        rows.append(row)
    return paired.sort_values(KEY_COLUMNS), pd.DataFrame(rows)


def make_former_failure_summary(
    final_all: pd.DataFrame, failures_path: Path
) -> pd.DataFrame:
    failures = pd.read_csv(failures_path)
    assert len(failures) == 24
    assert failures["experiment"].eq("squares").all()
    rows = []
    for failure in failures.itertuples(index=False):
        group = final_all.loc[
            final_all["experiment"].eq(failure.experiment)
            & final_all["algo"].eq(failure.algo)
            & np.isclose(final_all["resolution"], float(failure.resolution))
            & np.isclose(final_all["wiggle"], float(failure.wiggle))
            & final_all["seed"].eq(int(failure.seed))
        ]
        assert len(group) == 25
        row: dict[str, float | int | str] = {
            "experiment": failure.experiment,
            "algo": failure.algo,
            "N": int(round(float(failure.resolution) * 100)),
            "wiggle": float(failure.wiggle),
            "seed": int(failure.seed),
            "prior_failure_reason": failure.reason,
        }
        row.update(summarize_group(group))
        rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 24
    assert result[[f"finite_{metric}_rows" for metric in METRICS]].eq(25).all().all()
    assert result["final_missing_cells"].eq(0).all()
    return result.sort_values(["N", "wiggle", "algo"])


def make_tail_inventory(
    paired_complex: pd.DataFrame,
) -> pd.DataFrame:
    columns = KEY_COLUMNS + [
        "N",
        "hausdorff_final",
        "hausdorff_complex",
        "facet_gap_final",
        "area_error_final",
        "num_plic_fallback_cells_final",
        "hausdorff_material_change",
    ]
    tails = paired_complex.loc[paired_complex["hausdorff_final"].gt(1e-6), columns]
    return tails.sort_values(["experiment", "hausdorff_final"], ascending=[True, False])


def make_keep_all_check(
    setting_summary: pd.DataFrame, keep_all_path: Path
) -> pd.DataFrame:
    keep_all = pd.read_csv(keep_all_path)
    keep_all["N"] = keep_all["N"].astype(int)
    final = setting_summary.loc[
        setting_summary["experiment"].eq("zalesak")
        & setting_summary["algo"].eq("circular+corner"),
        ["N", "wiggle", "hausdorff_median", "hausdorff_max"],
    ]
    result = keep_all.merge(final, on=["N", "wiggle"], validate="one_to_one")
    result["final_minus_keep_median"] = (
        result["hausdorff_median"] - result["keep_hausdorff_median"]
    )
    result["final_minus_keep_max"] = (
        result["hausdorff_max"] - result["keep_hausdorff_max"]
    )
    return result


def plot_audit(
    setting_summary: pd.DataFrame,
    resolution_summary: pd.DataFrame,
    complex_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

    for column, (experiment, algo) in enumerate(CRITICAL_METHODS.items()):
        settings = setting_summary.loc[
            setting_summary["experiment"].eq(experiment)
            & setting_summary["algo"].eq(algo)
        ]
        for color, wiggle in zip(colors, EXPECTED_WIGGLES):
            line = settings.loc[settings["wiggle"].eq(wiggle)].sort_values("N")
            axes[0, column].plot(
                line["N"],
                line["hausdorff_median"],
                marker="o",
                linewidth=1.3,
                markersize=3.5,
                color=color,
                label=f"w={wiggle:g}",
            )
        axes[0, column].axhline(1e-6, color="#555555", linewidth=0.8, linestyle="--")
        axes[0, column].axhline(1e-8, color="#999999", linewidth=0.8, linestyle=":")
        axes[0, column].set_yscale("log")
        axes[0, column].set_xticks(EXPECTED_RESOLUTIONS)
        axes[0, column].set_xlabel("N")
        axes[0, column].set_ylabel("Hausdorff median")
        axes[0, column].set_title(f"{experiment.title()}: setting medians")
        axes[0, column].grid(True, which="both", alpha=0.18, linewidth=0.5)
        axes[0, column].legend(ncol=2, frameon=False)

        final_resolution = resolution_summary.loc[
            resolution_summary["experiment"].eq(experiment)
            & resolution_summary["algo"].eq(algo)
        ].sort_values("N")
        old_resolution = complex_summary.loc[
            complex_summary["experiment"].eq(experiment)
            & complex_summary["algo"].eq(algo)
            & complex_summary["scope"].eq("resolution")
        ].copy()
        old_resolution["N"] = old_resolution["N"].astype(int)
        old_resolution = old_resolution.sort_values("N")
        axes[1, column].plot(
            final_resolution["N"],
            final_resolution["hausdorff_at_most_1e-06"] / 125,
            marker="o",
            color="#009E73",
            linewidth=1.5,
            label="final, H<=1e-6",
        )
        axes[1, column].plot(
            final_resolution["N"],
            final_resolution["hausdorff_at_most_1e-08"] / 125,
            marker="s",
            color="#E69F00",
            linewidth=1.3,
            label="final, H<=1e-8",
        )
        axes[1, column].plot(
            old_resolution["N"],
            old_resolution["complex_hausdorff_at_most_1e-06"] / 125,
            marker="x",
            color="#666666",
            linewidth=1.1,
            linestyle="--",
            label="July 14, H<=1e-6",
        )
        axes[1, column].set_xticks(EXPECTED_RESOLUTIONS)
        axes[1, column].set_ylim(-0.03, 1.03)
        axes[1, column].set_xlabel("N")
        axes[1, column].set_ylabel("Fraction of cases at floor")
        axes[1, column].set_title(f"{experiment.title()}: threshold sensitivity")
        axes[1, column].grid(True, alpha=0.18, linewidth=0.5)
        axes[1, column].legend(frameon=False)

    fig.suptitle(
        "Perfect-reconstruction audit: final release vs July 14 profile", fontsize=11
    )
    fig.savefig(output_dir / "perfect_reconstruction_audit.pdf")
    fig.savefig(output_dir / "perfect_reconstruction_audit.png", dpi=300)
    plt.close(fig)


def run_pdf_qa(output_dir: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    pdf_path = output_dir / "perfect_reconstruction_audit.pdf"
    qa_path = output_dir / "pdf_qa.json"
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "submission" / "pdf_vector_qa.py"),
            str(pdf_path),
            "--json",
            str(qa_path),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(qa_path.read_text())
    assert payload["passed"] is True
    assert payload["pdf_count"] == 1
    assert payload["reports"][0]["image_objects"] == 0
    assert payload["reports"][0]["fonts"]
    assert all(font["embedded"] for font in payload["reports"][0]["fonts"])
    payload["reports"][0]["path"] = pdf_path.name
    payload["command_output"] = [
        line.replace(str(pdf_path.resolve()), pdf_path.name)
        for line in result.stdout.strip().splitlines()
    ]
    qa_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_checksum_ledger(output_dir: Path) -> None:
    bytecode = sorted(output_dir.rglob("*.pyc"))
    bytecode_dirs = sorted(output_dir.rglob("__pycache__"))
    if bytecode or bytecode_dirs:
        found = ", ".join(
            str(path.relative_to(output_dir)) for path in [*bytecode, *bytecode_dirs]
        )
        raise RuntimeError(
            f"bytecode artifacts must be removed before sealing: {found}"
        )

    paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256(path)}  {path.name}" for path in paths]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_all = read_cases(args.final_release)
    simplified_all = read_cases(args.july_simplified)
    complex_all = read_cases(args.july_complex)
    assert_final_contract(args.final_release, final_all)

    final = critical_rows(final_all)
    simplified = critical_rows(simplified_all)
    complex_rows = critical_rows(complex_all)

    setting_summary = make_setting_summary(final)
    resolution_summary = make_resolution_summary(final, setting_summary)
    threshold_summary = make_threshold_sensitivity(final, setting_summary)
    paired_complex, complex_summary = make_paired_comparison(final, complex_rows)
    paired_simplified, simplified_summary = make_simplified_crosscheck(
        final, simplified
    )
    former_failures = make_former_failure_summary(
        final_all, args.invalid_release / "failures.csv"
    )
    tails = make_tail_inventory(paired_complex)
    keep_all_check = make_keep_all_check(setting_summary, args.keep_all_summary)

    outputs = {
        "final_setting_summary.csv": setting_summary,
        "final_resolution_summary.csv": resolution_summary,
        "threshold_sensitivity.csv": threshold_summary,
        "complex_profile_case_comparison.csv": paired_complex,
        "complex_profile_summary.csv": complex_summary,
        "july17_case_crosscheck.csv": paired_simplified,
        "july17_crosscheck_summary.csv": simplified_summary,
        "formerly_failed_square_settings.csv": former_failures,
        "tail_inventory.csv": tails,
        "keep_all_linear_rescue_check.csv": keep_all_check,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    plot_audit(setting_summary, resolution_summary, complex_summary, args.output_dir)
    run_pdf_qa(args.output_dir)

    input_paths = {
        "final_case_metrics": args.final_release / "diagnostics" / "case_metrics.csv",
        "final_sweep_manifest": args.final_release / "sweep_manifest.json",
        "july17_case_metrics": args.july_simplified
        / "diagnostics"
        / "case_metrics.csv",
        "july14_case_metrics": args.july_complex / "diagnostics" / "case_metrics.csv",
        "invalid_failure_ledger": args.invalid_release / "failures.csv",
        "keep_all_linear_rescue_summary": args.keep_all_summary,
    }
    manifest = {
        "schema_version": 1,
        "critical_methods": CRITICAL_METHODS,
        "thresholds": THRESHOLDS,
        "material_change_definition": {
            "absolute_difference_gt": 1e-10,
            "relative_difference_gt": 0.01,
        },
        "inputs": {
            label: {"path": portable_path(path), "sha256": sha256(path)}
            for label, path in input_paths.items()
        },
        "outputs": sorted(
            [
                "README.md",
                "SHA256SUMS",
                "analysis_manifest.json",
                "analyze.py",
                *outputs,
                "pdf_qa.json",
                "perfect_reconstruction_audit.pdf",
                "perfect_reconstruction_audit.png",
            ]
        ),
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_checksum_ledger(args.output_dir)


if __name__ == "__main__":
    main()
