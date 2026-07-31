import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.submission.materialize_final_conservation_selection import (
    materialize_selection,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release(tmp_path: Path) -> tuple[Path, Path]:
    release = tmp_path / "submission_static_final"
    diagnostics = release / "diagnostics"
    bundle = release / "raw_runs" / "line_run"
    (bundle / "metrics").mkdir(parents=True)
    (bundle / "vtk").mkdir()
    diagnostics.mkdir(parents=True)

    (release / "submission_config.resolved.json").write_text(
        json.dumps({"source": {"target_commit": "a" * 40}}), encoding="utf-8"
    )
    (bundle / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (bundle / "vtk/mesh.vtk").write_text("mesh\n", encoding="utf-8")
    (bundle / "metrics/case_geometry.jsonl").write_text(
        json.dumps({"case_index": 3}) + "\n", encoding="utf-8"
    )
    (bundle / "metrics/case_metrics.csv").write_text(
        "case_index,hausdorff\n3,0\n", encoding="utf-8"
    )
    (bundle / "metrics/cell_metrics.csv").write_text(
        "case_index,cell_id\n3,0;0\n", encoding="utf-8"
    )

    inventory_fields = (
        "experiment",
        "algo",
        "resolution",
        "wiggle",
        "seed",
        "source_commit",
        "run_bundle",
    )
    with (diagnostics / "run_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=inventory_fields)
        writer.writeheader()
        writer.writerow(
            {
                "experiment": "lines",
                "algo": "linear",
                "resolution": "0.32",
                "wiggle": "0.2",
                "seed": "0",
                "source_commit": "a" * 40,
                "run_bundle": "raw_runs/line_run",
            }
        )

    files = sorted(path for path in release.rglob("*") if path.is_file())
    (release / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(release).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    specification = tmp_path / "selection_spec.json"
    specification.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "experiment": "lines",
                        "algo": "linear",
                        "resolution": 0.32,
                        "wiggle": 0.2,
                        "seed": 0,
                        "case_index": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return release, specification


def test_materializes_checksum_bound_final_release_selection(tmp_path):
    release, specification = _write_release(tmp_path)
    output = tmp_path / "validation" / "selection.json"

    payload = materialize_selection(release, specification, output)

    assert payload["release_binding"]["source_commit"] == "a" * 40
    assert payload["release_binding"]["sha256_manifest_digest"] == _sha256(
        release / "SHA256SUMS"
    )
    assert payload["cases"][0]["case_index"] == 3
    assert payload["cases"][0]["run_root"] == str(
        (release / "raw_runs/line_run").resolve()
    )
    assert json.loads(output.read_text()) == payload


def test_rejects_tampered_or_in_release_outputs(tmp_path):
    release, specification = _write_release(tmp_path)
    with pytest.raises(ValueError, match="outside FINAL_ROOT"):
        materialize_selection(
            release, specification, release / "validation/selection.json"
        )

    (release / "raw_runs/line_run/metrics/cell_metrics.csv").write_text(
        "case_index,cell_id\n3,tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_selection(
            release, specification, tmp_path / "validation/selection.json"
        )
