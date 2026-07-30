import json
import subprocess
from pathlib import Path

from submission.capture_environment import (
    capture_environment,
    capture_git_state,
    compare_declared_and_installed,
    parse_requirements,
    write_environment_capture,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    (path / "requirements.txt").write_text("numpy==1.23.4\n", encoding="utf-8")
    subprocess.run(["git", "add", "requirements.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def test_parse_and_compare_exact_requirements(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "# Runtime\nNumPy==1.23.4\nShapely==1.8.5.post1\n-e ../local\n",
        encoding="utf-8",
    )

    parsed = parse_requirements(requirements)
    comparison = compare_declared_and_installed(
        (requirement for requirement in parsed["requirements"]),
        [
            {"name": "numpy", "version": "1.24.4"},
            {"name": "Shapely", "version": "1.8.5.post1"},
        ],
    )

    assert parsed["unparsed"] == ["-e ../local"]
    assert comparison["missing"] == []
    assert comparison["version_mismatches"] == [
        {"name": "NumPy", "declared": "1.23.4", "installed": "1.24.4"}
    ]


def test_git_state_distinguishes_generated_artifacts_from_source(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.csv").write_text("x\n", encoding="utf-8")

    generated_only = capture_git_state(tmp_path)

    assert generated_only["dirty"] is True
    assert generated_only["source_dirty"] is False

    (tmp_path / "analysis.py").write_text("VALUE = 1\n", encoding="utf-8")
    with_source = capture_git_state(tmp_path)

    assert with_source["source_dirty"] is True
    assert any("analysis.py" in line for line in with_source["source_status"])


def test_capture_environment_records_runtime_git_and_fingerprints(tmp_path):
    _init_repo(tmp_path)

    record = capture_environment(
        tmp_path,
        include_scientific_stack=False,
        include_pip_check=False,
    )

    assert record["schema_version"] == 1
    assert record["repository"]["commit"]
    assert record["repository"]["source_dirty"] is False
    assert record["runtime"]["python_version"]
    assert record["installed_distributions"]
    assert record["input_fingerprints"] == [
        {
            "path": "requirements.txt",
            "size_bytes": 14,
            "sha256": "eed602a1fcab231dbf08eb1af701a921a4fba6a6aa74b59360fb1d0edb9b9cc5",
        }
    ]


def test_capture_environment_uses_an_environment_variable_allowlist(
    tmp_path, monkeypatch
):
    _init_repo(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("EXAMPLE_SECRET_TOKEN", "do-not-record")

    record = capture_environment(
        tmp_path,
        include_scientific_stack=False,
        include_pip_check=False,
    )

    assert record["environment_variables"]["OMP_NUM_THREADS"] == "3"
    assert "EXAMPLE_SECRET_TOKEN" not in record["environment_variables"]


def test_environment_capture_is_written_as_sorted_json(tmp_path):
    output_path = tmp_path / "release" / "environment.json"

    write_environment_capture(output_path, {"z": 1, "a": {"value": 2}})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "a": {"value": 2},
        "z": 1,
    }
    assert output_path.read_text(encoding="utf-8").splitlines()[1].startswith('  "a"')
