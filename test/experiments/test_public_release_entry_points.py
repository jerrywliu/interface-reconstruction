from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_submission_tree_has_canonical_benchmarks_without_legacy_wrappers():
    benchmarks = ("lines", "circles", "ellipses", "squares", "zalesak")

    for name in benchmarks:
        assert (REPO_ROOT / "experiments" / "static" / f"{name}.py").is_file()

    assert (REPO_ROOT / "submission" / "run_final_static_sweep.sh").is_file()
    assert not (REPO_ROOT / "archive").exists()


def test_prepackage_source_is_absent_from_submission_tree():
    retired_paths = (
        "facet.py",
        "run_old.py",
        "main/algos/interface_reconstruction.py",
        "main/algos/local_reconstruction.py",
        "main/algos/plic.py",
        "main/algos/static_interface_reconstruction.py",
        "main/structs/strand.py",
        "util/initialize/initialize_areas_old.py",
    )

    for relative in retired_paths:
        assert not (REPO_ROOT / relative).exists(), relative

def test_supported_source_does_not_import_removed_archive():
    for root_name in ("main", "util", "experiments", "submission"):
        for path in (REPO_ROOT / root_name).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "from archive" not in text, path
            assert "import archive" not in text, path


def test_figure_requirements_cover_committed_analysis_dependencies():
    requirements = (REPO_ROOT / "requirements-figures.txt").read_text().splitlines()

    assert "pandas==2.2.2" in requirements
    assert "reportlab==4.4.3" in requirements


def test_public_docs_do_not_expose_private_workspace_paths():
    documents = [
        REPO_ROOT / "submission" / "CLEAN_ENV_REPRODUCIBILITY_VALIDATION.md",
        REPO_ROOT
        / "submission"
        / "audits"
        / "square_active_partition_confidence_2026-07-31"
        / "README.md",
    ]

    for document in documents:
        assert "/Users/wei" not in document.read_text()
