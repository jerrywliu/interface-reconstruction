from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_archived_shape_wrappers_keep_canonical_configs_and_lvira_fallback():
    expected_configs = {
        "run_lines.sh": "static/line",
        "run_circles.sh": "static/circle",
        "run_ellipses.sh": "static/ellipse",
        "run_squares.sh": "static/square",
        "run_zalesak.sh": "static/zalesak",
    }

    for name, config in expected_configs.items():
        wrapper = REPO_ROOT / "archive" / "legacy_wrappers" / name
        text = wrapper.read_text()
        assert "Legacy convenience sweep" in text
        assert f"--config {config}" in text
        assert "--plic_fallback LVIRA" in text
        assert "fallback is LVIRA" in text
        assert "defaults to Youngs" not in text


def test_prepackage_source_is_isolated_from_supported_module_tree():
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

    archive = REPO_ROOT / "archive" / "legacy_v1"
    assert (archive / "README.md").is_file()
    assert (archive / "source" / "facet.py").is_file()
    assert (archive / "source" / "run_old.py").is_file()


def test_supported_source_does_not_import_from_archive():
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
