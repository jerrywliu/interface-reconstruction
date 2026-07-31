# Dependencies And Development Environment

## Environment Scope

The paper-facing code targets CPython 3.9. The July 2026 validation environment
used CPython 3.9.13, but its installed numerical packages did not exactly match all
checked-in pins. Consequently, Python 3.9 is the current CI and reproduction target,
not yet a multi-version public support guarantee.

`requirements.txt` is the frozen submission-era runtime input. It intentionally
remains unchanged because changing NumPy, SciPy, Matplotlib, Shapely, or VTK can
change numerical or rendering behavior. It currently contains both direct and
transitive pins.

## Installation Tiers

Create an isolated environment from the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the frozen runtime stack for reconstruction and ordinary Matplotlib figure
generation:

```bash
python -m pip install -r requirements.txt
```

Install the runtime stack plus automated tests:

```bash
python -m pip install -r requirements-test.txt
MPLBACKEND=Agg python -m pytest -q test
```

The Slack integration test is intentionally opt-in because it sends a real message
and attachments. Run it only with test credentials:

```bash
RUN_SLACK_INTEGRATION=1 python -m pytest -q \
  test/integration/test_slack_integration.py
```

Install the runtime stack plus reproducibility-analysis and review-packet
dependencies:

```bash
python -m pip install -r requirements-figures.txt
```

`pandas` is needed by the committed perfect-reconstruction audit, and `reportlab`
is needed by `experiments/static/build_figure_review_pdf.py`. Ordinary vector plots
use Matplotlib from the runtime requirements. PDF vector QA also needs the external
Poppler commands `pdfimages` and `pdffonts`. Compiling the manuscript requires a
separate LaTeX installation and is not part of the Python environment.

## What CI Establishes

The lightweight GitHub Actions job uses CPython 3.9 on Linux, installs
`requirements-test.txt`, runs the source-only freeze audit, and executes the full
repository test tree with a noninteractive Matplotlib backend. It does not run the
970-job scientific sweep, build the paper, exercise Slack, or establish bitwise
agreement across platforms.

## Reproducibility Policy

The exact environment that produced an accepted result set is the archival
authority for that result set and must be captured with
`submission/capture_environment.py`. The pinned requirements are the public clean
installation target.

The frozen requirements were validated in a clean CPython 3.9.13 environment on
macOS arm64. The dependency graph passed `pip check`, the full repository suite
passed with only the intentionally skipped Slack integration, three representative
paper-facing reconstructions agreed with the workstation stack to numerical
tolerance, and both deterministic vector figures passed PDF QA. Exact commands,
metrics, package versions, and limitations are recorded in
`submission/CLEAN_ENV_REPRODUCIBILITY_VALIDATION.md`.

The project does not claim bitwise-identical results across numerical stacks or
operating systems. Changes to the frozen numerical pins require a focused numerical
comparison and, when paper-facing outputs can change, regeneration of the affected
results and figures. Broader support still requires repeating the compact acceptance
suite in a clean Linux environment.

The additive test and figure requirement files do not alter any numerical runtime
pin; they make tooling roles explicit while preserving the frozen installation
input.
