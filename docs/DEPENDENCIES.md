# Dependencies And Development Environment

## Environment Scope

The paper-facing code targets CPython 3.9. The July 2026 validation environment
used CPython 3.9.13, but its installed numerical packages did not exactly match all
checked-in pins. Consequently, Python 3.9 is the current CI and reproduction target,
not yet a multi-version public support guarantee.

`requirements.txt` is the frozen submission-era runtime input. It intentionally
remains unchanged in this cleanup because changing NumPy, SciPy, Matplotlib,
Shapely, or VTK immediately before accepting a result set can change numerical or
rendering behavior. It currently contains both direct and transitive pins.

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
python -m pytest -q test
```

The Slack integration test is intentionally opt-in because it sends a real message
and attachments. Run it only with test credentials:

```bash
RUN_SLACK_INTEGRATION=1 python -m pytest -q \
  test/integration/test_slack_integration.py
```

Install the runtime stack plus the review-packet dependency:

```bash
python -m pip install -r requirements-figures.txt
```

`reportlab` is needed by `experiments/static/build_figure_review_pdf.py`; ordinary
vector plots use Matplotlib from the runtime requirements. PDF vector QA also needs
the external Poppler commands `pdfimages` and `pdffonts`. Compiling the manuscript
requires a separate LaTeX installation and is not part of the Python environment.

## What CI Establishes

The lightweight GitHub Actions job uses CPython 3.9 on Linux, installs
`requirements-test.txt`, runs the source-only freeze audit, and executes the full
repository test tree with a noninteractive Matplotlib backend. It does not run the
970-job scientific sweep, build the paper, exercise Slack, or establish bitwise
agreement across platforms.

## Clean-Environment Decisions Still Open

Before publishing a general-purpose environment specification:

1. Recreate `requirements.txt` in a clean CPython 3.9 environment and require
   `python -m pip check` to pass.
2. Compare a compact numerical acceptance suite against the accepted final release.
3. Decide whether the checked-in pins or the exact accepted-sweep environment is the
   archival authority when they differ.
4. Only then consider splitting the mixed runtime/transitive pins, adding a lock
   file, or declaring additional Python and operating-system support.

The additive test and figure files do not alter any existing runtime pin. They make
tooling roles explicit while preserving the frozen installation input.

Both additive files resolve with `pip --dry-run --ignore-installed` under the current
CPython 3.9 interpreter. That checks dependency compatibility, not installation or
numerical acceptance in a clean environment. The shared workstation environment
still fails `pip check` because of unrelated Torch/TorchSDE packages, so it is not
being presented as the public release environment.
