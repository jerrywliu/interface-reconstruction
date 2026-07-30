# Clean-Environment Reproducibility Validation

Date: 2026-07-30

Source commit: `525d0cf5b4df8ad6526429316872acf9eb656517`

Validation branch: `codex/clean-env-repro-20260730`

Validation worktree:
`/Users/wei/Code/Interface/active/interface-reconstruction-clean-env-repro-20260730`

Disposable environment:
`/tmp/interface-reconstruction-clean-env-20260730-py39`

The active source checkout and the incomplete final-release directory were not
imported, modified, or read. Every project command ran from the isolated worktree.
The existing workstation Python was used only for a read-only three-case numerical
comparison after the clean-environment validation was complete.

## Outcome

The frozen requirements install successfully in a clean CPython 3.9 environment on
Apple Silicon, and their dependency graph passes `pip check`. After declaring the
missing test and review-packet dependencies and making the live Slack test opt-in,
the complete repository suite passes with one expected skip:

```text
193 passed, 1 skipped in 41.20s
```

Three paper-facing reconstruction smokes completed with no missing final facets or
PLIC fallbacks. Both deterministic figure generators completed, and both PDFs pass
the vector gate with zero raster image objects and all fonts embedded.

## Platform

- OS: macOS 26.5.2, Darwin 25.5.0, arm64
- Python: CPython 3.9.13, ABI `cpython-39-darwin`
- Compiler: Apple Clang 14.0.0
- Poppler: 24.04.0 (`pdfimages`, `pdffonts`, and `pdftoppm`)
- NumPy BLAS/LAPACK: OpenBLAS ILP64
- SciPy BLAS/LAPACK: OpenBLAS
- Shapely GEOS: 3.10.3-CAPI-1.16.1
- Matplotlib FreeType: 2.6.1

The machine-readable environment capture is intentionally disposable rather than a
tracked public artifact. It reported 33 declared runtime requirements, 41 installed
distributions, zero missing requirements, zero version mismatches, and a successful
`pip check`.

## Installation Findings

The untouched `requirements.txt` installed without resolver or wheel failures and
`python -m pip check` printed `No broken requirements found.` Its core numerical,
geometry, and rendering imports all succeeded.

The runtime-only installation exposed two real documentation gaps:

- `python -m pytest --version` failed with `No module named pytest`.
- `import reportlab` failed with `ModuleNotFoundError`.

After installing `pytest==8.4.2` and `reportlab==4.4.3`, the first full test run had
one failure and 193 passes. The only failure was
`test/integration/test_slack_integration.py`, which was collected despite having no
credentials and attempted a real external integration. The branch therefore adds:

- `requirements-test.txt`, extending the runtime pins with `pytest==8.4.2`;
- `requirements-figures.txt`, extending them with `reportlab==4.4.3`;
- `docs/DEPENDENCIES.md`, documenting the CPython target and install tiers;
- an explicit `RUN_SLACK_INTEGRATION=1` opt-in for the live Slack test.

No numerical requirement, reconstruction implementation, experiment driver, or
figure generator was changed.

## Numerical Smoke

All runs used perturbed quadrilateral meshes, perturbation magnitude `0.1`, mesh
seed `0`, `LVIRA` unresolved-orientation fallback, and the paper-facing
`pre_f8_corner` profile. Zalesak additionally used
`exact_linear_support_only`, local-line arc-failure fallback, and disabled C0.

| Experiment | Method and case | Hausdorff | Facet gap | Area error | Final facets |
| --- | --- | ---: | ---: | ---: | --- |
| Lines | `linear`, case 6, N=100 | `3.088356475e-10` | `4.757175883e-10` | n/a | 180 linear |
| Squares | `linear+corner`, case 24, N=100 | `7.241358027e-10` | `8.757448205e-11` | `4.372023909e-12` | 120 linear, 13 corners |
| Zalesak | `circular+corner`, case 12, N=100 | `1.152662165e-08` | `4.800725101e-10` | `2.333402953e-11` | 108 arcs, 62 lines, 6 linear corners, 10 curved corners |

Every run wrote exactly one case row, complete cell diagnostics, exact case
geometry, merge provenance, zero unresolved PLIC fallback rows, and zero missing
final facets. The Zalesak driver printed handled arc-fit diagnostic messages; the
process completed successfully and its saved result/provenance checks passed.

### Comparison With The Validated Workstation Stack

The workstation stack uses NumPy 1.24.4, SciPy 1.8.1, and Matplotlib 3.8.3 instead
of the public pins. It is not a clean environment: `pip check` reports unrelated
Torch/TorchSDE problems. Running the same three commands from the isolated worktree
nevertheless provides a compact scientific comparison.

| Experiment | Agreement between stacks |
| --- | --- |
| Lines | Hausdorff and facet gap are bit-for-bit identical; case geometry and facet classes are identical. |
| Squares | Hausdorff, facet gap, and area error are bit-for-bit identical; case geometry and facet classes are identical. |
| Zalesak | Hausdorff, case geometry, counts, and facet classes are identical. Facet gap differs by `3.881e-13` absolute and area error by `3.144e-14` absolute. |

This compact comparison finds no reconstruction decision change and no meaningful
scientific regression from the checked-in pins.

## Vector Figure QA

The clean environment generated:

- `perfect_reconstruction_plic_stencil.pdf` (185,463 bytes): zero image objects,
  six embedded Type 1 fonts;
- `staged_reconstruction_zalesak.pdf` (386,220 bytes): zero image objects, two
  embedded CID TrueType fonts.

`submission/pdf_vector_qa.py` reported `PDF QA: 2/2 passed`. The generated 300-DPI
review PNGs were also inspected: panels, labels, legends, mesh lines, reconstruction
stages, and corner markers render without blank content, clipping, or overlap.

## Exact Commands

The environment was created outside the repository:

```bash
python3.9 -m venv /tmp/interface-reconstruction-clean-env-20260730-py39
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -m pip install --upgrade pip
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -m pip install -r requirements.txt
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -m pip check
```

The baseline missing-tool checks were:

```bash
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -m pytest --version
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -c 'import reportlab; print(reportlab.Version)'
```

The complete declared tool tiers and test suite were validated with:

```bash
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -m pip install \
  -r requirements-test.txt -r requirements-figures.txt
/tmp/interface-reconstruction-clean-env-20260730-py39/bin/python -m pip check
env -u SLACK_BOT_TOKEN -u SLACK_CHANNEL -u SLACK_WEBHOOK_URL \
  MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONPATH=. \
  /tmp/interface-reconstruction-clean-env-20260730-py39/bin/python \
  -m pytest -q test
```

The representative numerical commands followed this form, with the experiment,
method, case-count argument, and case index varied as shown in the table:

```bash
MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONPATH=. \
  /tmp/interface-reconstruction-clean-env-20260730-py39/bin/python \
  -m experiments.static.squares \
  --config static/square --resolution 1.0 \
  --facet_algo linear+corner \
  --save_name clean_env_repro_squares_linearpluscorner_case24 \
  --num_squares 25 --case_indices 24 \
  --mesh_type perturbed_quads --perturb_wiggle 0.1 --perturb_seed 0 \
  --perturb_fix_boundary 1 --plic_fallback LVIRA \
  --corner_behavior_profile pre_f8_corner
```

The Zalesak command additionally supplied:

```text
--do_c0 0 --arc_failure_fallback local_linear
--rescue_profile exact_linear_support_only
```

The vector figures and QA were generated with:

```bash
MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONPATH=. \
  /tmp/interface-reconstruction-clean-env-20260730-py39/bin/python \
  -m experiments.static.generate_plic_baseline_stencil_figure \
  --out /tmp/interface-reconstruction-clean-env-figures-20260730/perfect_reconstruction_plic_stencil \
  --case-index 4 --cell-x 14 --cell-y 13 \
  --resolution 0.32 --wiggle 0.3 --seed 0

MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONPATH=. \
  /tmp/interface-reconstruction-clean-env-20260730-py39/bin/python \
  -m experiments.static.generate_staged_reconstruction_figure \
  --output-dir /tmp/interface-reconstruction-clean-env-figures-20260730 \
  --prefix staged_reconstruction_zalesak \
  --case-index 22 --resolution 1.0 --wiggle 0.1 --seed 0

MPLBACKEND=Agg PYTHONPATH=. \
  /tmp/interface-reconstruction-clean-env-20260730-py39/bin/python \
  submission/pdf_vector_qa.py \
  /tmp/interface-reconstruction-clean-env-figures-20260730 \
  --json /tmp/interface-reconstruction-clean-env-figures-20260730/pdf_vector_qa.json
```

## Resolved Python Distributions

`pip freeze --all` in the clean environment returned:

```text
aiohttp==3.8.3
aiosignal==1.2.0
async-timeout==4.0.2
attrs==22.1.0
black==24.10.0
charset-normalizer==2.1.1
click==8.1.8
contourpy==1.0.5
cycler==0.11.0
exceptiongroup==1.3.1
fonttools==4.37.4
frozenlist==1.3.1
idna==3.4
iniconfig==2.1.0
kiwisolver==1.4.4
matplotlib==3.6.1
multidict==6.0.2
mypy-extensions==1.0.0
numpy==1.23.4
packaging==24.2
pathspec==0.12.1
Pillow==9.2.0
pip==26.0.1
platformdirs==4.3.6
pluggy==1.6.0
Pygments==2.20.0
pyparsing==3.0.9
pytest==8.4.2
python-dateutil==2.8.2
PyYAML==6.0
reportlab==4.4.3
scipy==1.9.2
setuptools==58.1.0
Shapely==1.8.5.post1
six==1.16.0
tomli==2.2.1
tqdm==4.67.1
typing_extensions==4.12.2
vtk==9.2.2
wslink==1.8.4
yarl==1.8.1
```

## Recommendation

Use two complementary records instead of choosing one and discarding the other:

1. Keep the current numerical pins as the public clean-install target. They install
   cleanly, pass the complete suite, generate compliant vector figures, and preserve
   the compact scientific results. Do not change these pins immediately before
   submission.
2. Treat the accepted final sweep's `environment.json` as the archival authority for
   the submitted numerical result set, because that sweep was produced with a
   different NumPy/SciPy/Matplotlib stack. Preserve its exact package, BLAS/LAPACK,
   GEOS, FreeType, VTK, Python, and platform capture beside the results.

Do not advertise bitwise cross-platform reproducibility. The evidence supports a
CPython 3.9 clean-install target on macOS arm64 and tolerance-level agreement across
the two tested local numerical stacks. Before declaring broader support, repeat the
compact acceptance suite in a clean Linux environment. If an exact installable
submission lock is desired, derive it from the accepted sweep snapshot only after
proving that snapshot resolves in a clean environment; do not replace the working
public pins merely because the workstation happened to contain newer packages.

## Limits

- One operating system and architecture were clean-installed.
- The compact scientific comparison covers three representative cases, not the full
  970-run sweep.
- The accepted sweep environment was compared through the existing validated
  workstation stack, not recreated in a second clean environment.
- Poppler and LaTeX remain external system dependencies.
- The incomplete final release was deliberately not inspected; its final audit and
  environment capture remain separate release gates.
