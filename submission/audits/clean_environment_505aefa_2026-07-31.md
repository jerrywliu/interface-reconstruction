# Clean-environment validation for `505aefa`

Validation date: 2026-07-30 PDT / 2026-07-31 UTC

Source commit: `505aefa454328d4ba34ade5e7247050a0acfc793`

Isolated branch and worktree:

- `codex/clean-env-repro-505aefa-20260731`
- `/Users/wei/Code/Interface/active/interface-reconstruction-clean-env-505aefa-20260731`

Fresh disposable environment:

- `/tmp/interface-reconstruction-clean-env-505aefa-20260731-py39`

The main checkout, Overleaf checkout, and active authoritative sweep were not
modified or used as execution roots. All project commands ran from the isolated
worktree at the exact source commit above.

## Verdict

The exact source commit passes the declared clean-install, test, representative
scientific-smoke, and vector-PDF gates. No blocker was found.

| Gate | Result |
| --- | --- |
| Install `requirements-test.txt` and `requirements-figures.txt` | pass |
| `python -m pip check` | pass, no broken requirements |
| Full supported `python -m pytest -q` | `236 passed, 1 skipped in 72.98s` |
| Line, square, and Zalesak scientific smokes | pass, 3/3 |
| Missing final facets | 0/499 active cells |
| Unresolved PLIC fallback records | 0 |
| Vector PDF QA | pass, 2/2 PDFs |

The skipped test is the credentialed live Slack integration, which is intentionally
opt-in and was run with Slack credentials unset.

## Environment

- macOS 26.5.2 (build 25F84), Darwin 25.5.0, arm64
- CPython 3.9.13, built with Apple Clang 14.0.0
- NumPy 1.23.4 using OpenBLAS ILP64
- SciPy 1.9.2 using OpenBLAS
- Shapely 1.8.5.post1 with GEOS 3.10.3-CAPI-1.16.1
- Matplotlib 3.6.1 with FreeType 2.6.1
- VTK 9.2.2
- pytest 8.4.2 and ReportLab 4.4.3
- Poppler 24.04.0 (`pdfimages` and `pdffonts`)
- pip 26.0.1; 41 installed distributions

The resolved package set matches the public clean-environment pins documented for
the earlier validation. A direct `import vtk` probe completed successfully and
reported VTK 9.2.2.

## Scientific smokes

All three cases used a perturbed quadrilateral mesh with `w=0.1`, seed 0, fixed
boundary nodes, `LVIRA` unresolved-orientation fallback, and the
`pre_f8_corner` profile. Zalesak additionally used
`exact_linear_support_only`, local-linear arc-failure fallback, and disabled `C0`.

| Experiment | Method and case | Hausdorff | Facet gap | Area error | Final facet classes |
| --- | --- | ---: | ---: | ---: | --- |
| Lines | `linear`, case 6, N=100 | `3.0883564752186087e-10` | `4.757175882868377e-10` | n/a | 180 linear |
| Squares | `linear+corner`, case 24, N=100 | `7.241358026991761e-10` | `8.757448204532568e-11` | `4.372023908903576e-12` | 120 linear, 13 linear corners |
| Zalesak | `circular+corner`, case 12, N=100 | `1.1526621649074481e-08` | `4.800725101102327e-10` | `2.3334029534955158e-11` | 108 arcs, 62 lines, 6 linear corners, 10 curved corners |

Each run wrote one case-metric row, complete cell diagnostics, exact case geometry,
merge provenance, facet metadata, VTP geometry, zero fallback rows, and zero missing
final facets.

### Agreement with the prior clean baseline

The saved outputs were compared with the same three cases from the independently
validated clean environment at `525d0cf5b4df`. This is a strict check of the
commit containing the square active-partition correction against the previous
clean-environment record.

- All three complete case-metric rows are byte-for-byte equal.
- Hausdorff, facet-gap, and available area-error deltas are exactly zero.
- All 499 sorted per-cell reconstruction decisions agree exactly, including merge
  identity, orientation status, final facet class/name, construction path, and
  fallback policy.
- All 499 serialized facet geometries and all three case-geometry records agree
  exactly.
- All comparable mesh, VTP, facet-metadata, and corner-tip artifacts have matching
  SHA-256 hashes: 6/6 for lines, 7/7 for squares, and 7/7 for Zalesak.

The square witness therefore confirms that the active-partition metric fix did not
change reconstruction decisions or facet geometry for this paper-facing case.

## Vector PDF QA

The exact documented deterministic generators produced high-resolution PNG review
siblings plus SVG and PDF vector outputs under
`/tmp/interface-reconstruction-clean-env-figures-505aefa-20260731`.

| PDF | Raster image objects | Fonts | Result |
| --- | ---: | ---: | --- |
| `perfect_reconstruction_plic_stencil.pdf` | 0 | 6/6 embedded | pass |
| `staged_reconstruction_zalesak.pdf` | 0 | 2/2 embedded | pass |

`submission/pdf_vector_qa.py` reported `PDF QA: 2/2 passed`. Rendered review images
were inspected and had no blank panels, clipping, missing labels, overlap, or
rasterized color key.

## Commands

The fresh environment was created and checked with:

```bash
VENV=/tmp/interface-reconstruction-clean-env-505aefa-20260731-py39
python3.9 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  -r requirements-test.txt -r requirements-figures.txt
"$VENV/bin/python" -m pip check
env -u SLACK_BOT_TOKEN -u SLACK_CHANNEL -u SLACK_WEBHOOK_URL \
  MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONPATH=. \
  "$VENV/bin/python" -m pytest -q
```

The scientific commands exactly repeated those in
`submission/CLEAN_ENV_REPRODUCIBILITY_VALIDATION.md`, changing only `--save_name`
to identify the target commit. The deterministic figure commands and vector gate
were also repeated unchanged from that document.

Generated smoke directories under `plots/` are ignored and remain untracked. The
fresh environment and generated figure directory are disposable; this report is
the only tracked validation artifact.
