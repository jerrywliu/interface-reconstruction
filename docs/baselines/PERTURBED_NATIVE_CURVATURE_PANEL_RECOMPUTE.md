# Perturbed curved-interface native-curvature panel recomputation

Date: 2026-08-22

## Scope

The circle and ellipse perturbed-mesh all-method panels were rebuilt from the
sealed static package using the common curvature observable:

```text
integral_Gamma_h |kappa_h - kappa_truth(nearest point)| ds
---------------------------------------------------------- .
                   length(Gamma_h)
```

Every reconstructed line and circular arc is read from its exact schema-v2
`facet_metadata.json` sidecar. The truth is an analytic circle or rotated
ellipse. Evaluation uses `geometric_curvature_error_external` with 16-point
arc-length quadrature. The sealed package is read only.

## Data sufficiency

The archive contains all required native geometry:

- 420 runs: 210 circle and 210 ellipse runs;
- seven methods, six resolutions, five perturbation magnitudes, and one seed;
- 25 cases per run, for 10,500 evaluated cases; and
- exact line endpoints and arc endpoints, centers, radii, and signed sweeps.

No reconstruction replay, method fallback, or silent relabeling was required.

## Old versus common curvature rows

The table reports medians over all 750 case rows for each benchmark/method.

| Benchmark | Method | Historical median | Native MAE median | New/old median ratio |
| --- | --- | ---: | ---: | ---: |
| Circles | Youngs | `1.0000e-1` | `1.0000e-1` | `1.000` |
| Circles | ELVIRA | `1.0000e-1` | `1.0000e-1` | `1.000` |
| Circles | LVIRA | `1.0000e-1` | `1.0000e-1` | `1.000` |
| Circles | Ours, linear per-cell | `1.0000e-1` | `1.0000e-1` | `1.000` |
| Circles | Ours, linear graph-coordinated | `1.0000e-1` | `1.0000e-1` | `1.000` |
| Circles | Ours, circular per-cell | `1.9358e-9` | `1.0710e-9` | `0.691` |
| Circles | Ours, circular graph-coordinated | `7.5041e-10` | `4.1973e-10` | `0.616` |
| Ellipses | Youngs | `5.0644e-2` | `4.4380e-2` | `0.864` |
| Ellipses | ELVIRA | `5.0660e-2` | `4.4474e-2` | `0.865` |
| Ellipses | LVIRA | `5.0679e-2` | `4.4443e-2` | `0.863` |
| Ellipses | Ours, linear per-cell | `5.0644e-2` | `4.4380e-2` | `0.864` |
| Ellipses | Ours, linear graph-coordinated | `5.0587e-2` | `4.4485e-2` | `0.865` |
| Ellipses | Ours, circular per-cell | `3.8715e-2` | `1.7473e-3` | `0.047` |
| Ellipses | Ours, circular graph-coordinated | `3.8636e-2` | `1.3731e-3` | `0.037` |

The PLIC and linear circle rows agree to roundoff because both observables see
zero reconstructed curvature against the constant `0.1` truth. The major
ellipse correction removes the historical signed-curvature mismatch and
equal-facet weighting; it is not a reconstruction change.

## Commands

The full native metric audit was run with:

```bash
python -m experiments.static.recompute_perturbed_native_curvature_panels \
  --workers 6 \
  --output results/submission/perturbed_native_curvature_panels_20260822
```

After metric completion exposed a plotting-only log-axis requirement, the
completed case table was reused to regenerate the panels:

```bash
python -m experiments.static.recompute_perturbed_native_curvature_panels \
  --workers 6 \
  --reuse-case-metrics \
  --output results/submission/perturbed_native_curvature_panels_20260822
```

Focused tests and vector QA:

```bash
python -m pytest -q \
  test/experiments/test_recompute_perturbed_native_curvature_panels.py \
  test/experiments/test_convergence_plotting.py

python -m submission.pdf_vector_qa \
  --json results/submission/perturbed_native_curvature_panels_20260822/vector_qa.json \
  results/submission/perturbed_native_curvature_panels_20260822/circle_reconstruction_perturbed_all_methods_5x2_axes.pdf \
  results/submission/perturbed_native_curvature_panels_20260822/ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf
```

Result: `5 passed`; both PDFs contain zero image objects and only embedded
fonts.

## Artifacts

All generated artifacts are under:

`results/submission/perturbed_native_curvature_panels_20260822/`

The bundle includes the two vector PDFs, PNG diagnostics, all 10,500 case
comparisons, the patched aggregate sweep CSV, the old/new summary, hashes, and
vector-QA output.
