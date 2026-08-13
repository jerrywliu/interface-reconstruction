# PCIC Cartesian Project Smoke

## Scope

This matched mini resolution study runs both frozen bare-PCIC conservative
corrections on cases 0--4 of all five project benchmarks at `N = 32, 64, 128`.
Every mesh is uniform, axis-aligned Cartesian. No policy was adjusted after
examining benchmark results.

The frozen choices are:

- infer phase from the oriented central PLIC;
- use `0.5` for the overcrowded LLS radius multiplier;
- select the nearest conservative center-translation root;
- select a multi-arc chord by PLIC proximity/alignment while preserving every
  paired component;
- report center translation and radius adjustment as distinct variants.

Command:

```bash
MPLBACKEND=Agg PYTHONPATH=. python -m \
  experiments.baselines.run_pcic_project_smoke \
  --output experiments/baselines/results/pcic_project_smoke_20260813_final \
  --benchmarks lines,squares,circles,ellipses,zalesak \
  --resolutions 32,64,128 --cases 0,1,2,3,4 \
  --corrections translate_center,adjust_radius
```

The run took `177.24 s` wall time. Timed reconstruction calls account for
`87.95 s`; fixture construction, metric sampling, serialization, and plotting
account for the remainder.

## Result contract

- `case_results.csv` has 150 rows: two corrections times five benchmarks,
  five cases, and three resolutions.
- `cells/*.csv` contains one row per mixed cell with status, conservation,
  local unsigned curvature error, component counts, crossings, fitted and
  corrected circle parameters, fallback details, and diagnostics.
- `geometry/*.json` preserves every native component through the shared
  external-baseline schema.
- `summary.csv` and `summary.json` aggregate each correction separately.
- `pcic_both_variants_all_benchmarks_summary.pdf` is a one-page vector
  comparison across all five benchmarks.
- `run_manifest.json` records hashes, exact cases, policies, metrics, boundary
  behavior, and runtime.

The curvature diagnostic is the unsigned local error
`|1/|R| - kappa_truth|` at each returned arc midpoint. A published
straight-line-limit fallback has zero estimated curvature. The geometry
diagnostic is sampled and has a nearest-sample floor near `1.25e-3`; it is a
smoke-test diagnostic rather than the paper's adaptive native metric.

## Coverage and outcomes

| Correction | Mixed | Reconstructed | Straight fallback | Unsupported | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Center translation | 7,380 | 6,840 | 289 | 135 | 116 |
| Radius adjustment | 7,380 | 6,436 | 289 | 135 | 520 |

All 135 unsupported cells per correction are line-interface cells without the
complete 7x7 Cartesian predictor halo required by bare PCIC. No one-sided or
project-method fallback was inserted.

Center translation is substantially more complete on cornered interfaces:

| Benchmark | Center-translation unresolved | Radius-adjustment unresolved |
| --- | ---: | ---: |
| Lines | 26 | 31 |
| Squares | 21 | 182 |
| Circles | 7 | 5 |
| Ellipses | 28 | 29 |
| Zalesak | 34 | 273 |

The radius-adjustment failures are conservative-correction bracketing failures
(`516/520`, with four value errors). Center translation has 86 bracketing
failures, 26 ambiguous multi-arc source choices, and four value errors. These
are retained as unresolved source-method outcomes rather than tuned away.

There are 18 multi-component center-translation cells and eight
multi-component radius-adjustment cells. The largest exact local conservation
residuals are `9.51e-10` and `9.33e-10`, respectively.

## Numerical signal

On circles, both corrections give approximately second-order mean facet-gap
convergence over `N = 32, 64, 128` (`1.95` center translation and `1.94`
radius adjustment). On ellipses, both give approximately third-order mean
facet-gap convergence (`2.99` and `2.98`). Their median local curvature errors
decrease from about `8.7e-4` at `N=32` to `4.4e-4` at `N=64`, then remain near
`4.8e-4` at `N=128`; this three-resolution smoke does not establish
second-order curvature convergence.

The cornered-benchmark error curves must be read together with completeness.
Radius adjustment can show smaller errors on the geometry it returns while
omitting many more difficult square and Zalesak cells. Center translation is
therefore the more robust of these two frozen PCIC interpretations for the
matched project matrix.

## Artifact QA

The summary PDF is one page, contains no raster objects, and embeds its font.
Focused runner, PCIC, adapter, fixture, and metric verification passes `47`
tests. Key SHA256 values are:

- summary PDF: `67b049feab7749853bfb85b69a7ae58851a022de5da9c315c01fd9f6caa8f210`;
- case CSV: `e266fd4515c488d18f9847127207612c77d235146155aee550f27a9a2181d286`;
- summary CSV: `90163be9c616096c8be5450b1497d8efb85ef0ea19924d150c964e2e800da1c2`;
- summary JSON: `6dc02c0e31c368ea01ecfce1344b33e4668b98f8bcdf5b93d9dcfda13cdc374f`;
- manifest: `e6efb2a7d35ec61e5bf5a73f3235ab12f48dcbc1438fb27ae1c453dc61fbc145`.
