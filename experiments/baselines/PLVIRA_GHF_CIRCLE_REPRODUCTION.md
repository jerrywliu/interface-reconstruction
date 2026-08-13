# PLVIRA GHF Cartesian circle reproduction

## Protocol

This is a deterministic static subset of the generalized-height-function
circle study in Popinet (2009), Section 6.1 and Figure 5, which is the GHF
method cited for PLVIRA by Remmerswaal and Veldman.

- Domain: `[-0.5, 0.5]^2`.
- Circle radius: `R = 0.2`; exact curvature `1/R = 5`.
- Resolutions: `N = 32, 64, 128, 256` (`R/h = 6.4, 12.8, 25.6, 51.2`).
- Four fixed translations, expressed in cell widths:
  `(-0.31, 0.17)`, `(0.13, -0.27)`, `(0.37, 0.41)`,
  `(-0.43, -0.11)`.
- Volume fractions: analytic circle-square intersections.
- Errors: relative per-mixed-cell curvature error, aggregated as arithmetic
  `L1`, RMS `L2`, and maximum `Linf` over all four translations.
- Grid class: uniform Cartesian only.

Machine-readable output:

- `results/plvira_ghf_circle/results.json`
- `results/plvira_ghf_circle/results.csv`

## Results

| `N` | `R/h` | samples | relative `L1` | order | relative `L2` | order | relative `Linf` | order |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 6.4 | 204 | `1.79837e-2` | -- | `7.13945e-2` | -- | `1.00000` | -- |
| 64 | 12.8 | 410 | `3.02178e-3` | `2.573` | `3.10373e-3` | `4.524` | `5.04029e-3` | `7.632` |
| 128 | 25.6 | 820 | `7.44699e-4` | `2.021` | `7.62640e-4` | `2.025` | `1.19464e-3` | `2.077` |
| 256 | 51.2 | 1638 | `1.85551e-4` | `2.005` | `1.89884e-4` | `2.006` | `2.93941e-4` | `2.023` |

The `128 -> 256` orders are `2.005`, `2.006`, and `2.023` for relative
`L1`, `L2`, and `Linf`, respectively. This reproduces the second-order
asymptotic GHF curvature scale reported by Popinet.

At `N=32`, `203/204` cells use a complete height function and one cell reaches
the published Algorithm 7 zero-curvature degeneracy fallback, producing the
unit relative `Linf` error. All `410 + 820 + 1638` samples for `N>=64` use a
complete height function; there are no zero or fitted fallbacks in those rows.
This coarse event is retained rather than hidden or tuned away. It exposes the
material difference between Popinet's published one-cell independence rule and
the later half-cell implementation rule documented in
`docs/baselines/PLVIRA_IMPLEMENTATION.md`.

## Reproduction command

```bash
PYTHONPATH=. python experiments/baselines/run_plvira_ghf_circle_convergence.py
```

This check validates the Cartesian GHF provider. It is not the
Remmerswaal--Veldman flower reconstruction curve: their linked flower program
uses exact level-set curvature and is therefore an exact-curvature oracle.
