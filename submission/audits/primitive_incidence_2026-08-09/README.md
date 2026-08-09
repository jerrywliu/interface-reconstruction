# Final primitive and fallback incidence audit

This audit reads the sealed submission sweep
`submission_static_20260731_012430_505aefa45432.sealed`, generated from source
commit `505aefa454328d4ba34ade5e7247050a0acfc793`.

## Scope

One paper-facing method is used for each benchmark so that the same geometry is
not counted repeatedly across nested variants:

| Benchmark | Selected method |
| --- | --- |
| Lines | `linear` |
| Circles | `circular` |
| Ellipses | `circular` |
| Squares | `linear+corner` |
| Zalesak | `circular+corner` |

The resulting population is exactly the one used by the manuscript merging
audit: 3,500 instances and 443,698 original mixed cells. The five final
primitive categories are mutually exclusive and close to every mixed cell.
Fallbacks are a separate provenance overlay because both fallback paths produce
a straight facet already counted in the linear category.

## Final primitive geometry

| Benchmark | Instances | Mixed cells | Linear | Circular | Line-line | Line-arc | Arc-arc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lines | 750 | 95,040 | 100.0000% | 0 | 0 | 0 | 0 |
| Circles | 750 | 52,855 | 0.0208% | 99.9792% | 0 | 0 | 0 |
| Ellipses | 750 | 120,180 | 0.0483% | 99.9517% | 0 | 0 | 0 |
| Squares | 625 | 62,581 | 87.4259% | 0 | 12.5741% | 0 | 0 |
| Zalesak | 625 | 113,042 | 31.8050% | 60.1192% | 2.5389% | 5.5289% | 0.0080% |
| All | 3,500 | 443,698 | 41.8695% | 54.2995% | 2.4203% | 1.4086% | 0.0020% |

## Fallback provenance

Cell entries are normalized by mixed cells in the benchmark. Instance entries
are normalized by the number of benchmark instances.

| Benchmark | Local-line cells | Instances | LVIRA cells | Instances |
| --- | ---: | ---: | ---: | ---: |
| Lines | 0 | 0 | 293 (0.3083%) | 2 (0.267%) |
| Circles | 1 (0.0019%) | 1 (0.133%) | 4 (0.0076%) | 2 (0.267%) |
| Ellipses | 10 (0.0083%) | 8 (1.07%) | 12 (0.0100%) | 6 (0.800%) |
| Squares | 0 | 0 | 22 (0.0352%) | 13 (2.08%) |
| Zalesak | 0 | 0 | 28 (0.0248%) | 16 (2.56%) |
| All | 11 (0.0025%) | 9 (0.257%) | 359 (0.0809%) | 39 (1.11%) |

The line LVIRA total is concentrated in two instances. The `N=150`, `w=0.3`,
case-index 3 instance contributes 291 cells; its Hausdorff error is
`1.5351545880206654e-07` and its facet-gap error is zero.

## Reproduction

Run from the repository root:

```bash
python submission/summarize_primitive_incidence.py \
  results/static/submission_static_20260731_012430_505aefa45432.sealed \
  --output-dir results/submission/primitive_incidence_505aefa
```

The generated directory contains benchmark, setting, and case-indexed CSVs,
the machine-readable summary, and the rendered Markdown report. The detailed
case ledger is generated locally and is not included in the compact submission
repository.
