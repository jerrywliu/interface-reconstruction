# Submission Code Structure

Updated: 2026-08-03

This document defines the supported public tree for the paper submission. It
separates the frozen reconstruction path from experiment orchestration and
submission verification.

## Frozen Production Method

The paper-facing method is:

```text
corner behavior: pre_f8_corner
rescue profile:  exact_linear_support_only
PLIC fallback:   LVIRA
```

These defaults are defined by `MergeMesh`, passed through
`util/reconstruction.py`, exposed by the benchmark CLIs, fixed by
`submission/submission_config.json`, and locked by
`test/experiments/test_sweep_profiles.py`.

The optional guarded continuity pass and connected-component diagnostic are not
part of the default reconstruction. Their submission role is documented in the
paper experiment map and dedicated diagnostics.

## Supported Tree

```text
main/
  algos/plic_normals.py       Youngs and ELVIRA normal estimates
  geoms/                      geometric solves for line, arc, and corner facets
  structs/facets/             facet data structures
  structs/polys/              per-cell and merged-zone fitting operations
  structs/meshes/             mesh topology, merging, orientation, and fitting
util/
  initialize/                 mesh and benchmark initialization
  metrics/                    geometry-aware error metrics
  plotting/                   VTK and Matplotlib output
  reconstruction.py           supported reconstruction dispatcher
  reconstruction_diagnostics.py
experiments/static/
  {lines,squares,circles,ellipses,zalesak}.py
  run_perturbed_sweeps.py     shared static sweep controller
  run_appendix_*.py           paper appendix companion studies
  generate_*.py               paper and review figure producers
experiments/submission/       claim-specific validation and diagnostics
submission/                   freeze, audit, figure, and package gates
test/                         unit, integration, release, and provenance tests
config/static/                paper benchmark configurations
docs/                         public navigation and reproduction maps
```

`run.py`, `util/advection.py`, and `config/advection/` remain supported research
surfaces because they exercise the same current reconstruction structures, but
they are not sources for paper results.

## Paper Workflow

1. `submission/run_final_static_sweep.sh` launches the exact full benchmark
   matrix and records raw, case, cell, merge, fallback, and geometry provenance.
2. `submission/audit_final_release.py` validates and seals the numerical result
   set.
3. `submission/run_final_figure_orchestrator` regenerates allowlisted direct
   vector PDFs from the sealed release and pinned source commits.
4. `submission/package_submission.py` binds the selected figures, paper source,
   code tree, checksums, and external deposit metadata into one deterministic
   package.

Every paper result and plot is mapped to its producer and run artifacts in
`docs/PAPER_EXPERIMENT_MAP.md`.

## Historical Code Policy

The submission branch removes the pre-package implementation, non-pytest
examples, hard-coded wrappers, superseded March camera-ready bundler, dated
ablations, historical sweep/metric-repair utilities, and stale experiment
notes. Their history remains available through Git; none is required to run the
submitted method, reproduce the paper experiments, or regenerate the figures.

## Deliberately Retained Material

The following items were not removed:

- alternative corner/rescue profiles used by regression and provenance tests;
- advection research code and configurations that use the current structures;
- artifact-specific finalizers used as adversarial fixtures by the figure
  trust-boundary tests;
- tracked historical static summary text files and all CSV/PDF/checksum audit
  evidence;
- submission diagnostics, including negative and failure evidence;
- result data and compact audit evidence.

No tracked paper result or diagnostic artifact was deleted. Two apparently
redundant raster previews were considered, then retained because one is
checksum-bound and the other is part of a documented before/after evidence
packet.

## Verification

From a clean CPython 3.9 environment:

```bash
python -m pytest -q test
python submission/check_submission_freeze.py --source-only
python -m compileall -q main util experiments submission
```

The source-only freeze check requires a clean committed tree, so run it after
committing the proposed public structure.
