# Code And Reproducibility Audit

Date: 2026-07-30

Scope: submission packaging and reproducibility only

Initial audited source: `3ce0ebc` (`codex/repro-audit-20260730` worktree)

This audit does not change reconstruction, static experiment drivers, sweep
selection, plotting, or numerical behavior. It reviews the release surface that
surrounds the frozen submission method.

## Executive Assessment

The final static sweep is scientifically well instrumented. Its canonical launcher
fixes the benchmark grid and method profiles, checks for uncommitted source, assigns
a collision-proof namespace, saves exact case geometry and case/cell/merge/fallback
diagnostics, snapshots the source, and copies every raw run into the release root.
The deterministic case and mesh seeds are explicit.

The remaining reproducibility risk is primarily environmental rather than
algorithmic. The launcher captures the exact sweep environment, but the checked-in
requirements do not match the validation environment and a clean-environment
numerical acceptance run is still required. The release audit can generate and
verify a whole-release checksum manifest after the approved figure packet is added.

## Status By Area

| Area | Status | Evidence and consequence |
|---|---|---|
| Frozen entry point | Good | `submission/run_final_static_sweep.sh` is the single submission launcher and spells out all 970 runs / 24,250 cases. |
| Source cleanliness | Good | `submission/check_submission_freeze.py --source-only` rejects tracked and untracked changes outside generated roots. |
| Deterministic cases | Good for the final sweep | The launcher fixes perturbation seed `0`; static drivers use benchmark-specific constant RNG seeds for case geometry and record both seeds in manifests. |
| Configuration discovery | Adequate for the launcher | The resolved submission config is archived. Direct module/config entry points still assume the repository root as the working directory. |
| Run manifests | Good | Per-run manifests record source commit/branch, command, numerical parameters, and relative artifact names. Sweep manifests record planned/successful counts and failures. |
| Raw-bundle self-containment | Good with one portability gap | Scientific raw runs, consolidated diagnostics, source snapshot, logs, resolved config, and environment capture live under one result root. Disposable raster previews are omitted because figures replay from exact VTK/metadata. The sweep manifest still stores some absolute artifact paths. |
| Dependency capture | Good for the accepted run | `environment.json` records Python/platform details, installed packages, NumPy/SciPy builds, GEOS, FreeType, VTK, and requirements fingerprints. General public support still needs clean-environment acceptance. |
| Environment consistency | Needs action | The audited Python 3.9.13 environment differs from checked-in pins, including NumPy `1.24.4` vs `1.23.4`, SciPy `1.8.1` vs `1.9.2`, and Matplotlib `3.8.3` vs `3.6.1`. `pip check` also fails because of unrelated Torch packages. |
| Test capture | Good for the declared target | `requirements-test.txt` and a CPython 3.9 GitHub Actions workflow run the source audit and full suite. A broader Python/platform matrix is not claimed. |
| Generated-file hygiene | Good | `.gitignore` excludes scratch roots and raster previews while keeping release CSV/JSON/PDF/VTK/manifests visible; `docs/GENERATED_FILES.md` records the policy. |
| Stale scripts | Classified | `docs/ENTRY_POINTS.md` distinguishes canonical, supported research, and legacy/superseded paths without deleting replay code. |

## Completed Low-Risk Cleanup

Added `submission/capture_environment.py`, a standard-library capture utility that
records:

- Git commit, tree, branch, timestamp, submodule state, and source cleanliness;
- Python implementation, version, ABI, executable, platform, architecture, locale,
  and selected numerical-threading environment variables;
- all installed Python distributions and a comparison with exact requirements pins;
- `pip check` output;
- NumPy/SciPy build configuration, Matplotlib backend and FreeType, Shapely GEOS,
  and VTK version when importable;
- SHA-256 fingerprints for the requirements, submission config, base config, and
  static benchmark configs.

The utility reads only an allowlist of environment variables and does not capture
credentials. It writes JSON atomically and does not modify any run behavior.

Suggested release usage after the final result root exists:

```bash
python submission/capture_environment.py \
  --output results/static/<final-release>/environment.json
```

The approved final launcher now runs this command before the sweep from the exact
execution environment and stores `environment.json` in the release root.

## Determinism Findings

The submission sweep separates two deterministic random streams:

- case geometry uses fixed benchmark constants (`41`, `42`, or `43` depending on
  the driver);
- perturbed mesh generation receives explicit seed `0` from the sweep controller.

Each run is a separate subprocess with a unique output name. Parallel scheduling
therefore does not share RNG state across cases, and controller collection order is
the precomputed specification order. Exact case geometry is saved, so a replay does
not have to infer the sampled interfaces.

The guarantee does not extend to every historical entry point. In particular,
`main/algos/static_interface_reconstruction.py` uses the process-global `random`
module without seeding. It is not used by the final submission launcher and should
be labeled legacy or retired in a later cleanup rather than silently treated as a
reproducible paper driver.

Bitwise-identical output across different numerical stacks is not established.
SciPy optimization, BLAS/LAPACK, GEOS/Shapely, VTK, Matplotlib, and font-library
versions can affect convergence paths, geometry predicates, serialized meshes, or
PDF rendering. The environment capture makes those differences diagnosable; it does
not substitute for rerunning a compact cross-environment acceptance suite.

## Packaging And Dependency Findings

`requirements.txt` pins runtime and transitive packages together and includes the
formatter `black`. Additive `requirements-test.txt` and
`requirements-figures.txt` now identify `pytest` and `reportlab` without changing
the frozen numerical stack. There is still no package metadata, lock file,
environment file, container definition, or broad supported-platform declaration.
The code is run in place through `python -m ...`, so installation metadata is not
required for the current launcher.

Recommended packaging follow-up after the result freeze:

1. Decide whether the authoritative submission environment is the checked-in pin
   set or the environment that produced the accepted final results.
2. Validate that environment from a clean virtual environment and record the exact
   Python version and platform.
3. Consider a lock file or optional dependency groups only after the accepted
   environment and numerical tolerance checks are recorded.
4. Add one clean-environment numerical acceptance job. A broad platform matrix can
   follow after submission.

Do not update numerical library pins immediately before the sweep without rerunning
the acceptance tests; that would be a scientific change even if the source code is
unchanged.

## Entry Points And Configuration

The final submission path is discoverable and explicit:

```bash
bash submission/run_final_static_sweep.sh
```

The launcher changes to the repository root before invoking modules, which avoids
the relative-path assumptions in `util/config.py`. Direct use of `run.py` and some
legacy shell scripts remains dependent on being launched from the repository root.
The root README points readers to `docs/ENTRY_POINTS.md`,
`docs/DEPENDENCIES.md`, and `docs/PAPER_EXPERIMENT_MAP.md`; the launcher and map,
not historical command examples later in the README, define the submission recipe.

For the final archive, retain these three distinct records:

- `submission_config.resolved.json`: intended scientific configuration;
- `sweep_manifest.json`: execution status and aggregate artifact locations;
- `diagnostics/run_manifests.jsonl`: actual parameters and source identity for each
  completed run.

The submission audit should compare the three instead of trusting any one file.

## Result And Raw-Bundle Findings

The release path is robust against accidental overwrite. Each completed run's exact
mesh, facets, metrics, case geometry, and provenance are copied into
`raw_runs/<namespaced-save-name>/`, marked read-only, and then used to populate
release-relative inventory rows. Per-case raster previews are excluded. The
temporary namespaced source run is removed only after the archived scientific file
inventory matches and consolidation succeeds. Required diagnostic files must exist
and parse before the controller counts a run as successful.

Remaining archival risks:

- `sweep_manifest.json` uses absolute paths for aggregate artifacts, so it is not
  fully relocatable even though the raw-run inventory is relative;
- a whole-release SHA-256 manifest is intentionally absent until approved figures
  are installed;
- read-only permission bits prevent casual edits but are not an integrity proof;
- `submission/audit_final_release.py` now verifies expected run/case counts,
  duplicate keys, required finite metrics, raw-bundle coverage, source/config
  consistency, and optional checksum generation/verification.

Before upload, generate a sorted SHA-256 manifest rooted at the release directory,
copy the bundle, verify the manifest at the destination, and retain the audit report
beside it. The checksum manifest should be generated only after figures and the
environment record are final.

## Generated Files And Stale Code

The repository now ignores disposable scratch roots and raster previews while
leaving release CSV, JSON, PDF, SVG, VTK, environment, checksum, and manifest files
visible. `docs/GENERATED_FILES.md` records this policy. `docs/ENTRY_POINTS.md`
classifies `run_old.py`, old initialization modules, historical camera-ready
launchers, and optional research tooling without deleting paths needed for replay.

## Tests And Verification Still Required

The new environment utility has focused tests for requirements comparison, Git
source/generated-state classification, input fingerprints, and atomic JSON output.
The complete repository suite passes `186/186` tests with this audit patch applied;
Python compilation and `git diff --check` also pass. The current environment does
not provide the declared `black` executable, so formatter availability is itself
part of the dependency-capture issue described above.

The final release still needs these operational checks:

1. Run the full repository suite from the exact sweep environment and archive the
   pytest summary.
2. Run `python -m pip check`; either make it clean in the isolated environment or
   document why every reported conflict is outside the submission dependency set.
3. Run the source freeze checker immediately before launch.
4. After the sweep, assert 970 successful runs, 24,250 unique case rows, zero missing
   required bundles, zero missing/nonfinite required metrics, and no unexpected
   duplicate run/case keys.
5. Replay a small deterministic sample from archived raw bundles and compare metrics
   within the accepted numerical tolerance.
6. Generate and verify the final release checksum manifest after figure promotion.

## Remaining Risks By Priority

### P0 Before Declaring The Release Reproducible

- Verify the automatically captured sweep environment in the release.
- Resolve or explicitly accept the requirements-versus-installed-version mismatch.
- Use a clean environment whose relevant dependency graph passes `pip check`.
- Audit final row counts, key uniqueness, missing/nonfinite values, failures, and raw
  bundle coverage.
- Add and verify a checksum manifest for the complete final release.

### P1 Before Public Code Release

- Complete clean-environment numerical acceptance before declaring broader
  Python/platform support.
- Make aggregate manifest paths release-relative.
- Add a compact public reproduction command for every paper table/figure after the
  experiment-to-code map is finalized.

### P2 After Submission

- Test a second operating system or container image for numerical tolerance-level
  reproducibility.
- Add schema validation and migrations for run/sweep manifests.
- Consider content-addressed raw bundles if long-term archival size permits.
