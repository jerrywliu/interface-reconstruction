# Code And Reproducibility Audit

Date: 2026-07-30

Scope: submission packaging and reproducibility only

Audited source: `3ce0ebc` (`codex/repro-audit-20260730` worktree)

This audit does not change reconstruction, static experiment drivers, sweep
selection, plotting, or numerical behavior. It reviews the release surface that
surrounds the frozen submission method.

## Executive Assessment

The final static sweep is scientifically well instrumented. Its canonical launcher
fixes the benchmark grid and method profiles, checks for uncommitted source, assigns
a collision-proof namespace, saves exact case geometry and case/cell/merge/fallback
diagnostics, snapshots the source, and copies every raw run into the release root.
The deterministic case and mesh seeds are explicit.

The remaining reproducibility risk is primarily environmental and archival rather
than algorithmic. The repository does not declare a supported Python/platform
envelope, the checked-in requirements do not match the environment currently used
for validation, and the release has no environment record or whole-release checksum
manifest. Those items should be resolved or explicitly captured before a submission
bundle is called independently reproducible.

## Status By Area

| Area | Status | Evidence and consequence |
|---|---|---|
| Frozen entry point | Good | `submission/run_final_static_sweep.sh` is the single submission launcher and spells out all 970 runs / 24,250 cases. |
| Source cleanliness | Good | `submission/check_submission_freeze.py --source-only` rejects tracked and untracked changes outside generated roots. |
| Deterministic cases | Good for the final sweep | The launcher fixes perturbation seed `0`; static drivers use benchmark-specific constant RNG seeds for case geometry and record both seeds in manifests. |
| Configuration discovery | Adequate for the launcher | The resolved submission config is archived. Direct module/config entry points still assume the repository root as the working directory. |
| Run manifests | Good | Per-run manifests record source commit/branch, command, numerical parameters, and relative artifact names. Sweep manifests record planned/successful counts and failures. |
| Raw-bundle self-containment | Good with two gaps | Raw runs, consolidated diagnostics, source snapshot, logs, and resolved config live under one result root. The sweep manifest stores absolute artifact paths, and the release has no file-level checksum inventory. |
| Dependency capture | Needs action | `requirements.txt` is pinned, but no Python version, OS/architecture, BLAS/LAPACK, GEOS, FreeType, or full installed-package record is archived. |
| Environment consistency | Needs action | The audited Python 3.9.13 environment differs from checked-in pins, including NumPy `1.24.4` vs `1.23.4`, SciPy `1.8.1` vs `1.9.2`, and Matplotlib `3.8.3` vs `3.6.1`. `pip check` also fails because of unrelated Torch packages. |
| Test capture | Partial | The repository suite passes, but there is no CI workflow, pytest configuration, declared test dependency set, or tested Python/platform matrix. |
| Generated-file hygiene | Needs cleanup after freeze | `.gitignore` covers `plots/*` and result PNGs, but not result CSV/JSON/PDF, `output/`, `tmp/`, `logs/`, or `.pytest_cache/`. This is the source of the large untracked worktree seen during July work. |
| Stale scripts | Needs classification, not deletion now | Legacy advection/reconstruction modules and several older camera-ready launchers remain callable. Removing or redirecting them immediately would be higher risk than documenting the canonical submission path. |

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

This command is intentionally not wired into the frozen launcher in this audit.
Until a separate approved launcher edit is made, run it from the exact environment
that executes the sweep and include `environment.json` in the final archive.

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
formatter `black`, but omits at least the test runner `pytest` and figure-packet
dependency `reportlab`. There is no `pyproject.toml`, package metadata, lock file,
environment file, container definition, or supported Python version declaration.
The code is normally run in-place through `python -m ...`, so installation metadata
is not required for the current launcher, but a fresh user cannot infer which pins
are runtime, test, plotting, or optional tooling.

Recommended packaging follow-up after the result freeze:

1. Decide whether the authoritative submission environment is the checked-in pin
   set or the environment that produced the accepted final results.
2. Validate that environment from a clean virtual environment and record the exact
   Python version and platform.
3. Split runtime, test, and figure-tool dependencies, or encode them as optional
   dependency groups in `pyproject.toml`.
4. Add one clean-environment installation smoke and the focused reconstruction test
   suite to CI. A broad platform matrix can follow after submission.

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
The root README also describes older experiment commands and known issues alongside
the newer workflow docs, so it should not be considered the authoritative
submission recipe.

For the final archive, retain these three distinct records:

- `submission_config.resolved.json`: intended scientific configuration;
- `sweep_manifest.json`: execution status and aggregate artifact locations;
- `diagnostics/run_manifests.jsonl`: actual parameters and source identity for each
  completed run.

The submission audit should compare the three instead of trusting any one file.

## Result And Raw-Bundle Findings

The release path is robust against accidental overwrite. Each completed run is
copied into `raw_runs/<namespaced-save-name>/`, marked read-only, and then used to
populate release-relative inventory rows. Required diagnostic files must exist and
parse before the controller counts a run as successful.

Remaining archival risks:

- `sweep_manifest.json` uses absolute paths for aggregate artifacts, so it is not
  fully relocatable even though the raw-run inventory is relative;
- only the source tarball has a recorded SHA-256 digest; raw runs, consolidated CSVs,
  figures, logs, resolved config, and environment capture do not;
- read-only permission bits prevent casual edits but are not an integrity proof;
- no single verifier currently checks expected run/case/cell counts, duplicate keys,
  missing/nonfinite metrics, raw-bundle existence, and all file hashes after copying
  the release to its archival destination.

Before upload, generate a sorted SHA-256 manifest rooted at the release directory,
copy the bundle, verify the manifest at the destination, and retain the audit report
beside it. The checksum manifest should be generated only after figures and the
environment record are final.

## Generated Files And Stale Code

The narrow ignore rules are useful during a source audit because unexpected files
remain visible, but they make ordinary development status difficult to read. A later
cleanup should choose one of two policies:

1. ignore all generated roots and rely on explicit release directories/manifests; or
2. keep result metadata visible but ignore timestamped run directories through a
   documented naming convention.

Apply that policy only after the final bundle is archived so it does not hide files
that still need review. Also classify `run_old.py`, old initialization modules,
historical camera-ready launchers, and optional VisIt tooling in a deprecation index.
Do not delete them based solely on names; some remain useful for historical replay.

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

- Capture the actual sweep environment and include it in the release.
- Resolve or explicitly accept the requirements-versus-installed-version mismatch.
- Use a clean environment whose relevant dependency graph passes `pip check`.
- Audit final row counts, key uniqueness, missing/nonfinite values, failures, and raw
  bundle coverage.
- Add and verify a checksum manifest for the complete final release.

### P1 Before Public Code Release

- Declare supported Python/platform versions and separate runtime/test/figure deps.
- Add CI for clean installation and focused/full tests.
- Make aggregate manifest paths release-relative.
- Document or retire legacy entry points and stale launchers.
- Adopt a deliberate generated-file ignore/archive policy.
- Add a compact public reproduction command for every paper table/figure after the
  experiment-to-code map is finalized.

### P2 After Submission

- Test a second operating system or container image for numerical tolerance-level
  reproducibility.
- Add schema validation and migrations for run/sweep manifests.
- Consider content-addressed raw bundles if long-term archival size permits.
