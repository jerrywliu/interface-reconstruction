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

The remaining reproducibility risk is primarily platform breadth rather than
algorithmic. The launcher captures the exact sweep environment, while the pinned
public stack now has a clean macOS arm64 CPython 3.9 install, full-suite, numerical
smoke, and vector-figure validation. The two stacks agree to the tested tolerances
but are not claimed to be bitwise portable. The release audit can generate and
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
| Dependency capture | Good for the accepted run and public target | `environment.json` records the exact sweep stack. The frozen requirements independently install cleanly on macOS arm64 CPython 3.9 and pass `pip check`; see `submission/CLEAN_ENV_REPRODUCIBILITY_VALIDATION.md`. |
| Environment consistency | Accepted with limited scope | The sweep stack differs from the public pins, but three paper-facing smokes showed no reconstruction-decision change and only negligible roundoff differences. Preserve both records and do not claim bitwise or cross-platform identity. |
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

Packaging policy after the clean-environment validation:

1. Treat the accepted sweep's captured environment as archival authority for the
   submitted result set.
2. Treat the checked-in pins as the public clean-install target validated on macOS
   arm64 CPython 3.9.
3. Consider a lock file or optional dependency groups only after submission.
4. Repeat the compact acceptance suite on clean Linux before claiming broader
   platform support.

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

### Command provenance schema

The current final release predates tokenized command capture and stores each
controller and child invocation using the lossy `" ".join(sys.argv)` representation.
The audit therefore permits command-string parsing only for the byte-verified
`505aefa` source release, schema version 1, a historical repository root without
whitespace, and a canonical serialization in which no token contains whitespace.
This exception cannot safely support relocation to a path containing spaces. The
executable is still compared as an exact lexical repository-relative path; the
historical root is provenance metadata and need not equal the verifier's checkout.

The controller and child-manifest producers now write an authoritative `argv` JSON
array for future runs and render the optional `command` display field with
`shlex.join`. When both fields are present, the audit requires `argv` to equal the
POSIX-tokenized `command` exactly. Any future source release, noncanonical command,
or path containing whitespace must provide `argv`; it cannot use the legacy
exception.

### Source snapshot trust

The audit disables Git replacement objects for every Git operation and verifies the
exact commit and blob object hashes itself. It enumerates the commit with
`git ls-tree -rz` and reads each allowed regular-file blob with `git cat-file`, so
archive attributes such as `export-ignore` cannot remove tracked source from the
oracle. A seek-aware wrapper limits gzip output to the canonical Git-tree payload
plus fixed per-member and global metadata allowances before `tarfile` can process
PAX or GNU metadata. A separate bounded single-member gzip pass forces CRC and
trailer validation and rejects concatenated members or compressed trailing data.
Before extracting any archived source payload, the audit also checks the complete
tar metadata pass against tree-derived bounds for file count, each file size, total
uncompressed bytes, path, and the complete executable mode; set-id and sticky bits
are rejected. Finally, it drains gzip through EOF and requires exactly two zero end
blocks plus zero padding to the next tar record boundary after the final member.
The archived bytes are then compared with the corresponding verified Git blobs.

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

## Completed And Remaining Verification

The new environment utility has focused tests for requirements comparison, Git
source/generated-state classification, input fingerprints, and atomic JSON output.
The isolated pinned environment passed `193` tests with the live Slack integration
skipped, passed `pip check`, completed three representative numerical smokes, and
generated two vector-only deterministic figures. Exact evidence is in
`submission/CLEAN_ENV_REPRODUCIBILITY_VALIDATION.md`. The integrated branch also
passes its expanded suite after all release tools are combined; that verification
is recorded in the branch handoff and project run memory.

The final release still needs these operational checks:

1. Run the full repository suite from the exact sweep environment and archive the
   pytest summary.
2. Preserve the clean pinned-environment report and document the unrelated
   Torch/TorchSDE conflicts in the sweep workstation's captured environment.
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
- Preserve the explicit dual-environment policy and accepted tolerance evidence.
- Audit final row counts, key uniqueness, missing/nonfinite values, failures, and raw
  bundle coverage.
- Add and verify a checksum manifest for the complete final release.

### P1 Before Public Code Release

- Repeat clean-environment numerical acceptance on Linux before declaring broader
  Python/platform support.
- Make aggregate manifest paths release-relative.
- Add a compact public reproduction command for every paper table/figure after the
  experiment-to-code map is finalized.

### P2 After Submission

- Test a second operating system or container image for numerical tolerance-level
  reproducibility.
- Add schema validation and migrations for run/sweep manifests.
- Consider content-addressed raw bundles if long-term archival size permits.
