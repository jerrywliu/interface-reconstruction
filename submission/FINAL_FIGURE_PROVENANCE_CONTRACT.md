# Final Figure Provenance Contract

The submission figure set is produced only by
`submission/final_figure_orchestrator.py`. Existing candidate directories and
caller-authored source assertions are not accepted.

## Source Attestation

The wrapper requires a full 40-hex approved generator commit and proves all of
the following before it creates any output:

1. The approved and scientific-release commits exist locally.
2. The approved generator commit descends from the scientific-release commit
   recorded in the audited final release.
3. The checkout `HEAD` and index tree equal the approved commit.
4. `git status --porcelain --untracked-files=all` is empty.
5. No tracked path has `assume-unchanged`, `skip-worktree`, or another
   nonstandard index flag.
6. Every tracked regular file, executable, and symlink has the Git blob bytes
   and mode stored in the approved commit.

The resulting `generator_checkout` record contains both commits, the commit
tree, tracked-file count, and a SHA-256 digest of the complete tracked-byte
inventory.

## Controlled Generation

The wrapper creates private execution and publication staging directories. Each
generator destination and both candidate roots must initially be nonexistent.
The wrapper invokes the commands itself, validates their scientific manifests,
and immediately copies the selected PDFs and manifests into private staging.

The fixed scientific contracts are:

- Final release: 970 completed runs, 24,250 cases, the exact method sets and
  resolution/perturbation grids in `submission_config.resolved.json`, seed 0,
  25 cases per setting, and the `LVIRA`/`pre_f8_corner`/
  `exact_linear_support_only` profile.
- Main text: all five experiments, exact method sets, paired endpoint variants,
  and representative cases `lines=6`, `squares=24`, `circles=12`,
  `ellipses=12`, `zalesak=12`.
- Resolution appendix: exact cases `0/22/12/12/20`, the designated best method,
  `N=16,32,64`, perturbations `0,0.1`, seed 0, and six newly completed runs per
  benchmark. All 30 `run_manifest.json` files are copied and hashed.
- Guarded C0 appendix: six ellipse and five Zalesak resolutions, five
  perturbations, three exact variants, seed 0, and 25 cases per setting. All
  165 settings must be newly completed; all 165 run manifests are copied and
  hashed. `plot_from_csv`, planned, collected, missing, and partial manifests
  are rejected.
- Deterministic PLIC: line case 4, cell `(14,13)`, `N=32`, perturbation `0.3`,
  seed 0.
- Deterministic staged reconstruction: Zalesak case 22, `N=100`, perturbation
  `0.1`, seed 0, radius 15, slot width 5, and relative slot top 10.

The ordinary plotting CLIs remain general-purpose. They do not require final
release arguments and do not establish submission provenance on their own.

## Candidate And Publication Gate

`submission/final_figure_candidates.json` is the exact allowlist: 14 unpaired
PDFs plus 12 paired slots, for 38 PDFs total. The all-method plot command may
create auxiliary PDFs; the wrapper copies only its five allowlisted outputs.

The sealed orchestration manifest records SHA-256 and size for every copied
manifest, command record, release anchor, and candidate. Acceptance then:

- rejects missing or unexpected candidate PDFs;
- requires exactly one page per candidate;
- runs `submission/pdf_vector_qa.py` fail-closed;
- renders fresh 300-DPI previews directly from accepted PDFs and verifies their
  dimensions;
- builds the indexed vector review PDF;
- measures its page count and verifies the candidate-to-review page map; and
- writes JSON/CSV source maps and vector-QA JSON.

Immediately before publication, every candidate and snapshot artifact is
rehashed. The complete staged tree is then checksummed and atomically renamed to
the requested output root. Any failure removes staging and leaves the output
root absent.

## Command

Run from the clean checkout at the approved generator commit:

```bash
GENERATOR_COMMIT="$(git rev-parse HEAD)"
python submission/final_figure_orchestrator.py \
  --repository "$PWD" \
  --release-root "$FINAL_ROOT" \
  --approved-generator-commit "$GENERATOR_COMMIT" \
  --output-root "$FINAL_FIGURE_ROOT"
```

`FINAL_FIGURE_ROOT` must not exist.
