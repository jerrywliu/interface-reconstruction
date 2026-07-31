# Paper Experiment and Figure Map

Updated: 2026-07-31

This is the paper-to-code index for the static reconstruction results in
Section 6 and its appendix. Run commands from the repository root. The target
submission outputs are vector PDFs; PNG files are review previews only.

The family-level execution checklist is
`submission/POST_SWEEP_FIGURE_MANIFEST.md`. The exact 38-candidate inventory is
`submission/final_figure_candidates.json`; each completed run also writes
`review/figure_candidate_source_map.csv`, which binds every candidate PDF to its
producer inputs. The tables below map the resulting slots to the active manuscript.

## Result Sources

| Tier | Location | Use |
| --- | --- | --- |
| Historical paper assets | March/May 2026 camera-ready bundles | Layout comparison only. Do not use as final numerical provenance. |
| July baseline | `results/static/static_paper_simplified_default_20260717_212413/` | Comparison/recovery only: 300 affected-method runs, 7,500 cases. Its source snapshot is reproducible but dirty, and its historical square/Zalesak `area_error` columns are not submission-equivalent. |
| Completed source release | `results/static/submission_static_20260731_012430_505aefa45432/` | Original completed result set: 970 all-method runs and 24,250 cases. Preserve for comparison; do not use it as an immutable downstream input. |
| Sealed final release | `results/static/submission_static_20260731_012430_505aefa45432.sealed/` | Canonical immutable numerical input. Its complete `SHA256SUMS` has digest `9b5cda54e469ee01bc6a9078bbe22b568ed1e211d16080be806dfe9458ff0b1e`. |

The machine-readable specification is `submission/submission_config.json`.
Figure promotion status and the paper filename contract are in
`submission/figure_provenance.csv` and `submission/FIGURE_PROMOTION_PLAN.md`.

Set these shell variables when a final release exists:

```bash
export SOURCE_RELEASE="$PWD/results/static/submission_static_20260731_012430_505aefa45432"
export FINAL_ROOT="$PWD/results/static/submission_static_20260731_012430_505aefa45432.sealed"
export SOURCE_COMMIT="$(python -c 'import json,os; print(json.load(open(os.environ["FINAL_ROOT"] + "/submission_config.resolved.json"))["source"]["target_commit"])')"
export GENERATOR_COMMIT="$(git rev-parse HEAD)"
export FIGURE_ROOT="$PWD/results/submission/final_figures_${GENERATOR_COMMIT:0:12}"
```

`SOURCE_RELEASE` is the preserved writable run output; `FINAL_ROOT` is its
read-only, checksum-sealed copy. All regenerated figure candidates live
separately under an authoritative `results/submission/final_figures_<commit>/`
root, where the suffix identifies the reviewed generator commit.

## Final Sweep

The approved launcher checks for committed source, records the exact commit,
creates a collision-proof namespace, and archives every raw run bundle inside
the release:

```bash
python submission/check_submission_freeze.py --source-only
bash submission/run_final_static_sweep.sh
```

The equivalent controller dry run was verified at exactly 970 runs and 24,250
cases. The launcher fixes:

- perturbation magnitudes `0, 0.05, 0.1, 0.2, 0.3`;
- seed `0` and 25 deterministic cases per setting;
- `pre_f8_corner + exact_linear_support_only + LVIRA`;
- five workers and pre-`C0` primary benchmark output.

Each final release has this contract:

```text
$FINAL_ROOT/
  SHA256SUMS
  submission_config.resolved.json
  sweep_manifest.json
  failures.csv
  perturbed_sweep.csv
  summary_plots/*.pdf
  logs/
  raw_runs/<save_name>/
  diagnostics/
    source_state.json
    source_snapshot.tar.gz
    run_inventory.csv
    run_manifests.jsonl
    case_geometry.jsonl
    case_metrics.csv
    cell_metrics.csv
    merge_events.csv
    unresolved_plic_fallbacks.csv
```

`case_metrics.csv` is the case-indexed source for tails and convergence;
`cell_metrics.csv` and `merge_events.csv` contain cell, facet, merge, rescue,
and final-primitive provenance. `run_inventory.csv` maps each row to a
release-relative raw bundle.

Compare a completed final release with the July affected-method baseline using
`submission/compare_release_results.py`. The utility matches exact case keys,
reports the expected candidate-only methods separately, and writes CSV, JSON,
and Markdown summaries without modifying either release. See
`submission/COMPARE_RELEASE_RESULTS.md` for the command and interpretation.

## Benchmark Map

The CLI config name in commands is `static/<name>`; its file is
`config/static/<name>.yaml`. Resolution `0.32` means `N=32` cells per side.

| Benchmark and paper role | Driver / config | Final grid and methods | Main metrics |
| --- | --- | --- | --- |
| Lines: linear perfect reconstruction | `experiments.static.lines` / `static/line` | `N=32,50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear` | Hausdorff, facet gap |
| Squares: line-line corners | `experiments.static.squares` / `static/square` | `N=50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `linear+corner`, `safe_circle`, `circular` | Hausdorff, facet gap; area error is secondary |
| Circles: constant curvature | `experiments.static.circles` / `static/circle`; radius 10 | `N=32,50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `safe_circle`, `circular` | Hausdorff, tangent error; gap/curvature in full panel |
| Ellipses: varying curvature and convergence | `experiments.static.ellipses` / `static/ellipse`; major axis 30 | Same grid/methods as circles | Hausdorff, tangent error; gap/curvature and fitted orders |
| Zalesak: arcs, slot edges, corners, and merging | `experiments.static.zalesak` / `static/zalesak`; radius 15, slot width 5 | `N=50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `safe_circle`, `circular`, `circular+corner` | Hausdorff, facet gap; area error is secondary |

To rerun one setting, use the corresponding driver. For example:

```bash
python -m experiments.static.zalesak \
  --config static/zalesak \
  --resolution 1.0 \
  --facet_algo 'circular+corner' \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0 \
  --perturb_fix_boundary 1 \
  --num_cases 25 \
  --plic_fallback LVIRA \
  --corner_behavior_profile pre_f8_corner \
  --rescue_profile exact_linear_support_only \
  --save_name <unique-save-name>
```

The raw result is `plots/<unique-save-name>/`; final sweeps additionally copy
it to `$FINAL_ROOT/raw_runs/<unique-save-name>/`.

## Figure Regeneration

The only submission-producing entry point is
`submission/run_final_figure_orchestrator`, documented in
`submission/FINAL_FIGURE_REGENERATION.md`. It privately materializes and audits
the sealed release, creates a physical run-bundle view, performs companion runs,
and accepts exactly the 38 allowlisted vector PDFs. It never follows a user-built
symlink view or fills missing geometry from historical results.

The generator commands below document the paper-to-code mapping and are useful
for local diagnosis. Their direct outputs are not submission candidates; the
orchestrator supplies their private input paths and acceptance context.

### Main-text panels

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --plots_root <orchestrator-private-physical-plots-view> \
  --out_dir "$FIGURE_ROOT/section6" \
  --figure_groups quantitative,representative \
  --endpoint_variants paired
```

| Family | Generated vector PDF | Paper asset |
| --- | --- | --- |
| Lines metrics | `section6/summary_plots/lines_maintext_metrics.pdf` | `line_reconstruction_maintext_metrics.pdf` |
| Lines representative | `section6/representative_cases/lines_maintext_representative_{with_endpoints,clean}.pdf` | `line_reconstruction_maintext_representative.pdf` |
| Squares metrics | `section6/summary_plots/squares_maintext_metrics.pdf` | `square_reconstruction_maintext_metrics.pdf` |
| Squares representative | `section6/representative_cases/squares_maintext_representative_{with_endpoints,clean}.pdf` | `square_reconstruction_maintext_representative.pdf` |
| Circles metrics | `section6/summary_plots/circles_maintext_metrics.pdf` | `circle_reconstruction_maintext_metrics.pdf` |
| Circles representative | `section6/representative_cases/circles_maintext_representative_{with_endpoints,clean}.pdf` | `circle_reconstruction_maintext_representative.pdf` |
| Ellipses metrics | `section6/summary_plots/ellipses_maintext_metrics.pdf` | `ellipse_reconstruction_maintext_metrics.pdf` |
| Ellipses representative | `section6/representative_cases/ellipses_maintext_representative_{with_endpoints,clean}.pdf` | `ellipse_reconstruction_maintext_representative.pdf` |
| Zalesak metrics | `section6/summary_plots/zalesak_maintext_metrics.pdf` | `zalesak_reconstruction_maintext_metrics.pdf` |
| Zalesak representative | `section6/representative_cases/zalesak_maintext_representative_{with_endpoints,clean}.pdf` | `zalesak_reconstruction_maintext_representative.pdf` |

The generator writes `$FIGURE_ROOT/section6/maintext_manifest.json`, including the
case and method selections. Current representative candidates are lines 6,
squares 24, circles 12, ellipses 12, and Zalesak 12.

### Full all-method panels

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --summary_dir "$FIGURE_ROOT/all_method_summary_plots" \
  --no-notify
```

| Generated vector PDF | Paper asset |
| --- | --- |
| `all_method_summary_plots/lines_all_methods_2x2.pdf` | `line_reconstruction_perturbed_all_methods_2x2.pdf` |
| `all_method_summary_plots/squares_all_methods_2x2.pdf` | `square_reconstruction_perturbed_all_methods_2x2.pdf` |
| `all_method_summary_plots/circles_all_methods_5x2_axes.pdf` | `circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` |
| `all_method_summary_plots/ellipses_all_methods_5x2_axes.pdf` | `ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` |
| `all_method_summary_plots/zalesak_all_methods_2x2.pdf` | `zalesak_reconstruction_perturbed_all_methods_2x2.pdf` |

### Resolution studies

These are deterministic companion runs at `N=16,32,64`, not rows in the
970-run release. The authoritative final-figure orchestrator materializes and
attests the release source commit itself. When running the companion commands
below manually for diagnosis, do not use a later integration checkout. Create a
clean detached worktree at the exact scientific source commit first:

```bash
export CONTROL_REPO="$(git rev-parse --show-toplevel)"
export COMPANION_WORKTREE="${CONTROL_REPO}-companion-${SOURCE_COMMIT:0:12}"
test ! -e "$COMPANION_WORKTREE"
git -C "$CONTROL_REPO" worktree add --detach "$COMPANION_WORKTREE" "$SOURCE_COMMIT"
cd "$COMPANION_WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python submission/check_submission_freeze.py --source-only
```

Keep `FINAL_ROOT` and `FIGURE_ROOT` as the absolute paths exported above. Run each
benchmark separately so its manifest remains intact:

```bash
python -m experiments.static.run_appendix_resolution_visuals \
  --only <benchmark> \
  --case_index <case> \
  --resolutions 0.16,0.32,0.64 \
  --wiggles 0,0.1 \
  --save_prefix "final_resolution_${SOURCE_COMMIT:0:12}_<benchmark>" \
  --endpoint_variants paired \
  --out_dir "$FIGURE_ROOT/resolution/<benchmark>"
```

| Benchmark / candidate case | Generated and paper asset |
| --- | --- |
| Lines / 0 | `lines_resolution_cartesian_vs_perturbed.pdf` |
| Squares / 22 | `squares_resolution_cartesian_vs_perturbed.pdf` |
| Circles / 12 | `circles_resolution_cartesian_vs_perturbed.pdf` |
| Ellipses / 12 | `ellipses_resolution_cartesian_vs_perturbed.pdf` |
| Zalesak / 20 | `zalesak_resolution_cartesian_vs_perturbed.pdf` |

These case choices remain an author-approval gate. Preserve each generated
`manifest.json`, logs, and the associated `plots/<save-prefix>_*` geometry.

### Guarded `C0` appendix panels

All four guarded-`C0` figures are active manuscript assets. They come from a
dedicated final-commit study because the primary 970-run sweep has `C0`
disabled. Manual execution uses the same detached source worktree established
above:

```bash
export C0_ROOT="$PWD/results/static/final_guarded_c0_${SOURCE_COMMIT:0:12}"
test ! -e "$C0_ROOT"
python -m experiments.static.run_appendix_c0_study \
  --out_dir "$C0_ROOT" \
  --out_csv "$C0_ROOT/csv/appendix_c0_sweep.csv" \
  --log_dir "$C0_ROOT/logs" \
  --save_prefix "final_guarded_c0_${SOURCE_COMMIT:0:12}" \
  --seeds 0 --ellipses 25 --zalesak 25 \
  --endpoint_variants paired
```

| Active paper asset | Generated vector candidate |
| --- | --- |
| `ellipses_appendix_c0_2x2.pdf` | `$C0_ROOT/summary_plots/ellipses_appendix_c0_2x2.pdf` |
| `ellipses_appendix_c0_representative.pdf` | `$C0_ROOT/representative_cases/ellipses_appendix_c0_representative_{with_endpoints,clean}.pdf` |
| `zalesak_appendix_c0_2x2.pdf` | `$C0_ROOT/summary_plots/zalesak_appendix_c0_2x2.pdf` |
| `zalesak_appendix_c0_representative.pdf` | `$C0_ROOT/representative_cases/zalesak_appendix_c0_representative_{with_endpoints,clean}.pdf` |

The representative variants require author selection; the two metric panels do
not. Historical pre-guard `C0` assets are ineligible.

### Deterministic explanatory figures

Manual execution uses the same detached source worktree established above:

```bash
python -m experiments.static.generate_plic_baseline_stencil_figure \
  --out "$FIGURE_ROOT/deterministic/perfect_reconstruction_plic_stencil"

python -m experiments.static.generate_staged_reconstruction_figure \
  --output-dir "$FIGURE_ROOT/deterministic" \
  --prefix staged_reconstruction_zalesak
```

The paper assets are `perfect_reconstruction_plic_stencil.pdf` and
`staged_reconstruction_zalesak.pdf`. Their sibling JSON/SVG outputs are the
important provenance files. The currently installed staged-Zalesak PDF contains
a raster color key and must be replaced by this regenerated vector PDF.

## Representative Variant Contract

`--endpoint_variants paired` produces two direct vector PDFs:

- `with_endpoints`: open circles mark every reconstructed facet endpoint;
- `clean`: main-panel endpoint/cell-crossing circles are hidden, but endpoints
  remain labeled in spyglass zooms and semantic reconstructed corners remain
  diamonds everywhere.

Only the author-selected variant is renamed to the unsuffixed paper asset.
Record the choice in the figure approval CSV and `submission/figure_provenance.csv`.

## Supporting Validation Map

Supporting analyses never write beneath `FINAL_ROOT`. First verify the sealed
release ledger and create one digest-named validation namespace with an explicit
binding to both the numerical release and the analysis code:

```bash
test -f "$FINAL_ROOT/SHA256SUMS"
python submission/audit_final_release.py "$FINAL_ROOT" --verify-sha256-manifest
export RELEASE_MANIFEST_SHA256="$(python -c 'import hashlib,os; print(hashlib.sha256(open(os.environ["FINAL_ROOT"] + "/SHA256SUMS", "rb").read()).hexdigest())')"
export ANALYSIS_COMMIT="$(git rev-parse HEAD)"
export VALIDATION_ROOT="$PWD/results/submission/final_validation_${SOURCE_COMMIT:0:12}_${RELEASE_MANIFEST_SHA256:0:12}"

python - "$FINAL_ROOT" "$VALIDATION_ROOT" "$SOURCE_COMMIT" \
  "$RELEASE_MANIFEST_SHA256" "$ANALYSIS_COMMIT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

release = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
payload = {
    "schema_version": 1,
    "release_name": release.name,
    "release_root": str(release),
    "source_commit": sys.argv[3],
    "sha256_manifest": "release_SHA256SUMS",
    "sha256_manifest_digest": sys.argv[4],
    "analysis_commit": sys.argv[5],
}
output.mkdir(parents=True, exist_ok=True)
binding = output / "release_binding.json"
encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if binding.exists() and binding.read_text() != encoded:
    raise SystemExit(f"existing validation binding disagrees: {binding}")
binding.write_text(encoded)
ledger = output / "release_SHA256SUMS"
source_ledger = release / "SHA256SUMS"
if ledger.exists() and ledger.read_bytes() != source_ledger.read_bytes():
    raise SystemExit(f"existing validation ledger disagrees: {ledger}")
shutil.copyfile(source_ledger, ledger)
PY
```

If `SHA256SUMS` is absent, stop and complete the one-time final-release sealing
step before declaring `FINAL_ROOT` immutable. Every command below writes under
`VALIDATION_ROOT`; the binding file and copied ledger travel with the outputs.

| Claim / experiment | Entry point and command | Data and provenance |
| --- | --- | --- |
| Ellipse empirical orders | `python -m experiments.submission.analyze_ellipse_convergence --case-metrics "$FINAL_ROOT/diagnostics/case_metrics.csv" --output-dir "$VALIDATION_ROOT/ellipse_convergence"` | `ellipse_convergence_{points,fits}.csv`, report JSON, vector PDF. This is the evidence bundle for the third-order author gate below. |
| Shared-vertex incidence | `python -m experiments.submission.topology_consistency_diagnostics --full --cell-metrics "$FINAL_ROOT/diagnostics/cell_metrics.csv" --run-inventory "$FINAL_ROOT/diagnostics/run_inventory.csv" --output-dir "$VALIDATION_ROOT/topology"` | Paper table, tolerance table, case/conflict CSVs, compressed vertex rows, and manifest. Relative raw-bundle paths resolve against the release root. |
| Exact conflict diagnosis | `python -m experiments.submission.diagnose_topology_conflicts --source-dir "$VALIDATION_ROOT/topology" --output-dir "$VALIDATION_ROOT/topology_diagnosis"` | Taxonomy, incident facets, data-derived incidence/counts, manifest, and vector examples. |
| Independent cells vs topology+merging | `python -m experiments.submission.run_topology_merging_ablation --output-dir "$VALIDATION_ROOT/topology_merging"` | 10 jobs / 250 Zalesak cases; case/summary CSVs, manifest, vector PDF. Record its exact run commit in the emitted manifest. |
| Guarded optional `C0` | `python -m experiments.submission.c0_conservation_validation --num-cases 25 --output "$VALIDATION_ROOT/c0_conservation"` | Matched 100-case off/on data, eligible joins, regressions, report, and manifest. This is a separate author/provenance gate, not evidence from the pre-`C0` release rows. |
| Five-benchmark conservation smoke | First materialize the exact selection, then run the analyzer using the commands below. | The selection verifies every consumed run input against `FINAL_ROOT/SHA256SUMS`; outputs report global, merged-zone, and constituent-cell residuals. |
| Zalesak remote-corner tail | Resolve the exact final run from the inventory, then invoke `diagnose_zalesak_failure` using the commands below. | Case-23 diagnostic PDF/SVG, entities/comparison CSVs, provenance JSON, and README; July is not an eligible input. |

Materialize and run the final-release conservation selection:

```bash
python -m experiments.submission.materialize_final_conservation_selection \
  --release-root "$FINAL_ROOT" \
  --output "$VALIDATION_ROOT/conservation/selection.json"
python -m experiments.submission.conservation_analyzer \
  --selection "$VALIDATION_ROOT/conservation/selection.json" \
  --output "$VALIDATION_ROOT/conservation/results"
```

Resolve the final Zalesak tail bundle without a handwritten path:

```bash
export ZALESAK_TAIL_RUN="$(python - "$FINAL_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
with (root / "diagnostics/run_inventory.csv").open(newline="") as stream:
    matches = [row for row in csv.DictReader(stream)
               if row["experiment"] == "zalesak"
               and row["algo"] == "circular+corner"
               and float(row["resolution"]) == 1.5
               and float(row["wiggle"]) == 0.2
               and int(row["seed"]) == 0]
assert len(matches) == 1, len(matches)
print(root / matches[0]["run_bundle"])
PY
)"
python -m experiments.submission.zalesak_failure.diagnose_zalesak_failure \
  --run-root "$ZALESAK_TAIL_RUN" \
  --artifact-root "$VALIDATION_ROOT/zalesak_failure"
```

## Manuscript Table Map

| Active table | Manuscript source | Code/data provenance | Approval gate |
| --- | --- | --- | --- |
| Methods comparison (`tab:methods_compare`) | `new_sections/problem_setup.tex` | Literature-supported classification checked against the frozen method/fallback contract in `submission/submission_config.json` and the reconstruction implementation. It is author-curated, not generated from sweep metrics. | Recheck citations and ensure checkmarks remain conditional on the identifiable oriented path. |
| Numerical parameters (`tab:reconstruction_parameters`) | `new_sections/appendix/algorithms.tex` | `config/static/base.yaml`, `main/structs/polys/base_polygon.py`, `main/structs/polys/neighbored_polygon.py`, `main/geoms/circular_facet.py`, and the exact source snapshot named by the final release. | Compare every displayed threshold against the pinned source commit before submission. |
| Benchmark geometry and sampling (`tab:static_benchmark_definitions`) | `new_sections/appendix/static_benchmarks/overview.tex` | `experiments/static/{lines,squares,circles,ellipses,zalesak}.py`, `config/static/*.yaml`, and `submission/submission_config.json`; the final run manifests record the realized parameters. | Cross-check all geometry ranges, seed, and 25-case sampling against the sealed release. |

## Author And Provenance Gates

- **Broad third-order wording is unresolved.** The current mapped evidence is the
  ellipse convergence analysis. It supports robust third-order behavior for
  facet-gap error, while prior final-data analysis found lower empirical orders
  for Hausdorff and tangent errors. Before submission, the authors must either
  narrow the abstract/introduction/method-overview wording to the demonstrated
  metric and benchmark or add separately mapped evidence for a broader claim.
  Any manuscript change remains blue until approved.
- **Guarded-`C0` numerical prose is unresolved.** The currently installed
  appendix values and captions predate the final guarded study. Regenerate all
  165 guarded settings, run the conservation validation above, and reconcile
  every reported median and continuity/conservation statement before promotion.
  Any manuscript change remains blue until approved.
- Record the exact paper commit audited for figures, tables, and prose in the
  review packet or approval ledger at review time using
  `git -C <paper-worktree> rev-parse HEAD`. Durable documentation must not pin a
  paper commit that becomes stale on the next writing edit.

The four guarded-`C0` assets listed above are active. They must be regenerated
from the guarded implementation at `SOURCE_COMMIT`; historical versions are
not eligible for promotion.

## Verification Notes

- CLI help was checked for every command above.
- Controller dry-run: 970 runs / 24,250 cases.
- Topology/merging and resolution-study dry runs emitted the expected commands.
- `python submission/check_submission_freeze.py --source-only` reported
  `SOURCE CLEAN` before this documentation commit.
- Matplotlib paper generators emit direct vector PDF siblings with embedded
  TrueType fonts. Promote those PDFs, not raster previews.
- `experiments.static.build_figure_review_pdf` is optional review tooling and
  requires `reportlab`, which is not currently declared in `requirements.txt`.
