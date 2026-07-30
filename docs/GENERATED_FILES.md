# Generated File Policy

The repository separates disposable local products from release evidence.

Ignored local products include Python/test caches, virtual environments, logs, and
the top-level `plots/`, `output/`, and `tmp/` workspaces. Raster previews under
`results/` are also ignored because they can be regenerated from saved geometry and
metrics.

The `results/` tree is otherwise intentionally visible. In particular, Git does
not blanket-ignore release CSV, JSON, JSONL, TXT, PDF, SVG, VTK, VTP, environment,
checksum, or manifest files. This keeps final provenance and scientific artifacts
visible during review even when they are not committed to the source repository.

Before archiving a release:

1. Run `git status --short --ignored` and confirm that only expected scratch files
   are ignored.
2. Run `python submission/check_submission_freeze.py --source-only` from a committed
   source tree.
3. Use the release audit and checksum process described in
   `submission/CODE_REPRODUCIBILITY_AUDIT.md`.
4. Archive the immutable result bundle outside Git rather than force-adding ignored
   previews.

To explain a particular ignore decision, run:

```bash
git check-ignore -v <path>
```
