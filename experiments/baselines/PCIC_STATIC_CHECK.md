# PCIC Cartesian static check

## Classification

`run_pcic_static_check.py` is a deterministic **source-method kernel check**.
It is not a reproduction of the randomized ellipse table in Maity,
Sundararajan, and Velusamy (2021), and its numbers must not be compared with or
substituted for that table.

The check exercises two now-operational parts of the Cartesian baseline:

1. the cited one-pass linear least-squares fit initialized by Parker--Young
   PLIC segments on a complete Cartesian halo; and
2. both separately named bare-PCIC conservative corrections on the same fixed
   circular volume-fraction field.

Run it with:

```bash
PYTHONPATH=. python -m experiments.baselines.run_pcic_static_check
```

The command exits unsuccessfully if the one-pass LLS direction does not improve
the fixed line fixture or either conservative correction has a target-cell
volume-fraction residual above `1e-9`.

## Why the primary PCIC ellipse table is not reproduced here

The paper's randomized Cartesian ellipse experiment is the appropriate
method-level gate, but the current source record does not support a unique run:

- the article presents fixed-radius center translation and fixed-center radius
  adjustment without attributing the reported bare-PCIC table to one of them;
- the fitted circle is unoriented, and the article does not state the disk vs.
  complement recovery rule;
- the minimum-radius reset gives two possible PLIC-chord centers without a
  selection rule;
- a fixed-radius translation can have multiple conservative roots, while no
  root-selection rule is stated; and
- the random realizations needed for an exact table replay are not published.

The implementation therefore retains the two correction variants as distinct
rows and freezes the missing execution policies before the matched project
study. A future statistical comparison may run both variants against the
paper's reported scale and order, but it must not select the better row after
observing the outcome.

## Separate cited-LLS ambiguity

Scardovelli and Zaleski (2003), Section 2.4, says that when more than five cells
are cut in the local block, the radius of influence is multiplied by an
unspecified number below one. The frozen port uses `0.5`, the neutral midpoint
of the stated interval. This and the other predeclared policies are documented
in `docs/baselines/PCIC_IMPLEMENTATION.md` and must be recorded with every
result.
