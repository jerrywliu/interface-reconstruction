# Experiments Overview

This document describes the **static experiment suites** that are run by the two sweep scripts:

- `experiments/static/run_linear_sweeps.py`
- `experiments/static/run_perturbed_sweeps.py`

If you just want the quick commands, jump to **Quick Commands**.

---

## Static Experiments Included

Both sweep scripts run the same set of static experiments:

1. **Circles** (`experiments/static/circles.py`)
2. **Ellipses** (`experiments/static/ellipses.py`)
3. **Lines** (`experiments/static/lines.py`)
4. **Squares** (`experiments/static/squares.py`)
5. **Zalesak’s Disk** (`experiments/static/zalesak.py`)

### Algorithms by Experiment

- **Circles / Ellipses / Lines**
  - `Youngs`
  - `LVIRA`
  - `safe_linear`
  - `linear`

- **Squares / Zalesak**
  - `Youngs`
  - `LVIRA`
  - `safe_linear`
  - `linear`
  - `safe_linear_corner`
  - `linear+corner`

---

## Sweep Scripts

### 1) Linear Sweeps

Script: `experiments/static/run_linear_sweeps.py`

**What it does**
- Runs all static experiments on **Cartesian meshes**.
- Sweeps **resolution** (and algorithms as listed above).
- Writes **summary plots** to `results/static/linear_*.png`.
- Sends summary plots to Slack (if Slack notify is enabled).

**Default resolutions**
- Circles / Ellipses / Lines:
  - `0.32, 0.50, 0.64, 1.00, 1.28, 1.50`
- Squares / Zalesak:
  - `0.50, 0.64, 1.00, 1.28, 1.50`

---

### 2) Perturbed-Quad Sweeps

Script: `experiments/static/run_perturbed_sweeps.py`

**What it does**
- Runs all static experiments on **perturbed Cartesian quad meshes**.
- Sweeps **resolution × wiggle × seed**.
- Writes a **CSV** of results to `results/static/perturbed_sweep_<timestamp>.csv`.
- Generates **summary plots** of *metric vs wiggle* (one line per resolution),
  saved under `results/static/perturbed_plots/`.
- Sends summary plots + CSV to Slack (if Slack notify is enabled).

**Default resolutions**
- Circles / Ellipses / Lines:
  - `0.32, 0.50, 0.64, 1.00, 1.28, 1.50`
- Squares / Zalesak:
  - `0.50, 0.64, 1.00, 1.28, 1.50`

**Default wiggles**
- `0.0, 0.05, 0.1, 0.2, 0.3`

**Default seeds**
- `0`

---

## Quick Commands

### Linear sweep (Cartesian meshes)
```bash
python -m experiments.static.run_linear_sweeps --subprocess
```

### Perturbed-quad sweep
```bash
python -m experiments.static.run_perturbed_sweeps
```

### Perturbed-quad sweep (single experiment)
```bash
# Only run lines
python -m experiments.static.run_perturbed_sweeps --only lines --notify
```

### Enable Slack notifications
```bash
python -m experiments.static.run_linear_sweeps --subprocess --notify
python -m experiments.static.run_perturbed_sweeps --notify
```

---

## Run One Experiment (Perturbed Quads)

Each static experiment can be run directly on perturbed quad meshes by passing mesh overrides.

### Lines (example)
```bash
python -m experiments.static.lines \
  --config static/line \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0
```

### Circles
```bash
python -m experiments.static.circles \
  --config static/circle \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0
```

### Ellipses
```bash
python -m experiments.static.ellipses \
  --config static/ellipse \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0
```

### Squares
```bash
python -m experiments.static.squares \
  --config static/square \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0
```

### Zalesak
```bash
python -m experiments.static.zalesak \
  --config static/zalesak \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0
```

## Where Outputs Go

- **Linear summary plots:** `results/static/linear_*.png`
- **Perturbed sweep CSV:** `results/static/perturbed_sweep_<timestamp>.csv`
- **Perturbed summary plots:** `results/static/perturbed_plots/*.png`
