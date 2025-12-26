# getArcFacet Logging Utilities

This module provides logging functionality for `getArcFacet` calls to help debug infinite loops and extract challenging test cases.

## Quick Start

### 1. Run experiments with logging enabled

```bash
# Run ellipse experiments with logging
python run_ellipses_with_logging.py --facet-algo circular

# Or with custom log file
python run_ellipses_with_logging.py --log-file my_calls.log
```

### 2. Analyze the log file

```bash
# View summary statistics
python -m util.logging.analyze_log get_arc_facet_calls.log

# Extract failed cases
python -m util.logging.analyze_log get_arc_facet_calls.log --extract-failed --output failed_cases.py

# Extract slow cases (>1 second)
python -m util.logging.analyze_log get_arc_facet_calls.log --extract-slow --threshold 1.0 --output slow_cases.py

# Extract cases with extreme area fractions
python -m util.logging.analyze_log get_arc_facet_calls.log --extract-extreme --min-frac 0.0 --max-frac 0.1 --output extreme_cases.py
```

## Programmatic Usage

### Enable/Disable Logging

```python
from util.logging.get_arc_facet_logger import enable_logging, disable_logging, get_stats

# Enable logging before running experiments
enable_logging(log_file='my_log.log')

# Run your code that calls getArcFacet
# ... your code here ...

# Disable logging and get statistics
disable_logging()
stats = get_stats()
print(f"Logged {stats['total_calls']} calls")
```

### Log File Format

The log file contains one JSON object per line, with the following structure:

```json
{
  "call_id": 1,
  "timestamp": 1234567890.123,
  "execution_time_seconds": 0.0015,
  "status": "success" | "failed" | "exception",
  "inputs": {
    "poly1": [[x1, y1], [x2, y2], ...],
    "poly2": [[x1, y1], [x2, y2], ...],
    "poly3": [[x1, y1], [x2, y2], ...],
    "a1": 0.5,
    "a2": 0.5,
    "a3": 0.5,
    "epsilon": 1e-10
  },
  "outputs": {
    "center": [x, y] | null,
    "radius": float | null,
    "num_intersects": 2
  }
}
```

## How It Works

The logging system uses **monkey patching** to intercept calls to `getArcFacet`:

1. Before importing any modules that use `getArcFacet`, call `enable_logging()`
2. This replaces `main.geoms.circular_facet.getArcFacet` with a wrapper function
3. The wrapper logs all calls, then calls the original function
4. After experiments, call `disable_logging()` to restore the original function

## Extracting Test Cases

The `analyze_log.py` script can extract test cases in a format compatible with `test_get_arc_facet_error.py`:

```bash
# Extract all failed cases
python -m util.logging.analyze_log get_arc_facet_calls.log \
    --extract-failed \
    --output test_cases.py
```

The output can be directly added to the `TEST_CASES` list in `test_get_arc_facet_error.py`.

## Notes

- Logging must be enabled **before** importing modules that use `getArcFacet`
- The log file is written incrementally (one line per call) so you can monitor progress
- All floating point values are preserved with full precision
- Execution time is measured for each call to identify slow cases



