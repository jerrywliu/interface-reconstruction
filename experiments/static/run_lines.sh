#!/bin/bash
########################################################
# Example usage:
# Youngs
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo Youngs --save_name line_youngs
# LVIRA
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo LVIRA --save_name line_lvira
# Our linear facets (without merging)
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo safe_linear --save_name line_safelinear
# Our linear facets (with merging)
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo linear --save_name line_mergelinear
########################################################

# Sweep
python3 -m experiments.static.lines --config static/line --sweep