#!/bin/bash
########################################################
# Example usage:
# Youngs
# python3 -m experiments.static.zalesak --config static/circle --num_cases 15 --facet_algo Youngs --save_name zalesak_youngs
# LVIRA
# python3 -m experiments.static.zalesak --config static/circle --num_cases 15 --facet_algo LVIRA --save_name zalesak_lvira
# Our linear facets without merging
# python3 -m experiments.static.zalesak --config static/circle --num_cases 15 --facet_algo safe_linear --save_name zalesak_safelinear
# Our linear facets with merging
# python3 -m experiments.static.zalesak --config static/circle --num_cases 15 --facet_algo linear --save_name zalesak_linear
# Our circular facets without merging
# python3 -m experiments.static.zalesak --config static/circle --num_cases 15 --facet_algo safe_circle --save_name zalesak_safecircle
# Our circular facets with merging
# python3 -m experiments.static.zalesak --config static/circle --num_cases 15 --facet_algo circular --save_name zalesak_mergecircle
########################################################

# Sweep
python3 -m experiments.static.zalesak --config static/circle --sweep --num_cases 15

# Plot only
python3 -m experiments.static.zalesak --config static/circle --plot_only --results_file results/static/zalesak_reconstruction_results.txt


