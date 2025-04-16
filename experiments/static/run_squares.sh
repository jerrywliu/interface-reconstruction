#!/bin/bash

# Run parameter sweep with default parameters (num_squares=25)
python3 -m experiments.static.squares --config static/square --sweep --num_squares 2

"""
# Youngs
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo Youngs --save_name square_youngs
# LVIRA
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo LVIRA --save_name square_lvira
# Our linear facets
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo linear --save_name square_linear

# Our linear facets with corner detection, without merging
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo safe_linear_corner --save_name square_safelinearcorner
# Our linear facets with corner detection, with merging
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo linear+corner --save_name square_linear+corner

# Our circular facets without merging, defaults to Youngs when orientation is ambiguous
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo safe_circle --save_name square_safecircle
# Our circular facets with merging
python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo circular --save_name square_mergecircle
"""