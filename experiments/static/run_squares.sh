#!/bin/bash

# Youngs
python3 -m experiments.static.squares --config static/circle --num_squares 15 --facet_algo Youngs --save_name square_youngs
# LVIRA
python3 -m experiments.static.squares --config static/circle --num_squares 15 --facet_algo LVIRA --save_name square_lvira
# Our linear facets
python3 -m experiments.static.squares --config static/circle --num_squares 15 --facet_algo linear --save_name square_linear

# Our circular facets without merging, defaults to Youngs when orientation is ambiguous
python3 -m experiments.static.squares --config static/circle --num_squares 15 --facet_algo safe_circle --save_name square_safecircle
# Our circular facets with merging
python3 -m experiments.static.squares --config static/circle --num_squares 15 --facet_algo circular --save_name square_mergecircle 