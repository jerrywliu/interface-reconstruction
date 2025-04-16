#!/bin/bash

# Run parameter sweep with default parameters (num_ellipses=25)
python3 -m experiments.static.ellipses --config static/circle --sweep

"""
# Youngs
python3 -m experiments.static.ellipses --config static/circle --num_ellipses 15 --facet_algo Youngs --save_name ellipse_youngs
# LVIRA
python3 -m experiments.static.ellipses --config static/circle --num_ellipses 15 --facet_algo LVIRA --save_name ellipse_lvira
# Our linear facets
python3 -m experiments.static.ellipses --config static/circle --num_ellipses 15 --facet_algo linear --save_name ellipse_linear

# Our circular facets without merging, defaults to Youngs when orientation is ambiguous
python3 -m experiments.static.ellipses --config static/circle --num_ellipses 15 --facet_algo safe_circle --save_name ellipse_safecircle
# Our circular facets with merging
python3 -m experiments.static.ellipses --config static/circle --num_ellipses 15 --facet_algo circular --save_name ellipse_mergecircle
"""
