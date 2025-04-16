#!/bin/bash

# Run parameter sweep with default parameters (num_circles=5, radius=10.0)
python3 -m experiments.static.circles --config static/circle --sweep

"""
# Youngs
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo Youngs --save_name circle_youngs
# LVIRA
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo LVIRA --save_name circle_lvira
# Our linear facets
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo linear --save_name circle_linear

# Our circular facets without merging, defaults to Youngs when orientation is ambiguous
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo safe_circle --save_name circle_safecircle
# Our circular facets with merging
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo circular --save_name circle_mergecircle
"""