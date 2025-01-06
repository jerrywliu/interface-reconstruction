#!/bin/bash

# Our circular facets without merging, defaults to Youngs when orientation is ambiguous
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo safe_circle --save_name circle_safecircle
# Our circular facets with merging
python3 -m experiments.static.circles --config static/circle --num_circles 15 --facet_algo circular --save_name circle_mergecircle