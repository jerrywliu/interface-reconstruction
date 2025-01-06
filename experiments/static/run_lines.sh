#!/bin/bash

# Run line experiments for different reconstruction algorithms
python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo Youngs --save_name line_youngs
python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo LVIRA --save_name line_lvira 
python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo safe_linear --save_name line_safelinear