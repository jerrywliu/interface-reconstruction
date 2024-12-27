#!/bin/bash

# Settings
# settings=("vortex" "x+o" "zalesak")
# settings=("x+o" "zalesak" "vortex")
settings=("x+o")
# Resolutions
# resolutions=(50 100)
# resolutions=(50 100 150)
resolutions=(100)
# resolutions=(32 64 128)
# Algos
# algos=("youngs" "lvira" "linear" "circular" "lcorner" "ccorner")
# algos=("youngs" "lvira" "linear" "circular")
# algos=("safecircle")
# algos=("safelinear" "safelinearcorner")
algos=("safelinearcorner")

for resolution in "${resolutions[@]}"; do
    for setting in "${settings[@]}"; do
        for algo in "${algos[@]}"; do
            command="python3 run.py --config advection/${setting}/${resolution}/${setting}_${resolution}_${algo}"
            echo "Running: $command"
            eval "$command"
        
        done
    done
done
