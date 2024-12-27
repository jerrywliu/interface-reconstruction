#!/bin/bash

# Settings
settings=("xpluso" "zalesak")
# Resolutions
# resolutions=(32 50 64 100 128 150)
# resolutions=(0.32 0.50 0.64 1.00 1.28 1.50)
resolutions=(1.50)
# Algos
# algos=("Youngs" "LVIRA" "linear" "circular" "linear+corner" "circular+corner")
algos=("circular")

for setting in "${settings[@]}"; do
    for resolution in "${resolutions[@]}"; do
        for algo in "${algos[@]}"; do
            # multiply resolution by 0.01
            # command="python3 run.py --config static/base --setting ${setting} --resolution $((resolution * 0.01)) --algo ${algo} --save_name static_${setting}_${resolution}_${algo}"
            command="python3 run.py --config static/base --setting ${setting} --resolution ${resolution} --facet_algo ${algo} --save_name static_${setting}_${resolution}_${algo}"
            echo "Running: $command"
            eval "$command"
        
        done
    done
done
