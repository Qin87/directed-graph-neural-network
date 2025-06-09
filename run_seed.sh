#!/bin/bash

# Define seeds to run
seeds=(100 200 300 400 500 600 700 800 900 1000)

# Run sequentially
for seed in "${seeds[@]}"
do
    echo "Running with seed $seed..."
    python3 -u -m src.run --seed="$seed" > "dirGNN${seed}.log" 2>&1
    echo "Finished seed $seed."
done

echo "All runs completed."
