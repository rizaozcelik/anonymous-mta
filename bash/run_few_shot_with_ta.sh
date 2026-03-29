#!/bin/bash

datasets=("property-tasks/fraction-sp3-c-high" "property-tasks/fraction-sp3-c-low" "property-tasks/logp-high" "property-tasks/logp-low" "property-tasks/no-h-donors-high" "property-tasks/no-h-donors-low" "property-tasks/no-rings-high" "property-tasks/no-rings-low" "property-tasks/tpsa-high" "property-tasks/tpsa-low")

lambdas=(0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
frac_molecules=(0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00)
for dataset in "${datasets[@]}"; do
    for lambda in "${lambdas[@]}"; do
        for frac in "${frac_molecules[@]}"; do
            python runners/few_shot_with_ta.py --model-name=lstm --task-name=$dataset --save-per-epoch=100 --lambda_=$lambda --fraction-of-dataset=$frac --training-strategy=few-shot-ft-with-ta
        done
    done
done
