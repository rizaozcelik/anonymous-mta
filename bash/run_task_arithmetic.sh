#!/bin/bash

datasets=("property-tasks/fraction-sp3-c-high" "property-tasks/fraction-sp3-c-low" "property-tasks/logp-high" "property-tasks/logp-low" "property-tasks/no-h-donors-high" "property-tasks/no-h-donors-low" "property-tasks/no-rings-high" "property-tasks/no-rings-low" "property-tasks/tpsa-high" "property-tasks/tpsa-low")

lambdas=(0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
for dataset in "${datasets[@]}"; do
    for lambda in "${lambdas[@]}"; do
        python runners/task_arithmetic.py --model-name=lstm --task-name=$dataset --lambda_=$lambda
    done
done
