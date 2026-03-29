#!/bin/bash

datasets=("property-tasks/fraction-sp3-c-high" "property-tasks/fraction-sp3-c-low" "property-tasks/logp-high" "property-tasks/logp-low" "property-tasks/no-h-donors-high" "property-tasks/no-h-donors-low" "property-tasks/no-rings-high" "property-tasks/no-rings-low" "property-tasks/tpsa-high" "property-tasks/tpsa-low")

frac_molecules=(0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00)
for dataset in "${datasets[@]}"; do
    for frac_molecule in "${frac_molecules[@]}"; do
        python  runners/finetuning.py --model-name=lstm --task-name=$dataset --save-per-epoch=100 --training-strategy=finetuning --fraction-of-dataset=$frac_molecule
    done
done