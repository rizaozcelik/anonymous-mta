#!/bin/bash
datasets=("multi-obj-tasks/fraction-sp3-c-high-and-no-h-donors-high" "multi-obj-tasks/no-h-donors-high-and-fraction-sp3-c-high" "multi-obj-tasks/logp-high-and-tpsa-low" "multi-obj-tasks/tpsa-low-and-logp-high" "multi-obj-tasks/no-rings-high-and-tpsa-low" "multi-obj-tasks/tpsa-low-and-no-rings-high")

for dataset in "${datasets[@]}"; do
    python  runners/multi_obj_finetuning.py --model-name=lstm --task-name=$dataset --save-per-epoch=100 --fraction-of-dataset=1.00
done