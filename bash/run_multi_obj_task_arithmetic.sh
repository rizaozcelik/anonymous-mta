#!/bin/bash
datasets=("multi-obj-tasks/fraction-sp3-c-high-and-no-h-donors-high" "multi-obj-tasks/logp-high-and-tpsa-low" "multi-obj-tasks/no-rings-high-and-tpsa-low")

for dataset in "${datasets[@]}"; do
    python  runners/multi_obj_task_arithmetic.py --model-name=lstm --task-name=$dataset 
done