from __future__ import annotations

import json

import torch

from library.models import get_chemical_language_model
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        ["--model-name", "--task-name", "--lambda_", "--training-strategy"]
    )

    model_name = args.model_name
    task_name = args.task_name
    lambda_value = args.lambda_
    training_strategy = args.training_strategy

    if training_strategy == "task-arithmetic":
        with open("./data/isomeric_token2label.json", "r") as f:
            token2label = json.load(f)
        with open("./manuscripting/tables/tasks_to_best_lambdas.json", "r") as f:
            tasks_to_best_lambdas = json.load(f)
        pt_loaddir = f"models/chemblv33/pretraining/{model_name}/model/last-epoch"
    elif training_strategy == "smi-enum-task-arithmetic":
        with open("./data/isomeric_token2label.json", "r") as f:
            token2label = json.load(f)
        with open(
            "./manuscripting/tables/isomeric_tasks_to_best_lambdas.json", "r"
        ) as f:
            tasks_to_best_lambdas = json.load(f)
        pt_loaddir = (
            f"models/chemblv33-isomeric/pretraining/{model_name}/model/last-epoch"
        )

    CLM = get_chemical_language_model(model_name)

    task1_name, task2_name = task_name.split("/")[1].split("-and-")
    task1_name = f"property-tasks/{task1_name}"
    task2_name = f"property-tasks/{task2_name}"

    best_lambda_task1 = tasks_to_best_lambdas[task1_name]
    best_lambda_task2 = tasks_to_best_lambdas[task2_name]

    model_base_save_dir = (
        f"models/{task_name}/{training_strategy}/lambda-{lambda_value:.2f}/{model_name}"
    )

    for setup_idx in range(5):
        task1_model_dir = f"models/{task1_name}/{training_strategy}/lambda-{best_lambda_task1:.2f}/{model_name}/setup-{setup_idx}/model/last-epoch"
        task2_model_dir = f"models/{task2_name}/{training_strategy}/lambda-{best_lambda_task2:.2f}/{model_name}/setup-{setup_idx}/model/last-epoch"

        setup_save_dir = f"{model_base_save_dir}/setup-{setup_idx}/model/last-epoch"

        pt_clm = CLM.from_checkpoint(pt_loaddir)
        task1_clm = CLM.from_checkpoint(task1_model_dir)
        task2_clm = CLM.from_checkpoint(task2_model_dir)

        with torch.no_grad():
            pt_state_dict, task1_state_dict, task2_state_dict = (
                pt_clm.state_dict(),
                task1_clm.state_dict(),
                task2_clm.state_dict(),
            )
            for key in pt_state_dict.keys():
                task1_vector = task1_state_dict[key] - pt_state_dict[key]
                task2_vector = task2_state_dict[key] - pt_state_dict[key]
                combined_vector = lambda_value * (task1_vector + task2_vector)
                pt_state_dict[key] = pt_state_dict[key] + combined_vector

        pt_clm.load_state_dict(pt_state_dict)
        pt_clm.save(setup_save_dir)

        del pt_clm, task1_clm, task2_clm
