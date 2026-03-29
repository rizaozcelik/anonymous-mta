from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch

from library.models import get_chemical_language_model
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        [
            "--model-name",
            "--task-name",
            "--training-strategy",
            "--n-generations",
            "--lambda_",
            "--lambda-2",
            "--fraction-of-dataset",
        ]
    )

    model_name = args.model_name
    dataset_name = args.task_name
    training_strategy = args.training_strategy
    n_generations = args.n_generations
    lambda_value_1 = args.lambda_
    # lambda_value_2 = args.lambda_2
    fraction_of_dataset = args.fraction_of_dataset

    # if lambda_value_2 is not None and lambda_value_1 is not None:
    # lambda_value = f"{lambda_value_1:.2f}-{lambda_value_2:.2f}"
    # elif lambda_value_1 is not None:
    if lambda_value_1 is not None:
        lambda_value = f"{lambda_value_1:.2f}"
    # else:
    # lambda_value = None

    if (
        training_strategy == "smiles-enumeration"
        or training_strategy == "smi-enum-task-arithmetic"
        or training_strategy == "few-shot-ft-with-smi-enum-ta"
    ):
        with open("./data/isomeric_token2label.json", "r") as f:
            token2label = json.load(f)
    else:
        with open("./data/token2label.json", "r") as f:
            token2label = json.load(f)

    for setup_idx in range(5):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        CLM = get_chemical_language_model(model_name)
        print("Started")
        start = time.time()

        if (
            training_strategy == "finetuning"
            or training_strategy == "smiles-enumeration"
        ):
            model_path = f"./models/{dataset_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/{model_name}/setup-{setup_idx}/"
        elif (
            training_strategy == "task-arithmetic"
            or training_strategy == "smi-enum-task-arithmetic"
        ):
            model_path = f"./models/{dataset_name}/{training_strategy}/lambda-{lambda_value}/{model_name}/setup-{setup_idx}/"
        elif (
            training_strategy == "few-shot-ft-with-ta"
            or training_strategy == "few-shot-ft-with-smi-enum-ta"
        ):
            model_path = f"./models/{dataset_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/lambda-{lambda_value}/{model_name}/setup-{setup_idx}/"

        model_weights_path = f"{model_path}/model/last-epoch"
        designs_save_path = f"{model_path}/designs/"
        if os.path.exists(f"{designs_save_path}/designs.txt"):
            print(
                "Already done: dataset_name",
                dataset_name,
                "training_strategy",
                training_strategy,
                "lambda_value",
                lambda_value_1,
                "fraction_of_dataset",
                fraction_of_dataset,
                "setup_idx",
                setup_idx,
            )
            continue

        clm = CLM.from_checkpoint(model_weights_path)
        batch_size = 4096
        print("Generating molecules for setup", setup_idx, "dataset", dataset_name)
        n_batches = n_generations // batch_size + 1
        designs, lls = clm.design_molecules(
            n_batches=n_batches,
            batch_size=batch_size,
            temperature=1.0,
            token2label=token2label,
        )

        os.makedirs(designs_save_path, exist_ok=True)
        with open(f"{designs_save_path}/designs.txt", "w") as f:
            f.write("\n".join(designs))

        np.savetxt(f"{designs_save_path}/lls.txt", lls)
        print("Finished", time.time() - start)
