from __future__ import annotations

import json
import os
import time

import torch

from library.models import get_chemical_language_model
from library.training import callbacks
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        [
            "--training-strategy",
            "--model-name",
            "--task-name",
            "--save-per-epoch",
            "--lambda_",
            "--fraction-of-dataset",
        ]
    )

    training_strategy = args.training_strategy
    model_name = args.model_name
    task_name = args.task_name
    save_per_epoch = args.save_per_epoch
    lambda_ = float(args.lambda_)
    fraction_of_dataset = args.fraction_of_dataset

    if training_strategy == "few-shot-ft-with-ta":
        with open("./data/token2label.json", "r") as f:
            token2label = json.load(f)
        ta_model_base_dir = (
            f"./models/{task_name}/task-arithmetic/lambda-{lambda_:0.2f}/{model_name}"
        )
    elif training_strategy == "few-shot-ft-with-smi-enum-ta":
        with open("./data/isomeric_token2label.json", "r") as f:
            token2label = json.load(f)
        ta_model_base_dir = f"./models/{task_name}/smi-enum-task-arithmetic/lambda-{lambda_:0.2f}/{model_name}"
    else:
        raise ValueError(
            f"Invalid training strategy: {training_strategy}. Use 'few-shot-ft-and-ta'"
        )

    model_save_base_dir = f"./models/{task_name}/{training_strategy}/frac-data-{fraction_of_dataset:0.2f}/lambda-{lambda_:0.2f}/{model_name}/"

    torch.manual_seed(42)
    CLM = get_chemical_language_model(model_name)

    print("Started")
    for setup_idx in range(5):
        start = time.time()
        print(f"Setup {setup_idx}")
        ta_model_dir = f"{ta_model_base_dir}/setup-{setup_idx}/model/last-epoch"
        model_save_dir = f"{model_save_base_dir}/setup-{setup_idx}/model"

        if os.path.exists(f"{model_save_dir}/last-epoch/model.pt"):
            print(f"Skipping setup {setup_idx}")
            continue

        ta_clm = CLM.from_checkpoint(ta_model_dir)

        ta_clm.fit(
            f"./data/{task_name}/setup-{setup_idx}/train.smiles",
            f"./data/{task_name}/setup-{setup_idx}/val.smiles",
            token2label=token2label,
            starting_epoch=0,
            fraction_of_dataset=fraction_of_dataset,
            smiles_enumeration=training_strategy == "few-shot-ft-with-smi-enum-ta",
            callbacks=[
                callbacks.EarlyStopping(
                    patience=5,
                    delta=1e-5,
                    criterion="val_loss",
                    mode="min",
                ),
                callbacks.ModelCheckpoint(
                    save_fn=ta_clm.save,
                    save_per_epoch=save_per_epoch,
                    basedir=model_save_dir,
                ),
                callbacks.HistoryLogger(savedir=model_save_dir),
                callbacks.NanTracker(),
            ],
        )

    print("All program took", time.time() - start)
