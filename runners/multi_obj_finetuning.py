from __future__ import annotations

import json
import time

import torch

from library.models import get_chemical_language_model
from library.training import callbacks
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        [
            "--model-name",
            "--task-name",
            "--save-per-epoch",
            "--fraction-of-dataset",
            "--training-strategy",
        ]
    )

    model_name = args.model_name
    task_name = args.task_name
    save_per_epoch = args.save_per_epoch
    fraction_of_dataset = args.fraction_of_dataset
    training_strategy = args.training_strategy

    task1_name, task2_name = task_name.split("/")[1].split("-and-")

    if training_strategy == "finetuning":
        with open("./data/token2label.json", "r") as f:
            token2label = json.load(f)
    elif training_strategy == "smiles-enumeration":
        with open("./data/isomeric_token2label.json", "r") as f:
            token2label = json.load(f)

    task1_model_dir = f"models/property-tasks/{task1_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/{model_name}"
    # task2_model_dir = f"models/property-tasks/{second_task}/finetuning/frac-data-{fraction_of_dataset:.2f}/{model_name}"

    # task1_dataset_dir = f"./data/property-tasks/{task_name}/"
    task2_dataset_dir = f"./data/property-tasks/{task2_name}/"

    model_save_dir = f"models/{task_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/{model_name}"

    print("Started")
    for setup_idx in range(5):
        start = time.time()
        print(f"Setup {setup_idx}")

        setup_save_dir = f"{model_save_dir}/setup-{setup_idx}/model"
        CLM = get_chemical_language_model(model_name)
        clm = CLM.from_checkpoint(
            f"{task1_model_dir}/setup-{setup_idx}/model/last-epoch"
        )
        clm.batch_size = 32

        torch.manual_seed(42)
        clm.fit(
            f"{task2_dataset_dir}/setup-{setup_idx}/train.smiles",
            f"{task2_dataset_dir}/setup-{setup_idx}/val.smiles",
            token2label=token2label,
            starting_epoch=0,
            fraction_of_dataset=fraction_of_dataset,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=5,
                    delta=1e-5,
                    criterion="val_loss",
                    mode="min",
                ),
                callbacks.ModelCheckpoint(
                    save_fn=clm.save,
                    save_per_epoch=save_per_epoch,
                    basedir=setup_save_dir,
                ),
                callbacks.HistoryLogger(savedir=setup_save_dir),
                callbacks.NanTracker(),
            ],
        )

        del clm

    print("All program took", time.time() - start)
