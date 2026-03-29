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
            "--training-strategy",
            "--fraction-of-dataset",
        ]
    )

    model_name = args.model_name
    dataset_name = args.task_name
    save_per_epoch = args.save_per_epoch
    training_strategy = args.training_strategy
    fraction_of_dataset = args.fraction_of_dataset

    ft_dataset_dir = f"./data/{dataset_name}/"
    model_save_dir = f"models/{dataset_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/{model_name}"

    if training_strategy == "finetuning":
        pt_dir = f"models/chemblv33/pretraining/{model_name}/model/last-epoch"
        with open("./data/token2label.json", "r") as f:
            token2label = json.load(f)
    elif training_strategy == "smiles-enumeration":
        pt_dir = f"models/chemblv33-isomeric/pretraining/{model_name}/model/last-epoch"
        with open("./data/isomeric_token2label.json", "r") as f:
            token2label = json.load(f)

    print("Started")
    for setup_idx in range(5):
        start = time.time()
        print(f"Setup {setup_idx}")

        setup_save_dir = f"{model_save_dir}/setup-{setup_idx}/model"
        CLM = get_chemical_language_model(model_name)
        clm = CLM.from_checkpoint(pt_dir)
        clm.batch_size = 32

        torch.manual_seed(42)
        clm.fit(
            f"{ft_dataset_dir}/setup-{setup_idx}/train.smiles",
            f"{ft_dataset_dir}/setup-{setup_idx}/val.smiles",
            token2label=token2label,
            starting_epoch=0,
            fraction_of_dataset=fraction_of_dataset,
            smiles_enumeration=training_strategy == "smiles-enumeration",
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
    print("All program took", time.time() - start)
