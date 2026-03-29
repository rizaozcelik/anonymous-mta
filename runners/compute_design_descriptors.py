from __future__ import annotations

import os
import time
from typing import List, Set

import pandas as pd
from rdkit import Chem

from library.evaluation import syntactic
from runners import setup


def read_finetuning_datasets(
    task_name: str, setup_idx: int, fraction_of_dataset: float
) -> List[str]:
    with open(f"./data/{task_name}/setup-{setup_idx}/train.smiles", "r") as f:
        ft_train_smiles = [line.strip() for line in f]

    with open(f"./data/{task_name}/setup-{setup_idx}/val.smiles", "r") as f:
        ft_val_smiles = [line.strip() for line in f]

    n_training_molecules = int(fraction_of_dataset * len(ft_train_smiles))
    n_validation_molecules = int(fraction_of_dataset * len(ft_val_smiles))
    ft_train_smiles = ft_train_smiles[:n_training_molecules]
    ft_val_smiles = ft_val_smiles[:n_validation_molecules]
    print(
        f"Read {n_training_molecules} training and {n_validation_molecules} validation molecules for {task_name} setup {setup_idx}"
    )

    return ft_train_smiles + ft_val_smiles


def read_task_arithmetic_datasets(task_name: str, setup_idx: int) -> List[str]:
    if task_name.endswith("-high"):
        task_name = task_name.replace("-high", "-low")
    elif task_name.endswith("-low"):
        task_name = task_name.replace("-low", "-high")
    else:
        raise ValueError(
            f"Invalid task name: {task_name}. Should end with '-high' or '-low'"
        )

    return read_finetuning_datasets(task_name, setup_idx, 1.0)


def fetch_finetuning_datasets(
    training_strategy: str, task_name: str, setup_idx: int, fraction_of_dataset: float
) -> Set[str]:
    if task_name.startswith("property-tasks"):
        if (
            training_strategy == "finetuning"
            or training_strategy == "smiles-enumeration"
        ):
            ft = read_finetuning_datasets(task_name, setup_idx, fraction_of_dataset)
        elif (
            training_strategy == "task-arithmetic"
            or training_strategy == "smi-enum-task-arithmetic"
        ):
            ft = read_task_arithmetic_datasets(task_name, setup_idx)
        elif (
            training_strategy == "few-shot-ft-with-ta"
            or training_strategy == "few-shot-ft-with-smi-enum-ta"
        ):
            ft = read_finetuning_datasets(task_name, setup_idx, fraction_of_dataset)
            ta_ft = read_task_arithmetic_datasets(task_name, setup_idx)
            ft = ft + ta_ft

    elif task_name.startswith("multi-obj-tasks"):
        task1_name, task2_name = task_name.split("/")[1].split("-and-")
        dataset1_name = f"property-tasks/{task1_name}"
        dataset2_name = f"property-tasks/{task2_name}"
        if (
            training_strategy == "finetuning"
            or training_strategy == "smiles-enumeration"
        ):
            task1_datasets = read_finetuning_datasets(
                dataset1_name, setup_idx, fraction_of_dataset
            )
            task2_datasets = read_finetuning_datasets(
                dataset2_name, setup_idx, fraction_of_dataset
            )
            ft = task1_datasets + task2_datasets
        elif (
            training_strategy == "task-arithmetic"
            or training_strategy == "smi-enum-task-arithmetic"
        ):
            task1_datasets = read_task_arithmetic_datasets(dataset1_name, setup_idx)
            task2_datasets = read_task_arithmetic_datasets(dataset2_name, setup_idx)
            ft = task1_datasets + task2_datasets

    return set(ft)


def get_canonical_designs_and_descriptors(designs, descriptors, training_set):
    can_designs_and_descriptors = list()
    for design in designs:
        can_smiles = syntactic.clean_design(design)
        try:
            descriptor_values = list()
            mol = Chem.MolFromSmiles(can_smiles)
            for descriptor_name, descriptor_fn in descriptors.items():
                descriptor_values.append(descriptor_fn(mol))

            is_novel_design = int(can_smiles not in training_set)
            can_designs_and_descriptors.append(
                (can_smiles, is_novel_design, *descriptor_values)
            )
        except Exception:
            can_designs_and_descriptors.append(("X",))

    return can_designs_and_descriptors


if __name__ == "__main__":
    args = setup.add_run_arguments(
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
    task_name = args.task_name
    training_strategy = args.training_strategy
    n_generations = args.n_generations
    lambda_value_1 = args.lambda_
    lambda_value_2 = args.lambda_2
    fraction_of_dataset = args.fraction_of_dataset

    if lambda_value_2 is not None and lambda_value_1 is not None:
        lambda_value = f"{lambda_value_1:.2f}-{lambda_value_2:.2f}"
    elif lambda_value_1 is not None:
        lambda_value = f"{lambda_value_1:.2f}"
    else:
        lambda_value = None

    pt_dir = f"models/chemblv33/{model_name}/model/last-epoch"
    with open("data/chemblv33/train.smiles", "r") as f:
        chembl_train_smiles = [line.replace(" ", "").strip() for line in f]

    with open("data/chemblv33/val.smiles", "r") as f:
        chembl_val_smiles = [line.replace(" ", "").strip() for line in f]

    chembl = set(chembl_train_smiles + chembl_val_smiles)
    start = time.time()
    for setup_idx in range(5):
        if (
            training_strategy == "finetuning"
            or training_strategy == "smiles-enumeration"
        ):
            designs_path = f"./models/{task_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/{model_name}/setup-{setup_idx}/designs"
        elif (
            training_strategy == "task-arithmetic"
            or training_strategy == "smi-enum-task-arithmetic"
        ):
            designs_path = f"./models/{task_name}/{training_strategy}/lambda-{lambda_value}/{model_name}/setup-{setup_idx}/designs"
        elif (
            training_strategy == "few-shot-ft-with-ta"
            or training_strategy == "few-shot-ft-with-smi-enum-ta"
        ):
            designs_path = f"./models/{task_name}/{training_strategy}/frac-data-{fraction_of_dataset:.2f}/lambda-{lambda_value}/{model_name}/setup-{setup_idx}/designs"
        elif training_strategy == "pretraining" or training_strategy == "pretraining-isomeric":
            designs_path = (
                f"./models/{task_name}/{training_strategy}/{model_name}/designs"
            )

        print(f"Setup {setup_idx}")
        with open(f"{designs_path}/designs.txt", "r") as f:
            designs = [line.strip() for line in f][:n_generations]

        if os.path.exists(f"{designs_path}/canonical_designs.csv"):
            df_can_designs = pd.read_csv(f"{designs_path}/canonical_designs.csv")
            if len(df_can_designs) == n_generations:
                print(
                    "Already done: dataset_name",
                    task_name,
                    "training_strategy",
                    training_strategy,
                    "lambda_value",
                    lambda_value,
                    "fraction_of_dataset",
                    fraction_of_dataset,
                    "setup_idx",
                    setup_idx,
                )
                continue

        if training_strategy == "pretraining":
            training_set = chembl
        else:
            finetuning_set = fetch_finetuning_datasets(
                training_strategy, task_name, setup_idx, fraction_of_dataset
            )
            training_set = finetuning_set.union(chembl)

        can_designs_and_descriptors = get_canonical_designs_and_descriptors(
            designs, setup.DESCRIPTORS, training_set
        )
        df_can_designs = pd.DataFrame(
            can_designs_and_descriptors,
            columns=["can_smiles", "is_novel", *setup.DESCRIPTORS.keys()],
        )
        df_can_designs.to_csv(f"{designs_path}/canonical_designs.csv", index=False)
        print(f"Finished setup {setup_idx} for {task_name}")

        if training_strategy == "pretraining":
            break

    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
