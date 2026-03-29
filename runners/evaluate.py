import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from library.evaluation import semantic
from runners import setup

PROPERTY_TO_THRESHOLD = {
    "no-h-donors": 1,
    "no-rings": 3,
    "logp": 3.5,
    "tpsa": 75,
    "fraction-sp3-c": 0.3,
}

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
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
    dataset_name = args.task_name
    training_strategy = args.training_strategy
    n_generations = args.n_generations
    lambda_1 = args.lambda_
    lambda_2 = args.lambda_2
    fraction_of_dataset = args.fraction_of_dataset

    if lambda_2 is not None and lambda_1 is not None:
        lambda_value = f"{lambda_1:.2f}-{lambda_2:.2f}"
    elif lambda_1 is not None:
        lambda_value = f"{lambda_1:.2f}"

    target_descriptor_names = dataset_name.split("/")[-1].split("-and-")
    comparisons = [
        "greater" if descriptor_name.endswith("-high") else "leq"
        for descriptor_name in target_descriptor_names
    ]
    target_descriptor_names = [
        name.replace("-high", "").replace("-low", "")
        for name in target_descriptor_names
    ]
    model_save_dir = f"models/{dataset_name}/{training_strategy}"

    scores_across_setups = []
    for setup_idx in range(5):
        start = time.time()
        if (
            training_strategy == "finetuning"
            or training_strategy == "smiles-enumeration"
        ):
            design_dir = f"{model_save_dir}/frac-data-{fraction_of_dataset:.2f}/{model_name}/setup-{setup_idx}/designs"
            scores_saving_dir = design_dir
        elif (
            training_strategy == "task-arithmetic"
            or training_strategy == "smi-enum-task-arithmetic"
        ):
            design_dir = f"{model_save_dir}/lambda-{lambda_value}/{model_name}/setup-{setup_idx}/designs"
            scores_saving_dir = design_dir
        elif (
            training_strategy == "few-shot-ft-with-ta"
            or training_strategy == "few-shot-ft-with-smi-enum-ta"
        ):
            design_dir = f"{model_save_dir}/frac-data-{fraction_of_dataset:.2f}/lambda-{lambda_value}/{model_name}/setup-{setup_idx}/designs"
            scores_saving_dir = design_dir
        elif training_strategy == "pretraining":
            design_dir = f"models/chemblv33/pretraining/{model_name}/designs"
            scores_saving_dir = f"models/{dataset_name}/pretraining/{model_name}"
            os.makedirs(scores_saving_dir, exist_ok=True)
        elif training_strategy == "pretraining-isomeric":
            design_dir = f"models/chemblv33-isomeric/pretraining/{model_name}/designs"
            scores_saving_dir = (
                f"models/{dataset_name}/pretraining-isomeric/{model_name}"
            )
            os.makedirs(scores_saving_dir, exist_ok=True)

        df_canonical_designs = pd.read_csv(f"{design_dir}/canonical_designs.csv")

        # print(
        #     f"Running syntactic {dataset_name} - setup {setup_idx} - {training_strategy}"
        # )
        # syntactic_scores = semantic.compute_success_rate(
        #     df_canonical_designs,
        #     descriptor_names=target_descriptor_names,
        #     descriptor_thresholds=[
        #         PROPERTY_TO_THRESHOLD[descriptor_name]
        #         for descriptor_name in target_descriptor_names
        #     ],
        #     comparisons=comparisons,
        # )

        # print(
        #     f"Running diversity {dataset_name} - setup {setup_idx} - {training_strategy}"
        # )
        df_novel_designs = df_canonical_designs[df_canonical_designs["is_novel"] == 1]
        df_novel_unique_designs = df_novel_designs.drop_duplicates("can_smiles")
        # diversity_scores = semantic.compute_diversity(
        #     designs_batch=df_novel_unique_designs["can_smiles"].tolist(),
        #     distance_threshold=0.65,
        # )

        print(f"Running KS {dataset_name} - setup {setup_idx} - {training_strategy}")
        ks_distances = semantic.compute_ks_distances(
            df_novel_unique_designs,
            # dataset_name=dataset_name,
            dataset_name="chemblv33",
            setup_idx=setup_idx,
            descriptor_names=list(setup.DESCRIPTORS.keys()),
        )
        if os.path.exists(f"{scores_saving_dir}/scores.json"):
            with open(f"{scores_saving_dir}/scores.json", "r") as f:
                merged_scores = json.load(f)
        else:
            merged_scores = dict()

        merged_scores = {
            **merged_scores,
            # **syntactic_scores,
            # **diversity_scores,
            **ks_distances,
        }
        with open(f"{scores_saving_dir}/scores.json", "w") as f:
            json.dump(merged_scores, f, indent=4)

        end = time.time()

        scores_across_setups.append(merged_scores)

        if (
            training_strategy == "pretraining"
            or training_strategy == "pretraining-isomeric"
        ):
            break

    if (
        training_strategy != "pretraining"
        and training_strategy != "pretraining-isomeric"
    ):
        metric_names = list(scores_across_setups[0].keys())
        average_scores_path = scores_saving_dir[: scores_saving_dir.index("/setup-")]

        if os.path.exists(f"{average_scores_path}/scores.json"):
            with open(f"{average_scores_path}/scores.json", "r") as f:
                mean_std_scores = json.load(f)
        else:
            mean_std_scores = {}
        for metric in metric_names:
            metric_values = [
                setup_scores[metric] for setup_scores in scores_across_setups
            ]
            mean_std_scores[f"{metric}_mean"] = np.mean(metric_values)
            mean_std_scores[f"{metric}_std"] = np.std(metric_values)

        with open(f"{average_scores_path}/scores.json", "w") as f:
            json.dump(mean_std_scores, f, indent=4)
