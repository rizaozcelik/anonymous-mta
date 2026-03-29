# %%
import json
import os

import pandas as pd

from runners import setup

datasets = setup.PROPERTY_TASKS
training_strategies = setup.TRAINING_STRATEGIES
frac_datas = setup.FRAC_DATAS
lambdas = setup.LAMBDAS

all_scores = list()
for dataset in datasets:
    for training_strategy in training_strategies:
        for frac_data in frac_datas:
            for lambda_value in lambdas:
                if (
                    training_strategy == "task-arithmetic"
                    or training_strategy == "smi-enum-task-arithmetic"
                ):
                    frac_data = None
                    scores_path = f"models/{dataset}/{training_strategy}/lambda-{lambda_value:0.2f}/lstm"
                elif (
                    training_strategy == "finetuning"
                    or training_strategy == "smiles-enumeration"
                ):
                    lambda_value = None
                    scores_path = f"models/{dataset}/{training_strategy}/frac-data-{frac_data:0.2f}/lstm"
                elif training_strategy == "pretraining" or training_strategy == "pretraining-isomeric":
                    lambda_value = None
                    frac_data = None
                    scores_path = f"models/{dataset}/{training_strategy}/lstm"
                elif (
                    training_strategy == "few-shot-ft-with-ta"
                    or training_strategy == "few-shot-ft-with-smi-enum-ta"
                ):
                    scores_path = f"models/{dataset}/{training_strategy}/frac-data-{frac_data:0.2f}/lambda-{lambda_value:0.2f}/lstm"

                if not os.path.exists(f"{scores_path}/scores.json") and os.path.exists(
                    scores_path
                ):
                    print(f"Missing: {scores_path}")
                    continue
                elif not os.path.exists(scores_path):
                    continue

                with open(f"{scores_path}/scores.json", "r") as f:
                    scores = json.load(f)

                if training_strategy == "pretraining" or training_strategy == "pretraining-isomeric":
                    new_scores = dict()
                    for metric_name in scores.keys():
                        new_scores[f"{metric_name}_mean"] = scores[metric_name]
                        new_scores[f"{metric_name}_std"] = 0
                    scores = new_scores

                scores["dataset"] = dataset
                scores["training_strategy"] = training_strategy
                scores["frac_data"] = frac_data
                scores["lambda"] = lambda_value
                all_scores.append(scores)


df_scores = pd.DataFrame(all_scores)
df_scores = df_scores.drop_duplicates(
    subset=["dataset", "training_strategy", "frac_data", "lambda"]
)
df_scores.to_csv("models/scores.csv", index=False)
