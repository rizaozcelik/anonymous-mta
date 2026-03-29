from __future__ import annotations

import argparse
from typing import List

from rdkit.Chem import Descriptors, rdMolDescriptors

DESCRIPTORS = {
    "fraction-sp3-c": rdMolDescriptors.CalcFractionCSP3,
    "logp": Descriptors.MolLogP,
    "no-h-donors": Descriptors.NumHDonors,
    "no-rings": Descriptors.RingCount,
    "tpsa": rdMolDescriptors.CalcTPSA,
}
PROPERTY_TASKS = [
    "property-tasks/fraction-sp3-c-high",
    "property-tasks/fraction-sp3-c-low",
    "property-tasks/logp-high",
    "property-tasks/logp-low",
    "property-tasks/no-h-donors-high",
    "property-tasks/no-h-donors-low",
    "property-tasks/no-rings-high",
    "property-tasks/no-rings-low",
    "property-tasks/tpsa-high",
    "property-tasks/tpsa-low",
]
MULTI_OBJ_TASKS = [
    "multi-obj-tasks/fraction-sp3-c-high-and-no-h-donors-high",
    "multi-obj-tasks/no-h-donors-high-and-fraction-sp3-c-high",
    "multi-obj-tasks/logp-high-and-tpsa-low",
    "multi-obj-tasks/tpsa-low-and-logp-high",
    "multi-obj-tasks/no-rings-high-and-tpsa-low",
    "multi-obj-tasks/tpsa-low-and-no-rings-high",
]
TRAINING_STRATEGIES = [
    "pretraining",
    "pretraining-isomeric",
    "finetuning",
    "smiles-enumeration",
    "task-arithmetic",
    "smi-enum-task-arithmetic",
    "few-shot-ft-with-ta",
    "few-shot-ft-with-smi-enum-ta",
]

LAMBDAS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]
FRAC_DATAS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

__ARGUMENTS = {
    "--model-name": {
        "help": "Name of the model to train",
        "choices": ["lstm"],
        "type": str,
    },
    "--training-strategy": {
        "help": "Training strategy to use",
        "choices": TRAINING_STRATEGIES,
        "type": str,
    },
    "--task-name": {
        "help": "Name of the dataset to use",
        "choices": [
            "chemblv33",
            "chemblv33-isomeric",
        ]
        + PROPERTY_TASKS
        + MULTI_OBJ_TASKS,
        "type": str,
    },
    "--fraction-of-dataset": {
        "help": "Fraction of molecules to use for training/validation",
        "type": float,
        "default": 1.0,
    },
    "--save-per-epoch": {
        "help": "How often to save models during training",
        "type": int,
        "default": 10,
    },
    "--n-generations": {
        "help": "How many designs to generate",
        "type": int,
        "default": 100_000,
    },
    "--lambda_": {
        "help": "Lambda value to scale the task vector",
        "type": float,
        "default": None,
        "choices": LAMBDAS,
    },
    "--lambda-2": {
        "help": "Lambda value to scale the task vector. Used for multi-objective tasks",
        "type": float,
        "default": None,
        "choices": LAMBDAS,
    },
}


def add_run_arguments(argument_list: List[str]):
    parser = argparse.ArgumentParser()

    for arg_name in argument_list:
        if arg_name not in __ARGUMENTS:
            raise ValueError(f"Invalid argument name: {arg_name}")
        parser.add_argument(arg_name, **__ARGUMENTS[arg_name])

    args, invalids = parser.parse_known_args()
    if len(invalids) > 0:
        raise ValueError(f"Invalid terminal arguments: {invalids}")
    return args
