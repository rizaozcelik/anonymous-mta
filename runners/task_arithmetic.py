import time

import torch

from library.models import get_chemical_language_model
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        ["--model-name", "--task-name", "--training-strategy", "--lambda_"]
    )

    model_name = args.model_name
    task_name = args.task_name
    training_strategy = args.training_strategy
    lambda_ = f"{args.lambda_:0.2f}"
    CLM = get_chemical_language_model(model_name)

    opposite_task_name = (
        task_name.replace("-high", "-low")
        if task_name.endswith("-high")
        else task_name.replace("-low", "-high")
    )

    if training_strategy == "task-arithmetic":
        pt_loaddir = f"models/chemblv33/pretraining/{model_name}/model/last-epoch"
        ft_base_loaddir = (
            f"./models/{opposite_task_name}/finetuning/frac-data-1.00/{model_name}"
        )
        model_base_save_dir = (
            f"models/{task_name}/task-arithmetic/lambda-{lambda_}/{model_name}/"
        )
    elif training_strategy == "smi-enum-task-arithmetic":
        pt_loaddir = (
            f"models/chemblv33-isomeric/pretraining/{model_name}/model/last-epoch"
        )
        ft_base_loaddir = f"./models/{opposite_task_name}/smiles-enumeration/frac-data-1.00/{model_name}"
        model_base_save_dir = f"models/{task_name}/smi-enum-task-arithmetic/lambda-{lambda_}/{model_name}/"

    print("Started")
    for setup_idx in range(5):
        start = time.time()
        print(f"Setup {setup_idx}")

        ft_loaddir = f"{ft_base_loaddir}/setup-{setup_idx}/model/last-epoch"
        setup_save_dir = f"{model_base_save_dir}/setup-{setup_idx}/model/last-epoch/"

        pt_clm = CLM.from_checkpoint(pt_loaddir)
        ft_clm = CLM.from_checkpoint(ft_loaddir)

        with torch.no_grad():
            pt_state_dict, ft_state_dict = pt_clm.state_dict(), ft_clm.state_dict()
            for key in pt_state_dict.keys():
                task_vector = ft_state_dict[key] - pt_state_dict[key]
                pt_state_dict[key] = pt_state_dict[key] - float(lambda_) * task_vector

        pt_clm.load_state_dict(pt_state_dict)
        pt_clm.save(setup_save_dir)

        del pt_clm, ft_clm

    print("All program took", time.time() - start)
