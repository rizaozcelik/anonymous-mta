# Molecular Task Arithmetic

This repository provides a modular framework for training, evaluating, and analyzing the models. All workflows are orchestrated via bash scripts that call Python scripts in the `runners/` directory.

> All scripts assume a Linux environment and bash shell.
> Set the `PYTHONPATH` as needed in your shell or within the bash scripts.

## Folder Structure

- `bash/`: Bash scripts to automate workflows.
- `data/`: Datasets and descriptors.
- `library/`: Core library code for modeling and evaluation.
- `models/`: Pretrained model checkpoints and outputs. The trained models will also be saved here. Only the pretrained models and designs are available in the repository for space limits (full data takes more than 200GBs). The pretrained models can be conditioned and sampled using the provided scripts.
- `runners/`: Python scripts for training, evaluation, and analysis.


## Installation

1. **Set up a Python environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Make sure torch is installed with GPU support.
## Running Workflows

All experiments and analyses are managed via bash scripts in `scripts/bash/`. These scripts call Python runner scripts in `runners/`. The available parameters and permitted values of the runners are defined in the `runners/setup.py` 

### Training

Five training scripts are available, each corresponding to a different training task/strategy: `run_few_shot_ft_with_ta.sh`, `run_finetuning.sh`, `run_multi_obj_finetuning.sh`, `run_multi_obj_task_arithmetic.sh`, and `run_task_arithmetic.sh`. See each script for the set parameter values and how to configure the run.

To run model training:
```bash
bash bash/<training_script>.sh
```


### Design
To run design tasks:
```bash
bash bash/run_design.sh
```

To compute the descriptors and extract unique and novel designs:
```bash
bash bash/run_descriptors.sh
```

### Evaluation

To evaluate models:
```bash
bash bash/run_evaluate.sh
bash bash/run_side_effect_evaluation.sh
``` 
