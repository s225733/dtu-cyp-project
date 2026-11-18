# MLP hyperparameter experiments

This document explains how to use `run_experiments.sh` to submit reproducible Morgan + RDKit MLP experiments on DTU HPC. It is intended to live at **`experiments/README.md`**, so it does not replace the starter repository's root `README.md`.

## Overview

The workflow separates configuration, training, and analysis:

| File | Responsibility |
| --- | --- |
| `experiments/mlp_sweep.csv` | Defines **one experiment per row**. |
| `run_experiments.sh` | Reads the CSV and submits one LSF job per experiment. |
| `finetune.sh` | Requests GPU resources, activates the HPC Python environment, and launches training. |
| `finetune.py` | Trains the model and writes metrics and checkpoints. |
| `compare_runs.py` | Compares completed experiments **locally**, after copying the results from HPC. |

`run_experiments.sh` is a **submission script**. It does not train a model on the login node. **Run it with `bash`, not `bsub`.** Its individual training jobs are submitted via `bsub -env all < finetune.sh`.

## Prerequisites

From the repository root on DTU HPC, ensure:

1. You have the project files, including `run_experiments.sh`, `finetune.sh`, and `experiments/mlp_sweep.csv`.
2. The HPC environment and Python dependencies have been set up in `.venv` (or set `CYP_VENV` to a different environment path). The Morgan + RDKit MLP needs `descriptastorus` and its runtime dependency `six`.
3. The input data have been prepared, including `data/regression/{train,val,test}.csv` and `data/classification/{train,val,test}.csv`.
4. An `outputs/` directory exists for LSF logs. If needed, create it **once** with `mkdir -p outputs`.
5. You are using a system with the LSF `bsub` command (DTU HPC, not your Fedora laptop).

The script checks that the CSV, `finetune.sh`, and `outputs/` exist before submitting jobs.

## Edit the experiment values

The file `experiments/mlp_sweep.csv` has these columns, **in this order**:

```csv
id,task,lr,dropout,hidden_dim,layers,batch_size,weight_decay,epochs
reg_reference,regression,0.001,0.4,512,3,64,0.01,100
reg_hidden256,regression,0.001,0.4,256,3,64,0.01,100
cls_reference,classification,0.001,0.1,512,3,64,0.01,100
```

| Column | Hydra setting | Meaning |
| --- | --- | --- |
| `id` | Used in the output path | Unique, descriptive experiment name; letters, digits, `_`, and `-` only. |
| `task` | `task` | `regression` or `classification`. |
| `lr` | `model.readout_lr` | MLP learning rate. |
| `dropout` | `readout.dropout` | Fraction of hidden activations dropped during training. |
| `hidden_dim` | `readout.hidden_dim` | Width of hidden layers. |
| `layers` | `readout.num_layers` | Number of linear layers, **including the output layer**. |
| `batch_size` | `train.batch_size` | Training examples per mini-batch. |
| `weight_decay` | `train.weight_decay` | Weight-decay regularization strength. |
| `epochs` | `train.epochs` | Maximum number of training epochs. |

All rows use `model=morgan_rdkit_mlp`, `seed=42`, and `device=cuda` through the existing `finetune.sh`. Other settings come from the Hydra YAML files. **Both dropout and weight decay are regularization choices**; gradient clipping (`train.grad_clip`) is held at its YAML default and is primarily for optimization stability.

The supplied sweep contains **18 jobs: 9 regression and 9 classification**. It varies hidden width, depth, batch size, and weight decay one at a time relative to each task's reference configuration. Previous validation experiments favored dropout `0.4` for regression and `0.1` for classification; these are starting points, not universal defaults.

You can edit the CSV using `nano experiments/mlp_sweep.csv` on HPC, but editing it locally, committing it on your experiment branch, and pulling on HPC makes the study easier to reproduce. Keep the header and column order unchanged. Avoid commas or spaces within values. The script validates IDs and tasks but does not comprehensively validate numeric inputs.

## Preview and submit on HPC

From the **repository root**:

```bash
# Check the script for shell syntax errors
bash -n run_experiments.sh

# Print the configurations without submitting any jobs
DRY_RUN=1 bash run_experiments.sh

# Submit all CSV rows as separate LSF jobs
bash run_experiments.sh

# Inspect jobs (including recent failures)
bjobs -a
```

The script exports `MODEL`, `TASK`, and `OVERRIDES` for each row. `finetune.sh` receives these environment variables and passes the overrides to Hydra. You do **not** need to manually edit `finetune.py` or its YAML defaults for every run.

**Each invocation submits the whole CSV again.** There is no automatic check to skip configurations that ran previously. Use dry-run mode first, especially after edits. Jobs may queue depending on GPU availability and allocation limits.

## Where results are saved

Each experiment gets a separate directory named after its ID and a timestamp, for example:

```text
runs/
  regression/
    morgan_rdkit_mlp/
      reg_hidden256/
        2026-11-10_13-42-40/
          config.json
          history.jsonl
          validation_metrics.json
          best.pt
```

The run's `config.json` records the actual settings, `history.jsonl` records training loss and validation metrics per epoch, `validation_metrics.json` contains the selected model's validation metrics, and `best.pt` is the checkpoint. Other files may also be written.

LSF logs are written to `outputs/cyp_<JOB_ID>.out` and `outputs/cyp_<JOB_ID>.err`. To investigate a failure:

```bash
tail -n 80 outputs/cyp_<JOB_ID>.err
```

Replace `<JOB_ID>` with the actual number. The run directory timestamp is a directory label, not proof of when training finished.

## Compare experiments on your laptop

Copy `runs/` from HPC to the repository on your Fedora laptop or home PC, then execute:

```bash
python3 compare_runs.py
```

The comparison utility reads saved configuration, history, and validation files and writes `mlp_experiments.json`. This analysis does **not** need a GPU or HPC compute resources.

Use **lower Macro ST-RAE** to select a regression configuration and **higher Macro MCC** to select a classification configuration. Compare training-loss and validation-score trends when investigating overfitting; a lower training loss alone is not evidence of better generalization.

## Methodology and cautions

- Choose hyperparameters using **validation metrics**, not held-out test performance.
- The current original `finetune.py` evaluates the held-out test data after each successful run unless you have added a validation-only option. For a rigorous study, disable test evaluation during sweeping and evaluate the selected final models on test only once.
- The script uses a fixed seed of `42`. It compares configurations; it does not measure variability across random seeds. Follow-up experiments with multiple seeds are useful if time permits.
- `runs/` may contain large model checkpoints. Keep them out of Git unless there is a deliberate storage policy; commit the CSV, scripts, and small analysis summaries instead.
- Working on a dedicated Git branch protects shared tracked code, but ignored local `runs/` and `outputs/` directories are **not** isolated by branches.
