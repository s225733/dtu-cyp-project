
#!/bin/bash
#BSUB -q gpuv100
#BSUB -J "cyp_mlp[1-20]%3"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/mlp_%J_%I.out
#BSUB -e outputs/mlp_%J_%I.err

set -euo pipefail

cd "${LS_SUBCWD:?Submit from project root}"
source "${CYP_VENV:-$PWD/.venv}/bin/activate"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Each LSF array element reads a different CSV row
LINE=$(sed -n "$((LSB_JOBINDEX + 1))p" \
    experiments/mlp_sweep.csv)

IFS=',' read -r ID TASK LR HIDDEN LAYERS DROPOUT BATCH \
    <<< "$LINE"

echo "Starting experiment: $ID"

python -u finetune.py \
    model=morgan_rdkit_mlp \
    task="$TASK" \
    seed=42 \
    device=cuda \
    model.readout_lr="$LR" \
    readout.hidden_dim="$HIDDEN" \
    readout.num_layers="$LAYERS" \
    readout.dropout="$DROPOUT" \
    train.batch_size="$BATCH" \
    hydra.run.dir="runs/$TASK/morgan_rdkit_mlp/$ID"

echo "Finished experiment: $ID"
