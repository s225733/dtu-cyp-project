#!/usr/bin/env bash
# Submit one LSF job per row in experiments/mlp_sweep.csv.
# Usage: DRY_RUN=1 bash run_experiments.sh   (preview)
#        bash run_experiments.sh             (submit)
set -euo pipefail

cd "$(dirname "$0")"
VALUES_FILE="experiments/mlp_sweep.csv"
[[ -f "$VALUES_FILE" ]] || { echo "Missing $VALUES_FILE" >&2; exit 1; }
[[ -f finetune.sh ]] || { echo "Missing finetune.sh" >&2; exit 1; }
[[ -d outputs ]] || { echo "Missing outputs/: create it once with mkdir -p outputs" >&2; exit 1; }

count=0
while IFS=, read -r ID TASK LR DROPOUT HIDDEN LAYERS BATCH WEIGHT_DECAY EPOCHS || [[ -n "${ID:-}" ]]; do
    [[ -z "$ID" || "$ID" == id || "$ID" == \#* ]] && continue
    [[ "$ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid ID: $ID" >&2; exit 1; }
    [[ "$TASK" == regression || "$TASK" == classification ]] || { echo "Invalid task: $TASK" >&2; exit 1; }

    export MODEL=morgan_rdkit_mlp TASK
    export OVERRIDES="model.readout_lr=$LR readout.dropout=$DROPOUT readout.hidden_dim=$HIDDEN readout.num_layers=$LAYERS train.batch_size=$BATCH train.weight_decay=$WEIGHT_DECAY train.epochs=$EPOCHS hydra.run.dir=runs/$TASK/morgan_rdkit_mlp/$ID/\${now:%Y-%m-%d_%H-%M-%S}"

    echo "$ID | $TASK | lr=$LR dropout=$DROPOUT hidden=$HIDDEN layers=$LAYERS batch=$BATCH decay=$WEIGHT_DECAY epochs=$EPOCHS"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        bsub -env all < finetune.sh
    fi
    ((count+=1))
done < "$VALUES_FILE"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Previewed $count experiment(s); no jobs submitted."
else
    echo "Submitted $count experiment(s)."
fi
