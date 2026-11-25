#!/usr/bin/env bash
# Submit one LSF job per row in the CheMeleon sweep CSV.
# Usage: DRY_RUN=1 bash chemeleon_run_experiments.sh   (preview)
#        bash chemeleon_run_experiments.sh             (submit)
set -euo pipefail   

cd "$(dirname "$0")"
VALUES_FILE="${VALUES_FILE:-experiments/chemeleon_sweep_2.csv}"
[[ -f "$VALUES_FILE" ]] || { echo "Missing $VALUES_FILE" >&2; exit 1; }
[[ -f finetune.sh ]] || { echo "Missing finetune.sh" >&2; exit 1; }
[[ -d outputs ]] || { echo "Missing outputs/: create it once with mkdir -p outputs" >&2; exit 1; }

count=0
while IFS=, read -r ID TASK FREEZE ENCODER_LR READOUT_LR DROPOUT HIDDEN LAYERS BATCH WEIGHT_DECAY EPOCHS || [[ -n "${ID:-}" ]]; do
    ID="${ID%$'\r'}"
    EPOCHS="${EPOCHS%$'\r'}"
    [[ -z "$ID" || "$ID" == id || "$ID" == \#* ]] && continue
    [[ "$ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid ID: $ID" >&2; exit 1; }
    [[ "$TASK" == regression || "$TASK" == classification ]] || { echo "Invalid task: $TASK" >&2; exit 1; }
    [[ "$FREEZE" == true || "$FREEZE" == false ]] || { echo "Invalid freeze value: $FREEZE" >&2; exit 1; }

    export MODEL=mecaChemeleon
    export TASK
    export OVERRIDES="model.freeze_encoder=$FREEZE model.encoder_lr=$ENCODER_LR model.readout_lr=$READOUT_LR readout.dropout=$DROPOUT readout.hidden_dim=$HIDDEN readout.num_layers=$LAYERS train.batch_size=$BATCH train.weight_decay=$WEIGHT_DECAY train.epochs=$EPOCHS hydra.run.dir=runs/$TASK/mecaChemeleon/$ID/\${now:%Y-%m-%d_%H-%M-%S}"

    echo "$ID | $TASK | $FREEZE | encoderlr=$ENCODER_LR readoutlr=$READOUT_LR dropout=$DROPOUT hidden=$HIDDEN layers=$LAYERS batch=$BATCH decay=$WEIGHT_DECAY epochs=$EPOCHS"
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
