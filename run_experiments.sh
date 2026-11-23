#!/usr/bin/env bash
# Unified LSF sweep runner for any Hydra model in conf/model/.
# Compatible with existing MLP, seed-repeat, and Atom-JEPA CSV files.
# Examples:
#   VALUES_FILE=experiments/mlp_sweep.csv DRY_RUN=1 bash run_experiments.sh
#   VALUES_FILE=experiments/atom_jepa_sweep1.csv MODEL=atom_jepa bash run_experiments.sh
# Model selection priority: row's "model" column > MODEL environment variable > inferred default.
set -euo pipefail

cd "$(dirname "$0")"
VALUES_FILE="${VALUES_FILE:-experiments/mlp_sweep.csv}"
[[ -f "$VALUES_FILE" ]] || { echo "Missing sweep CSV: $VALUES_FILE" >&2; exit 1; }
[[ -f finetune.sh ]] || { echo "Missing finetune.sh" >&2; exit 1; }
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -d outputs ]] || { echo "Missing outputs/: run mkdir -p outputs" >&2; exit 1; }
    command -v bsub >/dev/null || { echo "Missing LSF bsub command" >&2; exit 1; }
fi

# Parse and validate the ENTIRE CSV before submitting any jobs.
# Python's CSV reader avoids fragile Bash comma parsing and supports optional columns.
jobfile="$(mktemp)"
trap 'rm -f "$jobfile"' EXIT
python3 - "$VALUES_FILE" "${MODEL:-}" > "$jobfile" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path, model_arg = sys.argv[1:]
field_map = {
    'lr': 'model.readout_lr',
    'readout_lr': 'model.readout_lr',
    'encoder_lr': 'model.encoder_lr',
    'freeze_encoder': 'model.freeze_encoder',
    'dropout': 'readout.dropout',
    'hidden_dim': 'readout.hidden_dim',
    'layers': 'readout.num_layers',
    'batch_size': 'train.batch_size',
    'weight_decay': 'train.weight_decay',
    'epochs': 'train.epochs',
    'grad_clip': 'train.grad_clip',
}
reserved = {'id', 'task', 'model', 'seed', 'extra_overrides'}

with open(csv_path, newline='', encoding='utf-8-sig') as source:
    reader = csv.DictReader(source)
    if not reader.fieldnames or not {'id', 'task'} <= set(reader.fieldnames):
        sys.exit('CSV must contain id and task columns')
    if None in reader.fieldnames:
        sys.exit('CSV contains an empty header column')

    n = 0
    for row in reader:
        n += 1
        if None in row:
            sys.exit(f'Row {n + 1}: too many fields (check trailing commas)')
        row = {k: (v or '').strip() for k, v in row.items()}
        experiment_id = row['id']
        if not experiment_id or experiment_id.startswith('#'):
            continue
        task = row['task']
        model = row.get('model') or model_arg or (
            'atom_jepa' if 'encoder_lr' in row or 'freeze_encoder' in row
            else 'morgan_rdkit_mlp'
        )
        seed = row.get('seed') or '42'
        if not re.fullmatch(r'[A-Za-z0-9_-]+', experiment_id):
            sys.exit(f'Row {n + 1}: invalid ID {experiment_id!r}')
        if task not in {'regression', 'classification'}:
            sys.exit(f'Row {n + 1}: invalid task {task!r}')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', model):
            sys.exit(f'Row {n + 1}: invalid model {model!r}')
        if not re.fullmatch(r'\d+', seed):
            sys.exit(f'Row {n + 1}: invalid seed {seed!r}')
        if not Path(f'conf/model/{model}.yaml').is_file():
            sys.exit(f'Row {n + 1}: conf/model/{model}.yaml does not exist')

        overrides = []
        seen = set()
        for column, value in row.items():
            if column in reserved or not value:
                continue
            # Besides the familiar shorthand keys, accept actual Hydra dotted paths.
            if column not in field_map and not re.fullmatch(r'[+~]?[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+', column):
                sys.exit(f'Row {n + 1}: unsupported override column {column!r}')
            key = field_map.get(column, column)
            if key in seen:
                sys.exit(f'Row {n + 1}: duplicated override key {key}')
            seen.add(key)
            if any(ch.isspace() for ch in value):
                sys.exit(f'Row {n + 1}: spaces not supported in value of {column}')
            overrides.append(f'{key}={value}')

        extra = row.get('extra_overrides', '')
        if extra:
            overrides.extend(extra.split())
        overrides.append(
            f'hydra.run.dir=runs/{task}/{model}/{experiment_id}/${{now:%Y-%m-%d_%H-%M-%S}}'
        )
        output = '\t'.join((experiment_id, task, model, seed, ' '.join(overrides)))
        print(output)
PY

count=0
while IFS=$'\t' read -r ID TASK SELECTED_MODEL SEED OVERRIDES; do
    export MODEL="$SELECTED_MODEL" TASK SEED OVERRIDES
    printf '%s | %-14s | model=%-18s | seed=%s | %s\n' \
        "$ID" "$TASK" "$MODEL" "$SEED" "$OVERRIDES"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        bsub -env all < finetune.sh
    fi
    ((count+=1))
done < "$jobfile"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Previewed $count experiment(s); no jobs submitted."
else
    echo "Submitted $count experiment(s)."
fi
