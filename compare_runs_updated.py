#!/usr/bin/env python3
"""Compare MLP runs on your own computer, using only Python's standard library.

Put this file beside the copied runs/ directory, then run:
    python3 compare_runs.py

Optional paths:
    python3 compare_runs.py --runs /path/to/runs --output mlp_experiments.json

Outputs a complete JSON comparison, including every epoch's recorded history.
Also recognizes named sweep directories such as reg_hidden256/2026-11-10_13-42-40
and cross-references experiments/mlp_sweep.csv when available.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path


def run_identity(run_dir, runs_dir):
    """Return experiment ID and display timestamp for old/new Hydra layouts.

    Old: runs/<task>/<model>/<timestamp>/
    New: runs/<task>/<model>/<experiment_id>/<timestamp>/
    """
    parts = run_dir.relative_to(runs_dir).parts
    if len(parts) >= 4:
        experiment_id = parts[2]
        timestamp_folder = parts[-1]
    elif len(parts) == 3:
        experiment_id = None if is_timestamp(parts[2]) else parts[2]
        timestamp_folder = parts[-1]
    else:
        experiment_id = None
        timestamp_folder = parts[-1]
    return experiment_id, readable_timestamp(timestamp_folder)


def is_timestamp(name):
    try:
        datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
        return True
    except ValueError:
        return False


def readable_timestamp(folder_name):
    try:
        return datetime.strptime(folder_name, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return folder_name


def numeric(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def summary_score(task, metrics):
    """Return the appropriate macro metric and its optimization direction."""
    if task == "regression":
        return "macro_ST-RAE", metrics.get("macro_ST-RAE"), "min"
    if task == "classification":
        for name in ("macro_MCC", "mean_MCC", "macro_mcc", "mean_mcc"):
            if name in metrics:
                return name, metrics[name], "max"
        return "macro_MCC", None, "max"
    return None, None, None


def read_history(path):
    """Read the original JSON-lines records, including per-target validation metrics."""
    if not path.is_file():
        return [], ["history.jsonl missing"]

    history = []
    warnings = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                history.append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"Invalid history line {line_number}: {exc}")

    history.sort(key=lambda row: row.get("epoch") if numeric(row.get("epoch")) else math.inf)
    if not history:
        warnings.append("No readable training epochs")
    return history, warnings


def history_diagnostics(history, metric_name, direction, saved_score):
    """Compare training loss with validation performance *over time*.

    Loss and validation metric have different units: only compare their trends,
    never subtract training loss from ST-RAE or MCC.
    """
    valid = [row for row in history
             if numeric(row.get("epoch")) and numeric(row.get(metric_name))]
    loss_rows = [row for row in history if numeric(row.get("train_loss"))]

    result = {
        "epochs_recorded": len(history),
        "first_epoch": history[0].get("epoch") if history else None,
        "final_epoch": history[-1].get("epoch") if history else None,
        "initial_train_loss": loss_rows[0]["train_loss"] if loss_rows else None,
        "final_train_loss": loss_rows[-1]["train_loss"] if loss_rows else None,
        "best_epoch": None,
        "best_epoch_train_loss": None,
        "best_epoch_validation_score": None,
        "initial_validation_score": None,
        "final_validation_score": None,
        "validation_worsening_after_best": None,
        "validation_worsening_percent": None,
        "train_loss_decreased_after_best": None,
        "possible_overfitting_pattern": None,
        "matches_saved_validation_score": None,
    }
    if not valid or direction not in ("min", "max"):
        return result

    best = (min(valid, key=lambda row: row[metric_name]) if direction == "min"
            else max(valid, key=lambda row: row[metric_name]))
    first, last = valid[0], valid[-1]
    best_score = best[metric_name]
    final_score = last[metric_name]
    # Positive means final validation performance is worse than the best epoch.
    worsening = (final_score - best_score if direction == "min"
                 else best_score - final_score)
    if numeric(best.get("train_loss")) and numeric(last.get("train_loss")):
        loss_decreased = last["train_loss"] < best["train_loss"] - 1e-9
    else:
        loss_decreased = None

    result.update({
        "best_epoch": best["epoch"],
        "best_epoch_train_loss": best.get("train_loss"),
        "best_epoch_validation_score": best_score,
        "initial_validation_score": first[metric_name],
        "final_validation_score": final_score,
        "validation_worsening_after_best": max(0.0, worsening),
        "validation_worsening_percent": (
            100.0 * max(0.0, worsening) / abs(best_score)
            if best_score != 0 else None
        ),
        "train_loss_decreased_after_best": loss_decreased,
        # An indicator, not a proof: training loss keeps improving as
        # validation performance deteriorates after its best epoch.
        "possible_overfitting_pattern": (
            bool(loss_decreased and worsening > 1e-6)
            if loss_decreased is not None else None
        ),
        "matches_saved_validation_score": (
            math.isclose(saved_score, best_score, rel_tol=1e-5, abs_tol=1e-6)
            if numeric(saved_score) else None
        ),
    })
    return result


def validation_details(run_dir, config, metrics, metric_name, metric_value):
    """Describe the validation split and the saved checkpoint's performance.

    Target label counts are not unique molecule counts: each molecule can have
    labels for several CYP targets.
    """
    target_counts = {
        name: result["n"]
        for name, result in metrics.items()
        if isinstance(result, dict) and isinstance(result.get("n"), int)
    }

    # The training script exports train+validation split assignments here.
    # If unavailable, leave the unique validation-row count unknown.
    validation_rows = None
    split_path = run_dir / "split.csv"
    if split_path.is_file():
        try:
            with split_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if "split" in (reader.fieldnames or []):
                    validation_rows = sum(
                        row["split"].strip().lower() in ("val", "validation")
                        for row in reader
                    )
        except (OSError, UnicodeError, csv.Error):
            validation_rows = None

    hashes = config.get("split_sha256") or {}
    return {
        "split": "validation",
        "unique_rows_in_split_export": validation_rows,
        "labeled_samples_per_target": target_counts,
        "sha256": hashes.get("val") if isinstance(hashes, dict) else None,
        "selection_metric": metric_name,
        "selected_checkpoint_score": metric_value,
        "all_target_metrics": metrics,
    }


def experiment_variables(config, diagnostics):
    """Expose model and training choices without hiding them in nested configs."""
    model = config.get("model") or {}
    readout = config.get("readout") or {}
    train = config.get("train") or {}
    return {
        "network_architecture": {
            "model": model.get("name"),
            "featurizers": model.get("featurizers"),
            "readout_type": readout.get("type"),
            "hidden_dimension": readout.get("hidden_dim"),
            "number_of_layers": readout.get("num_layers"),
        },
        "optimization": {
            "learning_rate": model.get("readout_lr"),
            "batch_size": train.get("batch_size"),
            "configured_epochs": train.get("epochs"),
            "gradient_clipping": train.get("grad_clip"),
        },
        "regularization": {
            "dropout": readout.get("dropout"),
            "weight_decay": train.get("weight_decay"),
        },
        "training_loss": {
            "first_epoch": diagnostics.get("initial_train_loss"),
            "best_validation_epoch": diagnostics.get("best_epoch_train_loss"),
            "final_epoch": diagnostics.get("final_train_loss"),
        },
    }


def collect_runs(runs_dir):
    experiments = []
    skipped = 0
    for config_path in sorted(runs_dir.rglob("config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "validation_metrics.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable config: {config_path} ({exc})")
            skipped += 1
            continue

        model = config.get("model", {})
        if not isinstance(model, dict) or not str(model.get("name", "")).endswith("_mlp"):
            continue
        if not metrics_path.is_file():
            print(f"Skipping unfinished run (no validation metrics): {run_dir}")
            skipped += 1
            continue

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable metrics: {metrics_path} ({exc})")
            skipped += 1
            continue

        task = config.get("task")
        metric_name, metric_value, direction = summary_score(task, metrics)
        history, warnings = read_history(run_dir / "history.jsonl")
        diagnostics = history_diagnostics(history, metric_name, direction, metric_value)
        variables = experiment_variables(config, diagnostics)
        validation_set = validation_details(
            run_dir, config, metrics, metric_name, metric_value
        )

        experiment_id, timestamp = run_identity(run_dir, runs_dir)
        experiments.append({
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "task": task,
            "model_name": model.get("name"),
            "run_directory": str(run_dir.relative_to(runs_dir)),
            "experiment_variables": variables,  # Key settings grouped for easy comparison.
            "validation_set": validation_set,  # Split identity, size and target metrics.
            "model": model,  # All model settings, including readout_lr.
            "readout": config.get("readout", {}),  # Hidden size, layers, dropout, etc.
            "train": config.get("train", {}),  # Epochs, batch size, weight decay, etc.
            "seed": config.get("seed"),
            "device": config.get("device"),
            "summary_metric": {"name": metric_name, "value": metric_value,
                               "better": "lower" if direction == "min" else "higher"},
            "validation_metrics": metrics,  # Per-target selected-checkpoint metrics.
            "training_diagnostics": diagnostics,
            "epoch_history": history,  # All epoch records from history.jsonl.
            "warnings": warnings,
        })

    experiments.sort(key=lambda exp: (exp["timestamp"], str(exp["task"]), exp["run_directory"]))
    return experiments, skipped


def fmt(value, digits=4):
    if numeric(value):
        return f"{value:.{digits}f}"
    return "N/A" if value is None else str(value)


def print_ranked_results(experiments, task):
    selected = [exp for exp in experiments if exp["task"] == task]
    if not selected:
        return

    reverse = task == "classification"
    known = [exp for exp in selected if numeric(exp["summary_metric"]["value"])]
    unknown = [exp for exp in selected if not numeric(exp["summary_metric"]["value"])]
    known.sort(key=lambda exp: exp["summary_metric"]["value"], reverse=reverse)

    metric_name = "Macro ST-RAE ↓" if task == "regression" else "Macro MCC ↑"
    print(f"\n{task.upper()} — ranked by validation score ({metric_name})")
    print("=" * 100)
    for rank, exp in enumerate(known + unknown, 1):
        v = exp["experiment_variables"]
        arch = v["network_architecture"]
        opt = v["optimization"]
        reg = v["regularization"]
        losses = v["training_loss"]
        d = exp["training_diagnostics"]
        validation = exp["validation_set"]
        score = exp["summary_metric"]["value"]
        overfit = ("possible" if d["possible_overfitting_pattern"] is True else
                   "not confirmed" if d["possible_overfitting_pattern"] is False else "unknown")

        label = exp.get("experiment_id") or "manual / original layout"
        print(f"{rank:>2}. {label} | {exp['timestamp']} | {exp['model_name']} | "
              f"saved {metric_name}: {fmt(score, 6)}")
        print(f"    Architecture: {arch['number_of_layers']} layers, "
              f"hidden={arch['hidden_dimension']}, readout={arch['readout_type']}; "
              f"featurizers={arch['featurizers']}")
        print(f"    Optimization: LR={fmt(opt['learning_rate'], 6)}, "
              f"batch={opt['batch_size']}, epochs={opt['configured_epochs']}, "
              f"grad_clip={opt['gradient_clipping']}")
        print(f"    Regularization: dropout={reg['dropout']}, "
              f"weight_decay={reg['weight_decay']}")
        print(f"    Train loss: first={fmt(losses['first_epoch'], 6)}, "
              f"best-epoch={fmt(losses['best_validation_epoch'], 6)}, "
              f"last={fmt(losses['final_epoch'], 6)}")
        print(f"    Validation: best epoch={d['best_epoch']}, "
              f"best={fmt(d['best_epoch_validation_score'], 6)}, "
              f"last={fmt(d['final_validation_score'], 6)}, "
              f"worsening={fmt(d['validation_worsening_percent'], 2)}%, "
              f"overfit pattern={overfit}")
        print(f"    Validation set: rows={validation['unique_rows_in_split_export']}, "
              f"labeled n per target={validation['labeled_samples_per_target']}")
        print()
    print("Note: Training loss and validation metrics are different quantities; "
          "overfitting is a pattern, not a diagnosis.")
    print("      'Saved' uses validation_metrics.json; 'last' uses the final epoch in history.jsonl.")


def read_sweep(path):
    """Read the planned configurations; no dependency on the HPC submission script."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {"id", "task", "lr", "dropout", "hidden_dim", "layers",
                    "batch_size", "weight_decay", "epochs"}
        if not expected.issubset(reader.fieldnames or []):
            missing = sorted(expected - set(reader.fieldnames or []))
            raise ValueError(f"Invalid sweep CSV {path}; missing columns: {missing}")
        rows = []
        for row in reader:
            if not row.get("id") or row["id"].strip().startswith("#"):
                continue
            rows.append({key: value.strip() if isinstance(value, str) else value
                         for key, value in row.items()})
    return rows


def parameter_mismatches(plan, experiment):
    """Check CSV settings against the settings actually recorded by Hydra.

    Checking the saved configuration matters: a matching directory ID by itself
    does not guarantee that all intended overrides reached the training job.
    """
    config_values = {
        "lr": experiment["model"].get("readout_lr"),
        "dropout": experiment["readout"].get("dropout"),
        "hidden_dim": experiment["readout"].get("hidden_dim"),
        "layers": experiment["readout"].get("num_layers"),
        "batch_size": experiment["train"].get("batch_size"),
        "weight_decay": experiment["train"].get("weight_decay"),
        "epochs": experiment["train"].get("epochs"),
    }
    mismatches = []
    for key, actual in config_values.items():
        expected = plan.get(key)
        try:
            same = numeric(actual) and math.isclose(float(expected), float(actual),
                                                    rel_tol=1e-8, abs_tol=1e-10)
        except (ValueError, TypeError):
            same = str(expected) == str(actual)
        if not same:
            mismatches.append(f"{key}: planned={expected}, actual={actual}")
    return mismatches


def sweep_comparison(planned, experiments):
    """Match completed runs to the CSV experiment IDs.

    Multiple successful runs for one ID are kept in experiments but one best
    validation run per ID appears in this overview.
    """
    rows = []
    for plan in planned:
        task = plan["task"]
        direction = "min" if task == "regression" else "max"
        matches = [exp for exp in experiments if exp["experiment_id"] == plan["id"]
                   and exp["task"] == task
                   and numeric(exp["summary_metric"]["value"])]
        winner = (min(matches, key=lambda x: x["summary_metric"]["value"])
                  if matches and direction == "min" else
                  max(matches, key=lambda x: x["summary_metric"]["value"])
                  if matches else None)
        mismatches = parameter_mismatches(plan, winner) if winner else []
        status = ("completed_with_parameter_mismatch" if mismatches else "completed") if winner else "no_completed_run_found"
        rows.append({
            "id": plan["id"],
            "task": task,
            "configured_parameters": {k: v for k, v in plan.items() if k not in ("id", "task")},
            "status": status,
            "parameter_mismatches": mismatches,
            "completed_run_count": len(matches),
            "metric": winner["summary_metric"] if winner else None,
            "best_epoch": winner["training_diagnostics"]["best_epoch"] if winner else None,
            "initial_train_loss": winner["training_diagnostics"]["initial_train_loss"] if winner else None,
            "final_train_loss": winner["training_diagnostics"]["final_train_loss"] if winner else None,
            "best_run_directory": winner["run_directory"] if winner else None,
        })
    return rows


def print_sweep_coverage(rows):
    if not rows:
        return
    completed = sum(row["status"].startswith("completed") for row in rows)
    print(f"\nSWEEP COVERAGE — {completed}/{len(rows)} configurations have completed results")
    print("-" * 72)
    for row in rows:
        metric_value = (row["metric"] or {}).get("value")
        score = fmt(metric_value, 6) if metric_value is not None else "N/A"
        print(f"{row['id']:<19} {row['task']:<15} "
              f"{row['status']:<34} score={score} "
              f"runs={row['completed_run_count']}")
        for mismatch in row["parameter_mismatches"]:
            print(f"    WARNING: {mismatch}")
    print("Missing means no completed *local* validation result; check the HPC or sync runs/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--runs", type=Path, default=root / "runs", help="Copied HPC runs directory")
    parser.add_argument("--output", type=Path, default=root / "mlp_experiments.json", help="JSON output")
    parser.add_argument("--sweep", type=Path, default=root / "experiments/mlp_sweep.csv",
                        help="Optional planned experiment CSV for ID coverage")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error(f"Runs directory not found: {args.runs}")

    experiments, skipped = collect_runs(args.runs)
    try:
        planned = read_sweep(args.sweep)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.error(str(exc))
    sweep_rows = sweep_comparison(planned, experiments)
    result = {
        "total_experiments": len(experiments),
        "skipped_experiments": skipped,
        "sweep_file": str(args.sweep) if args.sweep.is_file() else None,
        "sweep_comparison": sweep_rows,
        "notes": {
            "possible_overfitting_pattern": (
                "True if training loss decreased after the best validation epoch while "
                "the validation score worsened; this is suggestive, not proof."
            ),
            "selection": "Use validation scores only, not test scores, for hyperparameter selection.",
        },
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Completed MLP runs: {len(experiments)}; skipped: {skipped}")
    print(f"JSON saved to: {args.output}")
    for task in ("regression", "classification"):
        print_ranked_results(experiments, task)
    print_sweep_coverage(sweep_rows)
    if not experiments:
        print("No completed MLP experiments found. Check validation_metrics.json files.")


if __name__ == "__main__":
    main()
