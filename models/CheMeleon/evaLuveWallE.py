from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightning import pytorch as pt
from chemprop import data, models, utils

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from load_dataset import loadData
""" 
task = "regression"
freeze = True

if task not in ["classification", "regression"]:
    raise ValueError("WTF man, you typed the task wrong, now you wasting our time, please lock in and type the task correctly")

checkpointDir = (Path(__file__).parent / "checkpoints" / {task} / f"frozen_{freeze}").resolve()
checkpointPath = (Path(__file__).parent / "checkpoints" / {task} / f"frozen_{freeze}" / f"best_{task}_model").resolve()

#The different targets
 """

SUCCESSCAP = 0.5

TARGETS = {
    "classification": ["CYP2D6_is_TDI","CYP3A4_is_TDI"],
    "regression": ["CYP1A2_pIC50_direct_inhibition","CYP2C9_pIC50_direct_inhibition","CYP2D6_pIC50_direct_inhibition","CYP3A4_pIC50_direct_inhibition"]
}

def get_predictions(checkPointPath, task, evalSplit, freeze):
    if task not in ["classification", "regression"]:
        raise ValueError("WTF man, you typed the task wrong, now you wasting our time, please lock in and type the task correctly")

    #The different targets
    if task == "classification" and freeze:
        Targets = ["CYP2D6_is_TDI","CYP3A4_is_TDI"]
    elif task == "regression":
        Targets = ["CYP1A2_pIC50_direct_inhibition","CYP2C9_pIC50_direct_inhibition","CYP2D6_pIC50_direct_inhibition","CYP3A4_pIC50_direct_inhibition"]


    #load from checkpoint
    model = models.MPNN.load_from_checkpoint(
        checkpoint_path=checkPointPath,
        map_location="cpu",
        #hparams_file
    )

    model.eval()

    trData, valData, tstData = loadData(task)

    #sets dataset to be the wanted dataset
    dataSet = valData if evalSplit=="val" else tstData

    loader = data.build_dataloader(
        dataSet,
        batch_size= 32,
        num_workers=0,
        shuffle=False,
        seed=42
    )

    trainer = pt.Trainer(
        accelerator="auto",
        logger=False,
        enable_checkpointing=False
    )

    #Predictions without training
    batches = trainer.predict(model, dataloaders=loader)

    #puts batches in a vertical matrix 
    predictions = np.concatenate(
        [np.asarray(batch) for batch in batches], 
        axis = 0 # joins vertically, 1 for horizontally
    )

    #loading original labels
    originPath = (Path(__file__).resolve().parents[2] / "data" / task / f"{evalSplit}.csv").resolve()
    df = pd.read_csv(originPath)

    #Bug measures
    if len(df) != len(predictions):
        raise ValueError("Something is hella wrong, the math ain't mathing and numbers ain't numbering")

    if predictions.shape[1] != len((TARGETS[task])):
        raise ValueError("The models aren't modelling, the number of outputs ain't matching")

    return df, predictions


def evaluate_classification(df, predictions, freeze):
    results = []

    for i, target in enumerate(TARGETS["classification"]):
        y = df[target].to_numpy(dtype=float)

        # identifies finite numbers with booleans
        mask = np.isfinite(y)

        trueY = y[mask].astype(int)
        probabilities = predictions[mask, i]

        if len(trueY) == 0:
            print(f"No valid labels for {target}")
            continue

        # Threshold to deem success or none success
        yPrediction = (probabilities >= SUCCESSCAP).astype(int)

        #Confusion matrix
        tn, fp, fn, tp = confusion_matrix(trueY, yPrediction, labels=[0, 1]).ravel()

        results.append({
            "Targets": target,
            "Samples": len(trueY),
            "Successes": int(tp+tn),
            "Fails": int(fp + fn),
            "Accuracy": accuracy_score(trueY, yPrediction),
            "Precision": precision_score(trueY, yPrediction, zero_division=0),
            "Recall": recall_score(trueY, yPrediction, zero_division=0),
            "F1": f1_score(trueY, yPrediction, zero_division=0),
            "MCC": matthews_corrcoef(trueY, yPrediction),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        })

        #save confusion matrix
        confMatrix = confusion_matrix(trueY, yPrediction, labels=[0,1])

        disp = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=["False", "True"])

        disp.plot(cmap="Reds", values_format="d")
        plt.title(f"Confusion Matrix - {target} + {freeze}")
        plt.tight_layout()
        plt.savefig(Path(__file__).resolve().parent / "figures" / "classification" / f"frozen_{freeze}" / f"{evalSplit}_{target}_confMatrix.png" ,dpi=200)
        plt.close()

    dfResults = pd.DataFrame(results)
    dfResults.to_csv(Path(__file__).resolve().parent / "figures" / "classification" / f"frozen_{freeze}" / f"{evalSplit}_Classification metrics" ,index=False)

    print("\nClassification results:")
    print(dfResults.to_string(index=False))

    if len(dfResults) == len(TARGETS["classification"]):
        print("\nMean MCC:", dfResults["MCC"].mean())


def evaluate_regression(df, predictions, freeze):
    results = []
    scores = []

    for i, target in enumerate(TARGETS["regression"]):
        y = df[target].to_numpy(dtype=float)

        mask = np.isfinite(y)
        trueY = y[mask]
        yPrediction = predictions[mask, i]

        if len(trueY) == 0:
            print(f"No valid labels for {target}")
            continue

        #Evaluation calculations
        mae = mean_absolute_error(trueY, yPrediction) # low mae better (mean absolute error)
        rmse = np.sqrt(mean_squared_error(trueY, yPrediction)) #low rmse better (rooted mean square error)
        r2 = r2_score(trueY, yPrediction) if len(trueY) > 1 else np.nan #r suqare, 1 = perfect predictions, negative = bad, 0 = target mean

        #Get confidence intervals and check validity
        low = df[target + "_conf_low"].to_numpy(dtype=float)[mask]
        high = df[target + "_conf_high"].to_numpy(dtype=float)[mask]

        if not (
            np.isfinite(low).all() and 
            np.isfinite(high).all() and
            (low <= high).all()
        ):
            raise ValueError(f"Invalid confidence intervals: {target}")

        #distance from p to low conf and to high conf
        def distance_to_highLow(p):
            return np.maximum(low-p, 0) + np.maximum(p - high, 0)

        #calculating relative error metric
        denominator = distance_to_highLow(trueY.mean()).sum()

        st_rae = (distance_to_highLow(yPrediction).sum() / denominator if denominator > 0 else np.nan) #(Soft-Threshold Relative Absolute Error)

        scores.append(st_rae)

        results.append({
            "Target": target,
            "Samples": len(trueY),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "ST-RAE": st_rae
        })

         # Scatter plot: actual vs predicted
        plt.figure()
        plt.scatter(trueY, yPrediction, alpha=0.5)

        lower = min(trueY.min(), yPrediction.min())
        upper = max(trueY.max(), yPrediction.max())

        plt.plot([lower, upper], [lower, upper], "--")
        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.title(target)
        plt.tight_layout()

        plt.savefig(Path(__file__).resolve().parent / "figures" / "regression" / f"frozen_{freeze}" / f"{evalSplit}_{target}.png" ,dpi=200)
        plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(Path(__file__).resolve().parent / "figures" / "regression" / f"frozen_{freeze}" / f"{evalSplit}_Regression metrics" ,index=False)

    print("\nRegression results:")
    print(results_df.to_string(index=False))

    # Do not average if a target score is undefined
    if len(scores) == 4 and np.isfinite(scores).all():
        print("\nMA-ST-RAE:", np.mean(scores))
    else:
        print("\nMA-ST-RAE: undefined")



task = "regression" # or "regression"
evalSplit = "val" # or "test"
freeze = True # or "False"
checkpointDir = (Path(__file__).resolve().parent / "checkpoints" / task / f"frozen_{freeze}").resolve()
checkpointPath = (Path(__file__).resolve().parent / "checkpoints" / task / f"frozen_{freeze}" / f"best_{task}_model-v2.ckpt").resolve()        

df, predictions = get_predictions(
        checkpointPath,
        task,
        evalSplit,
        freeze
    )




# Save predictions for individual molecules
dfPredictions = df.copy()

for j, target in enumerate(TARGETS[task]):
    dfPredictions[f"pred_{target}"] = predictions[:, j]

dfPredictions.to_csv(
    Path(__file__).resolve().parent / "predictions" / task / f"frozen_{freeze}" / "predictions.csv",
    index=False
)

if task == "classification":
    evaluate_classification(df, predictions, freeze)
else:
    evaluate_regression(df, predictions, freeze)

print("\nResults")