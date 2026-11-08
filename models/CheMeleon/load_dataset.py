import pandas as pd
from chemprop import data, featurizers

#choose the task type for the model either classication or regression
def loadData(task):
    if task not in ["classification", "regression"]:
        raise ValueError("Unknown task, please look closely at wut ya typed")
    
    trainData = pd.read_csv(f"data/{task}/train.csv")
    validData = pd.read_csv(f"data/{task}/val.csv")
    testData = pd.read_csv(f"data/{task}/test.csv")

    if task == "classification":
        targets = ["CYP2D6_is_TDI","CYP3A4_is_TDI"]
    else:
        targets = ["CYP1A2_pIC50_direct_inhibition","CYP2C9_pIC50_direct_inhibition","CYP2D6_pIC50_direct_inhibition","CYP3A4_pIC50_direct_inhibition"]

    # creates datapoints from the SMILES strings and target values(e.g. the CYP inhibition values) in the data and returns a labeled list of Molecule datapoint objects(Chemprop datapoints)
    def create_datapoints(df):
        return [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(df["SMILES"], df[targets].to_numpy(dtype=float))]


    trainDatapoints = create_datapoints(trainData)
    validDatapoints = create_datapoints(validData)
    testDatapoints = create_datapoints(testData)

    #Preparing the data to fit into the Chemprop model. The featurizer converts the SMILES strings into a graph representation of molecules, which is the input for Chemprop
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    trData = data.MoleculeDataset(trainDatapoints, featurizer)
    valData = data.MoleculeDataset(validDatapoints, featurizer)
    tstData = data.MoleculeDataset(testDatapoints, featurizer)

    print("task:", task)
    print("length of training data:", len(trData))
    print("length of validation data:", len(valData))
    print("length of test data:", len(tstData))
    print("targets:", targets)

    return trData, valData, tstData