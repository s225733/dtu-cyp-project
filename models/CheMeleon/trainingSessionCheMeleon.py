from pathlib import Path
from lightning import pytorch as pt
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data

from meccaCheMeleonModel import create_model
from load_dataset import loadData

#Configurations
Task = "regression" # or "regression"
Freeze = True
Epochs = 10

BatchSize = 32

#RELOADING
trainData, validData, testData = loadData(Task)

#data loaders
trainLoader = data.build_dataloader(trainData, batch_size=BatchSize, num_workers=0, seed=42)
validLoader = data.build_dataloader(validData, batch_size=BatchSize, num_workers=0, seed=42)
testLoader = data.build_dataloader(testData, batch_size=BatchSize, num_workers=0, seed=42, shuffle=False)

#Load CheMeleon model and attach to prediction head
model = create_model(Task, Freeze)

#save best model

checkpointDir = (Path(__file__).resolve().parent / "checkpoints" / Task / f"frozen_{Freeze}").resolve()

#Saves in checkpointDir, with name best_{Task}_model. It focuses on the validation loss and saves only the best model with the lowest validation loss
checkpoint_callback = ModelCheckpoint(
dirpath=checkpointDir,
filename=f"best_{Task}_model",
monitor="val_loss",
mode="min",
save_top_k=1
)


#Confuguring the trainer
trainer = pt.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=Epochs,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, trainLoader, validLoader)

print("Training completed!")
print("Best checkpoint:", checkpoint_callback.best_model_path)