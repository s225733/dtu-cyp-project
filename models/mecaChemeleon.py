from pathlib import Path

import torch
from torch import nn

from chemprop import data, featurizers
from chemprop import nn as chem_nn

from models.readouts import build_readout

class CheMeleonLoader:

    def __init__(self, chem_loader, device):
        self.chem_loader = chem_loader
        self.device = device

    def __len__(self):
        return len(self.chem_loader)

    def __iter__(self):
        for batch in self.chem_loader:
            #V_d = atom level descriptors, X_d = molecule level descriptor, y = true CYP labels.
            graph, V_d, X_d, y, *_ = batch 

            #Chemprop's .to() modifies the graph in place
            graph.to(self.device)

            # yield instead of return
            yield {
                "graph": graph,
                "y": y.to(self.device)
            }

class CheMeleonModel(nn.Module):

    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()

        self.cfg = cfg
        self.device_name = torch.device(device)

        project_root = Path(__file__).resolve().parent.parent
        checkpoint_path = Path(cfg.checkpoint)

        # checks if 
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"file not found on {checkpoint_path}")


        #load pretrained
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.encoder = chem_nn.BondMessagePassing(**checkpoint['hyper_parameters'])
        self.encoder.load_state_dict(checkpoint['state_dict'])
        self.encoder.requires_grad_(not cfg.freeze_encoder)

        #Molecular aggregation
        self.agg = chem_nn.MeanAggregation()

        #create prediction head
        self.readout = build_readout(readout, self.encoder.output_dim, n_targets)

        #Molecular graph featurizer
        self.featurizer = (featurizers.SimpleMoleculeMolGraphFeaturizer())

        #move model parameters
        self.to(self.device_name)

        print("CheMeleon initialized successfully!")
        print("Task:", cfg.task)
        print("Number of targets:", n_targets)
        print("Encoder frozen:", cfg.freeze_encoder)

    @property
    def device(self):
        return self.device_name

    def fit_preprocessing(self, train):
        # this function is needed for finetune.py, but chemeleon does not need it
        pass

    def loader(self, frame, targets, batch_size, shuffle = False):
        labels = frame[targets].to_numpy(dtype=float)

        smiles = frame["SMILES"].tolist()

        datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in  zip(smiles, labels)]
        dataset = data.MoleculeDataset(datapoints, self.featurizer)

        chem_loader = data.build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return CheMeleonLoader(chem_loader, self.device)

    def param_groups(self):
        groups = [{
            "params": list(self.readout.parameters()),
            "lr": self.cfg.readout_lr
        }]

        if not self.cfg.freeze_encoder:
            groups.append({
                "params": list(self.encoder.parameters()),
                "lr": self.cfg.encoder_lr
            })

        return groups

    def train(self, mode=True):
        super().train(mode)

        if self.cfg.freeze_encoder:
            self.encoder.eval()

        return self

    def forward(self, batch):
        graph = batch["graph"]

        atom_embeddings = self.encoder(graph)

        molecule_embeddings = self.agg(atom_embeddings, graph.batch)

        predictions = self.readout(molecule_embeddings)

        return predictions