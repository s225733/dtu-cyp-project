
import torch

from chemprop import nn, models, data

#task = regression or classification
def create_model(task, freeze=True):
    #load pretrained CheMeleon model
    agg = nn.MeanAggregation()
    chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])

    mp.load_state_dict(chemeleon_mp['state_dict']) 
    mp.requires_grad_(not freeze) # frozen state

    print("CheMeleon pretrained model loaded successfully!")
    print(mp)

    if task == "classification":
        fnn = nn.BinaryClassificationFFN(input_dim=mp.output_dim, n_tasks=2)
    elif task == "regression":
        fnn = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=4)
    else:
        raise ValueError("Unknown task, please look closely at wut ya typed")

    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    model_mpnn = models.MPNN(mp,agg,fnn,batch_norm=False, metrics=metric_list)

    print(f"CheMeleon model for {task} created successfully, the model is:", model_mpnn)

    return model_mpnn
