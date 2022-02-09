import torch.nn as nn


def LinearModel(num_in, num_out):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_in, num_out),
    )


def SingleLayer(num_in, num_features, num_out):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_in, num_features),
        nn.GELU(),
        nn.Linear(num_features, num_out),
    )
