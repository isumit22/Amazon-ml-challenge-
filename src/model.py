import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes=None, dropout=0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 128]
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out.squeeze()
