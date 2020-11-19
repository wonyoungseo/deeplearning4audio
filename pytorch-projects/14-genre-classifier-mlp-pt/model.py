import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, dropout_rate):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=1690, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 각 샘플 별로 flatten
        out = self.layers(x)
        return out