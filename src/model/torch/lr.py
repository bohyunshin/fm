import torch.nn as nn


class Model(nn.Module):
    """Logistic regression model using PyTorch"""

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
