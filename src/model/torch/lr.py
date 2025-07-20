import torch.nn as nn


class Model(nn.Module):
    """Logistic regression model using PyTorch"""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
