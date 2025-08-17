import torch.nn as nn

from torch import Tensor


class Model(nn.Module):
    def __init__(self, num_features: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_ids_batch: Tensor, feature_values_batch: Tensor):
        """
        Forward pass of the logistic regression model for sparse input.
        """
        batch_size, num_features = feature_values_batch.shape

        feature_ids = feature_ids_batch.flatten()
        feature_values = feature_values_batch.flatten()

        # Linear term: sum(wi * xi)
        linear_weights = self.linear.weight.squeeze(0)[feature_ids]
        linear_values = linear_weights * feature_values
        logits = linear_values.reshape(batch_size, -1).sum(dim=1, keepdim=True)
        return self.sigmoid(logits)
