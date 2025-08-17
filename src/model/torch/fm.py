import torch
import torch.nn as nn

from torch import Tensor


class Model(nn.Module):
    """
    Factorization Machine implementation in PyTorch.

    FM formula: y = w0 + sum(wi * xi) + sum(sum(<vi, vj> * xi * xj))
    where <vi, vj> is the dot product of latent vectors
    """

    def __init__(self, num_features: int, embedding_dim=32, **kwargs):
        """
        Args:
            num_features: Number of input features
            embedding_dim: Dimension of latent vectors for interactions
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

        # Linear weights (first-order interactions)
        self.linear = nn.Linear(num_features, 1, bias=False)

        # Embedding matrix for second-order interactions
        # Each feature gets an embedding vector of size embedding_dim
        self.embeddings = nn.Embedding(num_features, embedding_dim)

        # Initialize parameters
        self._init_weights()

        self.sigmoid = nn.Sigmoid()

    def _init_weights(self):
        """Initialize model parameters"""
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.01)
        nn.init.constant_(self.bias, 0)

    def forward(self, feature_ids_batch: Tensor, feature_values_batch: Tensor):
        """
        Forward pass of the Factorization Machine optimized for sparse input.

        Args:
            batch_data: Tuple from DataLoader: (features_batch, labels_batch)
                       where features_batch is list of (indices, values) tuples

        Returns:
            predictions: Output tensor of shape (batch_size, 1)
        """
        batch_size, num_features = feature_values_batch.shape

        feature_ids = feature_ids_batch.flatten()
        feature_values = feature_values_batch.flatten()

        # Global bias
        output = self.bias.expand(batch_size, 1).clone()

        # Linear term: sum(wi * xi)
        linear_weights = self.linear.weight.squeeze(0)[feature_ids]
        linear_values = linear_weights * feature_values
        output += linear_values.reshape(batch_size, -1).sum(dim=1, keepdim=True)

        # Interaction term: 0.5 * (sum(xi * vi)^2 - sum(xi^2 * vi^2))
        embeddings = self.embeddings(feature_ids)  # (nnz, embedding_dim)

        # calculation of sum(xi * vi)^2 term
        # tensor multiplication of (batch_size, num_features, embedding_dim) * (batch_size, num_features, 1)
        weighted_sum = (
            embeddings.reshape(batch_size, -1, self.embedding_dim)
            * feature_values.unsqueeze(1).reshape(batch_size, -1).unsqueeze(2)
        ).sum(dim=1)
        squared_of_sum = (weighted_sum**2).sum(dim=1, keepdim=True)

        # calculation of sum(xi^2 * vi^2) term
        squared_weighted_sum = (
            embeddings.reshape(batch_size, -1, self.embedding_dim).pow(2)
            * feature_values.unsqueeze(1).reshape(batch_size, -1).unsqueeze(2).pow(2)
        ).sum(dim=1)
        sum_of_squared = squared_weighted_sum.sum(dim=1, keepdim=True)

        interaction = 0.5 * (squared_of_sum - sum_of_squared)

        output += interaction

        return self.sigmoid(output)
