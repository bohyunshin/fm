import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Factorization Machine implementation in PyTorch.

    FM formula: y = w0 + sum(wi * xi) + sum(sum(<vi, vj> * xi * xj))
    where <vi, vj> is the dot product of latent vectors
    """

    def __init__(self, input_dim, embedding_dim=32):
        """
        Args:
            num_features: Number of input features
            embedding_dim: Dimension of latent vectors for interactions
        """
        super().__init__()

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

        # Linear weights (first-order interactions)
        self.linear = nn.Linear(input_dim, 1, bias=False)

        # Embedding matrix for second-order interactions
        # Each feature gets an embedding vector of size embedding_dim
        self.embeddings = nn.Embedding(input_dim, embedding_dim)

        # Initialize parameters
        self._init_weights()

        self.sigmoid = nn.Sigmoid()

    def _init_weights(self):
        """Initialize model parameters"""
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.01)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        """
        Forward pass of the Factorization Machine

        Args:
            x: Input tensor of shape (batch_size, num_features)

        Returns:
            predictions: Output tensor of shape (batch_size, 1)
        """
        batch_size, num_features = x.shape

        # Global bias term
        bias_term = self.bias

        # Linear term: sum(wi * xi)
        linear_term = self.linear(x)

        # Second-order interaction term
        # Get embeddings for all features
        feature_indices = torch.arange(num_features, device=x.device)
        feature_embeddings = self.embeddings(
            feature_indices
        )  # (num_features, embedding_dim)

        # Compute interactions efficiently using the FM formula:
        # sum(sum(<vi, vj> * xi * xj)) = 0.5 * (sum(xi * vi)^2 - sum(xi^2 * vi^2))

        # x_embedded: (batch_size, num_features, embedding_dim)
        x_embedded = x.unsqueeze(2) * feature_embeddings.unsqueeze(0)

        # Square of sum: (sum(xi * vi))^2
        square_of_sum = torch.sum(x_embedded, dim=1) ** 2  # (batch_size, embedding_dim)

        # Sum of squares: sum(xi^2 * vi^2)
        sum_of_squares = torch.sum(x_embedded**2, dim=1)  # (batch_size, embedding_dim)

        # Interaction term: 0.5 * (square_of_sum - sum_of_squares)
        interaction_term = 0.5 * torch.sum(
            square_of_sum - sum_of_squares, dim=1, keepdim=True
        )

        # Combine all terms
        output = bias_term + linear_term + interaction_term

        return self.sigmoid(output)
