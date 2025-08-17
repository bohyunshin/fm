from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from numpy.typing import NDArray


class SparseDataset(Dataset):
    def __init__(self, features_sparse, labels: np.ndarray):
        self.labels = torch.FloatTensor(labels)

        # Keep scipy sparse matrix in CSR format for efficient row access
        if hasattr(features_sparse, "tocsr"):
            self.features_sparse = features_sparse.tocsr()
        else:
            self.features_sparse = features_sparse

        # Pre-compute shape for tensor creation
        self.num_features = self.features_sparse.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract sparse row using CSR format (more efficient for row access)
        sparse_row = self.features_sparse[idx]  # CSR allows efficient row slicing

        # Return indices and values directly - no tensor creation needed
        if sparse_row.nnz > 0:
            col_indices = torch.from_numpy(sparse_row.indices).long()
            row_values = torch.from_numpy(sparse_row.data).float()
        else:
            col_indices = torch.empty(0, dtype=torch.long)
            row_values = torch.empty(0, dtype=torch.float32)

        return (col_indices, row_values), self.labels[idx]


def sparse_collate_fn(batch):
    """Custom collate function to handle variable-length sparse data"""
    features, labels = zip(*batch)

    # Stack labels normally
    labels = torch.stack(labels)

    # For features, we need to handle variable lengths
    # Option A: Pad to max length in batch
    max_nnz = max(len(feat[0]) for feat in features)

    batch_indices = []
    batch_values = []

    for feat in features:
        indices, values = feat
        # Pad with zeros if needed
        if len(indices) < max_nnz:
            pad_size = max_nnz - len(indices)
            indices = torch.cat([indices, torch.zeros(pad_size, dtype=torch.long)])
            values = torch.cat([values, torch.zeros(pad_size, dtype=torch.float32)])

        batch_indices.append(indices)
        batch_values.append(values)

    batch_indices = torch.stack(batch_indices)
    batch_values = torch.stack(batch_values)

    return (batch_indices, batch_values), labels


def split_sparse_data(
    metadata: pd.DataFrame,
    sparse_features,
    val_time_point: datetime,
    test_time_point: datetime,
    dt_column: str,
    y_column: str,
) -> Tuple:
    """Split sparse data based on timestamps"""

    # Create time-based masks
    train_mask = metadata[dt_column] < val_time_point
    val_mask = (metadata[dt_column] >= val_time_point) & (
        metadata[dt_column] < test_time_point
    )
    test_mask = metadata[dt_column] >= test_time_point

    # Split sparse features
    train_features = sparse_features[train_mask]
    val_features = sparse_features[val_mask]
    test_features = sparse_features[test_mask]

    # Split labels
    train_labels = metadata[train_mask][y_column].values
    val_labels = metadata[val_mask][y_column].values
    test_labels = metadata[test_mask][y_column].values

    return (
        (train_features, train_labels),
        (val_features, val_labels),
        (test_features, test_labels),
    )


def prepare_sparse_torch_dataloader(
    train_data: Tuple,
    val_data: Tuple,
    test_data: Tuple,
    batch_size: int = 128,
    num_workers: int = 1,
):
    """Prepare PyTorch DataLoaders for sparse data including test data"""

    train_features, train_labels = train_data
    val_features, val_labels = val_data
    test_features, test_labels = test_data

    train_dataset = SparseDataset(
        features_sparse=train_features,
        labels=train_labels,
    )
    val_dataset = SparseDataset(
        features_sparse=val_features,
        labels=val_labels,
    )
    test_dataset = SparseDataset(
        features_sparse=test_features,
        labels=test_labels,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=sparse_collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=sparse_collate_fn,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=sparse_collate_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader


class CustomDataset(Dataset):
    def __init__(self, features: NDArray, labels: NDArray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def split_data(
    data: pd.DataFrame,
    val_time_point: datetime,
    test_time_point: datetime,
    dt_column: str,
):
    train = data[lambda x: x[dt_column] < val_time_point]
    val = data[
        lambda x: (val_time_point <= x[dt_column]) & (x[dt_column] < test_time_point)
    ]
    test = data[lambda x: (test_time_point <= x[dt_column])]
    return (
        train.drop(dt_column, axis=1),
        val.drop(dt_column, axis=1),
        test.drop(dt_column, axis=1),
    )


def prepare_torch_dataloader(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: List[str],
    y_name: str,
    batch_size: int = 128,
    num_workers: int = 1,
):
    train_dataset = CustomDataset(
        features=train[feature_names].values,
        labels=train[y_name].values,
    )
    val_dataset = CustomDataset(
        features=val[feature_names].values,
        labels=val[y_name].values,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    return train_dataloader, val_dataloader, test
