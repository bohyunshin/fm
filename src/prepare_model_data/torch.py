from datetime import datetime
from typing import List

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from numpy.typing import NDArray


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
