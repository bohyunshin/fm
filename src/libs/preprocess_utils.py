from typing import List
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import torch


def sparse_one_hot_encoding(data: pd.DataFrame, categorical_columns: List[str]):
    encoder = OneHotEncoder(drop="first", sparse_output=True, dtype=int)

    # one hot encoding with sparse matrix
    sparse_matrix = encoder.fit_transform(data[categorical_columns])

    # convert csr sparse matrix to torch type
    sparse_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(sparse_matrix.nonzero()),  # indices
        torch.from_numpy(sparse_matrix.data).float(),  # values
        torch.Size(sparse_matrix.shape),  # size
    )
    return sparse_tensor


def fill_in_missing_values_and_create_flag(
    data: pd.DataFrame, target_columns: List[str], imputation_value: str = "MISSING"
):
    data = data[target_columns]
    for col in target_columns:
        data[col] = data[col].map(
            lambda x: imputation_value if x == -1 or x == "-1" or pd.isna(x) else x
        )
    return data


def convert_hour_to_category(hour: int):
    if 0 <= hour < 8:
        return "dawn"
    elif 8 <= hour < 13:
        return "morning"
    elif 13 <= hour < 19:
        return "afternoon"
    else:
        return "night"


def extract_timestamp_features(data: pd.DataFrame, timestamp_column: str):
    data["dt"] = data[timestamp_column].map(lambda x: datetime.fromtimestamp(x))
    data["weekday"] = data["dt"].map(lambda x: x.weekday())
    data["date"] = data["dt"].map(lambda x: x.date())
    data["month"] = data["dt"].map(lambda x: x.month)
    data["hour"] = data["dt"].map(lambda x: x.hour)
    data["hour_category"] = data["hour"].map(lambda x: convert_hour_to_category(x))
    data[timestamp_column] = data["dt"]
    return data, ["weekday", "hour_category"]
