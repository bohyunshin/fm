from typing import List, Tuple, Optional
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
        data[f"{col}_missing_flag"] = data[col].map(lambda x: 1 if x == -1 else 0)
        if data[f"{col}_missing_flag"].sum() == 0:
            data.drop(f"{col}_missing_flag", axis=1, inplace=True)
            continue
        data[col] = data[col].map(lambda x: imputation_value if x == -1 else x)
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


class Preprocessor:
    def __init__(
        self,
        categorical_columns: List[str],
        timestamp_column: str,
        y_column: str,
    ):
        self.categorical_columns = categorical_columns
        self.timestamp_column = timestamp_column
        self.y_column = y_column
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: Optional[List[str]] = None

    def fit_and_preprocess(
        self, data: pd.DataFrame, val_time_point: datetime, test_time_point: datetime
    ) -> Tuple:
        """Fit encoder on train data only, then transform all splits."""
        y = data[self.y_column]
        data_copy = data.copy()
        data_copy, time_based_feature_names = extract_timestamp_features(
            data=data_copy, timestamp_column=self.timestamp_column
        )
        dt = data_copy[self.timestamp_column]

        # Split data based on timestamps
        train_mask = dt < val_time_point
        val_mask = (dt >= val_time_point) & (dt < test_time_point)
        test_mask = dt >= test_time_point

        # Prepare categorical data for all splits
        cat_data = fill_in_missing_values_and_create_flag(
            data=data_copy,
            target_columns=self.categorical_columns + time_based_feature_names,
        )

        # Get column names
        all_categorical_columns = self.categorical_columns + time_based_feature_names
        missing_flag_columns = [
            col for col in cat_data.columns if col.endswith("_missing_flag")
        ]
        all_columns = all_categorical_columns + missing_flag_columns

        # Fit encoder only on training data
        train_cat_data = cat_data[train_mask]
        self.encoder = OneHotEncoder(
            drop="first", sparse_output=True, dtype=int, handle_unknown="ignore"
        )
        self.encoder.fit(train_cat_data[all_columns])
        self.feature_names = self.encoder.get_feature_names_out(all_columns).tolist()

        # Transform all splits using fitted encoder
        train_features = self.encoder.transform(train_cat_data[all_columns])
        val_features = self.encoder.transform(cat_data[val_mask][all_columns])
        test_features = self.encoder.transform(cat_data[test_mask][all_columns])

        # Prepare labels
        train_labels = y[train_mask].values
        val_labels = y[val_mask].values
        test_labels = y[test_mask].values

        return (
            (train_features, train_labels),
            (val_features, val_labels),
            (test_features, test_labels),
            self.feature_names,
        )

    def preprocess(self, data: pd.DataFrame):
        y = data[self.y_column]
        data, time_based_feature_names = extract_timestamp_features(
            data=data, timestamp_column=self.timestamp_column
        )
        dt = data[self.timestamp_column]
        data = fill_in_missing_values_and_create_flag(
            data=data,
            target_columns=self.categorical_columns + time_based_feature_names,
        )
        data = sparse_one_hot_encoding(
            data=data,
            categorical_columns=self.categorical_columns + time_based_feature_names,
        )
        feature_names = list(data.columns)
        return pd.concat([data, y, dt], axis=1), feature_names
