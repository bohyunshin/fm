from typing import List
from datetime import datetime

import pandas as pd


def one_hot_encoding(data: pd.DataFrame, categorical_columns: List[str]):
    data_encoded = pd.get_dummies(
        data, columns=categorical_columns, drop_first=True, dtype=int
    )
    return data_encoded


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
        data = one_hot_encoding(
            data=data,
            categorical_columns=self.categorical_columns + time_based_feature_names,
        )
        feature_names = list(data.columns)
        return pd.concat([data, y, dt], axis=1), feature_names
