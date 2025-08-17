import logging

from typing import List, Tuple, Optional
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from libs.preprocess_utils import (
    fill_in_missing_values_and_create_flag,
    extract_timestamp_features,
)
from preprocess.base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    def __init__(
        self,
        categorical_columns: List[str],
        timestamp_column: str,
        y_column: str,
        logger: logging.Logger,
        hash_size: Optional[int] = None,
    ):
        super().__init__(
            categorical_columns=categorical_columns,
            timestamp_column=timestamp_column,
            y_column=y_column,
            logger=logger,
            hash_size=hash_size,
        )

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
            len(self.feature_names),
        )
