import logging

from typing import List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from libs.preprocess_utils import fill_in_missing_values_and_create_flag
from preprocess.base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    def __init__(
        self,
        categorical_columns: List[str],
        y_column: str,
        logger: logging.Logger,
        timestamp_column: Optional[str] = None,
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
        self,
        data: pd.DataFrame,
        val_time_point: Optional[datetime] = None,
        test_time_point: Optional[datetime] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Tuple:
        """Fit encoder on train data only, then transform all splits."""
        y = data[self.y_column]
        n = len(data)
        indices = np.arange(n)
        # First split: separate test set
        temp_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_seed
        )

        # Second split: separate train and validation from remaining data
        # Adjust val_ratio for the remaining data
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)

        train_indices, val_indices = train_test_split(
            temp_indices, test_size=adjusted_val_ratio, random_state=random_seed
        )

        # Prepare categorical data for all splits
        cat_data = fill_in_missing_values_and_create_flag(
            data=data,
            target_columns=self.categorical_columns,
        )
        self.logger.info("Done filling missing values")

        train_features_hashed = self.hasher.transform(
            cat_data.iloc[train_indices, :][self.categorical_columns].to_dict(
                orient="records"
            )
        )
        val_features_hashed = self.hasher.transform(
            cat_data.iloc[val_indices, :][self.categorical_columns].to_dict(
                orient="records"
            )
        )
        test_features_hashed = self.hasher.transform(
            cat_data.iloc[test_indices, :][self.categorical_columns].to_dict(
                orient="records"
            )
        )

        self.logger.info("Done hashing categorical features")

        # Prepare labels
        train_labels = y.iloc[train_indices].values
        val_labels = y.iloc[val_indices].values
        test_labels = y.iloc[test_indices].values

        return (
            (train_features_hashed, train_labels),
            (val_features_hashed, val_labels),
            (test_features_hashed, test_labels),
            train_features_hashed.shape[1],  # Number of features after hashing
        )
