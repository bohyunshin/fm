import logging

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher


class BasePreprocessor(ABC):
    def __init__(
        self,
        categorical_columns: List[str],
        timestamp_column: str,
        y_column: str,
        logger: logging.Logger,
        hash_size: int = None,
    ):
        self.categorical_columns = categorical_columns
        self.timestamp_column = timestamp_column
        self.y_column = y_column
        self.logger = logger
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self.hasher = FeatureHasher(n_features=hash_size, input_type="dict")

    @abstractmethod
    def fit_and_preprocess(
        self, data: pd.DataFrame, val_time_point: datetime, test_time_point: datetime
    ) -> Tuple:
        raise NotImplementedError("This method should be implemented in subclasses.")
