import random
import logging

import pandas as pd

from data.base import BaseDataLoader


class DataLoader(BaseDataLoader):
    COLUMNS = [
        "label",
        *(f"I{i}" for i in range(1, 14)),
        *(f"C{i}" for i in range(1, 27)),
    ]

    def __init__(self, data_path: str, logger: logging.Logger):
        super().__init__(data_path=data_path, logger=logger)

    def load(self, is_test: bool):
        df = pd.read_csv(
            self.data_path,
            sep="\t",
            header=None,
        )
        df.columns = self.COLUMNS
        if is_test:
            not_clicked = random.sample(list(df[lambda x: x["label"] == 0].index), 5000)
            clicked = random.sample(list(df[lambda x: x["label"] == 1].index), 1000)
            df = df.loc[not_clicked + clicked]
        ratio_of_target = df["label"].value_counts() / df.shape[0]
        self.logger.info(
            f"Ratio of 1/0 in target column: {round(ratio_of_target[1], 4)} / {round(ratio_of_target[0], 4)}"
        )
        return df
