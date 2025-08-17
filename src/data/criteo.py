import random
import logging

import pandas as pd

from data.base import BaseDataLoader


class DataLoader(BaseDataLoader):
    COLUMNS = [
        "Sale",
        "SalesAmountInEuro",
        "time_delay_for_conversion",
        "click_timestamp",
        "nb_clicks_1week",
        "product_price",
        "product_age_group",
        "device_type",
        "audience_id",
        "product_gender",
        "product_brand",
        "product_category_1",
        "product_category_2",
        "product_category_3",
        "product_category_4",
        "product_category_5",
        "product_category_6",
        "product_category_7",
        "product_country",
        "product_id",
        "product_title",
        "partner_id",
        "user_id",
    ]

    def __init__(self, data_path: str, logger: logging.Logger):
        super().__init__(data_path, logger)

    def load(self, is_test: bool):
        df = pd.read_csv(
            self.data_path,
            sep="\t",
            header=None,
        )
        df.columns = self.COLUMNS
        if is_test:
            not_conversion = random.sample(
                list(df[lambda x: x["Sale"] == 0].index), 5000
            )
            conversion = random.sample(list(df[lambda x: x["Sale"] == 1].index), 5000)
            df = df.loc[not_conversion + conversion]
        ratio_of_target = df["Sale"].value_counts() / df.shape[0]
        self.logger.info(
            f"Ratio of 1/0 in target column: {round(ratio_of_target[1], 4)} / {round(ratio_of_target[0], 4)}"
        )
        return df
