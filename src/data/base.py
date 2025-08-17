from abc import ABC, abstractmethod

import logging


class BaseDataLoader(ABC):
    def __init__(self, data_path: str, logger: logging.Logger):
        self.data_path = data_path
        self.logger = logger

    @abstractmethod
    def load(self, is_test: bool):
        raise NotImplementedError("This method should be overridden by subclasses.")
