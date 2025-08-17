import pytest
from unittest.mock import patch
import sys
import os

# Add parent directory to path to import train_torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from train_torch import main


class MockDataLoader:
    def __init__(self, data_path, logger):
        self.data_path = data_path
        self.logger = logger
        self.mock_data = None

    def set_mock_data(self, data):
        self.mock_data = data

    def load(self, is_test=False):
        return self.mock_data


@pytest.mark.parametrize(
    "model,data_name",
    [
        ("lr", "criteo"),
        ("lr", "criteo_kaggle"),
        ("fm", "criteo"),
        ("fm", "criteo_kaggle"),
    ],
)
@patch("train_torch.load_data_module")
def test_main_with_different_models(
    mock_load_data_module,
    mock_args,
    mock_criteo_data,
    mock_criteo_kaggle_data,
    model,
    data_name,
):
    # Set the model for this test
    mock_args.model = model
    mock_args.data_name = data_name

    # Create mock data loader instance that returns the fixture data
    mock_loader_instance = MockDataLoader("/mock/path", None)
    if data_name == "criteo":
        mock_loader_instance.set_mock_data(mock_criteo_data)
    elif data_name == "criteo_kaggle":
        mock_loader_instance.set_mock_data(mock_criteo_kaggle_data)

    # Mock the data loader class to return our instance
    mock_load_data_module.return_value = lambda data_path, logger: mock_loader_instance

    # Run main and verify data loader was called correctly
    main(mock_args)

    mock_load_data_module.assert_called_once_with(data_name)
