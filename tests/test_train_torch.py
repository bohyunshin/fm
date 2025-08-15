import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import train_torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from train_torch import main


@pytest.mark.parametrize("model_type", ["lr", "fm"])
@patch("train_torch.DataLoader")
def test_main_with_model(mock_dataloader_class, mock_dataframe, mock_args, model_type):
    """Test main function with lr and fm models"""
    # Set the model type for this test
    mock_args.model = model_type

    # Setup mock DataLoader - return DataFrame like the real load() method
    mock_dataloader_instance = MagicMock()
    mock_dataloader_instance.load.return_value = mock_dataframe
    mock_dataloader_class.return_value = mock_dataloader_instance

    # Test with specified model
    try:
        main(mock_args)
        assert True  # If no exception raised, test passes
    except Exception as e:
        pytest.fail(f"main function failed with {model_type} model: {str(e)}")
