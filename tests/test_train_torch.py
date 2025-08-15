import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import train_torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from train_torch import main


@pytest.fixture
def mock_dataframe():
    """Create mock DataFrame that matches expected DataLoader format"""
    # Create mock data similar to what DataLoader.load() returns
    np.random.seed(42)
    n_samples = 1000

    # Create timestamps that align with fm.yaml config splits
    # val_time_point: "2020-10-01" (timestamp: 1601510400)
    # test_time_point: "2020-10-15" (timestamp: 1602720000)
    train_timestamps = np.random.randint(
        1577836800, 1601510400, n_samples // 3
    )  # 2020-01-01 to 2020-10-01 for train
    val_timestamps = np.random.randint(
        1601510400, 1602720000, n_samples // 3
    )  # 2020-10-01 to 2020-10-15 for val
    test_timestamps = np.random.randint(
        1602720000, 1604275200, n_samples - 2 * (n_samples // 3)
    )  # Remaining samples for test

    all_timestamps = np.concatenate([train_timestamps, val_timestamps, test_timestamps])
    np.random.shuffle(all_timestamps)

    data = {
        "Sale": np.random.randint(0, 2, n_samples),
        "SalesAmountInEuro": np.random.uniform(10, 100, n_samples),
        "time_delay_for_conversion": np.random.randint(1, 30, n_samples),
        "click_timestamp": all_timestamps,
        "nb_clicks_1week": np.random.randint(1, 50, n_samples),
        "product_price": np.random.uniform(5, 200, n_samples),
        "product_age_group": np.random.choice(["18-25", "26-35", "36-45"], n_samples),
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
        "audience_id": np.random.randint(1, 100, n_samples),
        "product_gender": np.random.choice(["M", "F", "U"], n_samples),
        "product_brand": np.random.choice(["brand_a", "brand_b", "brand_c"], n_samples),
        "product_category_1": np.random.choice(["cat1_a", "cat1_b"], n_samples),
        "product_category_2": np.random.choice(["cat2_a", "cat2_b"], n_samples),
        "product_category_3": np.random.choice(["cat3_a", "cat3_b"], n_samples),
        "product_category_4": np.random.choice(["cat4_a", "cat4_b"], n_samples),
        "product_category_5": np.random.choice(["cat5_a", "cat5_b"], n_samples),
        "product_category_6": np.random.choice(["cat6_a", "cat6_b"], n_samples),
        "product_category_7": np.random.choice(["cat7_a", "cat7_b"], n_samples),
        "product_country": np.random.choice(["US", "UK", "FR"], n_samples),
        "product_id": np.random.randint(1, 1000, n_samples),
        "product_title": [f"product_{i}" for i in range(n_samples)],
        "partner_id": np.random.randint(1, 50, n_samples),
        "user_id": np.random.randint(1, 10000, n_samples),
    }

    return pd.DataFrame(data)


@pytest.mark.parametrize("model_type", ["lr", "fm"])
@patch("train_torch.DataLoader")
def test_main_with_model(mock_dataloader_class, mock_dataframe, model_type):
    """Test main function with lr and fm models"""
    # Setup mock args
    args = MagicMock()
    args.model = model_type
    args.is_test = True
    args.criteo_data_path = "/mock/path"
    args.learning_rate = 0.01
    args.epochs = 2

    # Setup mock DataLoader - return DataFrame like the real load() method
    mock_dataloader_instance = MagicMock()
    mock_dataloader_instance.load.return_value = mock_dataframe
    mock_dataloader_class.return_value = mock_dataloader_instance

    # Test with specified model
    try:
        main(args)
        assert True  # If no exception raised, test passes
    except Exception as e:
        pytest.fail(f"main function failed with {model_type} model: {str(e)}")
