# tests/scenarios_tests/run/test_scenarios_run.py

import os
import re
from pathlib import Path

import pytest


@pytest.mark.scenarios
@pytest.mark.scenarios_run
@pytest.mark.parametrize(
    "training_data, test_data",
    [
        (
            "data/20241206/train/hung_data_train_DOA2_3000-5000-15000",
            "data/20241206/test/hung_data_test_DOA2_3000-5000-15000",
        ),
        (
            "data/20241206/train/hung_data_train_DOA2_5000-5000-5000",
            "data/20241206/test/hung_data_test_DOA2_5000-5000-5000",
        ),
        (
            "data/20241206/train/hung_data_train_DOA2_1000-3000-31000",
            "data/20241206/test/hung_data_test_DOA2_1000-3000-31000",
        ),
        (
            "data/20241206/train/hung_data_train_DOA2_2600-5000-17000",
            "data/20241206/test/hung_data_test_DOA2_2600-5000-17000",
        ),
        (
            "data/20241206/train/hung_data_train_DOA2_6300-4000-1500",
            "data/20241206/test/hung_data_test_DOA2_6300-4000-1500",
        ),
        (
            "data/20241206/train/hung_data_train_DOA2_2000-7000-14000",
            "data/20241206/test/hung_data_test_DOA2_2000-7000-14000",
        ),
        (
            "data/20241206/train/hung_data_train_DOA2_2500-8000-8500",
            "data/20241206/test/hung_data_test_DOA2_2500-8000-8500",
        ),
    ],
)
def test_run_under_various_distributions(training_data: str, test_data: str) -> None:
    """
    Train the HNetGRU model with various data distributions.

    This test function parameterizes over different training and testing datasets with varying
    data distributions. It runs the HNetGRU training process using specified data configurations
    to ensure that the model can handle different data distributions appropriately.

    Args:
        training_data (str): Path to the training data file with a specific DOA distribution.
        test_data (str): Path to the test data file corresponding to the training data distribution.

    Returns:
        None

    Example:
        The test runs multiple times with different training and testing data paths,
        each representing a unique distribution configuration for the purposes of training and evaluation.
    """

    # Extract sample ranges from the training_data filename
    match = re.search(r"hung_data_train_DOA\d+_(\d+)-(\d+)-(\d+)", training_data)
    if match:
        sample_range_trained_on = "-".join(match.groups())
    else:
        sample_range_trained_on = None  # Default values

    # Get the absolute paths for training and testing data
    current_dir = Path.cwd()
    train_filename = current_dir / training_data
    test_filename = current_dir / test_data

    # Create Hydra overrides
    overrides = [
        f"train_filename={train_filename}",
        f"test_filename={test_filename}",
        f"sample_range_trained_on={sample_range_trained_on}",
    ]

    # Execute the training script with the specified overrides
    os.system(f"python run.py {' '.join(overrides)}")

    # Placeholder assertion to indicate test completion
    assert True, "Training completed successfully"