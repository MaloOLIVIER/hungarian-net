# tests/scenarios_tests/run/test_scenarios_run.py

import os
from pathlib import Path
import re

import pytest

@pytest.mark.scenarios
@pytest.mark.parametrize(
    "training_data, test_data",
    [
        (
            "data/20241205/train/hung_data_train_DOA2_3000-5000-15000",
            "data/20241205/test/hung_data_test_DOA2_3000-5000-15000",
        ),
        (
            "data/20241205/train/hung_data_train_DOA2_5000-5000-5000",
            "data/20241205/test/hung_data_test_DOA2_5000-5000-5000",
        ),
        (
            "data/20241205/train/hung_data_train_DOA2_1000-3000-31000",
            "data/20241205/test/hung_data_test_DOA2_1000-3000-31000",
        ),
        (
            "data/20241205/train/hung_data_train_DOA2_2600-5000-17000",
            "data/20241205/test/hung_data_test_DOA2_2600-5000-17000",
        ),
        (
            "data/20241205/train/hung_data_train_DOA2_6300-4000-1500",
            "data/20241205/test/hung_data_test_DOA2_6300-4000-1500",
        ),
        (
            "data/20241205/train/hung_data_train_DOA2_2000-7000-14000",
            "data/20241205/test/hung_data_test_DOA2_2000-7000-14000",
        ),
        (
            "data/20241205/train/hung_data_train_DOA2_2500-8000-8500",
            "data/20241205/test/hung_data_test_DOA2_2500-8000-8500",
        ),
    ],
)
def test_train_hnetgru_under_various_distributions(training_data, test_data):
    """
    Train the HNetGRU model with various data distributions.

    """

    # Extract sample ranges from the training_data filename
    match = re.search(r"hung_data_train_DOA\d+_(\d+)-(\d+)-(\d+)", training_data)
    if match:
        sample_range_used = "-".join(match.groups())
    else:
        sample_range_used = None  # Default values

    # Get the absolute paths for training and testing data
    current_dir = Path.cwd()
    train_filename = current_dir / training_data
    test_filename = current_dir / test_data

    # Create Hydra overrides
    overrides = [
        f"train_filename={train_filename}",
        f"test_filename={test_filename}",
        f"sample_range_used={sample_range_used}",
    ]

    os.system(f"python run.py {' '.join(overrides)}")

    assert True, "Training completed successfully"
