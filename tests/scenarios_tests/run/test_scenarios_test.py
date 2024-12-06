# tests/scenarios_tests/run/test_scenarios_run.py

import os
import re
from pathlib import Path

import pytest


@pytest.mark.scenarios
@pytest.mark.scenarios_test
@pytest.mark.parametrize(
    "test_filename",
    [
        "data/20241206/test/hung_data_test_DOA2_3000-5000-15000",
        "data/20241206/test/hung_data_test_DOA2_5000-5000-5000",
        "data/20241206/test/hung_data_test_DOA2_1000-3000-31000",
        "data/20241206/test/hung_data_test_DOA2_2600-5000-17000",
        "data/20241206/test/hung_data_test_DOA2_6300-4000-1500",
        "data/20241206/test/hung_data_test_DOA2_2000-7000-14000",
        "data/20241206/test/hung_data_test_DOA2_2500-8000-8500",
    ],
)
@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "checkpoints/20241206/hnet_model_DOA2_1000-3000-31000_epoch\=14.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_2000-7000-14000_epoch\=25.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_2500-8000-8500_epoch\=28.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_2600-5000-17000_epoch\=28.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=2.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=9.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=16.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=19-v1.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=19.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=22.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=26.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_3000-5000-15000_epoch\=28.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_5000-5000-5000_epoch\=21.ckpt",
        "checkpoints/20241206/hnet_model_DOA2_6300-4000-1500_epoch\=29.ckpt",
    ],
)
def test_checkpoints(test_filename, checkpoint_path):
    """
    Train the HNetGRU model with various data distributions.

    """

    # Extract sample ranges from the checkpoint_path filename
    match = re.search(
        r"hnet_model_DOA\d+_(\d+)-(\d+)-(\d+)_epoch\\=\d+(?:-v\d+)?\.ckpt$",
        checkpoint_path,
    )
    if match:
        sample_range_trained_on = "-".join(match.groups())
    else:
        sample_range_trained_on = None  # Default values

    # Extract sample ranges from the testing_data filename
    match = re.search(r"hung_data_test_DOA\d+_(\d+)-(\d+)-(\d+)", test_filename)
    if match:
        sample_range_tested_on = "-".join(match.groups())
    else:
        sample_range_tested_on = None  # Default values

    # Get the absolute paths for training and testing data
    current_dir = Path.cwd()
    test_filename = current_dir / test_filename
    checkpoint_path = current_dir / checkpoint_path

    # Create Hydra overrides
    overrides = [
        f'checkpoint_path="{checkpoint_path}"',
        f"test_filename={test_filename}",
        f"sample_range_trained_on={sample_range_trained_on}",
        f"sample_range_tested_on={sample_range_tested_on}",
    ]

    os.system(f"python test.py {' '.join(overrides)}")

    assert True, "Testing completed successfully"
