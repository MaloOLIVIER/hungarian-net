# tests/scenarios_tests/model/test_train_hnet.py

import os
import pytest
import torch
from hungarian_net.train_hnet import main
from hungarian_net.dataset import HungarianDataset


@pytest.mark.parametrize(
    "training_data, test_data",
    [
        (
            "data/20241127/train/hung_data_train_DOA2_3000-5000-15000",
            "data/20241127/test/hung_data_test_DOA2_3000-5000-15000",
        ),
        (
            "data/20241127/train/hung_data_train_DOA2_5000-5000-5000",
            "data/20241127/test/hung_data_test_DOA2_5000-5000-5000",
        ),
        (
            "data/20241127/train/hung_data_train_DOA2_1000-3000-31000",
            "data/20241127/test/hung_data_test_DOA2_1000-3000-31000",
        ),
        (
            "data/20241127/train/hung_data_train_DOA2_2600-5000-17000",
            "data/20241127/test/hung_data_test_DOA2_2600-5000-17000",
        ),
        (
            "data/20241127/train/hung_data_train_DOA2_6300-4000-1500",
            "data/20241127/test/hung_data_test_DOA2_6300-4000-1500",
        ),
        (
            "data/20241127/train/hung_data_train_DOA2_2000-7000-14000",
            "data/20241127/test/hung_data_test_DOA2_2000-7000-14000",
        ),
        (
            "data/20241127/train/hung_data_train_DOA2_2500-8000-8500",
            "data/20241127/test/hung_data_test_DOA2_2500-8000-8500",
        ),
    ],
)
def test_train_model_under_various_distributions(
    max_doas, batch_size, nb_epochs, training_data, test_data
):
    """
    Train the HNetGRU model with various data distributions.
    
    Args:
        max_doas (int): Maximum number of Directions of Arrival (DOAs).
        batch_size (int): Number of samples per training batch.
        nb_epochs (int): Number of training epochs.
        training_data (str): Path to the training data file.
        test_data (str): Path to the testing data file.
    """

    assert False, "TODO: implement the test" 