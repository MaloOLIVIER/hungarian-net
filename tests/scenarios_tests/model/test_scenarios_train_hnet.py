# tests/scenarios_tests/model/test_train_hnet.py

import re
import pytest
from run import main


@pytest.mark.scenarios
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

    # Extract sample ranges from the training_data filename
    match = re.search(r"hung_data_train_DOA\d+_(\d+)-(\d+)-(\d+)", training_data)
    if match:
        sample_range_used = list(map(int, match.groups()))
    else:
        sample_range_used = None  # Default values

    # Mock nb_epochs to be 1 regardless of the input
    nb_epochs = 1

    main(
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        max_len=max_doas,
        sample_range_used=sample_range_used,
        filename_train=training_data,
        filename_test=test_data,
    )
