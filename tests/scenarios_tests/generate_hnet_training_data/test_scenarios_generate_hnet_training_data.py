# tests/scenarios_tests/generate_hnet_training_data/test_scenarios_generate_hnet_training_data.py

import pytest

from generate_hnet_training_data import main


@pytest.mark.scenarios_generate_data
def test_generate_data_with_various_distributions(sample_range, max_doas):
    """
    Parameterized test to generate data with different sample ranges and verify distributions.

    Args:
        sample_range (np.array): Array specifying the number of samples for each DOA combination.
        max_doas (int): Maximum number of Directions of Arrival (DOAs).
    """
    main(sample_range=sample_range, max_doas=max_doas)
