# tests/scenarios_tests/generate_hnet_training_data/test_scenarios_generate_hnet_training_data.py

import pytest

from generate_hnet_training_data import main
import numpy as np


@pytest.mark.scenarios
@pytest.mark.scenarios_generate_data
@pytest.mark.parametrize(
    "resolution_range",
    [("standard_resolution"), ("fine_resolution"), ("coarse_resolution")],
)
def test_generate_data_with_various_distributions(
    sample_range: np.ndarray, max_doas: int, resolution_range: str
) -> None:
    """
    Parameterized test to generate training data with different resolution settings.

    This test invokes the `main` function from the `generate_hnet_training_data` module with
    various resolution ranges to ensure that data is generated correctly under different
    configuration scenarios. It verifies that the data generation process accommodates
    standard, fine, and coarse resolutions as specified.

    Args:
        sample_range (np.ndarray): Array specifying the number of samples for each DOA combination.
        max_doas (int): Maximum number of Directions of Arrival (DOAs).
        resolution_range (str): The resolution setting for data generation. Can be "standard_resolution", "fine_resolution", or "coarse_resolution".

    Returns:
        None

    Example:
        The test will run three times with the following `resolution_range` values:
            - "standard_resolution"
            - "fine_resolution"
            - "coarse_resolution"
    """
    main(
        sample_range=sample_range, max_doas=max_doas, resolution_range=resolution_range
    )
