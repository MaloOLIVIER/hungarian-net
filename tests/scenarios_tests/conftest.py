# tests/scenarios_tests/conftest.py

import hydra
import numpy as np
import pytest

from hungarian_net.torch_modules.hnet_gru import HNetGRU


@pytest.fixture(params=[2])
def max_doas(request) -> int:
    """
    Fixture to provide different values for the maximum number of Directions of Arrival (DOAs).

    This fixture parameterizes the `max_doas` value, allowing tests to run with varying numbers
    of DOAs to ensure that the model behaves correctly across different configurations.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current value of `max_doas` for the test iteration.

    Example:
        When used in a test, `max_doas` will sequentially take the values 2, 4, and 8.
    """
    return request.param


@pytest.fixture(params=[256])
def batch_size(request) -> int:
    """
    Fixture to provide different batch sizes for testing.

    This fixture parameterizes the `batch_size`, allowing tests to verify model performance
    and behavior with various batch sizes. This helps in ensuring that the model scales
    appropriately with different amounts of data processed in each training iteration.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current value of `batch_size` for the test iteration.

    Example:
        When used in a test, `batch_size` will sequentially take the values 64, 128, and 256.
    """
    return request.param


@pytest.fixture
def model(max_doas) -> HNetGRU:
    """
    Fixture to initialize and provide an instance of the HNetGRU model.

    This fixture creates an instance of the `HNetGRU` model with a specified maximum number
    of DOAs (`max_doas`). By parameterizing `max_doas`, this fixture ensures that the model
    is tested under different configurations, enhancing the robustness of your test suite.

    Args:
        max_doas (int): The maximum number of Directions of Arrival (DOAs) to be used by the model.
                        This value is provided by the `max_doas` fixture.

    Returns:
        HNetGRU: An initialized instance of the `HNetGRU` model configured with the specified `max_doas`.

    Example:
        When used in a test, `model` will be an instance of `HNetGRU` with `max_len` set to values
        2, 4, and 8 across different test iterations.
    """
    return HNetGRU(max_len=max_doas)


@pytest.fixture(params=[1000])
def nb_epochs(request) -> int:
    """
    Fixture to provide different numbers of training epochs for testing.

    This fixture parameterizes the `nb_epochs`, allowing tests to evaluate model training
    over various training durations. This helps in assessing model convergence and
    performance consistency across different training lengths.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current number of training epochs for the test iteration.
    """
    return request.param


@pytest.fixture(
    params=[
        np.array(
            [3000, 5000, 15000]
        ),  # Uniform Distribution : Ensures a balanced number of samples across different DOA combinations.
        np.array(
            [5000, 5000, 5000]
        ),  # Flat Distribution : Allocates the same number of samples irrespective of `min(nb_ref, nb_pred)`.
        np.array(
            [1000, 3000, 31000]
        ),  # Exponential Distribution : Significantly increases the number of samples for higher values of `min(nb_ref, nb_pred)`.
        np.array(
            [2600, 5000, 17000]
        ),  # Slightly Increasing Distribution : Gradually increases the number of samples as `min(nb_ref, nb_pred)` increases.
        np.array(
            [6300, 4000, 1500]
        ),  # Reverse Exponential Distribution : Allocates more samples to lower `min` values and fewer to higher ones.
        np.array(
            [2000, 7000, 14000]
        ),  # Custom Mixed Emphasis 1 : Significantly increases the number of samples for higher values of `min(nb_ref, nb_pred)`.
        np.array(
            [2500, 8000, 8500]
        ),  # Custom Mixed Emphasis 2 : Combines elements of uniform and skewed distributions for balanced yet focused sampling.
        # total samples = 45000 every case
    ]
)
def sample_range(request) -> np.array:
    """
    Fixture to provide different sample ranges for data generation.

    This fixture parameterizes the `sample_range`, allowing tests to generate training data
    with various sample distributions. This helps in assessing the model's robustness and
    performance under different data distribution scenarios.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        np.array: The current `sample_range` for the test iteration.

    Example:
        When used in a test, `sample_range` will sequentially take the following values:
        - [3000, 5000, 15000] (Uniform Distribution)
        - [5000, 5000, 5000]  (Flat Distribution)
        - [1000, 3000, 31000] (Skewed Distribution)
        - [2600, 5000, 17000] (Slightly Increasing Distribution)
        - [6300, 4000, 1500]  (Reverse Exponential Distribution)
        - [2000, 7000, 14000] (Custom Mixed Emphasis 1)
        - [2500, 8000, 8500]  (Custom Mixed Emphasis 2)
    """
    return request.param
