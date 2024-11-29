# tests/nonregression_tests/conftest.py

import pytest
from hungarian_net.models import HNetGRU


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
        int: The current `batch_size` for the test iteration.

    Example:
        When used in a test, `batch_size` will sequentially take the values 64, 128, and 256.
    """
    return request.param


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


@pytest.fixture(params=[10])
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
        int: The current `nb_epochs` for the test iteration.
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
