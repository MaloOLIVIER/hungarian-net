# tests/consistency_tests/conftest.py

import pytest

from hungarian_net.torch_modules.attention_layer import AttentionLayer
from hungarian_net.torch_modules.hnet_gru import HNetGRU

# TODO: maybe rewrite docstrings


@pytest.fixture(params=[2, 4, 8])
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


@pytest.fixture(params=[64, 128, 256])
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


@pytest.fixture(params=[128])
def in_channels(request) -> int:
    """
    Fixture to provide different values for the number of input channels.
    """
    return request.param


@pytest.fixture(params=[128])
def out_channels(request) -> int:
    """
    Fixture to provide different values for the number of output channels.
    """
    return request.param


@pytest.fixture(params=[128])
def key_channels(request) -> int:
    """
    Fixture to provide different values for the number of key channels.
    """
    return request.param


@pytest.fixture
def attentionLayer(in_channels, out_channels, key_channels) -> AttentionLayer:
    """
    Fixture to initialize and provide an instance of the AttentionLayer model.

    This fixture creates an instance of the `AttentionLayer` model with specified input, output, and key channels.

    Args:
        in_channels (int): The number of input channels to the AttentionLayer.
        out_channels (int): The number of output channels from the AttentionLayer.
        key_channels (int): The number of key channels for the AttentionLayer.

    Returns:
        AttentionLayer: An initialized instance of the `AttentionLayer` model configured with the specified channels.
    """
    return AttentionLayer(in_channels, out_channels, key_channels)
