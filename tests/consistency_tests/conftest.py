# tests/consistency_tests/conftest.py

import pytest
from typing import Dict
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score

from hungarian_net.lightning_datamodules.hungarian_datamodule import HungarianDataModule
from unittest.mock import patch, MagicMock
from hungarian_net.lightning_modules.hnet_gru_lightning import HNetGRULightning
from hungarian_net.torch_modules.attention_layer import AttentionLayer
from hungarian_net.torch_modules.hnet_gru import HNetGRU
import numpy as np


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
def hnetgru(max_doas) -> HNetGRU:
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
        When used in a test, `model` will be an instance of `HNetGRU` with `max_doas` set to values
        2, 4, and 8 across different test iterations.
    """
    return HNetGRU(max_doas=max_doas)


@pytest.fixture(params=[128])
def in_channels(request) -> int:
    """
    Fixture to provide different values for the number of input channels.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current number of input channels for the test iteration.

    Example:
        When used in a test, `in_channels` will take the value 128.
    """
    return request.param


@pytest.fixture(params=[128])
def out_channels(request) -> int:
    """
    Fixture to provide different values for the number of output channels.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current number of output channels for the test iteration.

    Example:
        When used in a test, `out_channels` will take the value 128.
    """
    return request.param


@pytest.fixture(params=[128])
def key_channels(request) -> int:
    """
    Fixture to provide different values for the number of key channels.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current number of key channels for the test iteration.

    Example:
        When used in a test, `key_channels` will take the value 128.
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

    Example:
        When used in a test, `attentionLayer` will be an instance with 128 input, output, and key channels.
    """
    return AttentionLayer(in_channels, out_channels, key_channels)

@pytest.fixture(params=[1,2,4])
def num_workers(request) -> int:
    """
    Fixture to provide different values for the number of workers in the DataLoader.

    This fixture parameterizes the `num_workers` value, allowing tests to run with varying numbers
    of workers to ensure that the DataLoader behaves correctly across different configurations.

    Args:
        request (FixtureRequest): Pytest's fixture request object that provides access to the
                                  parameters specified in the `params` list.

    Returns:
        int: The current value of `num_workers` for the test iteration.

    Example:
        When used in a test, `num_workers` will sequentially take the values 1, 2, and 4.
    """
    return request.param

@pytest.fixture
def metrics(max_doas: int) -> MetricCollection:
    """
    Fixture to provide a MetricCollection for testing HNetGRULightning.

    This fixture creates a collection of metrics, including the MulticlassF1Score, configured
    based on the `max_doas` parameter. It ensures that the metrics are appropriately set up
    for evaluating classification performance in multi-class scenarios.

    Args:
        max_doas (int): The number of classes for the MulticlassF1Score metric.

    Returns:
        MetricCollection: A collection of metrics used by the HNetGRULightning module.

    Example:
        When used in a test, `metrics` will include a weighted MulticlassF1Score for the specified number of DOAs.
    """
    return MetricCollection({
        'f1': MulticlassF1Score(num_classes=max_doas, average='weighted', zero_division=1)
    })

@pytest.fixture
def hnet_gru_lightning(metrics: MetricCollection, max_doas: int) -> HNetGRULightning:
    """
    Fixture to provide an instance of HNetGRULightning for testing.

    This fixture initializes the `HNetGRULightning` module with the provided metrics and
    maximum number of DOAs. It sets up the Lightning module to be used in various tests,
    ensuring consistency across test cases.

    Args:
        metrics (MetricCollection): The collection of metrics to be used by the module.
        max_doas (int): The maximum number of Directions of Arrival (DOAs) for the model.

    Returns:
        HNetGRULightning: An initialized instance of the `HNetGRULightning` module.

    Example:
        When used in a test, `hnet_gru_lightning` will be an instance configured with the specified metrics and DOAs.
    """
    return HNetGRULightning(
        metrics=metrics,
        max_doas=max_doas
    )

@pytest.fixture
def data_dict() -> Dict:
    """
    Fixture to provide a mock data dictionary for testing HungarianDataset.

    This fixture returns a dictionary containing mock data samples, each consisting of:
    - Number of references (`nb_ref`)
    - Number of predictions (`nb_pred`)
    - Distance matrix (`dist_mat`)
    - Association matrix (`da_mat`)
    - Reference Cartesian coordinates (`ref_cart`)
    - Prediction Cartesian coordinates (`pred_cart`)

    The mock data is structured to simulate real dataset entries, enabling comprehensive testing
    of the HungarianDataModule's functionality.

    Returns:
        Dict: A dictionary containing mock data samples.

    Example:
        The `data_dict` fixture returns a dictionary with two entries:
            - Entry 0: 2 references and predictions with specific distance and association matrices.
            - Entry 1: 1 reference and prediction with simplified matrices.
    """
    return {
        0: (
            2,  # nb_ref
            2,  # nb_pred
            np.array([[0.0, 1.0], [1.0, 0.0]]),  # dist_mat
            np.array([[1, 0], [0, 1]]),          # da_mat
            np.array([[0, 0, 0], [1, 0, 0]]),   # ref_cart
            np.array([[0, 0, 0], [1, 0, 0]]),   # pred_cart
        ),
        1: (
            1,  # nb_ref
            1,  # nb_pred
            np.array([[0.0]]),  # dist_mat
            np.array([[1]]),     # da_mat
            np.array([[0, 0, 0]]),  # ref_cart
            np.array([[0, 0, 0]]),  # pred_cart
        ),
    }
    
@pytest.fixture
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def lightning_datamodule(mock_load_obj: MagicMock, batch_size: int, num_workers: int) -> HungarianDataModule:
    """
    Fixture to provide an instance of HungarianDataModule for testing.

    This fixture mocks the `load_obj` function to return the provided `data_dict` and initializes
    the `HungarianDataModule` with mock filenames, maximum DOAs, batch size, and number of workers.
    It ensures that the data loading process within the module uses the controlled mock data,
    facilitating isolated and reliable tests.

    Args:
        mock_load_obj (MagicMock): Mock of the `load_obj` function to return `data_dict`.
        batch_size (int): The batch size to be used by the DataLoader.
        num_workers (int): The number of worker processes for data loading.
        data_dict (Dict): The mock data dictionary to be returned by `load_obj`.

    Returns:
        HungarianDataModule: An initialized instance of `HungarianDataModule` with mock data.

    Example:
        When used in a test, `lightning_datamodule` will load data from `data_dict` with specified batch size and workers.
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    return HungarianDataModule(
        train_filename='mock_train',
        test_filename='mock_test',
        max_doas=2, #because mock_data_dict goes up to max_doas=2
        batch_size=batch_size,
        num_workers=num_workers
    )