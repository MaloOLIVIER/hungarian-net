# tests/consistency_tests/lightning_datamodules/test_consistency_hungarian_datamodule.py

import pytest
from typing import Dict
import numpy as np
from unittest.mock import patch, MagicMock
from hungarian_net.lightning_datamodules.hungarian_datamodule import HungarianDataModule, HungarianDataset

@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_dataset_init(mock_load_obj: MagicMock, data_dict: Dict) -> None:
    """
    Test the initialization of HungarianDataset.

    This test verifies that the HungarianDataset initializes correctly with the provided parameters.

    Args:
        mock_load_obj (MagicMock): Mocked load_obj function.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    dataset = HungarianDataset(train=True, max_doas=2, filename='mock_filename')

    # Assert
    assert dataset.data_dict == data_dict, "Data dictionary not loaded correctly."
    assert dataset.max_doas == 2, "max_doas not set correctly."

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_dataset_len(mock_load_obj: MagicMock, data_dict: Dict) -> None:
    """
    Test the __len__ method of HungarianDataset.

    This test verifies that the length of the dataset matches the number of entries
    in the data dictionary.

    Args:
        mock_load_obj (MagicMock): Mocked load_obj function.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    dataset = HungarianDataset(train=False, max_doas=2, filename='mock_filename')

    # Act
    length = len(dataset)

    # Assert
    assert length == len(data_dict), f"Dataset length {length} does not match expected {len(data_dict)}."

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_dataset_getitem(mock_load_obj: MagicMock, data_dict: Dict) -> None:
    """
    Test the __getitem__ method of HungarianDataset.

    This test verifies that the dataset returns the correct features and labels for a given index.

    Args:
        mock_load_obj (MagicMock): Mocked load_obj function.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    dataset = HungarianDataset(train=True, max_doas=2, filename='mock_filename')

    # Act & Assert for first item
    feat, label = dataset[0]
    expected_feat = data_dict[0][2]
    expected_label = [
        data_dict[0][3].reshape(-1),
        data_dict[0][3].sum(-1),
        data_dict[0][3].sum(-2),
    ]
    np.testing.assert_array_almost_equal(
        feat, expected_feat, decimal=6,
        err_msg="Features do not match expected values."
    )
    np.testing.assert_array_almost_equal(
        label[0], expected_label[0], decimal=6,
        err_msg="Labels do not match expected values."
    )
    np.testing.assert_array_almost_equal(
        label[1], expected_label[1], decimal=6,
        err_msg="Labels do not match expected values."
    )
    np.testing.assert_array_almost_equal(
        label[2], expected_label[2], decimal=6,
        err_msg="Labels do not match expected values."
    )

    # Act & Assert for second item
    feat, label = dataset[1]
    expected_feat = data_dict[1][2]
    expected_label = [
        data_dict[1][3].reshape(-1),
        data_dict[1][3].sum(-1),
        data_dict[1][3].sum(-2),
    ]
    np.testing.assert_array_almost_equal(
        feat, expected_feat, decimal=6,
        err_msg="Features do not match expected values."
    )
    np.testing.assert_array_almost_equal(
        label[0], expected_label[0], decimal=6,
        err_msg="Labels do not match expected values."
    )
    np.testing.assert_array_almost_equal(
        label[1], expected_label[1], decimal=6,
        err_msg="Labels do not match expected values."
    )
    np.testing.assert_array_almost_equal(
        label[2], expected_label[2], decimal=6,
        err_msg="Labels do not match expected values."
    )

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_dataset_compute_class_imbalance(mock_load_obj: MagicMock, data_dict: Dict) -> None:
    """
    Test the compute_class_imbalance method of HungarianDataset.

    This test verifies that the class imbalance is computed correctly based on the association matrices.

    Args:
        mock_load_obj (MagicMock): Mocked load_obj function.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    dataset = HungarianDataset(train=True, max_doas=2, filename='mock_filename')

    # Act
    class_counts = dataset.compute_class_imbalance()

    # Assert
    expected_counts = {
        0: 2,  # Two '0's in da_mat
        1: 3   # Three '1's in da_mat
    }
    assert class_counts == expected_counts, f"Expected {expected_counts}, got {class_counts}"

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.HungarianDataset')
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_datamodule_setup(mock_load_obj: MagicMock, mock_hungarian_dataset: MagicMock, lightning_datamodule: HungarianDataModule, data_dict: Dict) -> None:
    """
    Test the setup method of HungarianDataModule.

    This test verifies that the datasets for training, validation, and testing are initialized correctly
    based on the provided stage.

    Args:
        mock_load_obj (MagicMock): Mocked load_obj function.
        mock_hungarian_dataset (MagicMock): Mocked HungarianDataset class.
        lightning_datamodule (HungarianDataModule): Instance of HungarianDataModule.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    mock_hungarian_dataset.return_value = HungarianDataset(train=True, max_doas=2, filename='mock_train')

    # Act: Setup for 'fit' stage
    lightning_datamodule.setup(stage='fit')

    # Assert
    assert hasattr(lightning_datamodule, 'train_dataset'), "train_dataset not set."
    assert hasattr(lightning_datamodule, 'val_dataset'), "val_dataset not set."

    # Act: Setup for 'test' stage
    lightning_datamodule.setup(stage='test')

    # Assert
    mock_hungarian_dataset.assert_called_with(train=False, max_doas=2, filename='mock_test')
    assert hasattr(lightning_datamodule, 'test_dataset'), "test_dataset not set."

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_datamodule_train_dataloader(mock_load_obj: MagicMock, lightning_datamodule: HungarianDataModule, num_workers: int, batch_size: int, data_dict: Dict) -> None:
    """
    Test the train_dataloader method of HungarianDataModule.

    This test verifies that the training DataLoader is configured correctly.

    Args:
        lightning_datamodule (HungarianDataModule): Instance of HungarianDataModule.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    
    lightning_datamodule.setup(stage='fit')
    
    assert type(lightning_datamodule.train_dataset) is HungarianDataset
    assert lightning_datamodule.train_dataset.max_doas == HungarianDataset(train=True, max_doas=2, filename='mock_train').max_doas

    # Act
    train_loader = lightning_datamodule.train_dataloader()

    # Assert
    assert train_loader.batch_size == batch_size, "Batch size incorrect."
    assert train_loader.num_workers == num_workers, "Number of workers incorrect."
    assert train_loader.drop_last is True, "drop_last should be enabled."

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_datamodule_val_dataloader(mock_load_obj: MagicMock, lightning_datamodule: HungarianDataModule, num_workers: int, batch_size: int, data_dict: Dict) -> None:
    """
    Test the val_dataloader method of HungarianDataModule.

    This test verifies that the validation DataLoader is configured correctly.

    Args:
        lightning_datamodule (HungarianDataModule): Instance of HungarianDataModule.
        data_dict (Dict): Mocked data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    
    lightning_datamodule.setup(stage='fit')
    
    assert type(lightning_datamodule.val_dataset) is HungarianDataset
    assert lightning_datamodule.val_dataset.max_doas == HungarianDataset(train=False, max_doas=2, filename='mock_test').max_doas

    # Act
    val_loader = lightning_datamodule.val_dataloader()

    # Assert
    assert val_loader.batch_size == batch_size, "Batch size incorrect."
    assert val_loader.num_workers == num_workers, "Number of workers incorrect."
    assert val_loader.drop_last is False, "drop_last should be disabled."

@pytest.mark.consistency
@patch('hungarian_net.lightning_datamodules.hungarian_datamodule.load_obj')
def test_hungarian_datamodule_test_dataloader(mock_load_obj: MagicMock, lightning_datamodule: HungarianDataModule, num_workers: int, batch_size: int, data_dict: Dict) -> None:
    """
    Test the test_dataloader method of HungarianDataModule.

    This test verifies that the testing DataLoader is configured correctly.

    Args:
        lightning_datamodule (HungarianDataModule): Instance of HungarianDataModule.
        data_dict (Dict): Mock data dictionary.

    Returns:
        None
    """
    # Arrange
    mock_load_obj.return_value = data_dict
    
    lightning_datamodule.setup(stage='test')
    
    assert type(lightning_datamodule.test_dataset) is HungarianDataset
    assert lightning_datamodule.test_dataset.max_doas == HungarianDataset(train=False, max_doas=2, filename='mock_test').max_doas

    # Act
    test_loader = lightning_datamodule.test_dataloader()

    # Assert
    assert test_loader.batch_size == batch_size, "Batch size incorrect."
    assert test_loader.num_workers == num_workers, "Number of workers incorrect."
    assert test_loader.drop_last is False, "drop_last should be disabled."