# tests/consistency_tests/lightning_modules/test_consistency_hnet_gru_lightning.py

import pytest
import torch
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import MetricCollection
from hungarian_net.lightning_modules.hnet_gru_lightning import HNetGRULightning
from hungarian_net.torch_modules.hnet_gru import HNetGRU


@pytest.mark.consistency
def test_hnet_gru_lightning_initialization(
    hnet_gru_lightning: HNetGRULightning, metrics: MetricCollection
) -> None:
    """
    Test the initialization of HNetGRULightning.

    This test verifies that the HNetGRULightning module initializes correctly with the provided
    metrics, model, loss functions, optimizer, and scheduler.

    Args:
        hnet_gru_lightning (HNetGRULightning): Instance of HNetGRULightning.
        metrics (MetricCollection): Collection of metrics used by the module.

    Returns:
        None
    """
    assert hnet_gru_lightning.metrics == metrics, "Metrics not initialized correctly."
    assert isinstance(
        hnet_gru_lightning.model, HNetGRU
    ), "Model not initialized correctly."
    assert isinstance(
        hnet_gru_lightning.criterion1, torch.nn.BCEWithLogitsLoss
    ), "criterion1 not initialized correctly."
    assert isinstance(
        hnet_gru_lightning.criterion2, torch.nn.BCEWithLogitsLoss
    ), "criterion2 not initialized correctly."
    assert isinstance(
        hnet_gru_lightning.criterion3, torch.nn.BCEWithLogitsLoss
    ), "criterion3 not initialized correctly."
    assert hnet_gru_lightning.criterion_wts == [
        1.0,
        1.0,
        1.0,
    ], "Loss weights not initialized correctly."
    assert isinstance(
        hnet_gru_lightning.optimizer, torch.optim.Adam
    ), "Optimizer not initialized correctly."
    assert (
        hnet_gru_lightning.scheduler is None
    ), "Scheduler should be None when not provided."
    assert isinstance(
        hnet_gru_lightning.confusion_matrix, MulticlassConfusionMatrix
    ), "Confusion matrix not initialized correctly."


@pytest.mark.consistency
def test_hnet_gru_lightning_common_step(
    hnet_gru_lightning: HNetGRULightning, hnetgru: HNetGRU, max_doas: int, device: torch.device
) -> None:
    """
    Test the common_step method of HNetGRULightning.

    This test verifies that the common_step correctly computes the total loss and returns the expected outputs.

    Args:
        hnet_gru_lightning (HNetGRULightning): Instance of HNetGRULightning.
        hnetgru (HNetGRU): Instance of the HNetGRU model.
        max_doas (int): Maximum number of Degrees of Arrival (DOAs).

    Returns:
        None
    """
    # Arrange
    batch = (
        torch.randn(1, max_doas, max_doas).to(device=device),  # feat
        [
            torch.randn(max_doas, max_doas).reshape(-1).to(device=device),  # label
            torch.randn(max_doas, max_doas).sum(-1).to(device=device),
            torch.randn(max_doas, max_doas).sum(-2).to(device=device),
        ],
    )
    hnetgru_outputs = hnetgru(batch[0])

    # Act
    loss, output, target = hnet_gru_lightning.common_step(batch, batch_idx=0)

    # Assert
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor."
    assert hnetgru_outputs[0].shape == output[0].shape, "Output1 shape mismatch."
    assert hnetgru_outputs[1].shape == output[1].shape, "Output2 shape mismatch."
    assert hnetgru_outputs[2].shape == output[2].shape, "Output3 shape mismatch."
