# tests/consistency_tests/torch_modules/test_consistency_hnet_gru.py

import pytest
import torch

from hungarian_net.torch_modules.hnet_gru import HNetGRU


@pytest.mark.consistency
def test_HNetGRU_init(hnetgru, max_doas) -> None:
    """Test the initialization of the HNetGRU model.

    Args:
        hnetgru (HNetGRU): The model instance provided by the fixture.
        max_doas (int): The maximum number of Directions of Arrival, provided by the fixture.

    Returns:
        None
    """
    assert isinstance(hnetgru, HNetGRU), "Model is not an instance of HNetGRU"
    assert (
        hnetgru.max_len == max_doas
    ), f"Expected max_doas {max_doas}, got {hnetgru.max_len}"


@pytest.mark.consistency
def test_HNetGRU_forward(hnetgru, batch_size) -> None:
    """Test the forward pass of the HNetGRU model to ensure correct output shapes.

    Args:
        hnetgru (HNetGRU): The model instance provided by the fixture.
        batch_size (int): The size of the input batch, provided by the fixture.

    Returns:
        None
    """
    input_tensor = torch.randn(batch_size, hnetgru.max_len, hnetgru.max_len)
    # query - batch x seq x feature
    output1, output2, output3 = hnetgru.forward(input_tensor)
    # output1 - batch x (seq x feature)
    # output2 - batch x sequence
    # output3 - batch x feature

    assert output1.shape == (
        batch_size,
        hnetgru.max_len * hnetgru.max_len,
    ), f"Expected output1 shape {(batch_size, hnetgru.max_len, hnetgru.max_len)}, got {output1.shape}"
    assert output2.shape == (
        batch_size,
        hnetgru.max_len,
    ), f"Expected output2 shape {(batch_size, hnetgru.max_len)}, got {output2.shape}"
    assert output3.shape == (
        batch_size,
        hnetgru.max_len,
    ), f"Expected output3 shape {(batch_size, hnetgru.max_len)}, got {output3.shape}"
