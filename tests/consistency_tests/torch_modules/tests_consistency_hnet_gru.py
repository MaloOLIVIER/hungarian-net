# tests/consistency_tests/torch_modules/tests_consistency_hnet_gru.py

import pytest
import torch
from hungarian_net.torch_modules.hnet_gru import HNetGRU


@pytest.mark.consistency
def test_HNetGRU_init(model, max_doas) -> None:
    """Test the initialization of the HNetGRU model.

    Args:
        model (HNetGRU): The model instance provided by the fixture.
        max_doas (int): The maximum number of Directions of Arrival, provided by the fixture.

    Returns:
        None
    """
    assert isinstance(model, HNetGRU), "Model is not an instance of HNetGRU"
    assert (
        model.max_len == max_doas
    ), f"Expected max_doas {max_doas}, got {model.max_len}"

@pytest.mark.consistency
def test_HNetGRU_forward(model, batch_size) -> None:
    """Test the forward pass of the HNetGRU model to ensure correct output shapes.

    Args:
        model (HNetGRU): The model instance provided by the fixture.
        batch_size (int): The size of the input batch, provided by the fixture.

    Returns:
        None
    """
    input_tensor = torch.randn(batch_size, model.max_len, model.max_len)
    # query - batch x seq x feature
    output1, output2, output3 = model.forward(input_tensor)
    # output1 - batch x (seq x feature)
    # output2 - batch x sequence
    # output3 - batch x feature

    assert output1.shape == (
        batch_size,
        model.max_len * model.max_len,
    ), f"Expected output1 shape {(batch_size, model.max_len, model.max_len)}, got {output1.shape}"
    assert output2.shape == (
        batch_size,
        model.max_len,
    ), f"Expected output2 shape {(batch_size, model.max_len)}, got {output2.shape}"
    assert output3.shape == (
        batch_size,
        model.max_len,
    ), f"Expected output3 shape {(batch_size, model.max_len)}, got {output3.shape}"