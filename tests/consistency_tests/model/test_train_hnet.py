# tests/consistency_tests/model/test_train_hnet.py

import pytest
import torch
import numpy as np
from hungarian_net.train_hnet import HNetGRU, AttentionLayer

# TODO: maybe rewrite docstrings


def test_model_initialization(model, max_doas) -> None:
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


def test_forward_pass(model, batch_size) -> None:
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


def test_attention_layer_initialization(attentionLayer) -> None:
    """Test the initialization of the AttentionLayer.

    Args:
        attentionLayer (AttentionLayer): The AttentionLayer instance provided by the fixture.

    Returns:
        None
    """
    assert isinstance(
        attentionLayer, AttentionLayer
    ), f"AttentionLayer is not an instance of AttentionLayer class, got {attentionLayer.__repr__()}"


# TODO: write test for compute_weight_accuracy
