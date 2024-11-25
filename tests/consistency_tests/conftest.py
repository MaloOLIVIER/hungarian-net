# tests/conftest.py
import pytest
import numpy as np
import torch
from hungarian_net.train_hnet import HNetGRU, AttentionLayer

@pytest.fixture(params=[2, 4, 8])
def max_doas(request) -> int:
    """_summary_

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """
    return request.param

@pytest.fixture(params=[64, 128, 256])
def batch_size(request) -> int:
    return request.param

@pytest.fixture
def model(max_doas) -> HNetGRU:
    return HNetGRU(max_len=max_doas)