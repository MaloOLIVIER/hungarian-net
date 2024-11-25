# tests/conftest.py
import pytest
import numpy as np
import torch
from hungarian_net.train_hnet import HNetGRU, AttentionLayer

@pytest.fixture(params=[2, 4, 8])
def maxdoas_sources(request):
    return request.param

@pytest.fixture(params=[64, 128, 256])
def batch_size(request):
    return request.param

@pytest.fixture
def model(maxdoas_sources):
    return HNetGRU(max_len=maxdoas_sources)