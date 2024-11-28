# tests/nonregression_tests/model/test_nonregression_train_hnet.py

import os

import pytest
import torch
from pytest_mock import mocker

from hungarian_net.models import HNetGRU
from hungarian_net.train_hnet import main as train_main
from hungarian_net.train_hnet import set_seed
