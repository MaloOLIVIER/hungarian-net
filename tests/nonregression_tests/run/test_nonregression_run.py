# tests/nonregression_tests/run/test_nonregression_run.py

import os

import pytest
import torch
from pytest_mock import mocker

from hungarian_net.torch_modules.hnet_gru import HNetGRU
from run import main as train_main
from run import set_seed

# TODO: Performing a non-regression test by directly comparing a newly trained model with a reference model is ineffective due to inherent numerical computation errors that can cause discrepancies.
# TODO: In future iterations, it would be more effective to assess regression by evaluating the model's individual components (e.g., functions, classes, methods) to ensure each part operates as expected without being affected by numerical inaccuracies.


@pytest.mark.nonregression
def test_non_regression_train_hnet(mocker):

    set_seed()
