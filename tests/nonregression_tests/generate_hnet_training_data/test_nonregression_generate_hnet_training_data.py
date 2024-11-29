# tests/nonregression_tests/generate_hnet_training_data/test_nonregression_generate_hnet_training_data.py
import pytest
from pytest_mock import mocker

from hungarian_net.generate_hnet_training_data import main as generate_data_main
from hungarian_net.train_hnet import set_seed

# TODO: write a non regression test for the generation of data :
# to generate data under default parameters at this version of the code should output the same
# as ADAVANNE's version


@pytest.mark.nonregression
def test_non_regression_generate_hnet_training_data():
    """
    Non-regression test to ensure that generate_hnet_training_data produces the same data generated
    as the reference data generation when run with default parameters.
    """

    set_seed()


#     generate_data_main() # Generate data with default parameters
