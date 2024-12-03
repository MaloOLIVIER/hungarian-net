# tests/nonregression_tests/generate_hnet_training_data/tests_nonregression_generate_hnet_training_data.py
import pytest
from pytest_mock import mocker

from run import set_seed

# TODO: Performing a non-regression test by directly comparing generated data with reference data is ineffective due to inherent numerical computation errors that may cause discrepancies.
# TODO: In future iterations, it would be more effective to assess regression by evaluating the individual components of the data generation program (e.g., functions, classes, methods) to ensure each part operates correctly without being affected by numerical inaccuracies.


@pytest.mark.nonregression
def test_non_regression_generate_hnet_training_data():
    """
    Non-regression test to ensure that generate_hnet_training_data produces the same data generated
    as the reference data generation when run with default parameters.
    """

    set_seed()

    # TODO: implement the test
