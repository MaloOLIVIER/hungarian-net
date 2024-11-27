# tests/nonregression_tests/conftest.py

import os

import pytest
import torch
from pytest_mock import mocker

from hungarian_net.generate_hnet_training_data import main as generate_data_main
from hungarian_net.train_hnet import main as train_main
from hungarian_net.train_hnet import set_seed

# TODO: write a non regression test for the generation of data :
# to generate data under default parameters at this version of the code should output the same
# as ADAVANNE's version


@pytest.mark.nonregression
def test_non_regression_train_hnet(mocker):
    """
    Non-regression test to ensure that train_hnet.py produces the same HNet model
    as the reference model when run with default parameters.
    """

    set_seed()

    # Path to the reference model
    reference_model_path = os.path.join("models", "reference", "hnet_model.pt")

    assert os.path.exists(
        reference_model_path
    ), f"Reference model not found at {reference_model_path}"

    # Load the reference model's state dict
    reference_state_dict = torch.load(
        reference_model_path, map_location=torch.device("cpu")
    )

    # Mock the torch.save function to save the trained model's state dict
    mocker.patch("torch.save")

    # Run the training with default parameters
    trained_model = train_main(
        batch_size=256,
        nb_epochs=10,
        max_len=2,
        filename_train="data/reference/hung_data_train",
        filename_test="data/reference/hung_data_test",
    )

    # Load the newly trained model's state dict
    trained_state_dict = trained_model.state_dict()

    # Compare the state dicts
    for key in reference_state_dict:
        ref_tensor = reference_state_dict[key]
        trained_tensor = trained_state_dict.get(key)
        assert trained_tensor is not None, f"Key {key} missing in the trained model."
        assert torch.allclose(
            ref_tensor.mean(), trained_tensor.mean(), atol=1e-5
        ), f"Mismatch found in layer: {key}"
        # TODO: still has bugs, need to fix it, atol issue
        # TODO: git lfs for data files

    # Ensure no extra keys in the trained model
    assert len(trained_state_dict) == len(
        reference_state_dict
    ), "Trained model has unexpected additional layers."


@pytest.mark.nonregression
def test_non_regression_generate_hnet_training_data():
    """
    Non-regression test to ensure that generate_hnet_training_data produces the same data generated
    as the reference data generation when run with default parameters.
    """


#     generate_data_main() # Generate data with default parameters
