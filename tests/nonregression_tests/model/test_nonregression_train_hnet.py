# tests/nonregression_tests/model/test_nonregression_train_hnet.py

import os

import pytest
import torch
from pytest_mock import mocker

from hungarian_net.models import HNetGRU
from hungarian_net.train_hnet import main as train_main
from hungarian_net.train_hnet import set_seed


@pytest.mark.nonregression
def test_non_regression_train_hnet(mocker, max_doas, model, batch_size, nb_epochs):
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
    reference_model = HNetGRU(max_len=max_doas).to(torch.device("cpu"))
    reference_model.eval()

    reference_model.load_state_dict(
        torch.load(reference_model_path, map_location=torch.device("cpu"))
    )

    # Load the reference model's state dict
    reference_state_dict = reference_model.state_dict()

    # Mock the torch.save function to save the trained model's state dict
    mocker.patch("torch.save")

    # Run the training with default parameters
    trained_model = train_main(
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        max_len=max_doas,
        filename_train="data/reference/hung_data_train",
        filename_test="data/reference/hung_data_test",
    )
    trained_model.eval()

    # Load the newly trained model's state dict
    trained_state_dict = trained_model.state_dict()

    input_tensor = torch.randn(batch_size, model.max_len, model.max_len)

    # Get outputs from both models
    with torch.no_grad():
        ref_output = reference_model(input_tensor)
        ref_output = torch.concat(
            (ref_output[0].flatten(), ref_output[1].flatten(), ref_output[2].flatten())
        )

        trained_output = trained_model(input_tensor)
        trained_output = torch.concat(
            (
                trained_output[0].flatten(),
                trained_output[1].flatten(),
                trained_output[2].flatten(),
            )
        )

    print(
        "\nall close atol=1e-4, rtol=1e-2:",
        torch.allclose(ref_output, trained_output, atol=1e-4, rtol=1e-2),
    )
    print(
        "all close atol=1e-3, rtol=1e-1:",
        torch.allclose(ref_output, trained_output, atol=1e-3, rtol=1e-1),
    )
    print(
        "all close atol=1e-2, rtol=1e-1:",
        torch.allclose(ref_output, trained_output, atol=1e-2, rtol=1e-1),
    )
    print(
        "all close atol=1e-1, rtol=1e-1:",
        torch.allclose(ref_output, trained_output, atol=1e-1, rtol=1e-1),
    )
    print(
        "\nall close mean:",
        torch.allclose(ref_output.mean(), trained_output.mean(), atol=1e-4, rtol=1e-3),
    )
    print(
        "all close mean:",
        torch.allclose(ref_output.mean(), trained_output.mean(), atol=1e-3, rtol=1e-2),
    )
    print(
        "all close mean:",
        torch.allclose(ref_output.mean(), trained_output.mean(), atol=1e-2, rtol=1e-1),
    )

    # Compare outputs with the means of weights of the layers | beware of numeric computation errors
    assert torch.allclose(
        ref_output.mean(), trained_output.mean(), atol=1e-3, rtol=1e-2
    ), "Model outputs differ significantly."

    # Assert missing keys in the state dicts
    for key in reference_state_dict:
        trained_tensor = trained_state_dict.get(key)
        assert trained_tensor is not None, f"Key {key} missing in the trained model."

    # Ensure no extra keys in the trained model
    assert len(trained_state_dict) == len(
        reference_state_dict
    ), "Trained model has unexpected additional layers."
