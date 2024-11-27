# tests/consistency_tests/generate_hnet_training_data/test_generate_hnet_training_data.py

import numpy as np
import pytest

from hungarian_net.generate_hnet_training_data import compute_class_imbalance, sph2cart

# TODO: maybe rewrite docstrings


@pytest.mark.consistency
def test_sph2cart():
    azimuth = 0
    elevation = 0
    r = 1
    expected = np.array([1, 0, 0])
    result = sph2cart(azimuth, elevation, r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.consistency
@pytest.mark.parametrize(
    "azimuth, elevation, r, expected",
    [
        # Azimuth = 0 radians
        (0, -np.pi / 2, 1, np.array([0, 0, -1])),
        (0, 0, 1, np.array([1, 0, 0])),
        (0, np.pi / 2, 1, np.array([0, 0, 1])),
        # Azimuth = π/2 radians
        (np.pi / 2, -np.pi / 2, 1, np.array([0, 0, -1])),
        (np.pi / 2, 0, 1, np.array([0, 1, 0])),
        (np.pi / 2, np.pi / 2, 1, np.array([0, 0, 1])),
        # Azimuth = π radians
        (np.pi, -np.pi / 2, 1, np.array([0, 0, -1])),
        (np.pi, 0, 1, np.array([-1, 0, 0])),
        (np.pi, np.pi / 2, 1, np.array([0, 0, 1])),
        # Azimuth = π/3 radians
        (np.pi / 3, -np.pi / 2, 1, np.array([0, 0, -1])),
        (np.pi / 3, 0, 1, np.array([0.5, np.sqrt(3) / 2, 0])),
        (np.pi / 3, np.pi / 2, 1, np.array([0, 0, 1])),
    ],
)
def test_sph2cart_multiple_cases(azimuth, elevation, r, expected):
    """
    Test the sph2cart function with multiple azimuth, elevation, and radius values.
    """
    result = sph2cart(azimuth, elevation, r)
    assert np.allclose(
        result, expected, atol=1e-6
    ), f"Expected {expected}, got {result}"


@pytest.mark.consistency
def test_compute_class_imbalance():
    """
    Test compute_class_imbalance with a simple data_dict containing
    one association matrix with known counts of '0's and '1's.
    """
    # Arrange: Create a data_dict with one entry
    data_dict = {
        0: (
            2,
            2,  # nb_ref, nb_pred
            np.array([[0, 1], [1, 0]]),  # dist_mat
            np.array([[0, 1], [1, 0]]),  # da_mat
            np.array([[0, 0, 0], [0, 0, 0]]),  # ref_cart
            np.array([[0, 0, 0], [0, 0, 0]]),  # pred_cart
        )
    }

    # Expected class counts: two '0's and two '1's
    expected = {0: 2, 1: 2}

    # Act: Compute class imbalance
    result = compute_class_imbalance(data_dict)

    # Assert: Check if the result matches the expected counts
    assert result == expected, f"Expected {expected}, got {result}"
