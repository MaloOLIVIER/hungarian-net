# tests/consistency_tests/generate_hnet_training_data/test_consistency_generate_hnet_training_data.py

import numpy as np
import pytest

from generate_hnet_training_data import compute_class_imbalance, sph2cart

@pytest.mark.consistency
def test_sph2cart():
    """
    Test the `sph2cart` function with a simple case where azimuth and elevation are both zero.
    
    This test verifies that the spherical to Cartesian conversion is correct for 
    azimuth=0 degrees, elevation=0 degrees, and radius=1, which should map to
    the point (1, 0, 0) in Cartesian coordinates.
    
    Args:
        None
    
    Returns:
        None
    """
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
def test_sph2cart_multiple_cases(azimuth: float, elevation: float, r: float, expected: np.ndarray):
    """
    Test the `sph2cart` function with multiple combinations of azimuth, elevation, and radius values.
    
    This parameterized test ensures that the `sph2cart` function accurately converts 
    spherical coordinates to Cartesian coordinates across various scenarios.
    
    Args:
        azimuth (float): Azimuth angle in radians.
        elevation (float): Elevation angle in radians.
        r (float): Radius.
        expected (np.ndarray): Expected Cartesian coordinates.
    
    Returns:
        None
    """
    result = sph2cart(azimuth, elevation, r)
    assert np.allclose(
        result, expected, atol=1e-6
    ), f"Expected {expected}, got {result}"


@pytest.mark.consistency
def test_compute_class_imbalance():
    """
    Test the `compute_class_imbalance` function with a predefined data dictionary.
    
    This test verifies that the `compute_class_imbalance` function correctly counts the 
    number of '0's and '1's in the association matrices within the provided data dictionary.
    It uses a simple `data_dict` containing one association matrix with known counts for validation.
    
    Args:
        None
    
    Returns:
        None
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
