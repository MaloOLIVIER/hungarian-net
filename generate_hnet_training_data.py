# generate_hnet_training_data.py
"""
@inproceedings{Adavanne_2021,
title={Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers},
url={http://dx.doi.org/10.1109/WASPAA52581.2021.9632773},
DOI={10.1109/waspaa52581.2021.9632773},
booktitle={2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
publisher={IEEE},
author={Adavanne, Sharath and Politis, Archontis and Virtanen, Tuomas},
year={2021},
month=oct, pages={211-215} }
"""
import datetime
import os
import pickle
import random
import time
import typing as tp

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

DEFAULT_SAMPLE_RANGE = np.array([3000, 5000, 15000])


def sph2cart(azimuth: float, elevation: float, r: float) -> np.ndarray[float]:
    """
    Converts spherical coordinates to Cartesian coordinates.

    Args:
        azimuth (float): Azimuth angle in degrees.
        elevation (float): Elevation angle in degrees.
        r (float): Radius.

    Returns:
        np.ndarray[float]: Cartesian coordinates as an array [x, y, z].
    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])


def save_obj(obj: tp.Any, name: str) -> None:
    """
    Saves a Python object to a pickle file.

    Args:
        obj (Any): The Python object to save.
        name (str): The base name of the file (without extension).

    Returns:
        None
    """
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name: str) -> tp.Any:
    """
    Loads a Python object from a pickle file.

    Args:
        name (str): The base name of the file (without extension).

    Returns:
        Any: The loaded Python object.
    """
    with open(name, "rb") as f:
        return pickle.load(f)


def compute_class_imbalance(
    data_dict: dict[
        str,
        list[
            tp.Union[
                int,
                np.ndarray[float],
                np.ndarray[float],
                np.ndarray[float],
                np.ndarray[float],
                np.ndarray[float],
            ]
        ],
    ],
) -> dict[float, int]:
    """
    Computes the number of classes '0' and '1' in the association matrices of the training data.

    Args:
        data_dict (dict[str, list[tp.Union[int, np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]]]): Dictionary containing the Association matrix.

    Returns:
        dict[float, int]: A dictionary containing the number of classes '0' and '1' in the association matrices.
    """
    class_counts = {}
    for key, value in data_dict.items():
        nb_ref, nb_pred, dist_mat, da_mat, ref_cart, pred_cart = value
        for row in da_mat:
            for elem in row:
                if elem not in class_counts:
                    class_counts[elem] = 0
                class_counts[elem] += 1
    return class_counts


def generate_data(
    max_doas, sample_range, data_type="train", resolution_range="standard_resolution"
) -> dict[
    str,
    list[
        tp.Union[
            int,
            np.ndarray[float],
            np.ndarray[float],
            np.ndarray[float],
            np.ndarray[float],
            np.ndarray[float],
        ]
    ],
]:
    """
    Generates training or testing data based on the specified parameters.

    Args:
        max_doas (int): Maximum number of Directions of Arrival (DOAs).
        sample_range (np.ndarray): Array specifying the number of samples for each DOA combination.
                                   Should correspond to the minimum of `nb_ref` and `nb_pred`.
        data_type (str): Type of data to generate ('train' or 'test').
        resolution_range (str): Range of angular resolutions to consider:
                                 'standard_resolution', 'fine_resolution', or 'coarse_resolution'.

    Returns:
        dict[str, list[tp.Union[int, np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]]]: Generated data dictionary containing association matrices and related information.

    Raises:
        ValueError: If `resolution_range` is invalid.
    """
    data_dict = {}
    cnt = 0
    start_time = time.time()
    scale_factor = (
        1 if data_type == "train" else 0.1
    )  # Trainings get full samples, testing gets 10%

    # Metadata Container
    combination_counts = {}

    # Modify list_resolutions based on resolution_range and data_type
    if data_type == "train":
        if resolution_range == "standard_resolution":
            list_resolutions = [1, 2, 3, 4, 5, 10, 15, 20, 30]
        elif resolution_range == "fine_resolution":
            list_resolutions = [1] * 9  # [1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif resolution_range == "coarse_resolution":
            list_resolutions = [30] * 9  # [30, 30, 30, 30, 30, 30, 30, 30, 30]
        else:
            raise ValueError(
                "Invalid resolution_range: choose 'standard_resolution', 'fine_resolution', or 'coarse_resolution'."
            )
    else:
        # For test data, always use standard resolutions
        list_resolutions = [1, 2, 3, 4, 5, 10, 15, 20, 30]

    print(f"Resolution Range: {resolution_range}, List Resolutions: {list_resolutions}")

    # For each combination of associations ('nb_ref', 'nb_pred') = [(0, 0), (0, 1), (1, 0) ... (max_doas, max_doas)]
    # Generate random reference ('ref_ang') and prediction ('pred_ang') DOAs at different 'resolution'.
    for resolution in list_resolutions:  # Different angular resolution
        azi_range = range(-180, 180, resolution)
        ele_range = range(-90, 91, resolution)

        for nb_ref in range(max_doas + 1):  # Reference number of DOAs
            for nb_pred in range(max_doas + 1):  # Predicted number of DOAs

                # Update combination count with total_samples
                combination_key = (nb_ref, nb_pred)
                total_samples = int(scale_factor * sample_range[min(nb_ref, nb_pred)])
                combination_counts[combination_key] = (
                    combination_counts.get(combination_key, 0) + total_samples
                )

                # How many examples to generate for the given combination of ('nb_ref', 'nb_pred'), such that the overall dataset is not skewed.
                total_samples = int(scale_factor * sample_range[min(nb_ref, nb_pred)])
                for _ in range(total_samples):
                    if cnt % 100000 == 0:
                        elapsed_time = time.time() - start_time
                        print(
                            f"Number of examples generated : {cnt}, Time elapsed : {elapsed_time:.2f} seconds"
                        )
                    try:
                        # Generate random azimuth and elevation angles
                        ref_ang = np.array(
                            (
                                random.sample(azi_range, nb_ref),
                                random.sample(ele_range, nb_ref),
                            )
                        ).T
                        pred_ang = np.array(
                            (
                                random.sample(azi_range, nb_pred),
                                random.sample(ele_range, nb_pred),
                            )
                        ).T
                    except ValueError as e:
                        # print(f"Error with nb_ref={nb_ref}, nb_pred={nb_pred}: {e}")
                        continue

                    # Initialize both reference and predicted DOAs to a fixed dimension: (max_doas, 3)
                    # We initialize half of the samples to a fixed value of 10, the remaining half are randomly
                    # initialized in the range -100 to 100, this helps when Hnet is used as an alternative to
                    # permutation invariant training (PIT) of an associated model. During the initial phase of training
                    # the associated model with Hnet, the associated model can generate random number outputs,
                    # which when fed to Hnet that is only trained on constant valued inputs might not help converge.
                    # This small hack overcomes the problem, making the Hnet robust to varying inputs.
                    if random.random() > 0.5:
                        ref_cart, pred_cart = np.random.uniform(
                            low=-100, high=100, size=(max_doas, 3)
                        ), np.random.uniform(low=-100, high=100, size=(max_doas, 3))
                        # we make sure that the above random number is not in the good range of distances: [-1 to 1]
                        # Sets these values to 10, effectively removing them from the close proximity range.
                        # It ensures that no DOAs are too close to each other.
                        (
                            ref_cart[(ref_cart <= 1) & (ref_cart >= -1)],
                            pred_cart[(pred_cart <= 1) & (pred_cart >= -1)],
                        ) = (10, 10)
                    else:
                        ref_cart, pred_cart = 10 * np.ones((max_doas, 3)), 10 * np.ones(
                            (max_doas, 3)
                        )

                    # Convert Polar to Cartesian coordinates
                    ref_ang_rad, pred_ang_rad = (
                        ref_ang * np.pi / 180.0,
                        pred_ang * np.pi / 180.0,
                    )
                    ref_cart[:nb_ref, :] = sph2cart(
                        ref_ang_rad[:, 0], ref_ang_rad[:, 1], np.ones(nb_ref)
                    ).T
                    pred_cart[:nb_pred, :] = sph2cart(
                        pred_ang_rad[:, 0], pred_ang_rad[:, 1], np.ones(nb_pred)
                    ).T

                    # Compute distance matrix between predicted and cartesian
                    dist_mat = distance.cdist(ref_cart, pred_cart, "minkowski", p=2.0)

                    # Compute labels a.k.a. the association matrix
                    act_dist_mat = dist_mat[:nb_ref, :nb_pred]
                    row_ind, col_ind = linear_sum_assignment(act_dist_mat)
                    da_mat = np.zeros((max_doas, max_doas))
                    da_mat[row_ind, col_ind] = 1

                    # randomly shuffle dist and da matrices
                    rand_ind = random.sample(range(max_doas), max_doas)
                    if random.random() > 0.5:
                        dist_mat = dist_mat[rand_ind, :]
                        da_mat = da_mat[rand_ind, :]
                    else:
                        dist_mat = dist_mat[:, rand_ind]
                        da_mat = da_mat[:, rand_ind]

                    # Store the generated data
                    data_dict[cnt] = [
                        nb_ref,
                        nb_pred,
                        dist_mat,
                        da_mat,
                        ref_cart,
                        pred_cart,
                    ]
                    cnt += 1

    # Get current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    os.makedirs(f"data/{current_date}/{data_type}", exist_ok=True)

    # Human-readable filename
    out_filename = f"data/{current_date}/{data_type}/{resolution_range}_{data_type}_DOA{max_doas}_{'-'.join(map(str, sample_range))}.pkl"

    print(f"Saving data in: {out_filename}, #examples: {len(data_dict)}")
    save_obj(data_dict, out_filename)

    # Compute class imbalance
    class_imbalance = compute_class_imbalance(data_dict)
    print(f"{data_type.capitalize()} data class imbalance:", class_imbalance)

    # Print Additional Metadata
    print(f"\n{data_type.capitalize()} Data Metadata:")
    print("-" * 40)
    print(f"Total Samples: {len(data_dict)}\n")

    # Print Distribution of (nb_ref, nb_pred)
    print("Distribution of (nb_ref, nb_pred) Combinations:")
    for combo, count in sorted(combination_counts.items()):
        print(f"  {combo}: {count} samples")
    print("-" * 40)

    return data_dict


def main(
    sample_range=DEFAULT_SAMPLE_RANGE,
    max_doas=2,
    resolution_range="standard_resolution",
    testing="default",
) -> None:
    """
    Generates and saves training and testing datasets for the Hungarian Network (HNet) model.

    This function orchestrates the creation of training and testing data by invoking the `generate_data`
    function with specified parameters. It ensures that the datasets are balanced and adhere to the
    defined sample ranges for different Directions of Arrival (DOAs).

    Args:
        sample_range (np.ndarray[int], optional): Array specifying the number of samples for each DOA combination.
                                           Should correspond to the minimum of `nb_ref` and `nb_pred`.
                                           Defaults to `DEFAULT_SAMPLE_RANGE`.
        max_doas (int, optional): Maximum number of Directions of Arrival (DOAs) to consider. Defaults to 2.
        resolution_range (str, optional): Range of angular resolutions to consider:
                                         'standard_resolution', 'fine_resolution', or 'coarse_resolution'.
                                         Defaults to "standard_resolution".
        testing (str, optional): Determines the sample range for testing data.
                                 If not "default", uses the provided `sample_range`; otherwise, uses `DEFAULT_SAMPLE_RANGE`.
                                 Defaults to "default".

    Returns:
        None

    Raises:
        ValueError: If `sample_range` does not have the appropriate length corresponding to `max_doas`.

    Example:
        >>> main()
        Generating Training Data...
        Saving data in: data/20231010/train_resolution_train_DOA2_3000-5000-15000, #examples: 405000
        ...
        === Summary of Generated Datasets ===
        Training Data Samples: 405000
        Testing Data Samples: 40500
    """
    set_seed()

    print("\n=== Generating Hungarian Network Training Data ===")

    print("\nChecking Sample Range...")
    print(f"Sample Range: {sample_range}")

    print("\nGenerating Training Data...")
    # Generate training data
    train_data_dict = generate_data(
        max_doas, sample_range, data_type="train", resolution_range=resolution_range
    )

    sample_range_to_test = (
        sample_range
        if testing != "default"
        else DEFAULT_SAMPLE_RANGE  # Default sample range for testing
    )

    print("\nGenerating Testing Data...")
    # Generate testing data, same procedure as above
    test_data_dict = generate_data(
        max_doas,
        sample_range_to_test,
        data_type="test",
        resolution_range=resolution_range,
    )

    print("\n=== Summary of Generated Datasets ===")
    print(f"Training Data Samples: {len(train_data_dict)}")
    print(f"Testing Data Samples: {len(test_data_dict)}\n")


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int, optional): The seed value to set for all random number generators. Defaults to 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main(
        sample_range=np.array([30000, 50000, 150000]),
        resolution_range="fine_resolution",
    )
