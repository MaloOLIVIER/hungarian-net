# generate_hnet_training_data.py

import datetime
import random
import time
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def sph2cart(azimuth, elevation, r):
    """
    Convert spherical to cartesian coordinates
    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def compute_class_imbalance(data_dict):
    """Computes the number of classes '0' and '1' in the association matrices of the training data.

    Args:
        data_dict (dict): Dictionary containing the Association matrix.

    Returns:
        dict: A dictionary containing the number of classes '0' and '1' in the association matrices.
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


def generate_data(pickle_filename, max_doas, sample_range, data_type="train"):
    """
    Generates training or testing data based on the specified parameters.

    Args:
        pickle_filename (str): Base name for the output pickle file.
        max_doas (int): Maximum number of Directions of Arrival (DOAs).
        sample_range (np.array): Array specifying the number of samples for each DOA combination.
        data_type (str): Type of data to generate ('train' or 'test').

    Returns:
        dict: Generated data dictionary containing association matrices and related information.
    """
    data_dict = {}
    cnt = 0
    start_time = time.time()
    scale_factor = (
        1 if data_type == "train" else 0.1
    )  # Trainings get full samples, testing gets 10%

    # Metadata Container
    combination_counts = {}

    # For each combination of associations ('nb_ref', 'nb_pred') = [(0, 0), (0, 1), (1, 0) ... (max_doas, max_doas)]
    # Generate random reference ('ref_ang') and prediction ('pred_ang') DOAs at different 'resolution'.
    for resolution in [1, 2, 3, 4, 5, 10, 15, 20, 30]:  # Different angular resolution
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
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Human-readable filename
    out_filename = f"data/{current_date}_{pickle_filename}_{data_type}_DOA{max_doas}_{abs(hash(tuple(sample_range)))}"

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
    pickle_filename="hung_data", sample_range=np.array([3000, 5000, 15000]), max_doas=2
):
    """
    Generates and saves training and testing datasets for the Hungarian Network (HNet) model.

    This function orchestrates the creation of training and testing data by invoking the `generate_data`
    function with specified parameters. It ensures that the datasets are balanced and adhere to the
    defined sample ranges for different Directions of Arrival (DOAs).

    Args:
        pickle_filename (str, optional): Base name for the output pickle files. Defaults to "hung_data".
        sample_range (np.array, optional): Array specifying the number of samples for each DOA combination.
                                           Should correspond to the minimum of `nb_ref` and `nb_pred`.
                                           Defaults to [3000, 5000, 15000].
        max_doas (int, optional): Maximum number of Directions of Arrival (DOAs) to consider. Defaults to 2.

    Returns:
        None

    Raises:
        ValueError: If `sample_range` does not have the appropriate length corresponding to `max_doas`.

    Example:
        >>> main()
        Generating Training Data...
        Saving data in: data/hung_data_train, #examples: 405000
        ...
        === Summary of Generated Datasets ===
        Training Data Samples: 405000
        Testing Data Samples: 40500
    """

    print("\n=== Generating Hungarian Network Training Data ===")

    print("\nChecking Sample Range...")
    print(f"Sample Range: {sample_range}")

    print("\nGenerating Training Data...")
    # Generate training data
    train_data_dict = generate_data(
        pickle_filename, max_doas, sample_range, data_type="train"
    )

    print("\nGenerating Testing Data...")
    # Generate testing data, same procedure as above
    test_data_dict = generate_data(
        pickle_filename, max_doas, sample_range, data_type="test"
    )

    print("\n=== Summary of Generated Datasets ===")
    print(f"Training Data Samples: {len(train_data_dict)}")
    print(f"Testing Data Samples: {len(test_data_dict)}\n")


if __name__ == "__main__":
    main()
