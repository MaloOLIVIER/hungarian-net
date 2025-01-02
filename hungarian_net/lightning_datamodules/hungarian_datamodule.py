# hungarian_net/lightning_datamodules/hungarian_datamodule.py
import typing as tp

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from generate_hnet_training_data import load_obj


class HungarianDataset(Dataset):
    """
    A custom PyTorch dataset for loading and managing Hungarian Network training and testing data.

    This dataset handles the loading of pre-generated samples that include reference and predicted
    Directions of Arrival (DOAs), as well as their associated distance and association matrices.
    It also computes class imbalance weights to aid in the training process.

    Args:
        train (bool, optional): Flag indicating whether to load training data (`True`) or
                                 testing data (`False`). Defaults to `True`.
        max_doas (int, optional): Maximum number of DOAs. Determines the size of the
                                 distance and association matrices. Defaults to `None`.
        filename (str, optional): Path to the Pickle file containing the serialized data.
                                  Defaults to `None`.

    Attributes:
        data_dict (dict): Dictionary containing the loaded data samples.
        max_doas (int): Maximum number of DOAs.
        pos_wts (np.ndarray): Weights for position calculations based on class imbalance.
        f_scr_wts (np.ndarray): Weights for feature scores based on class imbalance.
    """

    def __init__(self, train=True, max_doas=None, filename=None) -> None:
        """
        Initializes the HungarianDataset.

        Loads the data from the specified Pickle file and computes class imbalance weights
        if the dataset is for training.

        Args:
            train (bool, optional): If `True`, loads training data; otherwise, loads testing data.
                                     Defaults to `True`.
            max_doas (int, optional): Maximum number of DOAs. Defaults to `None`.
            filename (str, optional): Path to the Pickle file containing the data. Defaults to `None`.

        Raises:
            ValueError: If the specified data file cannot be loaded correctly.
        """
        if train:
            self.data_dict = load_obj(filename)
        else:
            self.data_dict = load_obj(filename)
        self.max_doas = max_doas

        self.pos_wts = np.ones(self.max_doas**2)
        self.f_scr_wts = np.ones(self.max_doas**2)
        if train:
            loc_wts = np.zeros(self.max_doas**2)
            for i in range(len(self.data_dict)):
                label = self.data_dict[i][3]
                loc_wts += label.reshape(-1)
            self.f_scr_wts = loc_wts / len(self.data_dict)
            self.pos_wts = (len(self.data_dict) - loc_wts) / loc_wts

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of data samples.
        """
        return len(self.data_dict)

    def get_pos_wts(self) -> np.ndarray:
        """
        Retrieves the position weights based on class imbalance.

        Returns:
            np.ndarray: Array of position weights.
        """
        return self.pos_wts

    def get_f_wts(self) -> np.ndarray:
        """
        Retrieves the feature score weights based on class imbalance.

        Returns:
            np.ndarray: Array of feature score weights.
        """
        return self.f_scr_wts

    def compute_class_imbalance(self) -> dict[int, int]:
        """
        Computes the class imbalance in the dataset.

        This method iterates over the data dictionary and counts the occurrences
        of each class in the distance assignment matrix (da_mat).

        Returns:
            dict[int, int]: A dictionary where keys are class labels and values are the
                            counts of occurrences of each class.
        """
        class_counts = {}
        for key, value in self.data_dict.items():
            nb_ref, nb_pred, dist_mat, da_mat, ref_cart, pred_cart = value
            for row in da_mat:
                for elem in row:
                    if elem not in class_counts:
                        class_counts[int(elem)] = 0
                    class_counts[int(elem)] += 1
        return class_counts

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Retrieves the features and labels for a given index.

        Args:
            idx (int): Index of the desired data sample.

        Returns:
            tuple: A tuple containing:
                   - feat (numpy.ndarray): Feature matrix for the sample.
                   - label (list): List containing reshaped label arrays.
        """
        feat = self.data_dict[idx][2]
        label = self.data_dict[idx][3]

        label = [label.reshape(-1), label.sum(-1), label.sum(-2)]
        return feat, label

    def compute_weighted_accuracy(self, n1star: int, n0star: int) -> float:
        """
        Compute the weighted accuracy of the model.
        The weighted accuracy is calculated based on the class imbalance in the dataset.
        The weights for each class are determined by the proportion of the opposite class.
        Parameters:
            n1star (int): The number of true positives.
            n0star (int): The number of false positives.
        Returns:
            WA (float): The weighted accuracy of the model.

        References:
        Title: How To Train Your Deep Multi-Object Tracker
        Authors: Yihong Xu, Aljosa Osep, Yutong Ban, Radu Horaud, Laura Leal-Taixe, Xavier Alameda-Pineda
        Year: 2020

        URL: https://arxiv.org/abs/1906.06618
        """
        WA = 0

        class_counts = self.compute_class_imbalance()

        n0 = class_counts.get(0, 0)  # number of 0s
        n1 = class_counts.get(1, 0)  # number of 1s

        w0 = n1 / (n0 + n1)  # weight for class 0
        w1 = 1 - w0  # weight for class 1

        WA = (w1 * n1star + w0 * n0star) / (w1 * n1 + w0 * n0)

        return WA

class HungarianDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing data loading for the Hungarian Network.

    This DataModule encapsulates all data loading logic, including preparing datasets
    and configuring DataLoaders for training, validation, and testing.

    Args:
        train_filename (str, optional): Path to the training data Pickle file.
                                        Defaults to "data/reference/hung_data_train".
        test_filename (str, optional): Path to the testing data Pickle file.
                                       Defaults to "data/reference/hung_data_test".
        max_doas (int, optional): Maximum number of DOAs. Determines the size of the
                                   distance and association matrices. Defaults to `2`.
        batch_size (int, optional): Number of samples per batch. Defaults to `256`.
        num_workers (int, optional): Number of subprocesses to use for data loading.
                                     More workers can speed up data loading. Defaults to `4`.

    Attributes:
        train_filename (str): Filename for training data.
        test_filename (str): Filename for testing data.
        max_doas (int): Maximum number of DOAs.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
        train_dataset (HungarianDataset): Dataset for training.
        val_dataset (HungarianDataset): Dataset for validation.
        test_dataset (HungarianDataset): Dataset for testing.
    """

    def __init__(
        self,
        train_filename="data/reference/hung_data_train",
        test_filename="data/reference/hung_data_test",
        max_doas=2,
        batch_size=256,
        num_workers=4,
    ) -> None:
        super().__init__()
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.max_doas = max_doas
        self.batch_size = batch_size
        self.num_workers = num_workers

    # TODO: def transfer_batch_to_device(self, batch, device, dataloader_idx):

    def setup(self, stage: tp.Optional[str] = None) -> None:
        """
        Prepares the datasets for training, validation, and testing.

        This method is called by PyTorch Lightning to set up datasets. Depending on the
        stage (`fit`, `validate`, `test`, or `predict`), it initializes the appropriate
        datasets.

        Args:
            stage (str, optional): The stage for which to set up the data.
                                   Can be 'fit', 'validate', 'test', or 'predict'. Defaults to `None`.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = HungarianDataset(
                train=True, max_doas=self.max_doas, filename=self.train_filename
            )
            self.val_dataset = HungarianDataset(
                train=False, max_doas=self.max_doas, filename=self.test_filename
            )
        if stage == "test" or stage is None:
            self.test_dataset = HungarianDataset(
                train=False, max_doas=self.max_doas, filename=self.test_filename
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Configures the DataLoader with shuffling and dropping the last incomplete batch.

        Returns:
            DataLoader: Configured DataLoader for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Configures the DataLoader without shuffling and without dropping incomplete batches.

        Returns:
            DataLoader: Configured DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the testing dataset.

        Configures the DataLoader without shuffling and without dropping incomplete batches.

        Returns:
            DataLoader: Configured DataLoader for testing.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
