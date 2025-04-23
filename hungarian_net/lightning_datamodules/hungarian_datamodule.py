# hungarian_net/lightning_datamodules/hungarian_datamodule.py
import threading
import typing as tp

import numpy as np
from lightning import LightningDataModule
from functools import lru_cache
import os
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
        if filename is None or not isinstance(filename, str):
            raise ValueError("Filename must be a valid string path")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
            
        if max_doas is None or max_doas <= 0:
            raise ValueError("max_doas must be a positive integer")
        
        # All initialization happens in try block
        try:
            # Attempt to load data with error handling
            self.data_dict = self._load_data_safely(filename)
            
            # Validate data structure
            if not self.data_dict:
                raise ValueError("Loaded data dictionary is empty")
                
            # Check at least one sample for expected structure
            if len(self.data_dict) > 0:
                sample = self.data_dict[0]
                if not isinstance(sample, tuple) or len(sample) < 4:
                    raise ValueError("Data samples do not have the expected structure")
            
            self.max_doas = max_doas
            self._lock = threading.RLock()  # Lock for thread safety
            
            # Initialize weights with proper dimensions
            self.pos_wts = np.ones(self.max_doas**2)
            self.f_scr_wts = np.ones(self.max_doas**2)
            
            # Compute weights if training dataset
            if train:
                self._compute_weights()
                
        except Exception as e:
            # If initialization fails, clean up and re-raise
            self.data_dict = None
            self.max_doas = None
            self.pos_wts = None
            self.f_scr_wts = None
            raise ValueError(f"Failed to initialize dataset: {str(e)}") from e

    def _load_data_safely(self, filename: str) -> dict:
        """
        Computes class imbalance weights for training.
    
        This thread-safe method calculates two sets of weights:
        1. f_scr_wts: Feature score weights calculated as the proportion of positive samples
        2. pos_wts: Position weights calculated as the ratio of negative to positive samples
        
        These weights are used to handle class imbalance in the binary classification task.
        The method updates the instance variables 'pos_wts' and 'f_scr_wts' in-place.
        
        Both calculations include safeguards against division by zero through the use of
        np.divide with appropriate 'out' and 'where' parameters.
        
        Returns:
            None: Updates self.pos_wts and self.f_scr_wts in-place.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = load_obj(filename)
                if data is not None:
                    return data
                raise ValueError("Data loaded as None")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to load data after {max_retries} attempts: {str(e)}")
 
    def _compute_weights(self) -> None:
        """
        Computes class imbalance weights for training.
    
        This thread-safe method calculates two sets of weights:
        1. f_scr_wts: Feature score weights calculated as the proportion of positive samples
        2. pos_wts: Position weights calculated as the ratio of negative to positive samples
        
        These weights are used to handle class imbalance in the binary classification task.
        The method updates the instance variables 'pos_wts' and 'f_scr_wts' in-place.
        
        Both calculations include safeguards against division by zero through the use of
        np.divide with appropriate 'out' and 'where' parameters.
        
        Returns:
            None: Updates self.pos_wts and self.f_scr_wts in-place.
        """
        with self._lock:  # Isolation: Thread-safe operation
            loc_wts = np.zeros(self.max_doas**2)
            data_length = len(self.data_dict)
            
            if data_length == 0:
                return  # Nothing to compute
                
            for i in range(data_length):
                try:
                    label = self.data_dict[i][3]
                    loc_wts += label.reshape(-1)
                except (IndexError, ValueError) as e:
                    # Skip problematic samples but continue processing
                    continue
                    
            # Avoid division by zero
            self.f_scr_wts = np.divide(
                loc_wts, 
                data_length, 
                out=np.ones_like(loc_wts), 
                where=data_length!=0
            )
            
            # Avoid division by zero in pos_wts calculation
            self.pos_wts = np.divide(
                data_length - loc_wts, 
                loc_wts, 
                out=np.ones_like(loc_wts), 
                where=loc_wts!=0
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of data samples.
        """
        with self._lock:
            return len(self.data_dict) if self.data_dict is not None else 0

    def get_pos_wts(self) -> np.ndarray:
        """
        Retrieves the position weights based on class imbalance.

        Returns:
            np.ndarray: Array of position weights.
        """
        with self._lock:
            return self.pos_wts.copy() if self.pos_wts is not None else np.array([])

    def get_f_wts(self) -> np.ndarray:
        """
        Retrieves the feature score weights based on class imbalance.

        Returns:
            np.ndarray: Array of feature score weights.
        """
        with self._lock:
            return self.f_scr_wts.copy() if self.f_scr_wts is not None else np.array([])

    @lru_cache(maxsize=1)
    def compute_class_imbalance(self) -> dict[int, int]:
        """
        Computes the class imbalance in the dataset.

        This method iterates over the data dictionary and counts the occurrences
        of each class in the distance assignment matrix (da_mat).

        Returns:
            dict[int, int]: A dictionary where keys are class labels and values are the
                            counts of occurrences of each class.
        """
        with self._lock:
            if self.data_dict is None:
                return {}
                
            class_counts = {}
            
            for idx in range(len(self.data_dict)):
                try:
                    value = self.data_dict[idx]
                    if len(value) < 4:
                        continue  # Skip invalid entries
                        
                    nb_ref, nb_pred, dist_mat, da_mat, *_ = value
                    for row in da_mat:
                        for elem in row:
                            elem_int = int(elem)
                            if elem_int not in class_counts:
                                class_counts[elem_int] = 0
                            class_counts[elem_int] += 1
                except (IndexError, ValueError, TypeError):
                    # Skip problematic entries
                    continue
                    
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
        with self._lock:
            # Consistency: Validate index
            if self.data_dict is None:
                raise RuntimeError("Dataset not properly initialized")
                
            if not 0 <= idx < len(self.data_dict):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self.data_dict)}")
            
            try:
                # Atomicity: Either get both feature and label or fail
                sample = self.data_dict[idx]
                if len(sample) < 4:
                    raise ValueError(f"Sample at index {idx} has invalid structure")
                    
                feat = sample[2]
                label = sample[3]
                
                # Consistency: Validate shapes
                if feat is None or label is None:
                    raise ValueError(f"Invalid feature or label at index {idx}")
                    
                # Return a copy to avoid accidental modifications
                label_reshaped = [
                    label.reshape(-1).copy(), 
                    label.sum(-1).copy(), 
                    label.sum(-2).copy()
                ]
                return feat.copy(), label_reshaped
                
            except Exception as e:
                # Durability: On failure, return a safe empty result
                raise RuntimeError(f"Failed to retrieve item at index {idx}: {str(e)}") from e

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
        if not isinstance(n1star, int) or not isinstance(n0star, int):
            raise ValueError("Inputs must be integers")
            
        if n1star < 0 or n0star < 0:
            raise ValueError("Counts must be non-negative")
        
        try:
            class_counts = self.compute_class_imbalance()
            
            n0 = class_counts.get(0, 0)  # number of 0s
            n1 = class_counts.get(1, 0)  # number of 1s
            
            # Consistency: Handle edge cases
            if n0 + n1 == 0:
                return 0.0
                
            w0 = n1 / (n0 + n1)  # weight for class 0
            w1 = 1 - w0  # weight for class 1
            
            # Consistency: Handle edge cases
            denominator = (w1 * n1 + w0 * n0)
            if denominator == 0:
                return 0.0
                
            WA = (w1 * n1star + w0 * n0star) / denominator
            return float(WA)  # Ensure return type consistency
            
        except Exception as e:
            # Log error and return default
            return 0.0

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

        if not isinstance(max_doas, int) or max_doas <= 0:
            raise ValueError("max_doas must be a positive integer")
            
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
            
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")

        self.train_filename = train_filename
        self.test_filename = test_filename
        self.max_doas = max_doas
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._setup_lock = threading.Lock()

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
        with self._setup_lock:
            try:
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
            except Exception as e:
                # Log the error and re-raise
                raise RuntimeError(f"Error in setup: {str(e)}") from e

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Configures the DataLoader with shuffling and dropping the last incomplete batch.

        Returns:
            DataLoader: Configured DataLoader for training.
        """
        if self.train_dataset is None:
            raise RuntimeError("Setup was not called or failed to initialize train_dataset")

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
        if self.val_dataset is None:
            raise RuntimeError("Setup was not called or failed to initialize val_dataset")
 
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
        if self.test_dataset is None:
            raise RuntimeError("Setup was not called or failed to initialize test_dataset")

        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
