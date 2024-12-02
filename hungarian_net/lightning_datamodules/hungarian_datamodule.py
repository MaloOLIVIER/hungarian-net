import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from generate_hnet_training_data import load_obj


class HungarianDataset(Dataset):
    """
    Initializes the HungarianDataset.

    Args:
        train (bool, optional): If True, loads training data; otherwise, loads testing data. Defaults to True.
        max_len (int, optional): Maximum number of Directions of Arrival (DOAs). Defaults to 2.
        data_dir (str, optional): Directory where data files are stored. Defaults to "data".
    """

    def __init__(self, train=True, max_len=2, filename=None):
        if train:
            self.data_dict = load_obj(filename)
        else:
            self.data_dict = load_obj(filename)
        self.max_len = max_len

        self.pos_wts = np.ones(self.max_len**2)
        self.f_scr_wts = np.ones(self.max_len**2)
        if train:
            loc_wts = np.zeros(self.max_len**2)
            for i in range(len(self.data_dict)):
                label = self.data_dict[i][3]
                loc_wts += label.reshape(-1)
            self.f_scr_wts = loc_wts / len(self.data_dict)
            self.pos_wts = (len(self.data_dict) - loc_wts) / loc_wts

    def __len__(self):
        return len(self.data_dict)

    def get_pos_wts(self):
        return self.pos_wts

    def get_f_wts(self):
        return self.f_scr_wts

    def compute_class_imbalance(self):
        """
        Computes the class imbalance in the dataset.

        This method iterates over the data dictionary and counts the occurrences
        of each class in the distance assignment matrix (da_mat).

        Returns:
            dict: A dictionary where keys are class labels and values are the
                  counts of occurrences of each class.
        """
        class_counts = {}
        for key, value in self.data_dict.items():
            nb_ref, nb_pred, dist_mat, da_mat, ref_cart, pred_cart = value
            for row in da_mat:
                for elem in row:
                    if elem not in class_counts:
                        class_counts[elem] = 0
                    class_counts[elem] += 1
        return class_counts

    def __getitem__(self, idx):
        feat = self.data_dict[idx][2]
        label = self.data_dict[idx][3]

        label = [label.reshape(-1), label.sum(-1), label.sum(-2)]
        return feat, label

    def compute_weighted_accuracy(self, n1star, n0star):
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


# TODO: factorize HungarianDataset in HungarianDataModule


class HungarianDataModule(LightningDataModule):
    """
    LightningDataModule for HungarianDataset.

    Args:
        train_filename (str): Filename for training data.
        test_filename (str): Filename for testing data.
        max_len (int, optional): Maximum number of Directions of Arrival (DOAs). Defaults to 2.
        batch_size (int, optional): Batch size for data loaders. Defaults to 256.
        num_workers (int, optional): Number of workers for data loaders. Defaults to 4.
    """

    def __init__(
        self, train_filename, test_filename, max_len=2, batch_size=256, num_workers=4
    ):
        super().__init__()
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = HungarianDataset(
                train=True, max_len=self.max_len, filename=self.train_filename
            )
            self.val_dataset = HungarianDataset(
                train=False, max_len=self.max_len, filename=self.test_filename
            )
        if stage == "test" or stage is None:
            self.test_dataset = HungarianDataset(
                train=False, max_len=self.max_len, filename=self.test_filename
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
