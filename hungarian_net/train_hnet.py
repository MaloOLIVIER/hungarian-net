import datetime
import os
import random
import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.callbacks import ModelCheckpoint
from lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hungarian_net.dataset import HungarianDataset
from hungarian_net.models import HNetGRU


class HNetGRULightning(L.LightningModule):
    def __init__(self, max_len, sample_range_used, class_imbalance):
        super().__init__()
        self.model = HNetGRU(max_len=max_len)
        self.criterion1 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion2 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion3 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion_wts = [1.0, 1.0, 1.0]
        self.sample_range_used = sample_range_used
        self.class_imbalance = class_imbalance

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output1, output2, output3 = self(data)
        l1 = self.criterion1(output1, target[0])
        l2 = self.criterion2(output2, target[1])
        l3 = self.criterion3(output3, target[2])
        loss = sum(w * l for w, l in zip(self.criterion_wts, [l1, l2, l3]))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output1, output2, output3 = self(data)
        l1 = self.criterion1(output1, target[0])
        l2 = self.criterion2(output2, target[1])
        l3 = self.criterion3(output3, target[2])
        loss = sum(w * l for w, l in zip(self.criterion_wts, [l1, l2, l3]))
        self.log("val_loss", loss)
        # Calculate F1 Score or other metrics here
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


# @hydra.main(
#     config_path="configs",
#     config_name="run.yaml",
#     version_base="1.3",
# )
def main(
    batch_size=256,
    nb_epochs=1000,
    max_len=2,
    sample_range_used=[3000, 5000, 15000],
    filename_train="data/reference/hung_data_train",
    filename_test="data/reference/hung_data_test",
):
    """
    Train the Hungarian Network (HNetGRU) model.

    This function orchestrates the training process of the HNetGRU model, including data loading,
    model initialization, training loop with validation, and saving the best-performing model.

    Args:
        batch_size (int, optional):
            Number of samples per training batch. Defaults to 256.
        nb_epochs (int, optional):
            Total number of training epochs. Defaults to 1000.
        max_len (int, optional):
            Maximum number of Directions of Arrival (DOAs) the model can handle. Defaults to 2.
        sample_range_used (List[int], optional):
            List specifying the range of samples used for training. Defaults to [3000, 5000, 15000].
        filename_train (str, optional):
            Path to the training data file. Defaults to "data/reference/hung_data_train".
        filename_test (str, optional):
            Path to the testing data file. Defaults to "data/reference/hung_data_test".

    Steps:
        1. **Set Random Seed**:
            - Ensures reproducibility by setting seeds for Python's `random`, NumPy, and PyTorch.

        2. **Device Configuration**:
            - Checks for GPU availability and sets the computation device accordingly (CUDA or CPU).

        3. **Data Loading**:
            - Loads the training dataset using `HungarianDataset` with specified parameters.
            - Initializes a `DataLoader` for batching and shuffling training data.
            - Calculates class imbalance to handle potential data skew.
            - Loads the validation dataset similarly.

        4. **Model and Optimizer Initialization**:
            - Instantiates the `HNetGRU` model and moves it to the configured device.
            - Sets up the optimizer (`Adam`) for training the model parameters.

        5. **Loss Function Definition**:
            - Defines three separate Binary Cross-Entropy with Logits Loss functions (`criterion1`, `criterion2`, `criterion3`).
            - Assigns equal weights to each loss component.

        6. **Training Loop**:
            - Iterates over the specified number of epochs.
            - **Training Phase**:
                a. Sets the model to training mode.
                b. Iterates over training batches:
                    - Performs forward pass.
                    - Computes individual losses.
                    - Aggregates losses with defined weights.
                    - Backpropagates and updates model weights.
                c. Accumulates and averages training losses.

            - **Validation Phase**:
                a. Sets the model to evaluation mode.
                b. Iterates over validation batches without gradient computation:
                    - Performs forward pass.
                    - Computes losses.
                    - Calculates F1 scores with weighted averaging.
                c. Accumulates and averages validation losses and F1 scores.

            - **Early Stopping**:
                - Monitors validation F1 score to identify and save the best-performing model.
                - Saves model weights with a timestamped filename for version tracking.

            - **Metrics Logging**:
                - Prints comprehensive metrics after each epoch, including losses, F1 scores, and accuracy.
                - Tracks unweighted F1 scores separately for detailed analysis.

        7. **Final Output**:
            - Prints the best epoch and corresponding F1 score.
            - Returns the best-performing model instance.

    Returns:
        HNetGRU:
            The trained HNetGRU model with the best validation F1 score.
    """

    # TODO: Réécriture/factorisation du code sur le modèle de VibraVox de Julien HAURET
    # TODO: leverager TensorBoard, Hydra, Pytorch Lightning, RayTune, Docker

    set_seed()

    # Check wether to run on cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # load training dataset
    train_dataset = HungarianDataset(
        train=True, max_len=max_len, filename=filename_train
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    f_score_weights = np.tile(train_dataset.get_f_wts(), batch_size)
    print(train_dataset.get_f_wts())

    # Compute class imbalance
    class_imbalance = train_dataset.compute_class_imbalance()
    print("Class imbalance in training labels:", class_imbalance)

    # load validation dataset
    test_loader = DataLoader(
        HungarianDataset(train=False, max_len=max_len, filename=filename_test),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = HNetGRULightning(
        max_len=max_len,
        sample_range_used=sample_range_used,
        class_imbalance=class_imbalance,
    )

    logger = TensorBoardLogger("tb_logs", name="hnet_model")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    trainer = L.Trainer(
        max_epochs=nb_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        # gpus=1 if use_cuda else 0
    )

    trainer.fit(model, train_loader, test_loader)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
