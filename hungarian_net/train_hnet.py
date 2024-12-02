import datetime
import os
import random
import sys
import time
import warnings
from typing import List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from hungarian_net.dataset import HungarianDataModule, HungarianDataset
from hungarian_net.models import HNetGRULightning


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

    # Temporarly mock the dataloader
    filename_train = "data/20241202/train/hung_data_train_DOA2_3000-5000-15000"
    filename_test = "data/20241202/test/hung_data_test_DOA2_3000-5000-15000"

    lightning_datamodule = HungarianDataModule(
        train_filename=filename_train,
        test_filename=filename_test,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=4,
    )

    use_cuda = torch.cuda.is_available()
    lightning_module = HNetGRULightning(
        device=torch.device("cuda" if use_cuda else "cpu"), max_len=max_len
    )

    # Get current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    os.makedirs(f"models/{current_date}", exist_ok=True)

    # Human-readable filename
    dirpath = f"models/{current_date}/"
    out_filename = f"hnet_model_DOA{max_len}_{'-'.join(map(str, sample_range_used))}"

    logger = TensorBoardLogger("tb_logs", name="hungarian_net")
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=out_filename,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=nb_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        # gpus=1 if use_cuda else 0
    )

    trainer.fit(lightning_module, datamodule=lightning_datamodule)


def set_seed(seed=42):
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment():
    """
    Setup environment for training.

    """
    # Set Random Seed
    set_seed()

    # Check wether to run on cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    warnings.filterwarnings("ignore")

    # Set environment variables for full trace of errors
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Enable CUDNN backend
    torch.backends.cudnn.enabled = True

    # Enable CUDNN benchmarking to choose the best algorithm for every new input size
    # e.g. for convolutional layers chose between Winograd, GEMM-based, or FFT algorithms
    torch.backends.cudnn.benchmark = True

    return device


if __name__ == "__main__":
    device = setup_environment()
    main()
