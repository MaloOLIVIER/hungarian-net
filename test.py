import os
import random
import warnings
from pathlib import Path
from typing import List

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torchmetrics import MetricCollection


@hydra.main(
    config_path="configs",
    config_name="test.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """
    Instantiate all necessary modules, train and test the model.

    Args:
        cfg (DictConfig): Hydra configuration object, passed in by the @hydra.main decorator
    """

    # Instantiate LightningDataModule
    lightning_datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.lightning_datamodule
    )

    # Instantiate LightningModule
    metrics: MetricCollection = MetricCollection(
        dict(hydra.utils.instantiate(cfg.metrics))
    )
    lightning_module: L.LightningModule = hydra.utils.instantiate(
        cfg.lightning_module, metrics=metrics
    )

    # Instantiate Trainer
    callbacks: List[L.Callback] = list(hydra.utils.instantiate(cfg.callbacks).values())
    logger: Logger = hydra.utils.instantiate(cfg.logging.logger)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.test(
        lightning_module, ckpt_path=cfg.checkpoint_path, datamodule=lightning_datamodule
    )


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
    setup_environment()
    main()
