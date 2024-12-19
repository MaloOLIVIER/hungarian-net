# test.py
import os
import random
import warnings
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
    Instantiate all necessary modules and test the Hungarian Network model.

    This function sets up the data module, model, metrics, logger, and trainer based on the
    provided Hydra configuration. It then proceeds to evaluate the trained model on the
    test dataset using the specified checkpoint.

    Args:
        cfg (DictConfig): Hydra configuration object, passed in by the @hydra.main decorator.
                          Contains all configuration parameters for data loading, model initialization,
                          testing, logging, and more.

    Workflow:
        1. Instantiate the LightningDataModule using the configuration.
        2. Instantiate the MetricCollection for evaluation metrics.
        3. Instantiate the LightningModule (HNetGRULightning) with the metrics.
        4. Instantiate callbacks and logger based on the configuration.
        5. Instantiate the Trainer with callbacks and logger.
        6. Start the testing process using the specified checkpoint.
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
    logger.log_hyperparams(cfg)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.test(
        lightning_module, ckpt_path=cfg.checkpoint_path, datamodule=lightning_datamodule
    )


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across various libraries.

    This function ensures that the results are reproducible by setting the seed for Python's
    `random` module, NumPy, and PyTorch (both CPU and CUDA). It also configures PyTorch's
    backend for deterministic behavior.

    Args:
        seed (int, optional): The seed value to set for all random number generators.
                              Defaults to 42.

    Returns:
        None
    """
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment() -> torch.device:
    """
    Sets up the environment for testing, including seeding and device configuration.

    This function performs the following tasks:
        1. Sets the random seed for reproducibility.
        2. Determines whether to use CPU or GPU for testing.
        3. Configures environment variables and PyTorch backend settings to optimize performance.

    Returns:
        torch.device: The device (CPU or CUDA) that will be used for testing.

    Raises:
        None
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
