# tune.py
import os
import random
import warnings
from functools import partial
from typing import Any, List

import hydra
import lightning as L
import numpy as np
import ray
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torchmetrics import MetricCollection


@hydra.main(
    config_path="configs",
    config_name="run.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """
    Instantiate all necessary modules, train, and test the Hungarian Network model using Ray Tune for hyperparameter optimization.

    This function sets up the data module, model, metrics, logger, trainer, and Ray Tune's hyperparameter search configurations
    based on the provided Hydra configuration. It then initiates the Ray Tune hyperparameter tuning process to optimize
    the model's performance.

    Args:
        cfg (DictConfig): Hydra configuration object, passed in by the @hydra.main decorator.
                          Contains all configuration parameters for data loading, model initialization,
                          hyperparameter search, logging, and more.

    Workflow:
        1. Initialize Ray for distributed hyperparameter tuning.
        2. Define the hyperparameter search space.
        3. Set up the scheduler (ASHAScheduler) and CLI reporter for Ray Tune.
        4. Instantiate the LightningDataModule using the configuration and sampled batch size.
        5. Instantiate the MetricCollection for evaluation metrics.
        6. Instantiate the LightningModule (HNetGRULightning) with the sampled learning rate.
        7. Instantiate callbacks, including Ray Tune's TuneReportCallback.
        8. Instantiate the logger based on the configuration.
        9. Instantiate the Trainer with callbacks and logger.
        10. Define the training function to be executed by Ray Tune.
        11. Run hyperparameter tuning using Ray Tune.
        12. Retrieve and print the best trial's configuration and validation loss.
        13. Shutdown Ray after tuning is complete.
    """

    # Initialize Ray
    ray.init()

    # Define the hyperparameter search space
    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([64, 128, 256]),
    }

    # Set up the scheduler and reporter for Ray Tune
    scheduler = ASHAScheduler(
        metric="validation_loss",
        mode="min",
        max_t=cfg.nb_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["validation_loss", "training_iteration"])

    # Instantiate LightningDataModule
    lightning_datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.lightning_datamodule,
        batch_size=config["batch_size"].sample(),
    )

    # Instantiate LightningModule
    metrics: MetricCollection = MetricCollection(
        dict(hydra.utils.instantiate(cfg.metrics))
    )
    lightning_module: L.LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        metrics=metrics,
        optimizer=partial(torch.optim.Adam, lr=config["learning_rate"].sample()),
    )

    # Instantiate Trainer with Ray Tune callback
    tune_callback = TuneReportCallback(
        {"validation_loss": "validation_loss"}, on="validation_end"
    )
    callbacks: List[L.Callback] = list(
        hydra.utils.instantiate(cfg.callbacks).values()
    ) + [tune_callback]
    logger: Logger = hydra.utils.instantiate(cfg.logging.logger)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Define the training function for Ray Tune
    def train_tune(config: Any) -> None:
        """
        Training function to be used by Ray Tune for hyperparameter optimization.

        Args:
            config (Dict): Dictionary containing hyperparameters to tune.

        Returns:
            None
        """
        trainer.fit(lightning_module, datamodule=lightning_datamodule)
        trainer.test(ckpt_path="best", datamodule=lightning_datamodule)

    # Run hyperparameter tuning
    result = tune.run(
        train_tune,
        resources_per_trial={
            "cpu": cfg.num_workers,
            "gpu": 1 if torch.cuda.is_available() else 0,
        },
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_hnet_training",
    )

    best_trial = result.get_best_trial("validation_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(
        f"Best trial final validation loss: {best_trial.last_result['validation_loss']}"
    )

    # Shutdown Ray
    ray.shutdown()


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
    Sets up the environment for training, including seeding and device configuration.

    This function performs the following tasks:
        1. Sets the random seed for reproducibility.
        2. Determines whether to use CPU or GPU for training.
        3. Configures environment variables and PyTorch backend settings to optimize performance.

    Returns:
        torch.device: The device (CPU or CUDA) that will be used for training.

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
