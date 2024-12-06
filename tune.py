from functools import partial
import os
import random
import warnings
from typing import List

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
    Instantiate all necessary modules, train and test the model.

    Args:
        cfg (DictConfig): Hydra configuration object, passed in by the @hydra.main decorator
    """

    # TODO: leverager RayTune, Docker
    
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

    reporter = CLIReporter(
        metric_columns=["validation_loss", "training_iteration"]
    )

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
        cfg.lightning_module, metrics=metrics,
        optimizer=partial(torch.optim.Adam, lr=config["learning_rate"].sample()),
    )

    # Instantiate Trainer with Ray Tune callback
    tune_callback = TuneReportCallback({"validation_loss": "validation_loss"}, on="validation_end")
    callbacks: List[L.Callback] = list(hydra.utils.instantiate(cfg.callbacks).values()) + [tune_callback]
    logger: Logger = hydra.utils.instantiate(cfg.logging.logger)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial"
    )

    # Define the training function for Ray Tune
    def train_tune(config):
        trainer.fit(lightning_module, datamodule=lightning_datamodule)
        trainer.test(ckpt_path="best", datamodule=lightning_datamodule)

    # Run hyperparameter tuning
    result = tune.run(
        train_tune,
        resources_per_trial={"cpu": cfg.num_workers, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_hnet_training",
    )
    
    best_trial = result.get_best_trial("validation_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['validation_loss']}")

    # Shutdown Ray
    ray.shutdown()


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
