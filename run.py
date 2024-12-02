import datetime
import os
import random
import warnings

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from hungarian_net.lightning_datamodules.hungarian_datamodule import HungarianDataModule, HungarianDataset
from hungarian_net.lightning_modules.hnet_gru_lightning import HNetGRULightning


 @hydra.main(
     config_path="configs",
     config_name="run.yaml",
     version_base="1.3",
 )
def main(cfg: DictConfig):
    """ batch_size=256,
    nb_epochs=1000,
    max_len=2,
    sample_range_used=[3000, 5000, 15000],
    filename_train="data/reference/hung_data_train",
    filename_test="data/reference/hung_data_test", """

    

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

    # metrics: MetricCollection = MetricCollection(
    #    dict(hydra.utils.instantiate(cfg.metrics))
    # )

    use_cuda = torch.cuda.is_available()
    lightning_module = HNetGRULightning(
        metrics=None,
        device=torch.device("cuda" if use_cuda else "cpu"),
        max_len=max_len,
    )

    # Instantiate LightningModule
    lightning_module: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        metrics=metrics,
    )

    # Get current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    os.makedirs(f"checkpoints/{current_date}", exist_ok=True)

    # Human-readable filename
    dirpath = f"checkpoints/{current_date}/"
    out_filename = f"hnet_model_DOA{max_len}_{'-'.join(map(str, sample_range_used))}"

    """ checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=out_filename,
        monitor="validation_loss",
        save_top_k=1,
        mode="min",
    ) """

    # Instantiate Trainer
    callbacks: List[Callback] = list(hydra.utils.instantiate(cfg.callbacks).values())
    logger: Logger = hydra.utils.instantiate(cfg.logging.logger)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.fit(lightning_module, datamodule=lightning_datamodule)

    trainer.test(ckpt_path="best", datamodule=lightning_datamodule)


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
