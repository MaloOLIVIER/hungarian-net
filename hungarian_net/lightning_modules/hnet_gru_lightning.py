# hungarian_net/lightning_modules/hnet_gru_lightning.py
from functools import partial
from typing import Any, Dict, Tuple

import lightning as L
import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
from aim import Image
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix

from hungarian_net.torch_modules.hnet_gru import HNetGRU


class HNetGRULightning(L.LightningModule):
    """
    HNetGRULightning is a PyTorch Lightning module that encapsulates the HunNet GRU model,
    loss functions, optimizers, and evaluation metrics for training, validation, and testing.

    **Attributes:**
        model (HNetGRU): The GRU-based neural network model.
        criterion1 (nn.BCEWithLogitsLoss): Loss function for the first output.
        criterion2 (nn.BCEWithLogitsLoss): Loss function for the second output.
        criterion3 (nn.BCEWithLogitsLoss): Loss function for the third output.
        criterion_wts (List[float]): Weights for combining multiple loss components.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler.
        metrics (MetricCollection): Collection of evaluation metrics.
        confusion_matrix (MulticlassConfusionMatrix): Confusion matrix for multi-class classification.
    """

    def __init__(
        self,
        metrics: MetricCollection,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        max_doas: int = 2,
        optimizer: partial[optim.Optimizer] = partial(optim.Adam),
        scheduler: partial[optim.lr_scheduler] = None,
    ) -> None:
        super().__init__()

        # Automatically save hyperparameters except for non-serializable objects
        self.save_hyperparameters(ignore=["metrics", "device", "optimizer"])

        self._device = device
        self.model = HNetGRU(max_doas=max_doas).to(self._device)

        self.criterion1 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion2 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion3 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion_wts = [1.0, 1.0, 1.0]

        self.optimizer: torch.optim.Optimizer = optimizer(
            params=self.model.parameters()
        )

        self.scheduler: optim.lr_scheduler = (
            scheduler(self.optimizer) if scheduler is not None else None
        )

        self.metrics: MetricCollection = metrics
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=max_doas)

    def common_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Performs a forward pass and computes the loss.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple containing input data and target labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - Total loss.
                - Model outputs as a tuple of tensors.
                - Targets as a tuple of tensors.
        """
        data, target = batch
        data = data.to(self._device).float()

        # forward pass
        output = self.model(data)
        l1 = self.criterion1(output[0], target[0])
        l2 = self.criterion2(output[1], target[1])
        l3 = self.criterion3(output[2], target[2])

        loss = (
            self.criterion_wts[0] * l1
            + self.criterion_wts[1] * l2
            + self.criterion_wts[2] * l3
        )

        return loss, output, target

    def training_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Lightning training step.

        Args:
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor|Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Dictionary containing loss, outputs, and targets.
        """
        loss, output, target = self.common_step(batch, batch_idx)

        outputs = {"loss": loss, "output": output, "target": target}

        return outputs

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Lightning validation step.

        Args:
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor|Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Dictionary containing loss, outputs, and targets.
        """
        loss, output, target = self.common_step(batch, batch_idx)

        outputs = {"loss": loss, "output": output, "target": target}

        return outputs

    def test_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Lightning test step.

        Args:
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor|Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Dictionary containing loss, outputs, and targets.
        """
        loss, output, target = self.common_step(batch, batch_idx)

        outputs = {"loss": loss, "output": output, "target": target}

        return outputs

    def on_fit_start(self) -> None:
        """
        Called at the beginning of the fit loop.

        Resets all metrics to ensure no data leakage from previous runs.
        """
        # 1. Initialize or reset metrics
        self.metrics.reset()

    def on_test_start(self) -> None:
        """
        Called at the beginning of the testing loop.

        Resets test metrics and sets the model to evaluation mode.
        """
        # 1. Reset test metrics
        self.metrics.reset()

        # 2. Disable certain training-specific settings
        self.model.eval()

    def on_train_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """
        Method automatically called when the train batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the training_step method.
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data.
            batch_idx (int): Index of the batch.
        """

        self.common_logging("train", outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Method automatically called when the validation batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the validation_step method.
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        loss, output, target = outputs["loss"], outputs["output"], outputs["target"]

        self.common_logging("validation", outputs, batch, batch_idx)

        preds = torch.sigmoid(output[0]) > 0.5

        # confusion matrix
        self.confusion_matrix.update(preds, target[0])

        # Log confusion matrix
        fig_, ax_ = self.confusion_matrix.plot()

        self.logger.experiment.track(
            Image(fig_),
            name=f"validation/epoch={self.current_epoch}/confusion_matrix",
            step=self.global_step,
        )

    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Method automatically called when the test batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the test_step method.
            batch (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        loss, output, target = outputs["loss"], outputs["output"], outputs["target"]

        self.common_logging("test", outputs, batch, batch_idx)

        preds = torch.sigmoid(output[0]) > 0.5

        # confusion matrix
        self.confusion_matrix.update(preds, target[0])

        # Log confusion matrix
        fig_, ax_ = self.confusion_matrix.plot()

        self.logger.experiment.track(
            Image(fig_), name=f"test/confusion_matrix", step=self.global_step
        )

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.

        Retrieves the aggregated validation loss and steps the scheduler.
        Resets the confusion matrix for the next epoch.
        """
        # Retrieve the validation loss from callback metrics
        val_loss = self.trainer.callback_metrics.get("validation_loss")

        if val_loss is not None and self.scheduler is not None:
            # Step the scheduler based on the validation loss
            (
                self.scheduler.step(val_loss)
                if type(self.scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau
                else self.scheduler.step()
            )

            # Log the learning rate
            self.log("learning_rate", self.optimizer.param_groups[0]["lr"])

        self.confusion_matrix.reset()

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the testing epoch.

        Resets the confusion matrix after testing is complete.
        """
        self.confusion_matrix.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            Dict[str, Any]: Dictionary containing the optimizer and scheduler configurations.
        """
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
            }
        return {"optimizer": self.optimizer}

    def common_logging(
        self,
        stage: str,
        outputs: STEP_OUTPUT,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """
        Common logging for training, validation and test steps.

        Args:
            stage(str): Stage of the training ('train', 'validation', 'test').
            outputs(STEP_OUTPUT): Output of the {train, validation, test}_step method.
            batch(Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Batch data.
            batch_idx(int): Index of the batch.

        """
        loss, output, target = outputs["loss"], outputs["output"], outputs["target"]

        # Log loss
        self.log(f"{stage}_loss", loss, sync_dist=True)

        # Log epoch
        self.logger.experiment.track(
            self.current_epoch, name="epoch", step=self.global_step
        )

        preds = torch.sigmoid(output[0]) > 0.5

        metrics_to_log = self.metrics(preds, target[0])
        metrics_to_log = {f"{stage}/{k}": v for k, v in metrics_to_log.items()}

        # F1-score
        self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)
