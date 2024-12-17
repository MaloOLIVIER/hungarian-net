from functools import partial
from typing import Any, Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
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
    """

    def __init__(
        self,
        metrics: MetricCollection,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        max_len: int = 2,
        optimizer: partial[optim.Optimizer] = partial(optim.Adam),
        scheduler: partial[optim.lr_scheduler] = None,
    ):
        super().__init__()

        # Automatically save hyperparameters except for non-serializable objects
        self.save_hyperparameters(ignore=["metrics", "device", "optimizer"])

        self._device = device
        self.model = HNetGRU(max_len=max_len).to(self._device)

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
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=max_len)

    def common_step(
        self, batch, batch_idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass and computes the loss.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss, model outputs, and targets.
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

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Lightning training step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """

        loss, output, target = self.common_step(batch, batch_idx)

        outputs = {"loss": loss, "output": output, "target": target}

        return outputs

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Lightning validation step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """

        loss, output, target = self.common_step(batch, batch_idx)

        outputs = {"loss": loss, "output": output, "target": target}

        return outputs

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx
    ) -> Dict[str, torch.Tensor]:
        """
        Lightning test step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        loss, output, target = self.common_step(batch, batch_idx)

        outputs = {"loss": loss, "output": output, "target": target}

        return outputs

    def on_fit_start(self) -> None:
        """
        Called at the beginning of the fit loop.

        """
        # 1. Initialize or reset metrics
        self.metrics.reset()

    def on_test_start(self) -> None:
        """
        Called at the beginning of the testing loop.

        """
        # 1. Reset test metrics
        self.metrics.reset()

        # 2. Disable certain training-specific settings
        self.model.eval()

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """
        Method automatically called when the train batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the training_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """

        self.common_logging("train", outputs, batch, batch_idx)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Method automatically called when the validation batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the validation_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """
        loss, output, target = outputs["loss"], outputs["output"], outputs["target"]

        self.common_logging("validation", outputs, batch, batch_idx)
        
        preds = torch.sigmoid(output[0]) > 0.5
        
        # confusion matrix
        self.confusion_matrix.update(preds, target[0])
        
        # Log confusion matrix
        fig_, ax_ = self.confusion_matrix.plot()
        
        self.logger.experiment.add_figure(f"validation/epoch={self.current_epoch}/confusion_matrix", fig_, global_step=self.global_step)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Method automatically called when the test batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the test_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """
        loss, output, target = outputs["loss"], outputs["output"], outputs["target"]
        
        self.common_logging("test", outputs, batch, batch_idx)
        
        preds = torch.sigmoid(output[0]) > 0.5
        
        # confusion matrix
        self.confusion_matrix.update(preds, target[0])
        
        # Log confusion matrix
        fig_, ax_ = self.confusion_matrix.plot()
        
        self.logger.experiment.add_figure(f"test/confusion_matrix", fig_, global_step=self.global_step)

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.

        Retrieves the aggregated validation loss and steps the scheduler.
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

        Retrieves the aggregated test loss.
        """
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return self.optimizer

    def common_logging(
        self, stage: str, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """
        Common logging for training, validation and test steps.

        Args:
            stage(str): Stage of the training
            outputs(STEP_OUTPUT): Output of the {train,validation,test}_step method
            batch(Any): Batch
            batch_idx(int): Index of the batch

        """
        loss, output, target = outputs["loss"], outputs["output"], outputs["target"]

        # Log loss
        self.log(f"{stage}_loss", loss, sync_dist=True)

        preds = torch.sigmoid(output[0]) > 0.5

        metrics_to_log = self.metrics(preds, target[0])
        metrics_to_log = {f"{stage}/{k}": v for k, v in metrics_to_log.items()}

        # F1-score
        self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)
