from functools import partial
from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import f1_score
from torch import optim
from torchmetrics import MetricCollection


class HNetGRULightning(L.LightningModule):
    """ """

    def __init__(
        self,
        device,
        max_len: int = 2,
        optimizer: partial[torch.optim.Optimizer] = partial(optim.Adam),
    ):
        super().__init__()
        self._device = device
        self.model = HNetGRU(max_len=max_len).to(self._device)

        self.criterion1 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion2 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion3 = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion_wts = [1.0, 1.0, 1.0]

        self.optimizer: torch.optim.Optimizer = optimizer(
            params=self.model.parameters()
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
            zero_division=1,
        ).to(self._device)
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
            zero_division=1,
        ).to(self._device)
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
            zero_division=1,
        ).to(self._device)

    def common_step(self, batch, batch_idx):
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

    def training_step(self, batch, batch_idx):
        """
        Lightning training step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
        """

        train_loss, output, target = self.common_step(batch, batch_idx)

        preds = torch.sigmoid(output[0]) > 0.5

        train_f1 = self.train_f1(preds, target[0])

        # Log loss and F1-score
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_f1", train_f1, on_step=False, on_epoch=True, prog_bar=False)

        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
        """

        val_loss, output, target = self.common_step(batch, batch_idx)

        preds = torch.sigmoid(output[0]) > 0.5
        val_f1 = self.val_f1(preds, target[0])

        # Log loss and F1-score
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        """
        Lightning test step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
        """
        test_loss, output, target = self.common_step(batch, batch_idx)

        preds = torch.sigmoid(output[0]) > 0.5
        test_f1 = self.test_f1(preds, target[0])

        # Log loss and F1-score
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return test_loss

        return self.common_step(batch, batch_idx)

        # def on_fit_start(self) -> None:
        """
        Called at the beginning of the fit loop.

        - Checks the consistency of the DataModule's parameters
        """
        #    self.check_datamodule_parameter()

        # def on_test_start(self) -> None:
        """
        Called at the beginning of the testing loop.

        - Checks the consistency of the DataModule's parameters
        """

    #    self.check_datamodule_parameter()

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return self.optimizer

        # def common_logging(
        # self, stage: str, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
        # ) -> None:
        """
        Common logging for training, validation and test steps.

        Args:
            stage(str): Stage of the training
            outputs(STEP_OUTPUT): Output of the {train,validation,test}_step method
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx(int): Index of the batch

        """

        # Log loss
        # self.log(f"{stage}/loss", outputs["loss"], sync_dist=True)

        # Log metrics
        # predicted_phonemes = self.get_phonemes_from_logits(outputs["logits"])
        # target_phonemes = batch["phonemes_str"]
        # metrics_to_log = self.metrics(predicted_phonemes, target_phonemes)
        # metrics_to_log = {f"{stage}/{k}": v for k, v in metrics_to_log.items()}

        # self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return (
            self._get_name()
            + "(in_channels={}, out_channels={}, key_channels={})".format(
                self.conv_Q.in_channels,
                self.conv_V.out_channels,
                self.conv_K.out_channels,
            )
        )


class HNetGRU(nn.Module):
    def __init__(self, max_len=4, hidden_size=128):
        super().__init__()
        self.nb_gru_layers = 1
        self.max_len = max_len
        self.gru = nn.GRU(max_len, hidden_size, self.nb_gru_layers, batch_first=True)
        self.attn = AttentionLayer(hidden_size, hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, max_len)

    def forward(self, query):
        # query - batch x seq x feature

        out, _ = self.gru(query)
        # out - batch x seq x hidden

        out = out.permute((0, 2, 1))
        # out - batch x hidden x seq

        out = self.attn.forward(out)
        # out - batch x hidden x seq

        out = out.permute((0, 2, 1))
        out = torch.tanh(out)
        # out - batch x seq x hidden

        out = self.fc1(out)
        # out - batch x seq x feature

        out1 = out.view(out.shape[0], -1)
        # out1 - batch x (seq x feature)

        out2, _ = torch.max(out, dim=-1)
        # out2 - batch x seq x 1

        out3, _ = torch.max(out, dim=-2)
        # out3 - batch x 1 x feature

        return out1.squeeze(), out2.squeeze(), out3.squeeze()
