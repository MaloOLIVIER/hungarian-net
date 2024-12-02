from functools import partial
from typing import Any, Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import f1_score
from torch import optim
from torchmetrics import MetricCollection
from hungarian_net.torch_modules.attention_layer import AttentionLayer

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
