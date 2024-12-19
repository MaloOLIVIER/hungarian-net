# hungarian_net/torch_modules/hnet_gru.py
"""
@inproceedings{Adavanne_2021,
title={Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers},
url={http://dx.doi.org/10.1109/WASPAA52581.2021.9632773},
DOI={10.1109/waspaa52581.2021.9632773},
booktitle={2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
publisher={IEEE},
author={Adavanne, Sharath and Politis, Archontis and Virtanen, Tuomas},
year={2021},
month=oct, pages={211-215} }
"""

import torch
import torch.nn as nn

from hungarian_net.torch_modules.attention_layer import AttentionLayer


class HNetGRU(nn.Module):
    """
    HNetGRU model combining GRU and Attention layers for processing sound source localization.

    This model utilizes a GRU layer followed by an Attention layer to process input features,
    and produces multiple output representations for further analysis.

    Args:
        max_doas (int, optional): Maximum number of Directions of Arrival (DoAs) to estimate. Defaults to 2.
        hidden_size (int, optional): Number of hidden units in the GRU and Attention layers. Defaults to 128.

    Attributes:
        nb_gru_layers (int): Number of GRU layers.
        max_doas (int): Maximum number of DoAs to estimate.
        gru (nn.GRU): Gated Recurrent Unit layer for processing sequential data.
        attn (AttentionLayer): Custom Attention layer for feature weighting.
        fc1 (nn.Linear): Fully connected layer to project hidden states to output features.
    """

    def __init__(self, max_doas: int = 2, hidden_size: int = 128) -> None:
        super().__init__()
        self.nb_gru_layers = 1
        self.max_doas = max_doas
        self.gru = nn.GRU(max_doas, hidden_size, self.nb_gru_layers, batch_first=True)
        self.attn = AttentionLayer(hidden_size, hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, max_doas)

    def forward(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of HNetGRU.

        Processes the input query through GRU, applies the Attention mechanism,
        and projects the result to the desired number of DoAs. Additionally,
        computes global features using max pooling.

        Args:
            query (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns:
            tuple:
                - out1 (torch.Tensor): Flattened output tensor of shape (batch_size, sequence_length * feature_dim).
                - out2 (torch.Tensor): Tensor obtained by max pooling over the last dimension, shape (batch_size, sequence_length).
                - out3 (torch.Tensor): Tensor obtained by max pooling over the second dimension, shape (batch_size, feature_dim).
        """
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
