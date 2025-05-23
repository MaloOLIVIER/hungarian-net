# hungarian_net/torch_modules/attention_layer.py
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


class AttentionLayer(nn.Module):
    """
    Attention Layer for processing input features using multi-head attention mechanism.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        key_channels (int): Number of channels for the key and query.

    Attributes:
        conv_Q (nn.Conv1d): Convolution layer to generate queries.
        conv_K (nn.Conv1d): Convolution layer to generate keys.
        conv_V (nn.Conv1d): Convolution layer to generate values.
    """

    def __init__(self, in_channels: int, out_channels: int, key_channels: int) -> None:
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after applying attention mechanism.
        """
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self) -> str:
        """
        Returns a string representation of the AttentionLayer.

        Returns:
            str: String representation with layer configurations.
        """
        return (
            self._get_name()
            + "(in_channels={}, out_channels={}, key_channels={})".format(
                self.conv_Q.in_channels,
                self.conv_V.out_channels,
                self.conv_K.out_channels,
            )
        )
