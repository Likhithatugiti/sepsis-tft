"""
src/models/cnn_branch.py
-------------------------
1-D convolutional branch that encodes recent waveform windows
(last 6 hours of vital signs) into a compact feature vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class CNNBranch(nn.Module):
    """
    Multi-scale 1-D CNN over short waveform windows.

    Input:  (B, T_cnn, C_vitals)  — e.g. (B, 6, 8)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        in_channels: int = 8,
        channels: list[int] = (32, 64, 128),
        kernel_sizes: list[int] = (3, 3, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        layers = []
        c_in = in_channels
        for c_out, k in zip(channels, kernel_sizes):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c_in = c_out

        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)   # global average pool → (B, C, 1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for Conv1d
        x = rearrange(x, "b t c -> b c t")
        x = self.conv_stack(x)          # (B, C_last, T)
        x = self.pool(x).squeeze(-1)    # (B, C_last)
        return x
