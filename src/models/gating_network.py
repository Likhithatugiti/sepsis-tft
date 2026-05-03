"""
src/models/gating_network.py
-----------------------------
Gated Linear Unit (GLU) gating network used in the TFT architecture.
Implements skip · select · suppress gates for adaptive feature selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """Standard GLU: splits input in half, gates with sigmoid."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatingNetwork(nn.Module):
    """
    Variable selection / gating network from TFT.

    Given a concatenated context vector (CNN features + TFT hidden),
    produces soft gates to skip, select, or suppress each feature stream.

    Input:  (B, in_dim)
    Output: (B, out_dim)  — gated representation
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim)   # residual projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.elu(self.fc_in(x))
        gated = self.glu(h)
        skip = self.skip(x)
        return self.layer_norm(gated + skip)


class VariableSelectionNetwork(nn.Module):
    """
    Variable-level soft selection: learns per-feature importance weights.

    Used before feeding into the temporal self-attention block.

    Input:  (B, T, n_features, d_model)
    Output: (B, T, d_model)
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        # Flattened feature projection
        self.mlp = nn.Sequential(
            nn.Linear(n_features * d_model, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features),
        )
        self.feature_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_features, d_model)
        """
        B, T, F, D = x.shape
        flat = x.reshape(B, T, F * D)
        weights = self.softmax(self.mlp(flat))         # (B, T, F)
        selected = (weights.unsqueeze(-1) * x).sum(dim=2)  # (B, T, D)
        return self.layer_norm(self.feature_proj(selected))
