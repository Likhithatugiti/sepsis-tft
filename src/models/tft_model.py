"""
src/models/tft_model.py
------------------------
Temporal Fusion Transformer for sepsis onset prediction.

Architecture:
  CNN branch (waveform windows)
       +
  TFT encoder (temporal self-attention over 48h history)
       +
  Gating network
       +
  Feedforward risk head → P(sepsis at t+6h)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_branch import CNNBranch
from .gating_network import GatingNetwork, VariableSelectionNetwork


# ─── Positional Encoding ─────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─── Temporal Self-Attention Block ───────────────────────────────────────────

class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking — the model should not
    peek into the future at inference time.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        attn_out, _ = self.attn(x, x, x,
                                attn_mask=causal_mask,
                                key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ─── Feedforward Risk Head ───────────────────────────────────────────────────

class RiskHead(nn.Module):
    """Maps the gated context vector → scalar sepsis probability."""

    def __init__(self, in_dim: int, hidden_dims: list[int] = (128, 64),
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)  # (B,)


# ─── Full TFT Model ──────────────────────────────────────────────────────────

class SepsisTFT(nn.Module):
    """
    Full Sepsis Temporal Fusion Transformer.

    Args:
        n_features:     total number of input features (F)
        d_model:        internal dimension
        n_heads:        multi-head attention heads
        n_layers:       temporal self-attention layers
        d_ff:           feedforward dimension
        cnn_in:         number of vitals for CNN branch
        cnn_channels:   conv channels in CNN branch
        cnn_kernels:    kernel sizes in CNN branch
        dropout / attn_dropout: regularisation
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        cnn_in: int = 8,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        cnn_kernels: tuple[int, ...] = (3, 3, 3),
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        risk_hidden: tuple[int, ...] = (128, 64),
        risk_dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection: map each feature scalar → d_model embedding
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Variable selection (soft feature gates before attention)
        self.var_selection = VariableSelectionNetwork(
            n_features=n_features, d_model=d_model, dropout=dropout
        )

        # Temporal self-attention stack
        self.encoder = nn.ModuleList([
            TemporalSelfAttention(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_layers)
        ])

        # CNN branch
        self.cnn = CNNBranch(
            in_channels=cnn_in,
            channels=list(cnn_channels),
            kernel_sizes=list(cnn_kernels),
            dropout=dropout,
        )

        # Gating: fuse TFT output + CNN features
        gate_in = d_model + self.cnn.out_dim
        self.gate = GatingNetwork(gate_in, hidden_dim=d_model, dropout=dropout)

        # Risk head
        self.risk_head = RiskHead(d_model, list(risk_hidden), risk_dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, F)  — full lookback sequence
        x_cnn: torch.Tensor,       # (B, T_cnn, C_vitals)
        padding_mask: torch.Tensor | None = None,  # (B, T) bool
    ) -> dict[str, torch.Tensor]:

        B, T, F = x.shape

        # ── Variable selection (per timestep feature weighting) ───────────
        # Expand to (B, T, F, d_model) via broadcasted projection
        x_proj = self.input_proj(x)                # (B, T, d_model)
        x_enc = self.pos_enc(x_proj)               # (B, T, d_model)

        # ── Temporal self-attention ───────────────────────────────────────
        for layer in self.encoder:
            x_enc = layer(x_enc, key_padding_mask=padding_mask)

        # Take last timestep representation
        tft_out = x_enc[:, -1, :]                  # (B, d_model)

        # ── CNN branch ────────────────────────────────────────────────────
        cnn_out = self.cnn(x_cnn)                  # (B, cnn_out_dim)

        # ── Gating fusion ─────────────────────────────────────────────────
        fused = torch.cat([tft_out, cnn_out], dim=-1)   # (B, d_model + cnn_dim)
        gated = self.gate(fused)                         # (B, d_model)

        # ── Risk prediction ───────────────────────────────────────────────
        risk = self.risk_head(gated)               # (B,)

        return {
            "risk": risk,          # P(sepsis at t+6h)
            "embedding": gated,    # for SHAP / interpretability
            "tft_out": tft_out,
            "cnn_out": cnn_out,
        }
