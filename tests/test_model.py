"""
tests/test_model.py
--------------------
Unit tests for CNN branch, gating network, and full TFT model.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_branch import CNNBranch
from src.models.gating_network import GatingNetwork, VariableSelectionNetwork
from src.models.tft_model import SepsisTFT
from src.training.trainer import FocalLoss


# ─── Config ──────────────────────────────────────────────────────────────────
B = 4         # batch size
T = 48        # lookback hours
T_cnn = 6    # CNN window
F = 40        # total features
C = 8         # vital channels for CNN


# ─── CNN Branch ──────────────────────────────────────────────────────────────
class TestCNNBranch:
    def test_output_shape(self):
        model = CNNBranch(in_channels=C, channels=[32, 64, 128])
        x = torch.randn(B, T_cnn, C)
        out = model(x)
        assert out.shape == (B, 128), f"Expected ({B}, 128), got {out.shape}"

    def test_gradients_flow(self):
        model = CNNBranch(in_channels=C)
        x = torch.randn(B, T_cnn, C, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


# ─── Gating Network ──────────────────────────────────────────────────────────
class TestGatingNetwork:
    def test_output_shape(self):
        gate = GatingNetwork(in_dim=256, hidden_dim=128)
        x = torch.randn(B, 256)
        out = gate(x)
        assert out.shape == (B, 128)

    def test_variable_selection(self):
        vsn = VariableSelectionNetwork(n_features=F, d_model=64)
        x = torch.randn(B, T, F, 64)
        out = vsn(x)
        assert out.shape == (B, T, 64)


# ─── Full TFT Model ──────────────────────────────────────────────────────────
class TestSepsisTFT:
    @pytest.fixture
    def model(self):
        return SepsisTFT(
            n_features=F, d_model=64, n_heads=4, n_layers=1, d_ff=128,
            cnn_in=C, cnn_channels=[16, 32], cnn_kernels=[3, 3]
        )

    def test_forward_shape(self, model):
        x = torch.randn(B, T, F)
        x_cnn = torch.randn(B, T_cnn, C)
        out = model(x, x_cnn)
        assert out["risk"].shape == (B,), f"risk shape: {out['risk'].shape}"
        assert out["embedding"].shape[0] == B

    def test_risk_in_01(self, model):
        x = torch.randn(B, T, F)
        x_cnn = torch.randn(B, T_cnn, C)
        risk = model(x, x_cnn)["risk"]
        assert (risk >= 0).all() and (risk <= 1).all()

    def test_backward(self, model):
        x = torch.randn(B, T, F)
        x_cnn = torch.randn(B, T_cnn, C)
        y = torch.randint(0, 2, (B,)).float()
        loss_fn = FocalLoss()
        out = model(x, x_cnn)
        loss = loss_fn(out["risk"], y)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ─── Focal Loss ──────────────────────────────────────────────────────────────
class TestFocalLoss:
    def test_loss_positive(self):
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        preds = torch.sigmoid(torch.randn(16))
        targets = torch.randint(0, 2, (16,)).float()
        loss = loss_fn(preds, targets)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        targets = torch.ones(8)
        preds = torch.full((8,), 0.999)
        loss = loss_fn(preds, targets)
        assert loss.item() < 0.01
