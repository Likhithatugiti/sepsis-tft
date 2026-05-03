#!/usr/bin/env python
"""
scripts/train.py
-----------------
CLI entry point for training the Sepsis TFT.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --device cpu
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tft_model import SepsisTFT
from src.training.trainer import SepsisTrainer, make_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Sepsis TFT")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device_str = args.device or cfg.training.device
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data splits …")
    train_loader, val_loader, test_loader = make_loaders(
        cfg.data.splits_dir, batch_size=cfg.training.batch_size
    )

    # Infer n_features from data
    sample_X, _, _ = next(iter(train_loader))
    n_features = sample_X.shape[-1]
    logger.info(f"n_features={n_features}")

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("Building SepsisTFT …")
    model = SepsisTFT(
        n_features=n_features,
        d_model=cfg.model.tft.d_model,
        n_heads=cfg.model.tft.n_heads,
        n_layers=cfg.model.tft.n_encoder_layers,
        d_ff=cfg.model.tft.d_ff,
        cnn_in=cfg.model.cnn.in_channels,
        cnn_channels=cfg.model.cnn.channels,
        cnn_kernels=cfg.model.cnn.kernel_sizes,
        dropout=cfg.model.tft.dropout,
        attn_dropout=cfg.model.tft.attn_dropout,
        risk_hidden=cfg.model.risk_head.hidden_dims,
        risk_dropout=cfg.model.risk_head.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Training ───────────────────────────────────────────────────────────────
    trainer = SepsisTrainer(model, cfg, device)
    trainer.fit(train_loader, val_loader, output_dir=args.output)

    # ── Final evaluation ───────────────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    ckpt = torch.load(f"{args.output}/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Test AUROC: {test_metrics['auroc']:.4f} | "
        f"Test AUPRC: {test_metrics['auprc']:.4f}"
    )


if __name__ == "__main__":
    main()
