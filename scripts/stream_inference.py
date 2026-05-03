#!/usr/bin/env python
"""
scripts/stream_inference.py
-----------------------------
Launch the REST inference server or Kafka stream consumer.

Usage:
    # REST server
    python scripts/stream_inference.py --mode rest --checkpoint outputs/best_model.pt

    # Kafka consumer
    python scripts/stream_inference.py --mode kafka --checkpoint outputs/best_model.pt
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tft_model import SepsisTFT
from src.deployment.inference import ICUStreamConsumer, create_app

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint: str, cfg, n_features: int,
               device: torch.device) -> SepsisTFT:
    model = SepsisTFT(
        n_features=n_features,
        d_model=cfg.model.tft.d_model,
        n_heads=cfg.model.tft.n_heads,
        n_layers=cfg.model.tft.n_encoder_layers,
        d_ff=cfg.model.tft.d_ff,
        cnn_in=cfg.model.cnn.in_channels,
        cnn_channels=cfg.model.cnn.channels,
        cnn_kernels=cfg.model.cnn.kernel_sizes,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    logger.info(f"Loaded checkpoint (val_auroc={ckpt.get('val_auroc', '?'):.4f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["rest", "kafka"], default="rest")
    parser.add_argument("--n-features", type=int, default=40,
                        help="Total feature count (must match training)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, args.n_features, device)

    if args.mode == "rest":
        import uvicorn
        app = create_app(model, explainer=None, cfg=cfg, device=device)
        uvicorn.run(app, host=cfg.deployment.host, port=cfg.deployment.port)

    elif args.mode == "kafka":
        consumer = ICUStreamConsumer(model, cfg, device)
        asyncio.run(consumer.consume())


if __name__ == "__main__":
    main()
