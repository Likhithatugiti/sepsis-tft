"""
src/deployment/inference.py
----------------------------
Streaming inference server with:
  • REST endpoint via FastAPI
  • SHAP feature attribution
  • Kafka consumer for live ICU streams
  • Nightly EWC model update hook
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import shap

logger = logging.getLogger(__name__)


# ─── SHAP Explainer ──────────────────────────────────────────────────────────

class SHAPExplainer:
    """Wraps the TFT model for SHAP DeepExplainer attributions."""

    def __init__(self, model: torch.nn.Module, background_X: torch.Tensor,
                 background_X_cnn: torch.Tensor, feature_names: List[str],
                 device: torch.device):
        self.model = model
        self.feature_names = feature_names
        self.device = device

        # Use GradientExplainer (works with arbitrary PyTorch models)
        def wrapped(x_flat: torch.Tensor) -> torch.Tensor:
            B = x_flat.shape[0]
            T, F = background_X.shape[1], background_X.shape[2]
            T_cnn, C = background_X_cnn.shape[1], background_X_cnn.shape[2]
            x = x_flat[:, :T * F].reshape(B, T, F)
            x_cnn = x_flat[:, T * F:].reshape(B, T_cnn, C)
            out = self.model(x, x_cnn)
            return out["risk"].unsqueeze(-1)

        bg_flat = torch.cat([
            background_X.reshape(len(background_X), -1),
            background_X_cnn.reshape(len(background_X_cnn), -1),
        ], dim=1).to(device)

        self.explainer = shap.GradientExplainer(wrapped, bg_flat)
        self._T = background_X.shape[1]
        self._F = background_X.shape[2]
        self._T_cnn = background_X_cnn.shape[1]
        self._C = background_X_cnn.shape[2]

    def explain(self, x: torch.Tensor, x_cnn: torch.Tensor) -> Dict[str, Any]:
        x_flat = torch.cat([
            x.reshape(len(x), -1),
            x_cnn.reshape(len(x_cnn), -1),
        ], dim=1).to(self.device)

        shap_vals = self.explainer.shap_values(x_flat)
        # Average over timesteps to get per-feature importance
        tft_shap = shap_vals[:, :self._T * self._F].reshape(-1, self._T, self._F)
        mean_importance = np.abs(tft_shap).mean(axis=1)  # (B, F)

        return {
            "feature_names": self.feature_names,
            "importances": mean_importance.tolist(),
            "raw_shap": tft_shap.tolist(),
        }


# ─── REST Inference Server ────────────────────────────────────────────────────

def create_app(model: torch.nn.Module, explainer: SHAPExplainer,
               cfg, device: torch.device):
    """
    Build a FastAPI app for streaming inference.

    POST /predict
    {
      "patient_id": "P001",
      "vitals_sequence": [[hr, o2sat, temp, sbp, map, dbp, resp, etco2], ...],  # 48h
      "features_sequence": [[f1, ..., fN], ...],   # 48h × all features
      "waveform_window": [[...], ...]               # 6h vitals for CNN
    }
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install fastapi and pydantic: pip install fastapi pydantic")

    app = FastAPI(title="Sepsis Onset Predictor", version="1.0")

    class PredictRequest(BaseModel):
        patient_id: str
        features_sequence: List[List[float]]    # (48, F)
        waveform_window: List[List[float]]       # (6, 8)
        explain: bool = False

    class PredictResponse(BaseModel):
        patient_id: str
        sepsis_risk: float
        alert: bool
        lead_time_hours: int
        shap: Dict | None = None

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        x = torch.tensor([req.features_sequence], dtype=torch.float32).to(device)
        x_cnn = torch.tensor([req.waveform_window], dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            out = model(x, x_cnn)

        risk = float(out["risk"].item())
        alert = risk > cfg.deployment.risk_threshold

        shap_result = None
        if req.explain and explainer is not None:
            shap_result = explainer.explain(x, x_cnn)

        return PredictResponse(
            patient_id=req.patient_id,
            sepsis_risk=round(risk, 4),
            alert=alert,
            lead_time_hours=6,
            shap=shap_result,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ─── Kafka Streaming Consumer ─────────────────────────────────────────────────

class ICUStreamConsumer:
    """
    Consumes hourly vital sign messages from Kafka,
    maintains a rolling 48h buffer per patient,
    and emits risk scores when the buffer is full.
    """

    def __init__(self, model: torch.nn.Module, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.buffers: Dict[str, list] = {}   # patient_id → list of feature vectors
        self.cnn_buffers: Dict[str, list] = {}

    async def consume(self) -> None:
        try:
            from kafka import KafkaConsumer
            import json
        except ImportError:
            raise ImportError("Install kafka-python: pip install kafka-python")

        consumer = KafkaConsumer(
            self.cfg.deployment.streaming.kafka_topic,
            bootstrap_servers=self.cfg.deployment.streaming.kafka_bootstrap,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        logger.info("ICU stream consumer started …")
        for msg in consumer:
            await self._process_message(msg.value)

    async def _process_message(self, data: dict) -> None:
        pid = data["patient_id"]
        features = data["features"]         # list of F floats
        vitals = data["vitals"]             # list of 8 floats

        lookback = self.cfg.data.lookback   # 48
        cnn_win = self.cfg.preprocessing.waveform_window_hours  # 6

        buf = self.buffers.setdefault(pid, [])
        cbuf = self.cnn_buffers.setdefault(pid, [])
        buf.append(features)
        cbuf.append(vitals)

        # Keep rolling window
        if len(buf) > lookback:
            buf.pop(0)
        if len(cbuf) > cnn_win:
            cbuf.pop(0)

        if len(buf) >= lookback and len(cbuf) >= cnn_win:
            risk = await self._infer(buf[-lookback:], cbuf[-cnn_win:])
            if risk > self.cfg.deployment.risk_threshold:
                logger.warning(
                    f"🚨 ALERT | patient={pid} | "
                    f"sepsis_risk={risk:.3f} | lead=6h"
                )

    async def _infer(self, seq: list, cnn_seq: list) -> float:
        x = torch.tensor([seq], dtype=torch.float32).to(self.device)
        x_cnn = torch.tensor([cnn_seq], dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, x_cnn)
        return float(out["risk"].item())
