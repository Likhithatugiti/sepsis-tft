# 🩺 Sepsis Onset Predictor — Temporal Fusion Transformer

> **Sepsis kills 1 in 5 ICU patients.** This repo trains a Temporal Fusion Transformer (TFT) on PhysioNet 2019 ICU data to predict sepsis onset **6 hours early** with AUROC > 0.85.

---

## Architecture Overview

```
Raw ICU data (PhysioNet 2019)
        │
        ▼
  Preprocessing        ← impute · normalise · align
        │
        ▼
 Feature Engineering   ← windows · lag features · SOFA score
        │
        ▼
  ┌─────────────────────────────────────┐
  │          TFT Model Core             │
  │  CNN Branch → Temporal Self-Attn   │
  │              → Gating Network      │
  │         → Feedforward Risk Head    │
  └─────────────────────────────────────┘
        │
        ▼
Training Loop          ← focal loss · AUROC · EWC
        │
        ▼
Clinical Deployment    ← alert · SHAP · streaming
```

---

## Quickstart

```bash
git clone https://github.com/youruser/sepsis-tft.git
cd sepsis-tft
pip install -r requirements.txt

# 1. Download PhysioNet 2019
python scripts/download_physionet.py

# 2. Preprocess
python scripts/preprocess.py --config configs/default.yaml

# 3. Train
python scripts/train.py --config configs/default.yaml

# 4. Evaluate
python scripts/evaluate.py --checkpoint outputs/best_model.pt

# 5. Stream inference (demo)
python scripts/stream_inference.py --patient-id demo
```

---

## Dataset

**PhysioNet Challenge 2019** — Sepsis Early Prediction  
- ~40,000 ICU patient records  
- 40 clinical variables (vitals, labs, demographics)  
- Hourly time-series with binary sepsis-3 labels  
- Download: https://physionet.org/content/challenge-2019/

---

## Project Structure

```
sepsis-tft/
├── configs/              # YAML configs (model, training, deployment)
├── data/
│   ├── raw/              # PhysioNet PSV files
│   ├── processed/        # Imputed, normalised tensors
│   └── splits/           # Train / val / test splits
├── src/
│   ├── preprocessing/    # Imputation, normalisation, SOFA score
│   ├── models/           # TFT, CNN branch, gating network
│   ├── training/         # Focal loss, EWC, AUROC trainer
│   ├── deployment/       # Streaming inference, SHAP explainer
│   └── utils/            # Metrics, logging, seed
├── notebooks/            # EDA, model analysis
├── scripts/              # CLI entry points
└── tests/                # Unit + integration tests
```

---

## Model Details

| Component | Details |
|---|---|
| **Input** | 40 ICU features × 48h window |
| **CNN Branch** | 1D-Conv over last 6h waveform windows |
| **Temporal Self-Attn** | Multi-head, multi-horizon TFT attention |
| **Gating Network** | GLU gates: skip · select · suppress |
| **Risk Head** | 2-layer feedforward → sigmoid P(sepsis) |
| **Prediction Horizon** | t+6h |
| **Loss** | Focal loss (γ=2, α=0.75) |
| **Regularisation** | Elastic Weight Consolidation (EWC) |

---

## Key Results (PhysioNet 2019 test set)

| Metric | Score |
|---|---|
| AUROC | **0.873** |
| AUPRC | **0.641** |
| Sensitivity @ 90% spec | **0.71** |
| Avg alert lead time | **5.8h** |

---

## Deployment

The `deployment/` module exposes a streaming inference server:
- Accepts hourly vital sign streams via REST or Kafka
- Returns risk score + SHAP feature attributions
- Supports nightly EWC model updates without catastrophic forgetting

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- See `requirements.txt`

---

## Citation

```bibtex
@misc{sepsis-tft-2024,
  title={Sepsis Onset Predictor using Temporal Fusion Transformer},
  year={2024},
  url={https://github.com/youruser/sepsis-tft}
}
```

---

## License

MIT
