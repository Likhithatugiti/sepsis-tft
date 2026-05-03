"""
src/training/trainer.py
------------------------
Training loop with:
  • Focal loss for class imbalance
  • AUROC evaluation
  • Elastic Weight Consolidation (EWC) for continual learning
  • Cosine LR scheduler with warmup
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for highly imbalanced sepsis labels.
    FL(p) = -α (1-p)^γ log(p)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = logits.clamp(1e-6, 1 - 1e-6)
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


# ─── Elastic Weight Consolidation ────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation — prevents catastrophic forgetting
    when the model is updated nightly on new patient streams.

    Usage:
        ewc = EWC(model, dataloader, device)
        # In training loop, add ewc.penalty(model) to the loss
        # After each nightly update, call ewc.update(model, dataloader)
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 device: torch.device, lam: float = 400.0):
        self.lam = lam
        self.device = device
        self._params = {n: p.clone().detach()
                        for n, p in model.named_parameters() if p.requires_grad}
        self._fisher = self._compute_fisher(model, dataloader)

    def _compute_fisher(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters() if p.requires_grad
        }
        model.eval()
        n_batches = 0
        for batch in dataloader:
            x, x_cnn, y = [t.to(self.device) for t in batch]
            out = model(x, x_cnn)
            loss = nn.functional.binary_cross_entropy(out["risk"], y)
            model.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1

        return {n: f / max(n_batches, 1) for n, f in fisher.items()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._fisher:
                loss += (self._fisher[n] * (p - self._params[n]) ** 2).sum()
        return (self.lam / 2) * loss

    def update(self, model: nn.Module, dataloader: DataLoader) -> None:
        """Call after each nightly update."""
        self._params = {n: p.clone().detach()
                        for n, p in model.named_parameters() if p.requires_grad}
        self._fisher = self._compute_fisher(model, dataloader)
        logger.info("EWC Fisher information updated ✓")


# ─── Dataset helper ───────────────────────────────────────────────────────────

def make_loaders(
    splits_dir: str | Path,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    splits_dir = Path(splits_dir)

    def load(name: str) -> DataLoader:
        X = torch.from_numpy(np.load(splits_dir / f"{name}_X.npy"))
        X_cnn = torch.from_numpy(np.load(splits_dir / f"{name}_X_cnn.npy"))
        y = torch.from_numpy(np.load(splits_dir / f"{name}_y.npy"))
        ds = TensorDataset(X, X_cnn, y)
        shuffle = name == "train"
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    return load("train"), load("val"), load("test")


# ─── LR Scheduler with warmup ────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int,
                 total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        s = self.step_count
        if s <= self.warmup_steps:
            scale = s / max(self.warmup_steps, 1)
        else:
            progress = (s - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = lr * scale


# ─── Trainer ─────────────────────────────────────────────────────────────────

class SepsisTrainer:
    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.focal_loss = FocalLoss(
            gamma=cfg.training.focal_loss.gamma,
            alpha=cfg.training.focal_loss.alpha,
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.ewc: EWC | None = None
        self.best_auroc = 0.0
        self.patience_counter = 0

    def train_epoch(
        self, loader: DataLoader, scheduler: WarmupCosineScheduler
    ) -> Dict[str, float]:
        self.model.train()
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []

        for X, X_cnn, y in loader:
            X, X_cnn, y = X.to(self.device), X_cnn.to(self.device), y.to(self.device)

            out = self.model(X, X_cnn)
            risk = out["risk"]

            loss = self.focal_loss(risk, y)
            if self.ewc is not None:
                loss = loss + self.ewc.penalty(self.model)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )
            self.optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(y)
            n += len(y)
            all_preds.extend(risk.detach().cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        return {"loss": total_loss / n, "auroc": auroc}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []

        for X, X_cnn, y in loader:
            X, X_cnn, y = X.to(self.device), X_cnn.to(self.device), y.to(self.device)
            out = self.model(X, X_cnn)
            risk = out["risk"]
            loss = self.focal_loss(risk, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            all_preds.extend(risk.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        labels = np.array(all_labels)
        preds = np.array(all_preds)
        return {
            "loss": total_loss / n,
            "auroc": roc_auc_score(labels, preds),
            "auprc": average_precision_score(labels, preds),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path = "outputs",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg.training

        total_steps = cfg.epochs * len(train_loader)
        scheduler = WarmupCosineScheduler(
            self.optimizer, cfg.warmup_steps, total_steps
        )

        # Initialise EWC on first training data sample
        if cfg.ewc.enabled:
            logger.info("Computing initial EWC Fisher information …")
            self.ewc = EWC(self.model, train_loader, self.device, lam=cfg.ewc.lambda_)

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self.train_epoch(train_loader, scheduler)
            val_metrics = self.evaluate(val_loader)

            logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_auroc={train_metrics['auroc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auroc={val_metrics['auroc']:.4f} "
                f"val_auprc={val_metrics['auprc']:.4f}"
            )

            # Checkpoint
            if val_metrics["auroc"] > self.best_auroc:
                self.best_auroc = val_metrics["auroc"]
                self.patience_counter = 0
                torch.save(
                    {"epoch": epoch, "model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "val_auroc": self.best_auroc},
                    output_dir / "best_model.pt",
                )
                logger.info(f"  ✓ New best AUROC: {self.best_auroc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.early_stopping.patience:
                    logger.info("Early stopping triggered.")
                    break

        logger.info(f"Training complete. Best val AUROC: {self.best_auroc:.4f}")
