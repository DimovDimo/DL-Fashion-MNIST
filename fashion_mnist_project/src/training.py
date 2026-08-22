"""
training.py – Training loops, evaluation and inference utilities.

Contains:
  - evaluate_predictions(): generic evaluation + registry
  - predict_logits(): batch inference returning logits
  - torch_probabilities(): batch inference returning probabilities
  - train_autoencoder(): autoencoder training loop
  - reconstruction_errors(): per-image MSE errors
  - plot_confusion(): confusion matrix plotting
  - plot_confusion(): confusion matrix plotting
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, CFG, set_seed
from models import Augment, count_parameters


RESULTS: List[Dict[str, object]] = []


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, family: str,
    fit_s: float = 0.0, params: int = 0, notes: str = "",
    class_names: Sequence[str] = (), register: bool = True,
) -> Dict[str, object]:
    """Compute metrics and optionally register in the global RESULTS list."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    rec = {
        "model": model_name, "family": family,
        "accuracy": round(float(acc), 6), "error_rate": round(1.0 - float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "fit_s": round(fit_s, 1), "params": params, "notes": notes,
    }
    if register:
        RESULTS.append(rec)
    if class_names:
        print(classification_report(y_true, y_pred, target_names=list(class_names), digits=4))
    return rec


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device = DEVICE) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (logits, labels) arrays."""
    model = model.to(device).eval()
    all_logits, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        all_logits.append(model(xb).cpu())
        all_labels.append(yb)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def torch_probabilities(model: nn.Module, x: torch.Tensor, device: torch.device = DEVICE) -> np.ndarray:
    """Return softmax probabilities for a batch of images."""
    model = model.to(device).eval()
    logits = model(x.to(device))
    return torch.softmax(logits, dim=1).cpu().numpy()


def train_autoencoder(
    x_train: torch.Tensor, epochs: int, batch_size: int, latent: int,
    lr: float = 2e-3, device: torch.device = DEVICE, seed: int = 42,
) -> Tuple:
    """Train the autoencoder with MSE reconstruction loss."""
    from models import ConvAutoencoder
    set_seed(seed)
    model = ConvAutoencoder(latent=latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=True)
    history: List[float] = []
    model.train()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        history.append(total / n)
        print(f"  autoencoder epoch {ep:2d}/{epochs} | MSE {history[-1]:.5f}")
    return model, history


@torch.no_grad()
def reconstruction_errors(
    model, x: torch.Tensor, batch_size: int = 512, device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-image MSE reconstruction error and the reconstructions themselves."""
    model.eval()
    errs, recons = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        out = model(xb)
        errs.append(((out - xb) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        recons.append(out.cpu())
    return np.concatenate(errs), torch.cat(recons).numpy()


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str],
    title: str = "", ax=None,
) -> None:
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=list(class_names), yticklabels=list(class_names),
                ax=ax, cbar_kws={"label": "recall"})
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if ax is None:
        plt.show()
