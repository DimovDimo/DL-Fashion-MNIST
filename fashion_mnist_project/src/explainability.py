"""
explainability.py – Model explainability methods.

Contains:
  - grad_cam(): Grad-CAM heat-maps
  - integrated_gradients(): IG attribution with completeness check
  - occlusion_sensitivity(): perturbation-based attribution
  - shap_explanations(): SHAP gradient-based explanations
  - lime_explanation(): LIME local surrogate explanations
  - attention_rollout(): ViT attention roll-out
  - attribution_faithfulness(): quantitative attribution evaluation
  - plot_attribution_grid(): visualisation helper
  - sample_for_explanation(): test image sampling
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE


def plot_attribution_grid(
    images_u8: np.ndarray, maps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
    class_names: Sequence[str], title: str, cmap: str = "jet", symmetric: bool = False,
) -> None:
    """Three rows: original image, attribution map, and the map overlaid on the image."""
    n = len(images_u8)
    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.2))
    for i in range(n):
        vmax = float(np.abs(maps[i]).max()) + 1e-9
        kw = {"vmin": -vmax, "vmax": vmax} if symmetric else {}
        axes[0, i].imshow(images_u8[i], cmap="gray")
        axes[0, i].set_title(f"true {class_names[y_true[i]]}\npred {class_names[y_pred[i]]}", fontsize=6)
        axes[1, i].imshow(maps[i], cmap=cmap, **kw)
        axes[2, i].imshow(images_u8[i], cmap="gray")
        axes[2, i].imshow(maps[i], cmap=cmap, alpha=0.5, **kw)
        for r in range(3):
            axes[r, i].axis("off")
    fig.suptitle(title, y=1.03)
    plt.show()


def grad_cam(
    model: nn.Module, x: torch.Tensor, target_layer: nn.Module = None,
    class_idx: torch.Tensor = None, device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grad-CAM heat-maps for a batch of images."""
    model = model.to(device).eval()
    layer = target_layer
    if layer is None:
        from models import ResNetSmall, CNN
        if isinstance(model, ResNetSmall):
            layer = model.stage3
        elif isinstance(model, CNN):
            layer = model.features
        else:
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not convs:
                raise TypeError(f"{type(model).__name__} has no conv layer")
            layer = convs[-1]
    activations: Dict[str, torch.Tensor] = {}
    def forward_hook(_module, _inp, out):
        activations["value"] = out
    handle = layer.register_forward_hook(forward_hook)
    try:
        x = x.to(device)
        logits = model(x)
        targets = logits.argmax(dim=1) if class_idx is None else class_idx.to(device)
        score = logits.gather(1, targets.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, activations["value"])[0]
    finally:
        handle.remove()
    acts = activations["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), targets.detach().cpu().numpy()


def integrated_gradients(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor = None,
    baseline_value: float = 0.0, steps: int = 64, device: torch.device = DEVICE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Integrated Gradients attribution with completeness check."""
    model = model.to(device).eval()
    x = x.to(device)
    baseline = torch.full_like(x, baseline_value)
    with torch.no_grad():
        logits_x = model(x)
        logits_b = model(baseline)
    target = logits_x.argmax(dim=1) if target is None else target.to(device)
    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        score = model(point).gather(1, target.view(-1, 1)).sum()
        total_grad += torch.autograd.grad(score, point)[0]
    attributions = (x - baseline) * (total_grad / steps)
    attr_sum = attributions.sum(dim=(1, 2, 3))
    delta_f = (logits_x.gather(1, target.view(-1, 1)) - logits_b.gather(1, target.view(-1, 1))).squeeze(1)
    rel_err = (attr_sum - delta_f).abs() / (delta_f.abs() + 1e-8)
    diagnostics = {
        "steps": float(steps),
        "mean |sum(attr)|": float(attr_sum.abs().mean()),
        "mean |F(x) - F(baseline)|": float(delta_f.abs().mean()),
        "mean relative completeness error": float(rel_err.mean()),
    }
    return attributions.squeeze(1).detach().cpu().numpy(), diagnostics


def occlusion_sensitivity(
    model: nn.Module, x: torch.Tensor, target: int, patch: int = 7, stride: int = 2,
    fill: float = 0.0, device: torch.device = DEVICE,
) -> np.ndarray:
    """Slide an occluding patch over one image and record the drop in target-class probability."""
    model = model.to(device).eval()
    x = x.to(device)
    with torch.no_grad():
        base_prob = torch.softmax(model(x.unsqueeze(0)), dim=1)[0, target].item()
    positions = [(r, c) for r in range(0, 28 - patch + 1, stride) for c in range(0, 28 - patch + 1, stride)]
    batch = x.unsqueeze(0).repeat(len(positions), 1, 1, 1).clone()
    for i, (r, c) in enumerate(positions):
        batch[i, :, r:r + patch, c:c + patch] = fill
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[:, target].cpu().numpy()
    heat = np.zeros((28, 28), dtype=np.float64)
    counts = np.zeros((28, 28), dtype=np.float64)
    for (r, c), p in zip(positions, probs):
        heat[r:r + patch, c:c + patch] += base_prob - p
        counts[r:r + patch, c:c + patch] += 1
    return heat / np.maximum(counts, 1)


def shap_explanations(
    model: nn.Module, background: torch.Tensor, x: torch.Tensor, device: torch.device = DEVICE,
):
    """SHAP values via shap.GradientExplainer."""
    import shap
    try:
        explainer = shap.GradientExplainer(model.to(device).eval(), background.to(device))
        values = explainer.shap_values(x.to(device))
    except Exception as exc:
        print(f"[SHAP unavailable: {exc}]")
        return None
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[..., None]
    return arr


def attention_rollout(model, x: torch.Tensor, device: torch.device = DEVICE) -> np.ndarray:
    """Attention roll-out for Vision Transformer."""
    model = model.to(device).eval()
    maps = model.attention_maps(x.to(device))
    n_tokens = maps[0].shape[-1]
    eye = torch.eye(n_tokens, device=device).unsqueeze(0)
    joint = eye.repeat(x.shape[0], 1, 1)
    for attn in maps:
        a = attn.mean(dim=1)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        joint = a @ joint
    cls_to_patches = joint[:, 0, 1:]
    side = int(round(cls_to_patches.shape[1] ** 0.5))
    grid = cls_to_patches.reshape(-1, 1, side, side)
    grid = grid - grid.amin(dim=(2, 3), keepdim=True)
    grid = grid / (grid.amax(dim=(2, 3), keepdim=True) + 1e-8)
    up = F.interpolate(grid, size=(28, 28), mode="bilinear", align_corners=False)
    return up.squeeze(1).detach().cpu().numpy()


def attribution_faithfulness(
    model: nn.Module, test_ds, X_test_np: np.ndarray, y_test_np: np.ndarray,
    fisher_map: np.ndarray, class_names: Sequence[str],
    num_classes: int = 10, n_per_class: int = 8, steps: int = 32, seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Aggregate |Integrated-Gradients| maps per class and score against references."""
    rng = np.random.default_rng(seed)
    rows, class_maps = [], []
    for c in range(num_classes):
        idx = rng.choice(np.flatnonzero(y_test_np == c), size=n_per_class, replace=False)
        xb = test_ds.tensors[0][idx]
        attr, _ = integrated_gradients(model, xb, steps=steps)
        mean_map = np.abs(attr).mean(axis=0)
        class_maps.append(mean_map)
        garment = (X_test_np[idx].mean(axis=0) > 20)
        mass_on = float(np.abs(mean_map)[garment].sum() / (np.abs(mean_map).sum() + 1e-12) * 100)
        corr = float(np.corrcoef(mean_map.ravel(), fisher_map)[0, 1])
        rows.append({"class": class_names[c], "mass on garment %": mass_on, "corr with Fisher map": corr})
    return pd.DataFrame(rows).set_index("class").round(3), np.stack(class_maps)


def sample_for_explanation(
    test_ds, X_test_np: np.ndarray, y_test_np: np.ndarray,
    num_classes: int = 10, n_per_class: int = 1, seed: int = 42,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """One (or more) test image(s) per class."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y_test_np == c), size=n_per_class, replace=False)
        for c in range(num_classes)
    ])
    return test_ds.tensors[0][idx], X_test_np[idx], y_test_np[idx]
