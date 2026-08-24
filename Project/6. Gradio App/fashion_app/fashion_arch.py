"""
fashion_arch.py
================================================================================
Architectures, class names and preprocessing constants for the Fashion-MNIST
Gradio inference app.

The class definitions below are **verbatim copies** of the model classes defined
in the notebook `DL-Fashion-MNIST.ipynb` (Sections 4.1, 4.10 and 4.11), so that
the `*.pt` checkpoints exported by the notebook
(`artifacts/models/dl/<slug>.pt`, saved with `model.state_dict()`) can be
rebuilt and loaded here key-for-key.

Artifact layout produced by the notebook (see notebook Section 3.1b):
    artifacts/
      models/
        ml/        *.joblib  + *.json sidecar   (classical / boosting models)
        dl/        *.pt      + *.json sidecar   (deep models, state_dict)
        ensemble/  *.joblib  + *.json sidecar   (combiners: members/weights/meta)
      final_leaderboard.csv, run_summary.json, ...

Sidecar JSON for a deep model contains:
    {"model": "<display name>", "family": "Deep Learning", "accuracy": ...,
     "arch_class": "MLP" | "CNN" | "ResNetSmall" | "VisionTransformer",
     "arch_kwargs": {...}, "format": "state_dict", ...}
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# Dataset constants (notebook Section 1.2, CFG)
# --------------------------------------------------------------------------------
CLASS_NAMES: tuple = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
NUM_CLASSES = len(CLASS_NAMES)

# Training-split normalisation statistics (notebook Section 1.6: computed once on
# the training split of the cleaned 60k file -> mean=0.2860, std=0.3530).
# Every deep model in the notebook was trained on tensors normalised with these
# constants, so inference MUST use exactly the same values.
PIXEL_MEAN = 0.2860
PIXEL_STD = 0.3530


def normalize_batch(images_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (N,1,28,28) or (N,28,28) -> normalised float32 (N,1,28,28)."""
    if images_u8.dim() == 3:
        images_u8 = images_u8.unsqueeze(1)
    return images_u8.float().div_(255.0).sub_(PIXEL_MEAN).div_(PIXEL_STD)


# --------------------------------------------------------------------------------
# Deep architectures (verbatim from the notebook)
# --------------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully-connected baseline: no spatial prior, used as the control for the CNN comparison."""

    def __init__(self, in_features: int = 784, hidden: Sequence[int] = (512, 256),
                 num_classes: int = 10, p_drop: float = 0.3) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def conv_block(in_ch: int, out_ch: int, p_drop: float, pool: bool = True) -> nn.Sequential:
    """Two 3x3 conv+BN+ReLU layers, optionally followed by 2x2 max-pooling and dropout."""
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    """Compact VGG-style CNN with BatchNorm, dropout and global average pooling (~300k parameters)."""

    def __init__(self, num_classes: int = 10, p_drop: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, p_drop * 0.8),          # 28x28 -> 14x14
            conv_block(32, 64, p_drop),               # 14x14 -> 7x7
            conv_block(64, 128, p_drop, pool=False),  # 7x7 (kept, then globally pooled)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Basic two-convolution residual block with BatchNorm and an optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, p_drop: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNetSmall(nn.Module):
    """Compact 3-stage residual network sized for 28x28 grayscale images (~0.7 M parameters at width 32)."""

    def __init__(self, num_classes: int = 10, width: int = 32, p_drop: float = 0.3) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(ResidualBlock(w, w, 1, p_drop * 0.3), ResidualBlock(w, w, 1, p_drop * 0.3))
        self.stage2 = nn.Sequential(
            ResidualBlock(w, 2 * w, 2, p_drop * 0.5), ResidualBlock(2 * w, 2 * w, 1, p_drop * 0.5)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(2 * w, 4 * w, 2, p_drop * 0.7), ResidualBlock(4 * w, 4 * w, 1, p_drop * 0.7)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop), nn.Linear(4 * w, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


class PatchEmbedding(nn.Module):
    """Split a 28x28 image into non-overlapping patches and linearly project each one to `dim`."""

    def __init__(self, img_size: int = 28, patch: int = 7, in_ch: int = 1, dim: int = 128) -> None:
        super().__init__()
        if img_size % patch != 0:
            raise ValueError(f"image size {img_size} must be divisible by patch size {patch}")
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                     # (B, dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (B, n_patches, dim)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with a fused QKV projection."""

    def __init__(self, dim: int, heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"embedding dim {dim} must be divisible by the number of heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        out = self.proj_drop(self.proj(out))
        return (out, attn) if return_attention else out


class DropPath(nn.Module):
    """Stochastic depth: randomly drop the residual branch of a block for whole samples during training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], *([1] * (x.ndim - 1)), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + DropPath(Attn(LN(x))) then x + DropPath(MLP(LN(x)))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0,
                 drop: float = 0.1, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim), nn.Dropout(drop)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    """Vision Transformer tailored to 28x28x1 inputs (16 patch tokens + 1 CLS token)."""

    def __init__(
        self, img_size: int = 28, patch: int = 7, num_classes: int = 10, dim: int = 128,
        depth: int = 6, heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.1, drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch, 1, dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(drop)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, drop, dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


ARCH_REGISTRY = {
    "MLP": MLP,
    "CNN": CNN,
    "ResNetSmall": ResNetSmall,
    "VisionTransformer": VisionTransformer,
}


# --------------------------------------------------------------------------------
# Rebuild a model from a saved state_dict, with shape-based inference
# --------------------------------------------------------------------------------
def infer_arch_and_kwargs(state_dict: dict, arch_class: str | None = None,
                          arch_kwargs: dict | None = None) -> tuple:
    """Return (class_name, kwargs) for a saved state_dict.

    Priority: explicit `arch_class`/`arch_kwargs` from the sidecar JSON; anything
    missing is inferred from the tensor shapes in the state_dict itself, so even
    a checkpoint without a sidecar can be rebuilt.
    """
    keys = set(state_dict.keys())
    kwargs = dict(arch_kwargs or {})

    if arch_class and arch_class in ARCH_REGISTRY:
        name = arch_class
    elif any(k.startswith("patch_embed.") for k in keys):
        name = "VisionTransformer"
    elif any(k.startswith("stem.") for k in keys):
        name = "ResNetSmall"
    elif any(k.startswith("features.") for k in keys):
        name = "CNN"
    else:
        name = "MLP"

    if name == "MLP" and not kwargs:
        # hidden sizes from the 2-D linears, e.g. net.1 (784->512), net.4 (512->256), net.7 (256->10)
        linears = [(k, tuple(v.shape)) for k, v in state_dict.items()
                   if k.endswith(".weight") and v.dim() == 2]
        if linears and linears[0][1][1] == 784:
            kwargs["hidden"] = tuple(int(s[0]) for _, s in linears[:-1])   # drop the final classifier
    if name == "ResNetSmall":
        if "width" not in kwargs and "stem.0.weight" in state_dict:
            kwargs["width"] = int(state_dict["stem.0.weight"].shape[0])
    if name == "VisionTransformer":
        if "dim" not in kwargs and "pos_embed" in state_dict:
            kwargs["dim"] = int(state_dict["pos_embed"].shape[-1])
        if "depth" not in kwargs:
            kwargs["depth"] = len({k.split(".")[1] for k in keys
                                   if k.startswith("blocks.") and k.split(".")[1].isdigit()})
        if "heads" not in kwargs and kwargs.get("dim", 128) % 4 == 0:
            kwargs.setdefault("heads", 4)

    kwargs.setdefault("num_classes", NUM_CLASSES)
    return name, kwargs


def build_torch_model(state_dict: dict, arch_class: str | None = None,
                      arch_kwargs: dict | None = None) -> nn.Module:
    """Instantiate the right architecture and load the checkpoint weights (strict)."""
    name, kwargs = infer_arch_and_kwargs(state_dict, arch_class, arch_kwargs)
    model = ARCH_REGISTRY[name](**kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def torch_probabilities(model: nn.Module, x_norm: torch.Tensor, tta_mirror: bool = False) -> torch.Tensor:
    """softmax probabilities for a batch of normalised images (optionally mirror-averaged, notebook cfgx.tta)."""
    model.eval()
    p = torch.softmax(model(x_norm), dim=1)
    if tta_mirror:
        p = 0.5 * (p + torch.softmax(model(torch.flip(x_norm, dims=[3])), dim=1))
    return p
