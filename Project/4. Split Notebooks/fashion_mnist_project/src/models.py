"""
models.py – Neural network architecture definitions.

Contains:
  - MLP: Multi-layer perceptron
  - CNN: VGG-style CNN with global average pooling
  - ResNetSmall: Residual CNN for 28x28
  - VisionTransformer: ViT adapted for 28x28 grayscale
  - ConvAutoencoder: Compact conv autoencoder for anomaly detection
  - Augment: Data augmentation transform
  - count_parameters(): utility
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Augment(nn.Module):
    """Simple data augmentation: random horizontal flip + random translation."""

    def __init__(self, p_flip: float = 0.5, max_shift: int = 2):
        super().__init__()
        self.p_flip = p_flip
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p_flip > 0 and torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[-1])
        if self.max_shift > 0:
            shifts = torch.randint(-self.max_shift, self.max_shift + 1, (2,))
            x = torch.roll(x, shifts=(int(shifts[0]), int(shifts[1])), dims=(-2, -1))
        return x


class MLP(nn.Module):
    """Multi-layer perceptron for Fashion-MNIST."""

    def __init__(self, num_classes: int = 10, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """VGG-style CNN with global average pooling for Fashion-MNIST."""

    def __init__(self, num_classes: int = 10, p_drop: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(p_drop),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(p_drop),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _ResBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetSmall(nn.Module):
    """Small residual CNN for 28x28 inputs."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(w, w, 2, stride=1)
        self.stage2 = self._make_stage(w, w * 2, 2, stride=2)
        self.stage3 = self._make_stage(w * 2, w * 4, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(w * 4, num_classes)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [_ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(_ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class VisionTransformer(nn.Module):
    """Vision Transformer adapted for 28x28 grayscale inputs."""

    def __init__(self, patch: int = 7, num_classes: int = 10, dim: int = 128,
                 depth: int = 6, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        n_patches = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embed
        z = self.drop(z)
        z = self.encoder(z)
        z = self.norm(z)
        return self.head(z[:, 0])

    def attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return per-layer attention matrices for attention roll-out."""
        B = x.shape[0]
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embed
        z = self.drop(z)
        maps = []
        for layer in self.encoder.layers:
            z, attn = layer.self_attn(layer.norm1(z), layer.norm1(z), layer.norm1(z), need_weights=True)
            maps.append(attn)
            z = layer.norm2(z + layer.linear2(layer.dropout(layer.activation(layer.linear1(z)))))
        return maps


class ConvAutoencoder(nn.Module):
    """Compact conv autoencoder for 28x28x1 images."""

    def __init__(self, latent: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent),
        )
        self.decoder_fc = nn.Sequential(nn.Linear(latent, 32 * 7 * 7), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        h = self.decoder_fc(z).view(-1, 32, 7, 7)
        return self.decoder(h)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
