"""
config.py – Configuration dataclasses and seeding utilities.

Contains:
  - CFG: main configuration dataclass (v1 hyper-parameters)
  - CFGX: extended configuration dataclass (v2 additions)
  - set_seed(): reproducibility helper
  - DEVICE: torch device selection
  - has_module(): optional-dependency probe
  - Global constants (AVAILABLE, HAS_*, OPTIONAL_DEPS)
"""
from __future__ import annotations

import json
import os
import random
import importlib
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class CFG:
    """Single source of truth for every experimental hyper-parameter."""

    seed: int = 42
    kaggle_dataset: str = "zalando-research/fashionmnist"
    train_csv: str = "fashion-mnist_train.csv"
    test_csv: str = "fashion-mnist_test.csv"
    val_fraction: float = 0.10
    sk_train_subset: int = 12_000
    sk_eval_on_full_test: bool = True
    run_rbf_svm: bool = True
    batch_size: int = 256
    mlp_epochs: int = 20
    cnn_epochs: int = 25
    lr: float = 3e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    dropout: float = 0.30
    use_amp: bool = True
    augment: bool = True
    num_workers: int = 2
    artifacts_dir: str = "artifacts"
    class_names: Tuple[str, ...] = (
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass
class CFGX:
    """Configuration for everything added in the upgraded edition."""

    fast_mode: bool = False
    leakage_policy: str = "drop_from_train"
    drop_train_duplicates: bool = False
    eda_sample: int = 8_000
    corr_grid: int = 14
    embed_sample: int = 4_000
    pca_pre_components: int = 50
    tsne_perplexities: Tuple[int, ...] = (5, 30, 50)
    tsne_iter: int = 750
    umap_neighbors: Tuple[int, ...] = (5, 15, 50)
    umap_min_dist: float = 0.1
    run_3d_embeddings: bool = True
    cluster_k_grid: Tuple[int, ...] = (5, 8, 10, 12, 15)
    cluster_sample: int = 6_000
    iforest_contamination: float = 0.01
    iforest_estimators: int = 300
    autoencoder_epochs: int = 8
    autoencoder_latent: int = 32
    ae_batch_size: int = 256
    run_boosting: bool = True
    boost_pca_components: int = 80
    boost_train_subset: int = 20_000
    xgb_estimators: int = 600
    lgbm_estimators: int = 800
    cat_iterations: int = 800
    boost_use_gpu: bool = True
    run_tuning: bool = True
    optuna_trials: int = 25
    optuna_timeout_s: int = 600
    tuning_cv_folds: int = 3
    tuning_subset: int = 8_000
    run_resnet: bool = True
    run_vit: bool = True
    resnet_epochs: int = 30
    resnet_width: int = 32
    vit_epochs: int = 35
    vit_patch: int = 7
    vit_dim: int = 128
    vit_depth: int = 6
    vit_heads: int = 4
    vit_mlp_ratio: float = 2.0
    vit_dropout: float = 0.10
    vit_lr: float = 1e-3
    early_stopping_patience: int = 8
    min_delta: float = 1e-4
    weight_search_iters: int = 4_000
    tta: bool = True
    run_xai: bool = True
    ig_steps: int = 64
    occlusion_patch: int = 7
    occlusion_stride: int = 2
    shap_background: int = 64
    shap_samples: int = 8
    run_lime: bool = True
    bootstrap_iters: int = 2_000
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.fast_mode:
            self.eda_sample = 2_000
            self.embed_sample = 1_500
            self.tsne_perplexities = (30,)
            self.tsne_iter = 300
            self.umap_neighbors = (15,)
            self.run_3d_embeddings = False
            self.cluster_sample = 2_000
            self.autoencoder_epochs = 2
            self.boost_train_subset = 6_000
            self.xgb_estimators = 150
            self.lgbm_estimators = 200
            self.cat_iterations = 200
            self.optuna_trials = 5
            self.optuna_timeout_s = 120
            self.tuning_subset = 3_000
            self.resnet_epochs = 3
            self.vit_epochs = 3
            self.weight_search_iters = 500
            self.ig_steps = 16
            self.shap_background = 16
            self.bootstrap_iters = 300


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG used in this project so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def has_module(name: str) -> bool:
    """True if `name` can be imported in this runtime."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIONAL_DEPS: Dict[str, str] = {
    "xgboost": "3.7 XGBoost baseline",
    "lightgbm": "3.7 LightGBM baseline",
    "catboost": "3.7 CatBoost baseline",
    "optuna": "3.8 automated hyper-parameter search",
    "umap": "2.8 UMAP manifold learning (package name: umap-learn)",
    "shap": "6.4 SHAP explanations",
    "lime": "6.5 LIME explanations",
    "statsmodels": "8.2 exact McNemar test",
}

AVAILABLE: Dict[str, bool] = {name: has_module(name) for name in OPTIONAL_DEPS}

HAS_XGB = AVAILABLE["xgboost"]
HAS_LGBM = AVAILABLE["lightgbm"]
HAS_CATBOOST = AVAILABLE["catboost"]
HAS_OPTUNA = AVAILABLE["optuna"]
HAS_UMAP = AVAILABLE["umap"]
HAS_SHAP = AVAILABLE["shap"]
HAS_LIME = AVAILABLE["lime"]
HAS_STATSMODELS = AVAILABLE["statsmodels"]
