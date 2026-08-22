# python 3
"""
Restructure the monolithic DL-Fashion-MNIST notebook into:
  - Multiple .ipynb notebooks (one per major section)
  - Separate .py source modules with reusable code
"""

import json
import re
import os
import sys

INPUT_FILE = "/home/user/uploads/DL-Fashion-MNIST_extracted.txt"
NOTEBOOKS_DIR = "/home/user/fashion_mnist_project/notebooks"
SRC_DIR = "/home/user/fashion_mnist_project/src"

# ---------------------------------------------------------------------------
# 1. Parse the text file into a list of cells
# ---------------------------------------------------------------------------

def parse_cells(path: str) -> list[dict]:
    """Parse the extracted notebook text into a list of {index, kind, content} dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on cell headers
    cell_pattern = re.compile(r'^\[Cell #(\d+) — (CODE|MARKDOWN)\]\n-+\n', re.MULTILINE)
    matches = list(cell_pattern.finditer(text))
    
    cells = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        kind = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].rstrip('\n')
        cells.append({"index": idx, "kind": kind, "content": content})
    
    return cells

# ---------------------------------------------------------------------------
# 2. Build .ipynb notebook JSON
# ---------------------------------------------------------------------------

def make_notebook(cells_data: list[dict], title: str = "") -> dict:
    """Create a valid .ipynb JSON structure from a list of cells."""
    nb_cells = []
    for c in cells_data:
        if c["kind"] == "MARKDOWN":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": c["content"].split("\n"),
            })
            # Fix: each line needs \n except possibly last
            lines = c["content"].split("\n")
            nb_cells[-1]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        else:  # CODE
            lines = c["content"].split("\n")
            nb_cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else []),
                "outputs": [],
                "execution_count": None,
            })
    
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": nb_cells,
    }


def save_notebook(nb: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Written: {path}  ({len(nb['cells'])} cells)")


# ---------------------------------------------------------------------------
# 3. Identify cell ranges for each notebook section
# ---------------------------------------------------------------------------

# Based on the analysis of all 223 cells:
# Section mapping (cell index ranges):
SECTION_RANGES = {
    "00_introduction":        (1,   5),   # Cells 1-5: title, problem statement, scope
    "01_data_loading":        (6,  35),   # Cells 6-35: env, imports, config, download, cleaning, splitting
    "02_eda_basic":           (36, 53),   # Cells 36-53: class dist, samples, pixel stats, means, PCA
    "03_eda_advanced":        (54, 82),   # Cells 54-82: advanced EDA (intensity, metrics, manifold, clustering, outliers)
    "04_classical_ml":        (83, 131),  # Cells 83-131: classical ML baselines + boosting + tuning
    "05_deep_learning":       (132, 156),  # Cells 132-156: deep learning (MLP, CNN, ResNet, ViT)
    "06_ensembling":          (157, 179),  # Cells 157-179: ensembling
    "07_explainability":      (180, 189),  # Cells 180-189: explainability
    "08_unit_tests":          (190, 198),  # Cells 190-198: unit tests
    "09_statistical_validity": (199, 206), # Cells 199-206: statistical validity
    "10_comparison_conclusion": (207, 223),# Cells 207-223: comparison, discussion, refs, appendix
}


# ---------------------------------------------------------------------------
# 4. Main logic
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Parsing cells from extracted notebook...")
    cells = parse_cells(INPUT_FILE)
    print(f"  Found {len(cells)} cells total")
    
    # Build an index by cell number
    cell_by_idx = {c["index"]: c for c in cells}
    
    # -----------------------------------------------------------------------
    # Step A: Split into notebooks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Splitting into notebooks...")
    
    for nb_name, (start, end) in SECTION_RANGES.items():
        nb_cells = [cell_by_idx[i] for i in range(start, end + 1) if i in cell_by_idx]
        
        # For the first notebook, prepend a system-path cell
        if nb_name == "00_introduction":
            nb_cells_with_imports = nb_cells
        else:
            # Add sys.path setup as first code cell
            sys_path_cell = {
                "index": -1,
                "kind": "CODE",
                "content": (
                    "import sys, os\n"
                    "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\n"
                    "# Ensure the src directory with extracted modules is on the path\n"
                    "print('sys.path[0]:', sys.path[0])"
                ),
            }
            # Add it after the first markdown cell (title)
            insert_pos = 0
            for i, c in enumerate(nb_cells):
                if c["kind"] == "CODE":
                    insert_pos = i
                    break
                insert_pos = i + 1
            nb_cells_with_imports = nb_cells[:insert_pos] + [sys_path_cell] + nb_cells[insert_pos:]
        
        nb = make_notebook(nb_cells_with_imports, title=nb_name)
        save_notebook(nb, os.path.join(NOTEBOOKS_DIR, f"{nb_name}.ipynb"))
    
    print("\n" + "=" * 72)
    print("Done! Notebooks created in:", NOTEBOOKS_DIR)
    
    # -----------------------------------------------------------------------
    # Step B: Extract reusable code into .py modules
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Extracting reusable code into .py modules...")
    extract_modules(cells, cell_by_idx)
    
    print("\n" + "=" * 72)
    print("ALL DONE.")


# ---------------------------------------------------------------------------
# 5. Extract code into .py modules
# ---------------------------------------------------------------------------

def extract_modules(cells, cell_by_idx):
    """Extract function and class definitions from cells into organized .py modules."""
    
    # We'll collect all code cells and extract top-level defs
    all_code = ""
    code_cells = [c for c in cells if c["kind"] == "CODE"]
    for c in code_cells:
        all_code += c["content"] + "\n\n"
    
    # ---- config.py ----
    config_py = '''"""
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
'''
    
    write_module("config.py", config_py)
    
    # ---- data_loading.py ----
    data_loading_py = '''"""
data_loading.py – Dataset acquisition, integrity checking, cleaning and splitting.

Contains:
  - run_shell(): environment inspection helper
  - configure_kaggle_credentials(): Kaggle auth from Colab Secrets
  - find_csv(): locate CSV in nested dirs
  - load_fashion_csv(): parse Fashion-MNIST CSV efficiently
  - image_hashes(): hash-based duplicate detection
  - integrity_report(): 11-point data quality check
  - find_cross_split_duplicates(): byte-exact leakage detection
  - find_within_split_duplicates(): within-split duplicate groups
  - plot_leaked_images(): visualise leaked pairs
  - drop_rows(), apply_leakage_policy(), deduplicate_within()
  - leakage_verification_report(): post-cleaning verification
  - to_tensor_dataset(): uint8 -> normalised TensorDataset
  - make_loader(): DataLoader factory
  - make_flat_arrays(): flatten for scikit-learn
  - make_eda_sample(): stratified EDA working sample
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def run_shell(cmd: str) -> str:
    """Run a shell command and return its stdout, or an explanatory message."""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:
        return f"[not available: {exc}]"


def configure_kaggle_credentials() -> bool:
    """Populate KAGGLE_USERNAME / KAGGLE_KEY from Colab Secrets when available."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    try:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        return True
    except Exception:
        return False


def find_csv(data_dir: Path, filename: str) -> Path:
    """Locate a CSV inside the downloaded dataset directory."""
    matches = sorted(data_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {data_dir}")
    return matches[0]


def load_fashion_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Fashion-MNIST CSV into (images_uint8[N,28,28], labels_int64[N])."""
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else df.columns[0]
    labels = df[label_col].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=[label_col]).to_numpy(dtype=np.uint8)
    del df
    images = pixels.reshape(-1, 28, 28)
    return images, labels


def image_hashes(images: np.ndarray) -> np.ndarray:
    """Return a 1-D array of hashes, one per image."""
    flat = np.ascontiguousarray(images.reshape(len(images), -1))
    return np.array([hash(row.tobytes()) for row in flat], dtype=np.int64)


def integrity_report(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, num_classes: int = 10
) -> pd.DataFrame:
    """Run all data-quality checks and return them as a tidy pass/fail table."""
    tr_hash, te_hash = image_hashes(X_tr), image_hashes(X_te)
    tr_counts = np.bincount(y_tr, minlength=num_classes)
    te_counts = np.bincount(y_te, minlength=num_classes)
    n_degenerate = int((np.ptp(X_tr.reshape(len(X_tr), -1), axis=1) == 0).sum())

    checks = [
        ("Train shape is (60000, 28, 28)", X_tr.shape == (60000, 28, 28), str(X_tr.shape)),
        ("Test shape is (10000, 28, 28)", X_te.shape == (10000, 28, 28), str(X_te.shape)),
        ("No NaN in train", not np.isnan(X_tr.astype(np.float32)).any(), "uint8 cannot hold NaN"),
        ("Pixels within [0, 255]", int(X_tr.min()) >= 0 and int(X_tr.max()) <= 255,
         f"min={X_tr.min()}, max={X_tr.max()}"),
        ("Labels within [0, 9]", set(np.unique(y_tr)) <= set(range(num_classes)),
         f"unique={sorted(np.unique(y_tr).tolist())}"),
        ("Train perfectly balanced (6000/class)", bool((tr_counts == 6000).all()), str(tr_counts.tolist())),
        ("Test perfectly balanced (1000/class)", bool((te_counts == 1000).all()), str(te_counts.tolist())),
        ("Duplicate images inside train (informational)", True,
         f"{len(tr_hash) - len(np.unique(tr_hash))} exact duplicates"),
        ("Duplicate images inside test (informational)", True,
         f"{len(te_hash) - len(np.unique(te_hash))} exact duplicates"),
        ("No train/test leakage", len(np.intersect1d(tr_hash, te_hash)) == 0,
         f"{len(np.intersect1d(tr_hash, te_hash))} images shared between splits"),
        ("No constant (all-same-pixel) images", int(n_degenerate) == 0, f"{int(n_degenerate)} degenerate images"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def find_cross_split_duplicates(X_a: np.ndarray, X_b: np.ndarray) -> List[Tuple[int, int]]:
    """Return every (index_in_A, index_in_B) pair of byte-identical images."""
    flat_a = np.ascontiguousarray(X_a.reshape(len(X_a), -1))
    flat_b = np.ascontiguousarray(X_b.reshape(len(X_b), -1))
    index: Dict[bytes, List[int]] = {}
    for i in range(len(flat_a)):
        index.setdefault(flat_a[i].tobytes(), []).append(i)
    pairs: List[Tuple[int, int]] = []
    for j in range(len(flat_b)):
        for i in index.get(flat_b[j].tobytes(), ()):
            pairs.append((i, j))
    return pairs


def find_within_split_duplicates(X: np.ndarray) -> Dict[bytes, List[int]]:
    """Group the indices of byte-identical images inside a single split."""
    flat = np.ascontiguousarray(X.reshape(len(X), -1))
    groups: Dict[bytes, List[int]] = {}
    for i in range(len(flat)):
        groups.setdefault(flat[i].tobytes(), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def plot_leaked_images(
    pairs: Sequence[Tuple[int, int]], X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, class_names: Sequence[str], max_show: int = 10,
) -> None:
    """Show the training copy, the test copy and their pixel difference."""
    sel = list(pairs)[:max_show]
    if not sel:
        print("nothing to plot")
        return
    fig, axes = plt.subplots(3, len(sel), figsize=(1.45 * len(sel), 5.0), squeeze=False)
    for k, (i, j) in enumerate(sel):
        diff = X_train[i].astype(np.int16) - X_test[j].astype(np.int16)
        axes[0][k].imshow(X_train[i], cmap="gray", vmin=0, vmax=255)
        axes[0][k].set_title(f"train #{i}\\n{class_names[y_train[i]]}", fontsize=6)
        axes[1][k].imshow(X_test[j], cmap="gray", vmin=0, vmax=255)
        axes[1][k].set_title(f"test #{j}\\n{class_names[y_test[j]]}", fontsize=6)
        axes[2][k].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        axes[2][k].set_title(f"|diff| = {int(np.abs(diff).max())}", fontsize=6)
        for r in range(3):
            axes[r][k].axis("off")
    fig.suptitle("Leaked images: training copy (top), test copy (middle), pixel difference (bottom)", y=1.04)
    plt.show()


def drop_rows(X: np.ndarray, y: np.ndarray, indices: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return copies of X/y with the given row indices removed."""
    mask = np.ones(len(X), dtype=bool)
    idx = np.fromiter(sorted(set(int(i) for i in indices)), dtype=np.int64, count=-1)
    if len(idx):
        mask[idx] = False
    return X[mask], y[mask]


def apply_leakage_policy(
    policy: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Remove the cross-split duplicates according to policy."""
    train_hits = sorted({i for i, _ in pairs})
    test_hits = sorted({j for _, j in pairs})
    info: Dict[str, object] = {
        "policy": policy, "pairs found": len(pairs),
        "train rows removed": 0, "test rows removed": 0,
    }
    if policy == "drop_from_train":
        X_train, y_train = drop_rows(X_train, y_train, train_hits)
        info["train rows removed"] = len(train_hits)
    elif policy == "drop_from_test":
        X_test, y_test = drop_rows(X_test, y_test, test_hits)
        info["test rows removed"] = len(test_hits)
    elif policy == "keep":
        print("WARNING: leakage_policy = 'keep'")
    else:
        raise ValueError(f"unknown leakage policy '{policy}'")
    return X_train, y_train, X_test, y_test, info


def deduplicate_within(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Keep only the first copy of every byte-identical image inside one split."""
    groups = find_within_split_duplicates(X)
    redundant = [i for idxs in groups.values() for i in idxs[1:]]
    X2, y2 = drop_rows(X, y, redundant)
    return X2, y2, len(redundant)


def leakage_verification_report(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, num_classes: int = 10
) -> pd.DataFrame:
    """Re-run the checks that motivated the cleaning."""
    pairs_after = find_cross_split_duplicates(X_train, X_test)
    hash_overlap = len(np.intersect1d(image_hashes(X_train), image_hashes(X_test)))
    tr_counts = np.bincount(y_train, minlength=num_classes)
    te_counts = np.bincount(y_test, minlength=num_classes)
    shares = tr_counts / tr_counts.sum()
    checks = [
        ("No byte-exact train/test duplicates", len(pairs_after) == 0, f"{len(pairs_after)} pairs remain"),
        ("No hash-level train/test overlap", hash_overlap == 0, f"{hash_overlap} hashes shared"),
        ("Official test set still has 10,000 rows", len(X_test) == 10_000, f"{len(X_test)} rows"),
        ("Test set still perfectly balanced", bool((te_counts == 1_000).all()), str(te_counts.tolist())),
        ("Training set still ~balanced", bool(np.abs(shares - 1/num_classes).max() < 0.002),
         f"max deviation {100 * float(np.abs(shares - 1/num_classes).max()):.3f} pp"),
        ("Training set lost < 0.5 % of its rows", (60_000 - len(X_train)) / 60_000 < 0.005,
         f"{60_000 - len(X_train)} rows removed"),
        ("Labels still within [0, 9]", set(np.unique(y_train)) <= set(range(num_classes)),
         f"unique={sorted(np.unique(y_train).tolist())}"),
        ("Pixels still within [0, 255]", int(X_train.min()) >= 0 and int(X_train.max()) <= 255,
         f"min={int(X_train.min())}, max={int(X_train.max())}"),
        ("dtypes unchanged", X_train.dtype == np.uint8 and y_train.dtype == np.int64,
         f"{X_train.dtype} / {y_train.dtype}"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def to_tensor_dataset(
    images: np.ndarray, labels: np.ndarray, mean: float, std: float,
) -> TensorDataset:
    """Convert uint8 HxW images + int labels into a normalised float32 TensorDataset."""
    x = torch.from_numpy(images).float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)
    y = torch.from_numpy(labels).long()
    return TensorDataset(x, y)


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    """Build a DataLoader."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=num_workers > 0,
    )


def make_flat_arrays(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten (N,28,28) uint8 images into (N,784) float32 in [0,1]."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0, labels


def make_eda_sample(
    images: np.ndarray, labels: np.ndarray, n: int, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a stratified working sample for heavy EDA routines."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, n // len(classes))
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=min(per_class, int((labels == c).sum())), replace=False)
        for c in classes
    ])
    rng.shuffle(idx)
    imgs, labs = images[idx], labels[idx]
    flat = imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0
    return imgs, labs, flat
'''
    
    write_module("data_loading.py", data_loading_py)
    
    # ---- eda.py ----
    eda_py = '''"""
eda.py – Exploratory Data Analysis functions (basic and advanced).

Contains all plotting and analysis functions from Sections 2.1–2.10:
  - plot_class_distribution()
  - plot_samples_per_class()
  - pixel_intensity_overview()
  - per_class_intensity_table()
  - plot_class_means_and_variance()
  - plot_class_similarity()
  - plot_pca_scatter()
  - plot_global_intensity()
  - plot_per_class_intensity()
  - intensity_ks_matrix()
  - class_mean_variance_panels()
  - fisher_discriminability()
  - pixel_correlation_analysis()
  - pca_embedding_analysis()
  - embedding_quality(), register_embedding()
  - make_tsne(), tsne_perplexity_sweep()
  - plot_embedding_3d()
  - umap_sweep()
  - kmeans_grid()
  - cluster_composition()
  - isolation_forest_outliers()
  - autoencoder_outlier_report()
  - detector_agreement()
"""
from __future__ import annotations

import inspect
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


def plot_class_distribution(
    label_sets: Dict[str, np.ndarray], class_names: Sequence[str], figsize: Tuple[int, int] = (12, 4)
) -> pd.DataFrame:
    """Bar-plot the class distribution of several splits side by side."""
    counts = pd.DataFrame(
        {name: np.bincount(y, minlength=len(class_names)) for name, y in label_sets.items()},
        index=list(class_names),
    )
    ax = counts.plot(kind="bar", figsize=figsize, width=0.8, edgecolor="black", linewidth=0.4)
    ax.set_title("Fashion-MNIST class distribution (perfectly balanced by design)")
    ax.set_xlabel("class")
    ax.set_ylabel("number of images")
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fontsize=7, padding=1)
    plt.show()
    return counts


def plot_samples_per_class(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    n_per_class: int = 8, seed: int = 42,
) -> None:
    """Show a grid with one row per class and n_per_class random examples per row."""
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 1.05, n_classes * 1.15))
    for c in range(n_classes):
        idx = rng.choice(np.flatnonzero(labels == c), size=n_per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{c} {class_names[c]}", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Random samples per class (raw 28x28 grayscale)", y=1.005)
    plt.show()


def pixel_intensity_overview(images: np.ndarray, sample_size: int = 4_000, seed: int = 42) -> pd.Series:
    """Plot the global pixel histogram (log scale) and return summary statistics."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(sample_size, len(images)), replace=False)
    flat = images[idx].reshape(-1).astype(np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].hist(flat, bins=64, color="#4C72B0", edgecolor="black", linewidth=0.3)
    axes[0].set_yscale("log")
    axes[0].set_title("Pixel-value histogram (log-count)")
    axes[0].set_xlabel("pixel intensity [0-255]")
    axes[0].set_ylabel("count (log)")
    nonzero = flat[flat > 0]
    axes[1].hist(nonzero, bins=64, color="#DD8452", edgecolor="black", linewidth=0.3)
    axes[1].set_title("Non-zero pixels only (the garment itself)")
    axes[1].set_xlabel("pixel intensity [1-255]")
    axes[1].set_ylabel("count")
    plt.show()
    stats = pd.Series({
        "mean": flat.mean(), "std": flat.std(), "median": float(np.median(flat)),
        "min": flat.min(), "max": flat.max(),
        "% exactly 0 (background)": 100.0 * (flat == 0).mean(),
        "% saturated at 255": 100.0 * (flat == 255).mean(),
        "mean of non-zero pixels": nonzero.mean(),
    }, name="pixel statistics (raw 0-255)").round(3)
    return stats


def per_class_intensity_table(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Mean intensity, std and fraction of non-background pixels for each class."""
    flat = images.reshape(len(images), -1).astype(np.float32)
    rows = []
    for c, name in enumerate(class_names):
        sub = flat[labels == c]
        rows.append({
            "class": name, "mean intensity": sub.mean(), "std intensity": sub.std(),
            "ink coverage % (pixels > 20)": 100.0 * (sub > 20).mean(),
            "mean brightness of garment": sub[sub > 0].mean(),
        })
    return pd.DataFrame(rows).set_index("class").round(2)


def plot_class_means_and_variance(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> np.ndarray:
    """Plot the average image of every class plus the per-pixel std map."""
    flat = images.reshape(len(images), -1).astype(np.float32)
    means = np.stack([flat[labels == c].mean(axis=0) for c in range(len(class_names))])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for c, ax in enumerate(axes.ravel()):
        ax.imshow(means[c].reshape(28, 28), cmap="viridis")
        ax.set_title(f"{c}: {class_names[c]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Mean image per class", y=1.0)
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].imshow(flat.mean(axis=0).reshape(28, 28), cmap="viridis")
    axes[0].set_title("Global mean image")
    axes[0].axis("off")
    im1 = axes[1].imshow(flat.std(axis=0).reshape(28, 28), cmap="magma")
    axes[1].set_title("Per-pixel standard deviation")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    dead = (flat.max(axis=0) == 0).reshape(28, 28)
    axes[2].imshow(dead, cmap="gray_r")
    axes[2].set_title(f"Always-black pixels ({int(dead.sum())} of 784)")
    axes[2].axis("off")
    plt.show()
    return means


def plot_class_similarity(class_means: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    """Correlation heat-map between class-mean images."""
    corr = np.corrcoef(class_means)
    corr_df = pd.DataFrame(corr, index=list(class_names), columns=list(class_names))
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": "Pearson correlation of class-mean images"}, annot_kws={"size": 7})
    plt.title("Similarity between class templates")
    plt.show()
    return corr_df


def plot_pca_scatter(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    n_samples: int = 6_000, seed: int = 42
) -> PCA:
    """Fit a 2-component PCA on raw pixels and scatter-plot the classes."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(n_samples, len(images)), replace=False)
    flat = images[idx].reshape(len(idx), -1).astype(np.float32) / 255.0
    pca_full = PCA(n_components=100, random_state=seed).fit(flat)
    z = pca_full.transform(flat)[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    palette = sns.color_palette("tab10", len(class_names))
    for c, name in enumerate(class_names):
        m = labels[idx] == c
        axes[0].scatter(z[m, 0], z[m, 1], s=5, alpha=0.55, color=palette[c], label=name)
    axes[0].set_title("PCA of raw pixels (first 2 components)")
    axes[0].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[0].legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].plot(range(1, len(cum)+1), cum, marker="o", ms=3)
    axes[1].axhline(0.90, ls="--", c="red", lw=1, label="90% variance")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("number of principal components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].legend()
    plt.show()
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"{n90} principal components explain 90% of the pixel variance.")
    return pca_full


def plot_global_intensity(flat01: np.ndarray, bins: int = 64) -> pd.Series:
    """Four complementary views of the global pixel-intensity distribution."""
    vals = (flat01.reshape(-1) * 255.0).astype(np.float32)
    fg = vals[vals > 0]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    axes[0].hist(vals, bins=bins, color="#4C72B0", edgecolor="black", linewidth=0.3)
    axes[0].set_title("All pixels (linear count)")
    axes[0].set_xlabel("intensity")
    axes[1].hist(vals, bins=bins, color="#4C72B0", edgecolor="black", linewidth=0.3)
    axes[1].set_yscale("log")
    axes[1].set_title("All pixels (log count)")
    axes[1].set_xlabel("intensity")
    axes[2].hist(fg, bins=bins, color="#DD8452", edgecolor="black", linewidth=0.3, density=True)
    axes[2].set_title("Foreground only (> 0), density")
    axes[2].set_xlabel("intensity")
    order = np.sort(vals[np.random.default_rng(0).choice(len(vals), size=min(200_000, len(vals)), replace=False)])
    axes[3].plot(order, np.linspace(0, 1, len(order)), lw=1.6, color="#55A868")
    axes[3].set_title("Empirical CDF of all pixels")
    axes[3].set_xlabel("intensity")
    axes[3].set_ylabel("F(x)")
    fig.suptitle("Global pixel-intensity distribution", y=1.04)
    plt.show()
    return pd.Series({
        "mean": float(vals.mean()), "std": float(vals.std()),
        "skewness": float(sp_stats.skew(vals)), "excess kurtosis": float(sp_stats.kurtosis(vals)),
        "% background (== 0)": float(100.0 * (vals == 0).mean()),
        "% saturated (== 255)": float(100.0 * (vals == 255).mean()),
        "foreground mean": float(fg.mean()), "foreground std": float(fg.std()),
        "foreground median": float(np.median(fg)),
    }, name="global intensity statistics").round(3)


def plot_per_class_intensity(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str], bins: int = 48
) -> pd.DataFrame:
    """Per-class histograms, ECDFs and a violin plot of foreground intensity."""
    n_classes = len(class_names)
    palette = sns.color_palette("tab10", n_classes)
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.4), sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        v = (flat01[labels == c] * 255.0).reshape(-1)
        ax.hist(v[v > 0], bins=bins, color=palette[c], density=True, edgecolor="black", linewidth=0.2)
        ax.set_title(f"{c}: {class_names[c]}", fontsize=9)
        ax.set_xlabel("intensity")
    fig.suptitle("Foreground-intensity density per class", y=1.02)
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        v = (flat01[labels == c] * 255.0).reshape(-1)
        v = v[v > 0]
        v = np.sort(rng.choice(v, size=min(40_000, len(v)), replace=False))
        axes[0].plot(v, np.linspace(0, 1, len(v)), lw=1.3, color=palette[c], label=class_names[c])
    axes[0].set_title("ECDF of foreground intensity per class")
    axes[0].set_xlabel("intensity")
    axes[0].set_ylabel("F(x)")
    axes[0].legend(fontsize=7, ncol=2)
    per_image_df = pd.DataFrame(
        {"class": [class_names[c] for c in labels], "mean intensity": flat01.mean(axis=1) * 255.0}
    )
    try:
        sns.violinplot(data=per_image_df, x="class", y="mean intensity", hue="class", ax=axes[1],
                       palette=palette, inner="quartile", cut=0, order=list(class_names), legend=False)
    except TypeError:
        sns.violinplot(data=per_image_df, x="class", y="mean intensity", ax=axes[1],
                       palette=palette, inner="quartile", cut=0, order=list(class_names))
    axes[1].set_title("Distribution of the per-image mean intensity")
    axes[1].set_ylabel("mean intensity of the image")
    axes[1].tick_params(axis="x", rotation=40)
    plt.show()
    rows = []
    for c in range(n_classes):
        allv = (flat01[labels == c] * 255.0).reshape(-1)
        fg = allv[allv > 0]
        rows.append({
            "class": class_names[c], "image-mean": float(flat01[labels == c].mean() * 255),
            "fg mean": float(fg.mean()), "fg std": float(fg.std()),
            "fg skew": float(sp_stats.skew(fg)),
            "% background": float(100.0 * (allv == 0).mean()),
            "ink coverage % (>20)": float(100.0 * (allv > 20).mean()),
        })
    return pd.DataFrame(rows).set_index("class").round(2)


def intensity_ks_matrix(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    max_pixels: int = 20_000, seed: int = 42
) -> pd.DataFrame:
    """Pairwise two-sample KS statistic between per-image mean-intensity distributions."""
    rng = np.random.default_rng(seed)
    per_image = flat01.mean(axis=1)
    n = len(class_names)
    mat = np.zeros((n, n))
    for i in range(n):
        a = per_image[labels == i]
        a = rng.choice(a, size=min(max_pixels, len(a)), replace=False)
        for j in range(i + 1, n):
            b = per_image[labels == j]
            b = rng.choice(b, size=min(max_pixels, len(b)), replace=False)
            d = float(sp_stats.ks_2samp(a, b).statistic)
            mat[i, j] = mat[j, i] = d
    df = pd.DataFrame(mat, index=list(class_names), columns=list(class_names))
    plt.figure(figsize=(7.6, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                annot_kws={"size": 7}, cbar_kws={"label": "KS distance"})
    plt.title("KS distance between per-image mean-intensity distributions")
    plt.show()
    return df.round(3)


def class_mean_variance_panels(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot class means, per-pixel stds and deviation-from-global maps."""
    n_classes = len(class_names)
    means = np.stack([flat01[labels == c].mean(axis=0) for c in range(n_classes)])
    stds = np.stack([flat01[labels == c].std(axis=0) for c in range(n_classes)])
    global_mean = flat01.mean(axis=0)
    for title, mats, cmap, kw in [
        ("Class MEAN images", means, "viridis", {}),
        ("Class per-pixel STD images", stds, "magma", {}),
        ("Class mean MINUS global mean", means - global_mean, "coolwarm", {"vmin": -0.45, "vmax": 0.45}),
    ]:
        fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
        for c, ax in enumerate(axes.ravel()):
            im = ax.imshow(mats[c].reshape(28, 28), cmap=cmap, **kw)
            ax.set_title(f"{c}: {class_names[c]}", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        fig.suptitle(title, y=1.01)
        plt.show()
    return means, stds


def fisher_discriminability(
    flat01: np.ndarray, labels: np.ndarray, n_classes: int = 10, eps: float = 1e-8
) -> np.ndarray:
    """Per-pixel Fisher ratio Var_between / Var_within."""
    global_mean = flat01.mean(axis=0)
    between = np.zeros(flat01.shape[1], dtype=np.float64)
    within = np.zeros(flat01.shape[1], dtype=np.float64)
    n = len(flat01)
    for c in range(n_classes):
        sub = flat01[labels == c]
        n_c = len(sub)
        between += n_c * (sub.mean(axis=0) - global_mean) ** 2
        within += n_c * sub.var(axis=0)
    return (between / n) / (within / n + eps)


def pixel_correlation_analysis(
    flat01: np.ndarray, grid: int = 14, max_rows: int = 4_000, seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Correlation structure of the pixel features."""
    rng = np.random.default_rng(seed)
    rows_idx = rng.choice(len(flat01), size=min(max_rows, len(flat01)), replace=False)
    X = flat01[rows_idx]
    X = X + rng.normal(0, 1e-6, X.shape).astype(np.float32)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)
    if 28 % grid != 0:
        grid = 14
    step = 28 // grid
    coarse_imgs = X.reshape(-1, grid, step, grid, step).mean(axis=(2, 4)).reshape(len(X), grid * grid)
    coarse = np.nan_to_num(np.corrcoef(coarse_imgs, rowvar=False))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    im0 = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Full 784 x 784 pixel-correlation matrix")
    axes[0].set_xlabel("pixel index")
    axes[0].set_ylabel("pixel index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    centre = 14 * 28 + 14
    im1 = axes[1].imshow(corr[centre].reshape(28, 28), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Correlation of every pixel with the centre pixel (14,14)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    yy, xx = np.mgrid[0:28, 0:28]
    pos = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(784, k=1)
    dist_bins = np.round(d[iu]).astype(int)
    corr_vals = corr[iu]
    decay = (pd.DataFrame({"distance (px)": dist_bins, "correlation": corr_vals})
             .groupby("distance (px)")["correlation"].agg(["mean", "std", "count"]).reset_index())
    axes[2].plot(decay["distance (px)"], decay["mean"], marker="o", ms=3, color="#4C72B0")
    axes[2].fill_between(decay["distance (px)"], decay["mean"] - decay["std"],
                         decay["mean"] + decay["std"], alpha=0.2, color="#4C72B0")
    axes[2].axhline(0, c="grey", lw=1)
    axes[2].set_title("Mean pixel correlation vs. spatial distance")
    axes[2].set_xlabel("euclidean distance between pixels (px)")
    axes[2].set_ylabel("Pearson correlation")
    fig.suptitle("Pixel-correlation structure", y=1.03)
    plt.show()
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(pd.DataFrame(coarse).round(2), cmap="RdBu_r", vmin=-1, vmax=1, annot=False,
                cbar_kws={"label": "block-averaged correlation"})
    plt.title(f"Correlation between {grid}x{grid} block-averaged pixels")
    plt.xlabel("coarse pixel block")
    plt.ylabel("coarse pixel block")
    plt.show()
    return corr, decay.round(3)


def pca_embedding_analysis(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    n_components: int = 50, seed: int = 42, show_3d: bool = True,
) -> Tuple[PCA, np.ndarray]:
    """Fit a PCA, plot the 2D and 3D projections, the spectrum, and the leading eigen-garments."""
    pca = PCA(n_components=n_components, random_state=seed).fit(flat01)
    Z = pca.transform(flat01)
    palette = sns.color_palette("tab10", len(class_names))
    fig = plt.figure(figsize=(16, 4.6))
    ax0 = fig.add_subplot(1, 3, 1)
    for c in range(len(class_names)):
        m = labels == c
        ax0.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.55, color=palette[c], label=class_names[c])
    ax0.set_title("PCA - components 1 & 2")
    ax0.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax0.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax0.legend(fontsize=6, ncol=2, markerscale=2)
    ax1 = fig.add_subplot(1, 3, 2)
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(range(1, len(cum)+1), cum, marker="o", ms=3)
    ax1.axhline(0.90, ls="--", c="red", lw=1, label="90% variance")
    ax1.set_title("Cumulative explained variance")
    ax1.set_xlabel("components")
    ax1.legend()
    ax2 = fig.add_subplot(1, 3, 3, projection="3d") if show_3d else fig.add_subplot(1, 3, 3)
    if show_3d:
        for c in range(len(class_names)):
            m = labels == c
            ax2.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=4, alpha=0.5, color=palette[c], label=class_names[c])
        ax2.set_title("PCA - components 1, 2 & 3 (3D)")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
    else:
        ax2.axis("off")
    fig.suptitle("Principal-component structure of raw pixel space", y=1.03)
    plt.show()
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.8))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(pca.components_[k].reshape(28, 28), cmap="RdBu_r")
        ax.set_title(f"PC{k+1}\\n{pca.explained_variance_ratio_[k]*100:.1f}% var", fontsize=8)
        ax.axis("off")
    fig.suptitle("The first 12 principal components as images", y=1.02)
    plt.show()
    return pca, Z


def embedding_quality(
    X_high: np.ndarray, X_low: np.ndarray, labels: np.ndarray, k: int = 10, seed: int = 42
) -> Dict[str, float]:
    """Score a low-dimensional embedding with trustworthiness and kNN accuracy."""
    n = len(X_low)
    tw = float(trustworthiness(X_high, X_low, n_neighbors=k))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_low[perm[:cut]], labels[perm[:cut]])
    acc = float(knn.score(X_low[perm[cut:]], labels[perm[cut:]]))
    return {"trustworthiness": round(tw, 4), "knn_accuracy_in_embedding": round(acc, 4)}


def make_tsne(n_components: int, perplexity: float, n_iter: int, seed: int) -> TSNE:
    """Construct a TSNE object compatible across scikit-learn versions."""
    params = inspect.signature(TSNE.__init__).parameters
    iter_kw = "max_iter" if "max_iter" in params else "n_iter"
    kwargs = {"n_components": n_components, "perplexity": perplexity, "init": "pca",
              "learning_rate": "auto", "random_state": seed, iter_kw: n_iter}
    return TSNE(**kwargs)


def tsne_perplexity_sweep(
    Z_high: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    perplexities: Sequence[int], n_iter: int, seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Run t-SNE for several perplexities and plot them side by side."""
    out: Dict[int, np.ndarray] = {}
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5.2 * len(perplexities), 4.8), squeeze=False)
    palette = sns.color_palette("tab10", len(class_names))
    for ax, p in zip(axes[0], perplexities):
        t0 = time.time()
        Z2 = make_tsne(2, float(p), n_iter, seed).fit_transform(Z_high)
        out[int(p)] = Z2
        for c in range(len(class_names)):
            m = labels == c
            ax.scatter(Z2[m, 0], Z2[m, 1], s=4, alpha=0.6, color=palette[c], label=class_names[c])
        ax.set_title(f"t-SNE, perplexity={p}  ({time.time()-t0:.0f}s)")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][-1].legend(fontsize=6, ncol=2, markerscale=2, loc="best")
    fig.suptitle("t-SNE perplexity sweep (PCA-50 input)", y=1.02)
    plt.show()
    return out


def plot_embedding_3d(Z3: np.ndarray, labels: np.ndarray, class_names: Sequence[str], title: str) -> None:
    """Render a 3-component embedding from two viewing angles."""
    palette = sns.color_palette("tab10", len(class_names))
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim) in enumerate([(20, 45), (20, 135)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for c in range(len(class_names)):
            m = labels == c
            ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=4, alpha=0.55, color=palette[c], label=class_names[c])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title}  (elev={elev}, azim={azim})", fontsize=10)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        if k == 1:
            ax.legend(fontsize=6, ncol=2, markerscale=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.show()


def umap_sweep(
    Z_high: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    neighbor_grid: Sequence[int], min_dist: float, seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Run UMAP for several n_neighbors values."""
    import umap
    out: Dict[int, np.ndarray] = {}
    fig, axes = plt.subplots(1, len(neighbor_grid), figsize=(5.2 * len(neighbor_grid), 4.8), squeeze=False)
    palette = sns.color_palette("tab10", len(class_names))
    for ax, k in zip(axes[0], neighbor_grid):
        t0 = time.time()
        reducer = umap.UMAP(n_components=2, n_neighbors=int(k), min_dist=min_dist, random_state=seed)
        Z2 = reducer.fit_transform(Z_high)
        out[int(k)] = Z2
        for c in range(len(class_names)):
            m = labels == c
            ax.scatter(Z2[m, 0], Z2[m, 1], s=4, alpha=0.6, color=palette[c], label=class_names[c])
        ax.set_title(f"UMAP, n_neighbors={k}, min_dist={min_dist}  ({time.time()-t0:.0f}s)")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][-1].legend(fontsize=6, ncol=2, markerscale=2, loc="best")
    fig.suptitle("UMAP neighbourhood-size sweep (PCA-50 input)", y=1.02)
    plt.show()
    return out


def kmeans_grid(
    Z: np.ndarray, labels: np.ndarray, k_grid: Sequence[int], seed: int = 42
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """Fit k-means for several k and score each partition."""
    rows, assignments = [], {}
    for k in k_grid:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(Z)
        lab = km.labels_
        assignments[int(k)] = lab
        rows.append({
            "k": int(k), "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Z, lab, sample_size=min(3_000, len(Z)), random_state=seed)),
            "ARI vs. true labels": float(adjusted_rand_score(labels, lab)),
            "NMI vs. true labels": float(normalized_mutual_info_score(labels, lab)),
        })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(df["k"], df["inertia"], marker="o")
    axes[0].set_title("Elbow curve (inertia)"); axes[0].set_xlabel("k")
    axes[1].plot(df["k"], df["silhouette"], marker="o", color="#DD8452")
    axes[1].set_title("Silhouette (no labels used)"); axes[1].set_xlabel("k")
    axes[2].plot(df["k"], df["ARI vs. true labels"], marker="o", label="ARI")
    axes[2].plot(df["k"], df["NMI vs. true labels"], marker="s", label="NMI")
    axes[2].axvline(10, ls="--", c="grey", lw=1)
    axes[2].text(10.1, axes[2].get_ylim()[0], "true k = 10", fontsize=8, color="grey")
    axes[2].set_title("Agreement with the official taxonomy"); axes[2].set_xlabel("k"); axes[2].legend()
    fig.suptitle("k-means over a grid of cluster counts", y=1.04)
    plt.show()
    return df.round(4), assignments


def cluster_composition(
    cluster_labels: np.ndarray, true_labels: np.ndarray, images: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Contingency heat-map (cluster x true class) plus the mean image of every cluster."""
    k = int(cluster_labels.max()) + 1
    cont = np.zeros((k, len(class_names)))
    for i in range(k):
        cont[i] = np.bincount(true_labels[cluster_labels == i], minlength=len(class_names))
    cont_norm = cont / np.maximum(cont.sum(axis=1, keepdims=True), 1)
    df = pd.DataFrame(cont_norm, columns=list(class_names), index=[f"cluster {i}" for i in range(k)])
    plt.figure(figsize=(10, 5.6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 7},
                cbar_kws={"label": "share of the cluster"})
    plt.title("Composition of each k-means cluster (rows sum to 1)")
    plt.xlabel("true class")
    plt.show()
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(1.5 * k, 4.4))
    for i, ax in enumerate(np.array(axes).ravel()[:k]):
        ax.imshow(images[cluster_labels == i].mean(axis=0), cmap="gray")
        dom = int(np.argmax(cont[i]))
        ax.set_title(f"c{i}: {class_names[dom]}\\n({cont_norm[i, dom]*100:.0f}% pure, n={int(cont[i].sum())})", fontsize=7)
        ax.axis("off")
    for ax in np.array(axes).ravel()[k:]:
        ax.axis("off")
    fig.suptitle("Mean image of every discovered cluster", y=1.03)
    plt.show()
    return df.round(3)


def isolation_forest_outliers(
    Z: np.ndarray, labels: np.ndarray, images: np.ndarray, class_names: Sequence[str],
    contamination: float = 0.01, n_estimators: int = 300, seed: int = 42, n_show: int = 10,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit an Isolation Forest and visualise the most anomalous images."""
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=seed, n_jobs=-1).fit(Z)
    scores = iso.score_samples(Z)
    flag = iso.predict(Z) == -1
    order = np.argsort(scores)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.35, 3.4))
    for j in range(n_show):
        i_out, i_in = order[j], order[-(j + 1)]
        axes[0, j].imshow(images[i_out], cmap="gray")
        axes[0, j].set_title(f"{class_names[labels[i_out]]}\\n{scores[i_out]:.3f}", fontsize=6)
        axes[0, j].axis("off")
        axes[1, j].imshow(images[i_in], cmap="gray")
        axes[1, j].set_title(f"{class_names[labels[i_in]]}\\n{scores[i_in]:.3f}", fontsize=6)
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("most anomalous")
    fig.suptitle("Isolation Forest - top: most anomalous, bottom: most typical", y=1.06)
    plt.show()
    per_class = pd.DataFrame({
        "class": list(class_names),
        "anomaly rate %": [100.0 * flag[labels == c].mean() for c in range(len(class_names))],
        "mean anomaly score": [float(scores[labels == c].mean()) for c in range(len(class_names))],
    }).set_index("class").round(3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    per_class["anomaly rate %"].plot(kind="barh", ax=axes[0], color="#C44E52", edgecolor="black", linewidth=0.4)
    axes[0].axvline(100 * contamination, ls="--", c="grey", lw=1)
    axes[0].set_title(f"Share of images flagged per class (target = {100*contamination:.1f}%)")
    axes[1].hist(scores, bins=60, color="#4C72B0", edgecolor="black", linewidth=0.3)
    axes[1].axvline(np.quantile(scores, contamination), ls="--", c="red", lw=1.2, label="decision threshold")
    axes[1].set_title("Distribution of anomaly scores")
    axes[1].set_xlabel("score (lower = more anomalous)")
    axes[1].legend()
    plt.show()
    return scores, per_class


def autoencoder_outlier_report(
    model, images_u8: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    mean: float, std: float, history: Sequence[float], n_show: int = 10, top_q: float = 0.99,
) -> np.ndarray:
    """Reconstruction-error histogram, worst/best reconstructions and per-class error."""
    import torch
    from data_loading import reconstruction_errors
    x = torch.from_numpy(images_u8).float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)
    err, recon = reconstruction_errors(model, x)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].plot(range(1, len(history) + 1), history, marker="o")
    axes[0].set_title("Autoencoder training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")
    axes[1].hist(err, bins=70, color="#55A868", edgecolor="black", linewidth=0.3)
    axes[1].axvline(np.quantile(err, top_q), ls="--", c="red", lw=1.2, label=f"{top_q:.0%} quantile")
    axes[1].set_yscale("log"); axes[1].set_title("Per-image reconstruction error")
    axes[1].set_xlabel("MSE"); axes[1].legend()
    per_class_err = [float(err[labels == c].mean()) for c in range(len(class_names))]
    axes[2].barh(list(class_names), per_class_err, color="#4C72B0", edgecolor="black", linewidth=0.4)
    axes[2].set_title("Mean reconstruction error per class")
    fig.suptitle("Convolutional-autoencoder anomaly analysis", y=1.04)
    plt.show()
    order = np.argsort(-err)
    fig, axes = plt.subplots(4, n_show, figsize=(n_show * 1.35, 6.4))
    for j in range(n_show):
        i_bad, i_good = order[j], order[-(j + 1)]
        axes[0, j].imshow(images_u8[i_bad], cmap="gray")
        axes[0, j].set_title(f"{class_names[labels[i_bad]]}\\n{err[i_bad]:.3f}", fontsize=6)
        axes[1, j].imshow(recon[i_bad, 0], cmap="gray")
        axes[2, j].imshow(images_u8[i_good], cmap="gray")
        axes[2, j].set_title(f"{class_names[labels[i_good]]}\\n{err[i_good]:.3f}", fontsize=6)
        axes[3, j].imshow(recon[i_good, 0], cmap="gray")
        for r in range(4):
            axes[r, j].axis("off")
    fig.suptitle("Rows 1-2: worst-reconstructed | Rows 3-4: best-reconstructed", y=1.03)
    plt.show()
    return err


def detector_agreement(iso_scores: np.ndarray, ae_err: np.ndarray, top_frac: float = 0.01) -> pd.DataFrame:
    """Overlap between the two anomaly rankings."""
    k = max(1, int(top_frac * len(iso_scores)))
    set_iso = set(np.argsort(iso_scores)[:k].tolist())
    set_ae = set(np.argsort(-ae_err)[:k].tolist())
    inter = len(set_iso & set_ae)
    rho = float(sp_stats.spearmanr(-iso_scores, ae_err)[0])
    expected = k * k / len(iso_scores)
    return pd.DataFrame([{
        "top-k size": k, "images flagged by both": inter,
        "expected overlap if independent": round(expected, 2),
        "Jaccard index": round(inter / (2 * k - inter), 4) if (2 * k - inter) else 0.0,
        "Spearman rank correlation": round(rho, 4),
        "enrichment vs. chance": round(inter / expected, 2) if expected > 0 else float("nan"),
    }])
'''
    
    write_module("eda.py", eda_py)
    
    # ---- models.py ----
    models_py = '''"""
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
'''
    
    write_module("models.py", models_py)
    
    # ---- training.py ----
    training_py = '''"""
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
'''
    
    write_module("training.py", training_py)
    
    # ---- explainability.py ----
    explainability_py = '''"""
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
        axes[0, i].set_title(f"true {class_names[y_true[i]]}\\npred {class_names[y_pred[i]]}", fontsize=6)
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
'''
    
    write_module("explainability.py", explainability_py)
    
    # ---- statistics_utils.py ----
    statistics_py = '''"""
statistics_utils.py – Statistical comparison utilities.

Contains:
  - wilson_interval(): Wilson score CI for a binomial proportion
  - mcnemar_full(): McNemar test (exact + asymptotic)
  - holm_bonferroni(): Holm step-down correction
  - pairwise_significance(): all-pair McNemar + Holm
  - paired_bootstrap(): paired bootstrap CI
  - bootstrap_difference(): CI of accuracy difference
  - cochrans_q(): Cochran's Q omnibus test
  - final_leaderboard(): merge results with CIs
  - plot_literature_comparison(): visual comparison with published results
  - delta_table(), delta_table_v2(): head-to-head comparison tables
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def mcnemar_full(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    name_a: str = "A", name_b: str = "B",
) -> Dict[str, object]:
    """Full McNemar analysis of two classifiers on the same test set."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    n_disc = n01 + n10
    if n_disc == 0:
        chi2_stat, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / n_disc
        p_chi2 = float(sp_stats.chi2.sf(chi2_stat, df=1))
        p_exact = float(sp_stats.binomtest(min(n01, n10), n=n_disc, p=0.5).pvalue)
    out = {
        "model A": name_a, "model B": name_b,
        "acc A": float(a_ok.mean()), "acc B": float(b_ok.mean()),
        "A right / B wrong": n01, "A wrong / B right": n10,
        "chi2 (corrected)": round(chi2_stat, 3),
        "p (exact binomial)": p_exact, "p (chi2)": p_chi2,
        "odds ratio": round(n10 / n01, 3) if n01 else float("inf"),
    }
    return out


def mcnemar_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    name_a: str = "A", name_b: str = "B",
) -> pd.DataFrame:
    """Simplified McNemar returning a DataFrame for ensemble gain reports."""
    result = mcnemar_full(y_true, pred_a, pred_b, name_a, name_b)
    return pd.DataFrame([{
        "n01": result["A right / B wrong"],
        "n10": result["A wrong / B right"],
        "p-value": result["p (exact binomial)"],
        "significant at 0.05": result["p (exact binomial)"] < 0.05,
    }])


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm's step-down procedure for family-wise error rate control."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted < alpha, adjusted


def pairwise_significance(
    predictions: Dict[str, np.ndarray], y_true: np.ndarray, alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar for every model pair, then Holm correction."""
    names = list(predictions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = mcnemar_full(y_true, predictions[a], predictions[b], a, b)
            rows.append({
                "model A": a, "model B": b,
                "acc A": r["acc A"], "acc B": r["acc B"],
                "delta (pp)": 100 * (r["acc B"] - r["acc A"]),
                "p raw": r["p (exact binomial)"],
            })
    df = pd.DataFrame(rows)
    reject, adj = holm_bonferroni(df["p raw"].to_numpy(), alpha=alpha)
    df["p Holm-adjusted"] = adj
    df["significant"] = reject
    grid = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)
    for _, r in df.iterrows():
        grid.loc[r["model A"], r["model B"]] = r["p Holm-adjusted"]
        grid.loc[r["model B"], r["model A"]] = r["p Holm-adjusted"]
    plt.figure(figsize=(1.05 * len(names) + 4, 0.85 * len(names) + 3))
    sns.heatmap(grid, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=0, vmax=0.2,
                annot_kws={"size": 6}, cbar_kws={"label": "Holm-adjusted p-value"})
    plt.title("Pairwise McNemar tests, Holm-corrected")
    plt.show()
    return df.sort_values("p Holm-adjusted").reset_index(drop=True), grid


def paired_bootstrap(
    correct: np.ndarray, names: Sequence[str], n_iter: int = 2_000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap the test set, keeping all models paired."""
    rng = np.random.default_rng(seed)
    n = correct.shape[1]
    boot = np.empty((n_iter, correct.shape[0]), dtype=np.float64)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot[b] = correct[:, idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    df = pd.DataFrame({
        "model": list(names), "accuracy": correct.mean(axis=1),
        "bootstrap CI low": lo, "bootstrap CI high": hi,
        "bootstrap std (pp)": 100 * boot.std(axis=0),
    }).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df, boot


def bootstrap_difference(
    boot: np.ndarray, names: Sequence[str], name_a: str, name_b: str, alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile CI of acc(B) - acc(A) from paired bootstrap replicates."""
    i, j = list(names).index(name_a), list(names).index(name_b)
    diff = boot[:, j] - boot[:, i]
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return {
        "comparison": f"{name_b} - {name_a}",
        "mean delta (pp)": float(100 * diff.mean()),
        "CI low (pp)": float(100 * lo), "CI high (pp)": float(100 * hi),
        "P(B better than A)": float((diff > 0).mean()),
    }


def cochrans_q(correct: np.ndarray) -> Dict[str, float]:
    """Cochran's Q omnibus test for k paired binary classifiers."""
    k, n = correct.shape
    col_totals = correct.sum(axis=1).astype(float)
    row_totals = correct.sum(axis=0).astype(float)
    total = col_totals.sum()
    numerator = (k - 1) * (k * (col_totals ** 2).sum() - total ** 2)
    denominator = k * total - (row_totals ** 2).sum()
    q = float(numerator / denominator) if denominator else 0.0
    p = float(sp_stats.chi2.sf(q, df=k - 1))
    return {"k models": k, "n images": n, "Cochran Q": round(q, 2), "df": k - 1, "p-value": p}


def final_leaderboard(records: Sequence[Dict[str, object]], ci_table: pd.DataFrame) -> pd.DataFrame:
    """Merge the results registry with confidence intervals."""
    df = pd.DataFrame(list(records)).drop_duplicates(subset=["model"], keep="last")
    df = df.merge(ci_table[["model", "CI low (95%)", "CI high (95%)"]], on="model", how="left")
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df[["rank", "model", "family", "accuracy", "CI low (95%)", "CI high (95%)",
               "macro_f1", "error_rate", "fit_s", "params", "notes"]]


def plot_literature_comparison(df: pd.DataFrame) -> None:
    """Horizontal bar chart contrasting our models with published results."""
    d = df.sort_values("accuracy")
    colors = ["#C44E52" if o == "ours" else "#B0B0B0" for o in d["origin"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(d["model"] + "  [" + d["origin"] + "]", d["accuracy"], color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7.5)
    ax.set_xlim(0.75, 1.02)
    ax.set_xlabel("test accuracy on the official 10,000-image Fashion-MNIST test set")
    ax.set_title("This notebook (red) vs. published results (grey)")
    ax.axvline(0.8350, ls=":", c="green", lw=1.2)
    ax.text(0.836, 0.2, "human-level estimate", color="green", fontsize=8, rotation=90, va="bottom")
    plt.show()
'''
    
    write_module("statistics_utils.py", statistics_py)
    
    # ---- notebook_tests.py ----
    tests_py = '''"""
notebook_tests.py – Inline unit test suite for the Fashion-MNIST pipeline.

Contains:
  - SkipTest exception class
  - run_test_suite(): test runner
  - All test_* functions for data, tensors, models, inference, bookkeeping
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SkipTest(Exception):
    """Raised by a test whose prerequisites are absent."""
    pass


def run_test_suite(tests: Sequence[Callable[[], None]], verbose: bool = True) -> pd.DataFrame:
    """Execute every test function, catching failures."""
    rows = []
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            status, message = "PASS", (fn.__doc__ or "").strip().split("\\n")[0]
        except SkipTest as exc:
            status, message = "SKIP", str(exc)
        except AssertionError as exc:
            status, message = "FAIL", f"AssertionError: {exc}"
        except Exception as exc:
            status, message = "ERROR", f"{type(exc).__name__}: {exc}"
        rows.append({"test": fn.__name__, "status": status, "detail": message,
                     "seconds": round(time.time() - t0, 3)})
        if verbose:
            symbol = {"PASS": "PASS ", "SKIP": "SKIP ", "FAIL": "FAIL ", "ERROR": "ERROR"}[status]
            print(f"[{symbol}] {fn.__name__:<46s} {rows[-1]['seconds']:>6.2f}s  {message[:70]}")
    return pd.DataFrame(rows)
'''
    
    write_module("notebook_tests.py", tests_py)
    
    # ---- __init__.py ----
    write_module("__init__.py", '"""Fashion-MNIST DL project – source modules."""\n')


def write_module(name: str, content: str):
    path = os.path.join(SRC_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    lines = content.count("\n") + 1
    print(f"  Written: {path}  ({lines} lines)")


if __name__ == "__main__":
    main()
