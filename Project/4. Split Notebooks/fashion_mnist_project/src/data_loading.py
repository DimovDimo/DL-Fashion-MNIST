"""
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
        axes[0][k].set_title(f"train #{i}\n{class_names[y_train[i]]}", fontsize=6)
        axes[1][k].imshow(X_test[j], cmap="gray", vmin=0, vmax=255)
        axes[1][k].set_title(f"test #{j}\n{class_names[y_test[j]]}", fontsize=6)
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
