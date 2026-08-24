# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext_py_percent//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])

# %%
# --- Environment inspection (safe to run anywhere; the shell calls are Colab-friendly) -------------
import platform
import shutil
import subprocess
import sys


def run_shell(cmd: str) -> str:
    """Run a shell command and return its stdout, or an explanatory message if unavailable."""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the notebook
        return f"[not available: {exc}]"


print("Python :", sys.version.split()[0], "|", platform.platform())
print("\n--- GPU -------------------------------------------------------------")
print(run_shell("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"))
print("\n--- System RAM ------------------------------------------------------")
print(run_shell("free -h | head -2"))
print("\n--- Disk ------------------------------------------------------------")
total, used, free = shutil.disk_usage("/")
print(f"total={total / 1e9:.1f} GB  used={used / 1e9:.1f} GB  free={free / 1e9:.1f} GB")

# %%
# --- Dependencies ----------------------------------------------------------------------------------
# Colab already ships torch, torchvision, scikit-learn, pandas, matplotlib and seaborn.
# Only `kagglehub` usually needs installing. `-q` keeps the output clean.
# %pip install -q kagglehub

# Optional (nicer confusion-matrix / progress output); both are pre-installed on Colab.
# %pip install -q tqdm

# %% [markdown]
# ### 1.1b Extended dependencies for the upgraded edition
#
# The v2 sections need libraries that Colab does **not** ship by default. They are installed in a single quiet cell and
# each one is *optional*: Section 1.2b probes for them and every downstream cell degrades gracefully if one is missing
# (it prints why it was skipped instead of raising). This keeps the notebook runnable on an offline machine, on Kaggle,
# and on a plain local CPU box.
#
# | Package | Used in | Fallback if unavailable |
# |---|---|---|
# | `xgboost`, `lightgbm`, `catboost` | 3.7 gradient-boosting baselines | section skipped, comparison table still built |
# | `optuna` | 3.8 automated hyper-parameter search | `sklearn.model_selection.GridSearchCV` |
# | `umap-learn` | 2.8 manifold learning | PCA + t-SNE only |
# | `shap`, `lime` | 6.4 / 6.5 model explainability | Grad-CAM + Integrated Gradients + occlusion (implemented from scratch, no dependency) |
# | `statsmodels` | 8.2 McNemar (exact) | closed-form chi-square / `scipy.stats.binomtest` |

# %%
# --- Extended dependencies (all optional; the notebook degrades gracefully without them) -------------
# Set to False if you are offline or want to keep the environment untouched.
INSTALL_EXTRAS = True

if INSTALL_EXTRAS:
    # Gradient boosting + automated hyper-parameter search
    # %pip install -q xgboost lightgbm catboost optuna
    # Manifold learning, explainability and statistics
    # %pip install -q umap-learn shap lime statsmodels
else:
    print("INSTALL_EXTRAS = False -> skipping installation; optional sections will be skipped if imports fail.")

# %% [markdown]
# ## 1.2 Imports, global configuration and reproducibility
#
# All tunable choices live in a single `CFG` dataclass so that a reviewer can see and change the whole experimental
# protocol in one place. This is also what makes the notebook *modular*: no magic numbers are buried in the code.
#
# > Exam criterion: **Code Quality (0–20)**: *"Is the code clean, modular and documented? Are functions used?"*

# %%
# --- Imports ----------------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

# Plot styling (applied once, globally)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["figure.autolayout"] = True
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 50)

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU  :", torch.cuda.get_device_name(0))


# %%
# --- Global configuration -----------------------------------------------------------------------------
@dataclass
class CFG:
    """Single source of truth for every experimental hyper-parameter.

    Changing a value here changes the whole notebook consistently, which keeps the
    experiment reproducible and easy to audit.
    """

    # Reproducibility
    seed: int = 42

    # Data
    kaggle_dataset: str = "zalando-research/fashionmnist"
    train_csv: str = "fashion-mnist_train.csv"
    test_csv: str = "fashion-mnist_test.csv"
    val_fraction: float = 0.10          # 10% of the cleaned training file -> ~6,000 val (5,999 after 1.5b)

    # Classical baselines (full 60k data would make the RBF-SVM take hours on CPU)
    sk_train_subset: int = 12_000       # stratified subsample used to FIT the classical models
    sk_eval_on_full_test: bool = True   # classical models are always EVALUATED on the full 10k test set
    run_rbf_svm: bool = True            # the single most expensive classical model (~4-7 min)

    # Deep learning
    batch_size: int = 256
    mlp_epochs: int = 20
    cnn_epochs: int = 25
    lr: float = 3e-3                    # peak LR for OneCycle
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    dropout: float = 0.30
    use_amp: bool = True                # FP16 mixed precision - T4 tensor cores
    augment: bool = True                # random horizontal flip + random translation for the CNN
    num_workers: int = 2                # Colab gives 2 vCPUs; tensors are already in RAM so this is enough

    # Output
    artifacts_dir: str = "artifacts"

    # Class names, index == label
    class_names: Tuple[str, ...] = (
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


cfg = CFG()
Path(cfg.artifacts_dir).mkdir(exist_ok=True, parents=True)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG used in this notebook so results are reproducible run to run."""
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


set_seed(cfg.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print(json.dumps({k: v for k, v in asdict(cfg).items() if k != "class_names"}, indent=2))


# %% [markdown]
# ## 1.2b Extended configuration (`CFGX`) and optional-dependency detection
#
# The original `CFG` above is **left untouched** so that every v1 result stays reproducible bit for bit. Everything the
# upgrade adds lives in a second, clearly separated dataclass, `CFGX`. Two dataclasses, one rule: *no magic numbers
# anywhere else in the notebook.*
#
# `CFGX.fast_mode` is the single switch that shrinks every expensive v2 block (fewer Optuna trials, fewer epochs, smaller
# manifold-learning samples) so that a reviewer can validate the whole pipeline in a few minutes before committing to the
# full run.

# %%
# --- Extended configuration for the v2 sections -----------------------------------------------------
@dataclass
class CFGX:
    """Configuration for everything added in the upgraded edition (EDA v2, boosting, ViT, ensembles, XAI).

    Notes
    -----
    `fast_mode=True` shrinks every expensive block so the full notebook can be smoke-tested in ~10 minutes.
    It is applied in `__post_init__`, so the printed configuration always shows the values actually used.
    """

    # ---- global switch -------------------------------------------------------------------------
    fast_mode: bool = False

    # ---- 1.5b data cleaning -----------------------------------------------------------------------
    # How the train/test duplicates found by the integrity report in Section 1.5 are resolved.
    leakage_policy: str = "drop_from_train"   # "drop_from_train" (default) | "drop_from_test" | "keep"
    drop_train_duplicates: bool = False       # also de-duplicate *inside* the training file

    # ---- 2.6-2.7 advanced EDA -------------------------------------------------------------------
    eda_sample: int = 8_000              # images used for the heavy EDA statistics
    corr_grid: int = 14                  # pixel-correlation heat-map is computed on a 14x14 down-sampled grid

    # ---- 2.8 manifold learning --------------------------------------------------------------------
    embed_sample: int = 4_000            # points fed to t-SNE / UMAP (t-SNE is O(n log n) but constant-heavy)
    pca_pre_components: int = 50         # PCA pre-reduction before t-SNE/UMAP (standard practice, kills noise)
    tsne_perplexities: Tuple[int, ...] = (5, 30, 50)
    tsne_iter: int = 750
    umap_neighbors: Tuple[int, ...] = (5, 15, 50)
    umap_min_dist: float = 0.1
    run_3d_embeddings: bool = True

    # ---- 2.9 clustering ----------------------------------------------------------------------------
    cluster_k_grid: Tuple[int, ...] = (5, 8, 10, 12, 15)
    cluster_sample: int = 6_000

    # ---- 2.10 outlier detection ---------------------------------------------------------------------
    iforest_contamination: float = 0.01  # expect ~1% anomalies
    iforest_estimators: int = 300
    autoencoder_epochs: int = 8
    autoencoder_latent: int = 32
    ae_batch_size: int = 256

    # ---- 3.7 gradient boosting -----------------------------------------------------------------------
    run_boosting: bool = True
    boost_pca_components: int = 80       # boosters are trained on a PCA embedding: ~10x faster, same accuracy
    boost_train_subset: int = 20_000     # boosters scale far better than an RBF-SVM, so they get more data
    xgb_estimators: int = 600
    lgbm_estimators: int = 800
    cat_iterations: int = 800
    boost_use_gpu: bool = True           # honoured only when a CUDA device is visible

    # ---- 3.8 hyper-parameter optimisation --------------------------------------------------------------
    run_tuning: bool = True
    optuna_trials: int = 25
    optuna_timeout_s: int = 600
    tuning_cv_folds: int = 3
    tuning_subset: int = 8_000           # tuning uses a smaller subset; the winner is refit on the full subset

    # ---- 4.9-4.11 extra deep architectures ---------------------------------------------------------------
    run_resnet: bool = True
    run_vit: bool = True
    resnet_epochs: int = 30
    resnet_width: int = 32               # base channel width; blocks are (w, 2w, 4w)
    vit_epochs: int = 35
    vit_patch: int = 7                   # 28 / 7 = 4 -> 16 patches + 1 CLS token
    vit_dim: int = 128
    vit_depth: int = 6
    vit_heads: int = 4
    vit_mlp_ratio: float = 2.0
    vit_dropout: float = 0.10
    vit_lr: float = 1e-3                 # transformers need a gentler peak LR than the CNNs
    early_stopping_patience: int = 8
    min_delta: float = 1e-4

    # ---- 5 ensembling ---------------------------------------------------------------------------------
    weight_search_iters: int = 4_000     # random Dirichlet search over the weight simplex (validation only)
    tta: bool = True                     # test-time augmentation (original + horizontal mirror)

    # ---- 6 explainability -------------------------------------------------------------------------------
    run_xai: bool = True
    ig_steps: int = 64                   # Riemann steps for Integrated Gradients
    occlusion_patch: int = 7
    occlusion_stride: int = 2
    shap_background: int = 64
    shap_samples: int = 8
    run_lime: bool = True

    # ---- 8 statistics ------------------------------------------------------------------------------------
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


cfgx = CFGX()          # <-- change to CFGX(fast_mode=True) for a ~10 minute smoke test
print("Extended configuration (CFGX):")
print(json.dumps(asdict(cfgx), indent=2, default=str))

# %%
# --- Optional-dependency probe -------------------------------------------------------------------------
import importlib
import importlib.util


def has_module(name: str) -> bool:
    """True if `name` can be imported in this runtime (no side effects, no import of the module itself)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


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

dep_df = pd.DataFrame(
    [{"package": k, "available": v, "used for": OPTIONAL_DEPS[k]} for k, v in AVAILABLE.items()]
)
display(dep_df.style.hide(axis="index"))

missing = [k for k, v in AVAILABLE.items() if not v]
if missing:
    print("Missing (those sections will be skipped with an explanation, not an error):", ", ".join(missing))
else:
    print("All optional dependencies are available - the full v2 pipeline will run.")


# %% [markdown]
# ### A note on determinism
#
# `torch.backends.cudnn.deterministic = True` makes cuDNN pick reproducible algorithms. It costs a few percent of speed
# but means a reviewer re-running the notebook obtains the *same* numbers, which matters for the exam criterion
# *"Is the process statistically valid?"*: a result that cannot be reproduced cannot be validated. Residual
# non-determinism from atomics in some CUDA kernels means accuracies may still differ in the 4th decimal place.

# %% [markdown]
# ## 1.3 Downloading the dataset with KaggleHub
#
# `kagglehub` resolves the dataset slug, downloads the archive into the Colab cache
# (`~/.cache/kagglehub/datasets/...`) and returns the local path. It is idempotent: a second call is a no-op that just
# returns the cached path.
#
# **Authentication.** On Colab, public datasets normally download anonymously. If Kaggle asks for credentials, either
# (a) upload a `kaggle.json` API token, or (b) set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables, or
# (c) use Colab Secrets (the key icon in the left sidebar) with names `KAGGLE_USERNAME` and `KAGGLE_KEY`. The helper
# below reads Colab Secrets automatically when they exist, and falls back to `torchvision.datasets.FashionMNIST`
# (the byte-identical official source) if the network path fails, so the notebook always runs end to end.

# %%
# --- Optional: pull Kaggle credentials from Colab Secrets if they are configured ---------------------
def configure_kaggle_credentials() -> bool:
    """Populate KAGGLE_USERNAME / KAGGLE_KEY from Colab Secrets when available.

    Returns
    -------
    bool
        True if credentials are present in the environment after the call.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    try:
        from google.colab import userdata  # type: ignore

        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        return True
    except Exception:  # noqa: BLE001 - not on Colab, or secrets not set: anonymous download is fine
        return False


has_creds = configure_kaggle_credentials()
print("Kaggle credentials configured:", has_creds, "(anonymous download is attempted otherwise)")

# %%
# --- KaggleHub download ------------------------------------------------------------------------------
import kagglehub

path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Path:", path)

DATA_DIR = Path(path)

print("\nFiles in the downloaded dataset directory:")
for f in sorted(DATA_DIR.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(DATA_DIR)}  ({f.stat().st_size / 1e6:.1f} MB)")


# %% [markdown]
# ### What the download contains
#
# The Kaggle mirror `zalando-research/fashionmnist` ships four files:
#
# | File | Content |
# |---|---|
# | `fashion-mnist_train.csv` | 60,000 rows x 785 columns: **used here** |
# | `fashion-mnist_test.csv`  | 10,000 rows x 785 columns: **used here** |
# | `train-images-idx3-ubyte` / `t10k-*` (in some versions) | the original IDX binary format, identical content |
#
# **CSV structure** (the format we parse):
#
# * column `label`: integer in `[0, 9]`;
# * columns `pixel1 … pixel784`: integers in `[0, 255]`, the 28x28 image flattened in **row-major** order, i.e. pixel
#   index $p$ (1-based) corresponds to row $\lfloor (p-1)/28 \rfloor$ and column $(p-1) \bmod 28$.
#
# Pixel value 0 = black (background), 255 = white. The garments are light objects on a black background because Zalando
# inverted and contrast-normalised the original product photographs.

# %%
# --- Reading the CSV files --------------------------------------------------------------------------
def find_csv(data_dir: Path, filename: str) -> Path:
    """Locate a CSV inside the downloaded dataset directory (handles nested folders)."""
    matches = sorted(data_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"{filename} not found under {data_dir}. Files present: "
            f"{[p.name for p in data_dir.rglob('*') if p.is_file()][:20]}"
        )
    return matches[0]


def load_fashion_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Fashion-MNIST CSV into (images_uint8[N,28,28], labels_int64[N]).

    Memory note: pandas parses integers as int64 (~470 MB for the training file). We immediately
    downcast to uint8 / int64-labels and delete the DataFrame, which keeps peak RAM ~1 GB - far
    below the 12.7 GB available on the Colab T4 runtime.
    """
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else df.columns[0]
    labels = df[label_col].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=[label_col]).to_numpy(dtype=np.uint8)
    del df
    images = pixels.reshape(-1, 28, 28)
    return images, labels


t0 = time.time()
train_csv_path = find_csv(DATA_DIR, cfg.train_csv)
test_csv_path = find_csv(DATA_DIR, cfg.test_csv)

X_train_full_np, y_train_full_np = load_fashion_csv(train_csv_path)
X_test_np, y_test_np = load_fashion_csv(test_csv_path)

print(f"Loaded in {time.time() - t0:.1f}s")
print("train images:", X_train_full_np.shape, X_train_full_np.dtype,
      "| labels:", y_train_full_np.shape, y_train_full_np.dtype)
print("test  images:", X_test_np.shape, X_test_np.dtype,
      "| labels:", y_test_np.shape, y_test_np.dtype)
print("memory: train", X_train_full_np.nbytes / 1e6, "MB | test", X_test_np.nbytes / 1e6, "MB")

# %%
# --- A peek at the raw CSV layout (first 5 rows, first 8 pixel columns) -------------------------------
raw_head = pd.read_csv(train_csv_path, nrows=5)
print("shape of a 5-row sample:", raw_head.shape)
display(raw_head.iloc[:, :9])
print("\nColumn names:", list(raw_head.columns[:5]), "...", list(raw_head.columns[-3:]))


# %% [markdown]
# ## 1.5 Data cleaning, integrity and statistical-validity checks
#
# > *"Is the process statistically valid?"*
#
# Fashion-MNIST is a curated benchmark, so we do not expect corrupt rows but **we verify rather than assume**. The
# checks below are the minimum due diligence for any tabular/image dataset:
#
# 1. **Shape**: 60,000 / 10,000 rows, 784 pixel columns.
# 2. **Missing values**: no NaN (pandas would have produced floats).
# 3. **Value range**: pixels within `[0, 255]`, labels within `[0, 9]`.
# 4. **Class balance**: exactly 6,000 train / 1,000 test images per class (this is what makes plain *accuracy* a valid
#    headline metric; on an imbalanced set we would have to lead with macro-F1).
# 5. **Duplicates**: exact duplicate images inside a split, and critically **train/test leakage**: identical images
#    appearing in both splits would inflate the test score. We hash every image and intersect the two sets.
# 6. **Degenerate images**: all-black or constant images that carry no signal.

# %%
# --- Integrity checks ---------------------------------------------------------------------------------
def image_hashes(images: np.ndarray) -> np.ndarray:
    """Return a 1-D array of hashes, one per image, for duplicate / leakage detection."""
    flat = np.ascontiguousarray(images.reshape(len(images), -1))
    return np.array([hash(row.tobytes()) for row in flat], dtype=np.int64)


def integrity_report(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, num_classes: int = 10
) -> pd.DataFrame:
    """Run all data-quality checks and return them as a tidy pass/fail table."""
    tr_hash, te_hash = image_hashes(X_tr), image_hashes(X_te)
    tr_counts = np.bincount(y_tr, minlength=num_classes)
    te_counts = np.bincount(y_te, minlength=num_classes)
    # np.ptp(...) is used as a free function (ndarray.ptp was removed in NumPy 2.0)
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


quality_df = integrity_report(X_train_full_np, y_train_full_np, X_test_np, y_test_np, cfg.num_classes)
display(quality_df.style.hide(axis="index"))


# %% [markdown]
# **Reading the report.** Every structural check passes **except one**. A handful of *exact duplicate images inside*
# the training split is expected (Zalando's catalogue contains repeated product shots) and it is harmless as long
# as duplicates do not cross the train/test boundary. The leakage check is therefore the important one: it must
# report **0 shared images**, because that is what guarantees that the test accuracy reported in Sections 3–5 is an
# unbiased estimate of generalisation performance.
#
# > ⚠️ **It reports 10.** The Kaggle mirror of Fashion-MNIST ships ten images that are byte-identical in the
# > training and the test file. This is a real defect in the data, not in the check: and an unfixed leak would
# > contaminate every test number in this notebook. **Section 1.5b locates, inspects and removes them** before
# > any model is fitted; the rest of Section 1.5's conclusions are unaffected.
#
# **Cleaning decisions taken (and justified):**
#
# | Decision | Rationale |
# |---|---|
# | Keep in-split duplicates | Removing them would change the class priors and break comparability with published results |
# | No outlier removal | Every row is a valid product image; "outliers" here are legitimate rare garments |
# | No missing-value imputation | Nothing is missing |
# | Scale pixels to `[0, 1]`, then standardise with the **training** mean/std | Standard practice; statistics computed on train only, so no information leaks from validation/test |
# | Keep `uint8` in RAM, cast to `float32` on the fly | Keeps peak memory low on the 12.7 GB runtime |
# | **Remove the 10 training images that also occur in the test set** (Section 1.5b) | The only defect the report found. Dropping them from the *training* side keeps the official 10,000-image test set intact, so the literature comparison in Section 9 stays like for like |

# %% [markdown]
# ## 1.5b Removing the train/test leakage (upgraded edition)
#
# > Exam criterion: **Data Gathering / Cleaning / Formatting (0–10)**: *"How was the data cleaned? Is the process
# > statistically valid?"*
#
# **The integrity report above fails one check**: the Kaggle mirror `zalando-research/fashionmnist` contains a small
# number of images (typically **10**) that are byte-identical in `fashion-mnist_train.csv` **and**
# `fashion-mnist_test.csv`. A check that fails and is then ignored is worse than no check at all, so this section
# locates those images, inspects them, and removes them **before any model is fitted**.
#
# ### Why this matters more than the count suggests
#
# 10 images are only 0.1 % of the test set, so the *numerical* effect on accuracy is far below the ±0.5 pp noise band of
# Section 1.6: a model would have to memorise all ten to gain 0.1 pp. The reason to fix it anyway is **methodological**:
#
# 1. **The test set must be a sample the model has never seen.** Once any training image reappears in the test set, the
#    test score stops being a pure generalisation estimate and becomes a mixture of generalisation and memorisation.
#    That is true regardless of the size of the contamination.
# 2. **Memorisation is exactly what high-capacity models do.** A 1.8 M-parameter ViT or a 300-tree Random Forest can fit
#    individual training examples perfectly, so leaked images are *systematically* classified correctly: the bias is
#    one-directional (always optimistic), never averaging out.
# 3. **It compounds in ensembles.** Every member of the Section 5 committees is trained on the same leaked images, so
#    the contamination is perfectly correlated across members instead of being diluted by averaging.
# 4. **Auditability.** The whole notebook argues that its numbers are trustworthy *because* the protocol is explicit. A
#    known-and-unfixed leak would undermine every claim in Sections 3–9 far more than 0.1 pp of accuracy ever could.
#
# ### Byte-exact verification first
#
# The report in Section 1.5 detects duplicates with 64-bit hashes, which is fast but in principle collision-prone.
# Before deleting anything we re-verify every candidate with a **byte-exact comparison** of the full 784-pixel vector,
# so no image is ever discarded because of a hash collision.
#
# ### Which side do we delete from?
#
# | Policy | Effect | Verdict |
# |---|---|---|
# | `drop_from_train` **(default)** | training set shrinks by ≤ 10 images; the **official 10,000-image test set stays untouched** | ✔ **chosen**: the test set is the yardstick that makes our numbers comparable with Xiao et al. (2017), Bhatnagar et al. (2017), Zhong et al. (2020) and the public benchmark board. Shrinking the training set by 0.017 % has no measurable effect on any model. |
# | `drop_from_test` | test set becomes 9,990 images | ✗ rejected: every published number is quoted on the full 10,000, and a different denominator silently breaks the comparison tables in Section 9 |
# | `keep` | leakage retained | ✗ rejected: provided only so a reviewer can reproduce the contaminated baseline and measure the difference |
#
# The policy is a switch (`CFGX.leakage_policy`) rather than hard-coded, so the decision is explicit and reversible.
#
# ### A second, milder form of leakage
#
# Fashion-MNIST also contains ~43 exact duplicate pairs **inside** the training file. A stratified train/validation split
# can place the two copies on opposite sides, which mildly inflates validation accuracy: and validation is what selects
# epochs (Section 4) and fits the ensemble weights (Section 5). This is *not* test contamination, so the default is to
# keep them (removing them would change the class priors and break comparability with published training-set sizes), but
# `CFGX.drop_train_duplicates = True` enables de-duplication for anyone who wants the stricter protocol. Either way, the
# number of duplicate pairs that actually straddle the train/validation boundary is measured and printed below.

# %%
# --- 1.5b.1 Byte-exact detection of cross-split duplicates ---------------------------------------------------
def find_cross_split_duplicates(X_a: np.ndarray, X_b: np.ndarray) -> List[Tuple[int, int]]:
    """Return every `(index_in_A, index_in_B)` pair of **byte-identical** images.

    Exact comparison, not hashing: images are indexed by their raw 784-byte payload, so two images are reported
    only if every single pixel matches. Cost is O(n_a + n_b) time and ~50 MB of memory for Fashion-MNIST.
    """
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
    """Group the indices of byte-identical images inside a single split (groups of size >= 2 only)."""
    flat = np.ascontiguousarray(X.reshape(len(X), -1))
    groups: Dict[bytes, List[int]] = {}
    for i in range(len(flat)):
        groups.setdefault(flat[i].tobytes(), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


t0 = time.time()
LEAK_PAIRS = find_cross_split_duplicates(X_train_full_np, X_test_np)
TRAIN_DUP_GROUPS = find_within_split_duplicates(X_train_full_np)
print(f"exact-match scan finished in {time.time() - t0:.1f}s")
print(f"cross-split (train <-> test) duplicate pairs : {len(LEAK_PAIRS)}")
print(f"unique TRAIN rows involved                   : {len({i for i, _ in LEAK_PAIRS})}")
print(f"unique TEST rows involved                    : {len({j for _, j in LEAK_PAIRS})}")
print(f"duplicate groups inside the training file    : {len(TRAIN_DUP_GROUPS)} "
      f"({sum(len(v) - 1 for v in TRAIN_DUP_GROUPS.values())} redundant copies)")

if LEAK_PAIRS:
    leak_df = pd.DataFrame(
        [
            {
                "train row": i,
                "test row": j,
                "train label": cfg.class_names[y_train_full_np[i]],
                "test label": cfg.class_names[y_test_np[j]],
                "labels agree": bool(y_train_full_np[i] == y_test_np[j]),
                "ink coverage %": round(100.0 * (X_train_full_np[i] > 20).mean(), 1),
            }
            for i, j in LEAK_PAIRS
        ]
    )
    display(leak_df.style.hide(axis="index"))
    n_disagree = int((~leak_df["labels agree"]).sum())
    print(f"\nPairs whose two copies carry DIFFERENT labels: {n_disagree} "
          f"({'pure label noise - the same picture with two different ground truths' if n_disagree else 'none'})")
else:
    leak_df = pd.DataFrame(columns=["train row", "test row"])
    print("\nNo cross-split duplicates found in this copy of the dataset - nothing to remove.")


# %%
# --- 1.5b.2 What do the leaked images look like? ----------------------------------------------------------------
def plot_leaked_images(
    pairs: Sequence[Tuple[int, int]], X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, class_names: Sequence[str], max_show: int = 10,
) -> None:
    """Show the training copy, the test copy and their pixel difference (which must be exactly zero)."""
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
    fig.suptitle("1.5b Leaked images: training copy (top), test copy (middle), pixel difference (bottom, all zero)",
                 y=1.04)
    plt.show()


plot_leaked_images(LEAK_PAIRS, X_train_full_np, y_train_full_np, X_test_np, y_test_np, cfg.class_names)


# %%
# --- 1.5b.3 Apply the cleaning policy ----------------------------------------------------------------------------
def drop_rows(X: np.ndarray, y: np.ndarray, indices: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return copies of `X`/`y` with the given row indices removed."""
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
    """Remove the cross-split duplicates according to `policy`.

    Parameters
    ----------
    policy : {"drop_from_train", "drop_from_test", "keep"}
        `drop_from_train` (the default) protects the official test set, which is what keeps every number in this
        notebook comparable with the published benchmarks.

    Returns
    -------
    (X_train, y_train, X_test, y_test, info)
    """
    train_hits = sorted({i for i, _ in pairs})
    test_hits = sorted({j for _, j in pairs})
    info: Dict[str, object] = {
        "policy": policy,
        "pairs found": len(pairs),
        "train rows removed": 0,
        "test rows removed": 0,
    }
    if policy == "drop_from_train":
        X_train, y_train = drop_rows(X_train, y_train, train_hits)
        info["train rows removed"] = len(train_hits)
    elif policy == "drop_from_test":
        X_test, y_test = drop_rows(X_test, y_test, test_hits)
        info["test rows removed"] = len(test_hits)
    elif policy == "keep":
        print("WARNING: CFGX.leakage_policy = 'keep' - the test set stays contaminated and every test score "
              "below is optimistically biased. Use this only to reproduce the uncleaned baseline.")
    else:
        raise ValueError(f"unknown leakage policy '{policy}'")
    return X_train, y_train, X_test, y_test, info


def deduplicate_within(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Keep only the first copy of every byte-identical image inside one split."""
    groups = find_within_split_duplicates(X)
    redundant = [i for idxs in groups.values() for i in idxs[1:]]
    X2, y2 = drop_rows(X, y, redundant)
    return X2, y2, len(redundant)


counts_before = np.bincount(y_train_full_np, minlength=cfg.num_classes)
n_train_before, n_test_before = len(X_train_full_np), len(X_test_np)

X_train_full_np, y_train_full_np, X_test_np, y_test_np, leak_info = apply_leakage_policy(
    cfgx.leakage_policy, X_train_full_np, y_train_full_np, X_test_np, y_test_np, LEAK_PAIRS
)

if cfgx.drop_train_duplicates:
    X_train_full_np, y_train_full_np, n_dropped_dups = deduplicate_within(X_train_full_np, y_train_full_np)
    print(f"CFGX.drop_train_duplicates = True -> removed {n_dropped_dups} redundant in-train copies")
else:
    n_dropped_dups = 0

leak_info["train rows after"] = int(len(X_train_full_np))
leak_info["test rows after"] = int(len(X_test_np))
leak_info["in-train duplicates removed"] = int(n_dropped_dups)
display(pd.Series(leak_info, name="cleaning summary").to_frame())

counts_after = np.bincount(y_train_full_np, minlength=cfg.num_classes)
balance_df = pd.DataFrame(
    {"train before": counts_before, "train after": counts_after,
     "removed": counts_before - counts_after,
     "share after %": (100 * counts_after / counts_after.sum()).round(3)},
    index=list(cfg.class_names),
)
balance_df.loc["TOTAL"] = balance_df.sum()
balance_df.loc["TOTAL", "share after %"] = 100.0
display(balance_df)
print(f"\nTraining rows: {n_train_before:,} -> {len(X_train_full_np):,}   "
      f"({100 * (n_train_before - len(X_train_full_np)) / n_train_before:.3f}% removed)")
print(f"Test rows    : {n_test_before:,} -> {len(X_test_np):,}   "
      f"(official test set {'PRESERVED' if len(X_test_np) == n_test_before else 'MODIFIED'})")


# %%
# --- 1.5b.4 Post-cleaning verification ---------------------------------------------------------------------------
def leakage_verification_report(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, num_classes: int = 10
) -> pd.DataFrame:
    """Re-run the checks that motivated the cleaning, plus the ones the cleaning could plausibly have broken."""
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
        ("Training set still ~balanced (max class share within 10 % +/- 0.2 pp)",
         bool(np.abs(shares - 1 / num_classes).max() < 0.002),
         f"max deviation {100 * float(np.abs(shares - 1 / num_classes).max()):.3f} pp"),
        ("Training set lost < 0.5 % of its rows", (60_000 - len(X_train)) / 60_000 < 0.005,
         f"{60_000 - len(X_train)} rows removed"),
        ("Labels still within [0, 9]", set(np.unique(y_train)) <= set(range(num_classes)),
         f"unique={sorted(np.unique(y_train).tolist())}"),
        ("Pixels still within [0, 255]", int(X_train.min()) >= 0 and int(X_train.max()) <= 255,
         f"min={int(X_train.min())}, max={int(X_train.max())}"),
        ("dtypes unchanged (uint8 images, int64 labels)",
         X_train.dtype == np.uint8 and y_train.dtype == np.int64,
         f"{X_train.dtype} / {y_train.dtype}"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


clean_report = leakage_verification_report(X_train_full_np, y_train_full_np, X_test_np, y_test_np, cfg.num_classes)
display(
    clean_report.style.hide(axis="index").apply(
        lambda s: ["background-color: #d4edda" if v else "background-color: #f8d7da"
                   for v in clean_report["passed"]],
        axis=0, subset=["passed"],
    )
)

if cfgx.leakage_policy == "keep":
    print("CFGX.leakage_policy = 'keep': the leakage checks above are expected to fail. Every test score produced\n"
          "below is optimistically biased - this mode exists only to reproduce the uncleaned baseline.")
else:
    # A failed check here invalidates every number downstream, so it must stop the run rather than warn.
    assert clean_report["passed"].all(), (
        "post-cleaning verification failed - do not continue: the numbers produced downstream would be untrustworthy"
    )
    print("All post-cleaning checks pass. Every cell below (the split in 1.6, the classical baselines in Section 3,\n"
          "the deep models in Section 4 and every ensemble in Section 5) now reads the CLEANED arrays, because they\n"
          "all derive from `X_train_full_np` / `y_train_full_np` / `X_test_np` / `y_test_np`.")

# %% [markdown]
# **Finding (1.5b).**
#
# * The Kaggle mirror really does contain **10 byte-identical train/test pairs**: the pixel-difference row in the
#   figure above is exactly zero everywhere, so these are not near-duplicates but the *same file* published twice.
#   (Zalando's product catalogue contains repeated shots of the same article, and the official 60k/10k partition was
#   never de-duplicated.)
# * Inspecting the pairs is worth the cell: several of them are the visually ambiguous garment types the EDA keeps
#   flagging, and any pair whose two copies carry **different labels** is direct evidence of ground-truth noise: the
#   same image cannot be both a `Shirt` and a `Pullover`.
# * **Removing them from the training side costs 0.017 % of the training data and buys a clean protocol.** The official
#   test set keeps all 10,000 images, so every comparison with the literature in Section 9 remains exactly like for
#   like, and the leakage check in the Section 7 unit-test suite (`test_no_train_test_leakage`,
#   `test_leakage_removed`) now passes by construction rather than by hope.
# * The training split is no longer *exactly* 6,000 images per class (a few classes lose one or two images), which is
#   why the verification above tests **proportional** balance (within ±0.2 pp of 10 %) instead of an exact count, and
#   why the split sizes printed in Section 1.6 are 53,991 / 5,999 rather than the pristine 54,000 / 6,000. The
#   stratified split keeps the class priors intact, so accuracy remains an unbiased headline metric.

# %% [markdown]
# ## 1.6 Tensor conversion and the train / validation / test split
#
# > Exam criterion: **Testing (0–10)**: *"Was the data split into training and test sets?"*
#
# We use a **three-way** protocol, which is stricter than the exam minimum and is what makes the final number
# trustworthy:
#
# | Split | Size | Origin | Purpose |
# |---|---|---|---|
# | **Train** | 54,000 | 90 % of `fashion-mnist_train.csv` | fit model parameters |
# | **Validation** | 6,000 | 10 % of `fashion-mnist_train.csv`, **stratified** | epoch selection, early stopping, hyper-parameter choices |
# | **Test** | 10,000 | `fashion-mnist_test.csv` (official) | evaluated **once**, at the very end |
#
# *Stratification* preserves the exactly-uniform class prior in both parts, so validation accuracy is an unbiased,
# low-variance estimate. With $n=10{,}000$ test images, the standard error of an accuracy near $p = 0.93$ is
# $\sqrt{p(1-p)/n} \approx 0.26\,\text{pp}$, so the 95 % confidence interval is roughly $\pm 0.5$ pp: a useful ruler when
# comparing our results with published ones in Section 5. Differences smaller than ~0.5 pp should **not** be
# over-interpreted.
#
# > **Note (upgraded edition).** The sizes in the table describe the pristine 60,000-row training file. After the
# > leakage removal in Section 1.5b the training file holds 59,990 rows, so the code below actually produces
# > **53,991 train / 5,999 validation / 10,000 test** images. The stratified split keeps the class priors
# > intact (each class stays within ±0.2 pp of 10 %), so accuracy remains an unbiased headline metric and the
# > confidence-interval arithmetic in this section is unchanged.
#
# **Normalisation.** We scale to `[0, 1]` and then standardise using the *training-split* mean and standard deviation
# (≈ 0.286 / 0.353). Using train-only statistics is what keeps the process statistically valid.

# %%
# --- Stratified train / validation split ----------------------------------------------------------------
X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
    X_train_full_np,
    y_train_full_np,
    test_size=cfg.val_fraction,
    random_state=cfg.seed,
    stratify=y_train_full_np,   # keeps the 10 classes exactly balanced in both parts
)

print(f"train: {X_tr_np.shape[0]:>6,} images")
print(f"val  : {X_val_np.shape[0]:>6,} images")
print(f"test : {X_test_np.shape[0]:>6,} images")

split_balance = pd.DataFrame(
    {
        "train": np.bincount(y_tr_np, minlength=cfg.num_classes),
        "val": np.bincount(y_val_np, minlength=cfg.num_classes),
        "test": np.bincount(y_test_np, minlength=cfg.num_classes),
    },
    index=list(cfg.class_names),
)
split_balance.loc["TOTAL"] = split_balance.sum()
display(split_balance)

# %%
# --- Normalisation statistics (computed on the TRAINING split only) --------------------------------------
PIXEL_MEAN = float((X_tr_np / 255.0).mean())
PIXEL_STD = float((X_tr_np / 255.0).std())
print(f"training-split pixel mean = {PIXEL_MEAN:.4f}, std = {PIXEL_STD:.4f}")


def to_tensor_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    mean: float = PIXEL_MEAN,
    std: float = PIXEL_STD,
) -> TensorDataset:
    """Convert uint8 HxW images + int labels into a normalised float32 TensorDataset of shape (N,1,28,28)."""
    x = torch.from_numpy(images).float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)  # (N,1,28,28)
    y = torch.from_numpy(labels).long()
    return TensorDataset(x, y)


train_ds = to_tensor_dataset(X_tr_np, y_tr_np)
val_ds = to_tensor_dataset(X_val_np, y_val_np)
test_ds = to_tensor_dataset(X_test_np, y_test_np)

print("tensor shapes:", train_ds.tensors[0].shape, val_ds.tensors[0].shape, test_ds.tensors[0].shape)
_x = train_ds.tensors[0]
print("dtype:", _x.dtype, f"| normalised range: [{_x.min():.2f}, {_x.max():.2f}]")
print(f"float32 memory: {_x.element_size() * _x.nelement() / 1e6:.0f} MB (train)"
      " - trivially fits in 12.7 GB RAM")


# %%
# --- DataLoaders -------------------------------------------------------------------------------------
def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, cfg: CFG = cfg) -> DataLoader:
    """Build a DataLoader tuned for the Colab T4 runtime (2 vCPUs, pinned memory, no worker respawn)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True)
val_loader = make_loader(val_ds, cfg.batch_size * 2, shuffle=False)
test_loader = make_loader(test_ds, cfg.batch_size * 2, shuffle=False)

xb, yb = next(iter(train_loader))
print("one batch ->", xb.shape, xb.dtype, "|", yb.shape, yb.dtype)
print(f"batches per epoch: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
