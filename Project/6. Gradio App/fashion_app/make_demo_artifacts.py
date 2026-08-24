"""
make_demo_artifacts.py: build a small but authentic demo `artifacts/` tree
================================================================================
Trains miniature versions of the notebook's models on the official Fashion-MNIST
data (downloaded from the Zalando GitHub mirror) and saves them in EXACTLY the
format produced by DL-Fashion-MNIST.ipynb:

    artifacts/models/ml/*.joblib (+ *.json sidecars)      sklearn pipelines & boosters
    artifacts/models/dl/*.pt       (+ *.json sidecars)    torch state_dicts
    artifacts/models/ensemble/*.joblib (+ sidecars)       combiners
    artifacts/models/{ml,dl,ensemble}/BEST.json
    artifacts/final_leaderboard.csv, run_summary.json, artifact_manifest.csv

It also writes a few demo `sample_photos/`. Run once (~5-10 min on 2 CPU cores):
    python make_demo_artifacts.py
"""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fashion_arch import (  # noqa: E402
    CLASS_NAMES, MLP, CNN, ResNetSmall, VisionTransformer, normalize_batch,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "_data"
ART = HERE / "artifacts"
MODELS = ART / "models"
ML_DIR, DL_DIR, ENS_DIR = MODELS / "ml", MODELS / "dl", MODELS / "ensemble"
PHOTOS = HERE / "sample_photos"

N_TRAIN = 5_000          # demo subset (the notebook used ~54,000 — accuracies here are lower)
N_TEST = 1_500
SEED = 42

URLS = {
    "train-images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train-labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test-images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test-labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------- data
def _download(key: str) -> bytes:
    cache = DATA_DIR / f"{key}.gz"
    if not cache.exists():
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        print("downloading", URLS[key])
        urllib.request.urlretrieve(URLS[key], cache)
    return gzip.decompress(cache.read_bytes())


def _parse_images(buf: bytes) -> np.ndarray:
    magic, n, h, w = np.frombuffer(buf[:16], dtype=">u4")
    assert magic == 2051, magic
    return np.frombuffer(buf[16:], dtype=np.uint8).reshape(int(n), int(h), int(w)).copy()


def _parse_labels(buf: bytes) -> np.ndarray:
    magic, n = np.frombuffer(buf[:8], dtype=">u4")
    assert magic == 2049, magic
    return np.frombuffer(buf[8:], dtype=np.uint8).astype(np.int64).copy()


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xtr = _parse_images(_download("train-images"))
    ytr = _parse_labels(_download("train-labels"))
    Xte = _parse_images(_download("test-images"))
    yte = _parse_labels(_download("test-labels"))
    rng = np.random.default_rng(SEED)
    idx_tr = rng.choice(len(Xtr), N_TRAIN, replace=False)
    idx_te = rng.choice(len(Xte), N_TEST, replace=False)
    return Xtr[idx_tr], ytr[idx_tr], Xte[idx_te], yte[idx_te]


# ---------------------------------------------------------------- persistence
def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def sidecar(name: str, family: str, acc: float, f1: float, **extra) -> dict:
    return {
        "model": name, "family": family, "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4), "weighted_f1": round(float(f1), 4),
        "val_accuracy": float("nan"), "fit_seconds": 0.0, "predict_seconds": 0.0,
        "params": None, "notes": "demo artifact (trained on a small subset)",
        "selection_metric": "best_validation_accuracy" if family == "Deep Learning" else "test_accuracy",
        "saved_at": datetime.now().replace(microsecond=0).isoformat(),
        "python": "demo", "numpy": np.__version__, **extra,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


REGISTRY: List[dict] = []


def save_ml(name: str, family: str, obj, acc: float, f1: float, features: str = "flat") -> None:
    path = ML_DIR / f"{slugify(name)}.joblib"
    joblib.dump(obj, path)
    side = sidecar(name, family, acc, f1, file=path.name, format="joblib",
                   kind="sklearn_estimator", features=features)
    write_json(ML_DIR / f"{slugify(name)}.json", side)
    REGISTRY.append({"model": name, "family": family, "folder": "ml", "accuracy": acc})


def save_dl(name: str, model: nn.Module, acc: float, f1: float,
            arch_class: str, arch_kwargs: dict) -> None:
    path = DL_DIR / f"{slugify(name)}.pt"
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)
    side = sidecar(name, "Deep Learning", acc, f1, file=path.name, format="state_dict",
                   kind="torch_module", arch_class=arch_class, arch_kwargs=arch_kwargs,
                   n_parameters=int(sum(p.numel() for p in model.parameters())))
    write_json(DL_DIR / f"{slugify(name)}.json", side)
    REGISTRY.append({"model": name, "family": "Deep Learning", "folder": "dl", "accuracy": acc})


def save_ens(name: str, combiner: str, members: List[str], acc: float, f1: float,
             weights=None, meta=None) -> None:
    path = ENS_DIR / f"{slugify(name)}.joblib"
    joblib.dump({"name": name, "combiner": combiner, "members": members,
                 "weights": (np.asarray(weights).tolist() if weights is not None else None),
                 "meta": meta}, path)
    side = sidecar(name, "Ensemble", acc, f1, file=path.name, format="joblib",
                   kind="ensemble_combiner", combiner=combiner, n_members=len(members),
                   members=members)
    write_json(ENS_DIR / f"{slugify(name)}.json", side)
    REGISTRY.append({"model": name, "family": "Ensemble", "folder": "ensemble", "accuracy": acc})


# ---------------------------------------------------------------- torch training
def train_torch(model: nn.Module, xtr: torch.Tensor, ytr: torch.Tensor,
                xval: torch.Tensor, yval: torch.Tensor, epochs: int, lr: float,
                wd: float = 5e-4) -> Tuple[nn.Module, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_state, best_acc = None, -1.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xval).argmax(1) == yval).float().mean())
        print(f"    epoch {ep + 1}/{epochs}  val acc = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def torch_probs(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.softmax(model(x), dim=1).numpy()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    for d in (ML_DIR, DL_DIR, ENS_DIR, PHOTOS, PHOTOS / "white_bg"):
        d.mkdir(parents=True, exist_ok=True)

    print("loading Fashion-MNIST …")
    Xtr_u8, ytr, Xte_u8, yte = load_data()
    print("train", Xtr_u8.shape, "test", Xte_u8.shape)

    # splits & tensors -------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(Xtr_u8, ytr, test_size=0.15,
                                                      random_state=SEED, stratify=ytr)
    xtr_t = normalize_batch(torch.from_numpy(X_train))
    ytr_t = torch.from_numpy(y_train)
    xval_t = normalize_batch(torch.from_numpy(X_val))
    yval_t = torch.from_numpy(y_val)
    xte_t = normalize_batch(torch.from_numpy(Xte_u8))

    flat = lambda X: X.reshape(len(X), -1).astype(np.float32) / 255.0  # noqa: E731
    F_train, F_val, F_test = flat(X_train), flat(X_val), flat(Xte_u8)

    def scores(y_true, P):
        pred = P.argmax(1)
        return accuracy_score(y_true, pred), f1_score(y_true, pred, average="macro")

    # ---------------- classical models (sklearn pipelines like the notebook)
    print("\n[ml] Logistic Regression / Linear SVM / RBF SVM / Random Forest")
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, n_jobs=1))])
    logreg.fit(F_train, y_train)
    a, f = scores(yte, logreg.predict_proba(F_test))
    save_ml("Logistic Regression", "Classical ML", logreg, a, f)
    print("  logreg acc", a)

    linsvm = Pipeline([("scaler", StandardScaler()),
                       ("clf", LinearSVC(C=0.01, dual="auto", max_iter=5000))])
    linsvm.fit(F_train, y_train)
    a, f = scores(yte, np.eye(10)[linsvm.predict(F_test)])  # app derives probs from decision_function
    save_ml("Linear SVM", "Classical ML", linsvm, a, f)
    print("  linear svm acc", a)

    rbf = Pipeline([("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.90, random_state=SEED)),
                    ("clf", SVC(C=10.0, gamma="scale", kernel="rbf", probability=True))])
    rbf.fit(F_train[:2500], y_train[:2500])
    a, f = scores(yte, rbf.predict_proba(F_test))
    save_ml("RBF SVM (PCA-90%)", "Classical ML", rbf, a, f)
    print("  rbf svm acc", a)

    rf = Pipeline([("clf", RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                                  n_jobs=2, random_state=SEED))])
    rf.fit(F_train[:2500], y_train[:2500])
    a, f = scores(yte, rf.predict_proba(F_test))
    save_ml("Random Forest (100 trees)", "Classical ML", rf, a, f)
    print("  random forest acc", a)

    # ---------------- gradient boosting on PCA-80 features (notebook layout)
    print("\n[ml] gradient boosting on PCA-80 (+ exported pca_boost.joblib)")
    pca_boost = PCA(n_components=80, random_state=SEED).fit(F_train)
    joblib.dump(pca_boost, ML_DIR / "pca_boost.joblib")   # <- enables the app to use PCA models
    P_train, P_val, P_test = pca_boost.transform(F_train), pca_boost.transform(F_val), pca_boost.transform(F_test)

    import xgboost as xgb
    import lightgbm as lgb
    xgbm = xgb.XGBClassifier(n_estimators=120, learning_rate=0.2, max_depth=6,
                             objective="multi:softprob", tree_method="hist",
                             random_state=SEED, n_jobs=2, verbosity=0)
    xgbm.fit(P_train, y_train)
    a, f = scores(yte, xgbm.predict_proba(P_test))
    save_ml("XGBoost (PCA-80)", "Gradient Boosting", xgbm, a, f, features="pca")
    print("  xgboost acc", a)

    lgbm = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=63,
                              subsample=0.9, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=2, verbose=-1)
    lgbm.fit(F_train, y_train)
    a, f = scores(yte, lgbm.predict_proba(F_test))
    save_ml("LightGBM (raw 784 px)", "Gradient Boosting", lgbm, a, f, features="flat")
    print("  lightgbm acc", a)

    # ---------------- deep models (torch state_dicts like the notebook)
    print("\n[dl] MLP")
    mlp = MLP()
    mlp, vacc = train_torch(mlp, xtr_t, ytr_t, xval_t, yval_t, epochs=4, lr=1e-3)
    a, f = scores(yte, torch_probs(mlp, xte_t))
    save_dl("MLP (512-256)", mlp, a, f, "MLP", {})
    print("  mlp acc", a)

    print("[dl] CNN")
    cnn = CNN()
    cnn, vacc = train_torch(cnn, xtr_t, ytr_t, xval_t, yval_t, epochs=4, lr=1.5e-3)
    a, f = scores(yte, torch_probs(cnn, xte_t))
    save_dl("CNN (VGG-style, GAP)", cnn, a, f, "CNN", {})
    print("  cnn acc", a)

    print("[dl] ResNet-small")
    resnet = ResNetSmall(width=32)
    resnet, vacc = train_torch(resnet, xtr_t, ytr_t, xval_t, yval_t, epochs=3, lr=1.5e-3)
    a, f = scores(yte, torch_probs(resnet, xte_t))
    save_dl("ResNet-small (residual CNN)", resnet, a, f, "ResNetSmall", {"width": 32})
    print("  resnet acc", a)

    print("[dl] ViT-tiny")
    vit = VisionTransformer(dim=128, depth=6, heads=4)
    vit, vacc = train_torch(vit, xtr_t, ytr_t, xval_t, yval_t, epochs=3, lr=8e-4)
    a, f = scores(yte, torch_probs(vit, xte_t))
    save_dl("ViT-tiny (16 patches)", vit, a, f, "VisionTransformer",
            {"dim": 128, "depth": 6, "heads": 4, "patch": 7})
    print("  vit acc", a)

    # ---------------- ensembles (member probs on val/test like the notebook)
    print("\n[ensemble] voting & stacking")
    deep = {"MLP (512-256)": mlp, "CNN (VGG-style, GAP)": cnn,
            "ResNet-small (residual CNN)": resnet, "ViT-tiny (16 patches)": vit}
    deep_members = list(deep)
    val_probs = {n: torch_probs(m, xval_t) for n, m in deep.items()}
    test_probs = {n: torch_probs(m, xte_t) for n, m in deep.items()}
    val_probs["Logistic Regression"] = logreg.predict_proba(F_val)
    test_probs["Logistic Regression"] = logreg.predict_proba(F_test)
    val_probs["LightGBM (raw 784 px)"] = lgbm.predict_proba(F_val)
    test_probs["LightGBM (raw 784 px)"] = lgbm.predict_proba(F_test)

    soft_test = np.mean([test_probs[n] for n in deep_members], axis=0)
    a, f = scores(yte, soft_test)
    save_ens("DL soft voting (equal)", "soft_vote", deep_members, a, f)
    print("  dl soft voting acc", a)

    hybrid_members = deep_members + ["Logistic Regression", "LightGBM (raw 784 px)"]
    w = np.array([0.22, 0.28, 0.28, 0.10, 0.06, 0.06])          # hand-set for the demo
    hyb_test = np.tensordot(w, np.stack([test_probs[n] for n in hybrid_members]), axes=(0, 0))
    a, f = scores(yte, hyb_test)
    save_ens("Hybrid ML+DL weighted voting", "weighted_vote", hybrid_members, a, f, weights=w)
    print("  hybrid weighted acc", a)

    X_meta_val = np.hstack([val_probs[n] for n in hybrid_members])
    X_meta_test = np.hstack([test_probs[n] for n in hybrid_members])
    meta = LogisticRegression(max_iter=1000, C=1.0)
    meta.fit(X_meta_val, y_val)
    a, f = scores(yte, meta.predict_proba(X_meta_test))
    save_ens("Hybrid ML+DL stacking", "stacking", hybrid_members, a, f, meta=meta)
    print("  hybrid stacking acc", a)

    # ---------------- best pointers & manifests (notebook layout)
    for folder in ("ml", "dl", "ensemble"):
        rows = [r for r in REGISTRY if r["folder"] == folder]
        if rows:
            best = max(rows, key=lambda r: r["accuracy"])
            slug = slugify(best["model"])
            write_json(MODELS / folder / "BEST.json", {
                "folder": folder, "best_model": best["model"],
                "accuracy": best["accuracy"], "macro_f1": 0.0,
                "file": slug + (".pt" if folder == "dl" else ".joblib"),
                "saved_at": datetime.now().isoformat()})

    lb = pd.DataFrame(REGISTRY).sort_values("accuracy", ascending=False)
    lb.to_csv(ART / "final_leaderboard.csv", index=False)
    write_json(ART / "run_summary.json", {
        "demo": True, "note": "miniature demo artifacts mimicking DL-Fashion-MNIST.ipynb "
                              "(small subset, few epochs)", "models": len(REGISTRY),
        "best_model": lb.iloc[0]["model"], "best_accuracy": float(lb.iloc[0]["accuracy"])})
    manifest = pd.DataFrame([{"family_folder": r["folder"], "file": slugify(r["model"]) +
                              (".pt" if r["folder"] == "dl" else ".joblib")} for r in REGISTRY])
    manifest.to_csv(ART / "models_manifest.csv", index=False)

    # ---------------- demo sample photos
    print("\n[sample photos]")
    rng = np.random.default_rng(7)
    picks = rng.choice(len(Xte_u8), size=min(8, len(Xte_u8)), replace=False)
    for k, i in enumerate(picks):
        big = Image.fromarray(Xte_u8[i]).resize((224, 224), Image.LANCZOS)
        big.save(PHOTOS / f"fmnist_style_{k + 1}_{CLASS_NAMES[yte[i]].replace('/', '_').replace(' ', '_')}.png")
    # photos on a WHITE background (tests the auto-invert path)
    for k, i in enumerate(rng.choice(len(Xte_u8), size=3, replace=False)):
        arr = 255 - Xte_u8[i]                       # dark garment on white
        Image.fromarray(arr).resize((224, 224), Image.LANCZOS).save(
            PHOTOS / "white_bg" / f"product_shot_{k + 1}_inverted.png")

    print("\nDone. Artifacts ->", ART)
    print("Sample photos ->", PHOTOS)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f}s")
