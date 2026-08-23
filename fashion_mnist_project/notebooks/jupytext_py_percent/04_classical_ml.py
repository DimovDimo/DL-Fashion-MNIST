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

# %% [markdown]
# <a id="sec3"></a>
# # 3. Traditional machine-learning baselines
#
# > Exam criterion: **Testing (0–10)**: *"Were different models compared? Which metrics were used?"*
#
# **Why bother with classical models in a deep-learning exam?** Because a deep model without a baseline is an
# unfalsifiable claim. These four baselines quantify how much of the task is solvable with (a) a linear decision
# boundary, (b) a kernel method, and (c) an axis-aligned ensemble: and therefore how much *extra* value the
# convolutional prior actually adds (RQ1).
#
# ## 3.1 Experimental protocol
#
# | Design choice | Value | Justification |
# |---|---|---|
# | Training subset | `CFG.sk_train_subset = 12,000` **stratified** samples | An RBF-SVM is $O(n^2)$–$O(n^3)$: fitting on all 54,000 images takes 1–3 hours on Colab's 2 vCPUs, vs. ~3 minutes on 12,000. Learning curves for Fashion-MNIST are already flat by ~10k, so the accuracy cost is ~1 pp. |
# | Evaluation set | the **full official 10,000-image test set** | keeps every number in this notebook directly comparable with the deep models and with published benchmarks |
# | Preprocessing | `StandardScaler` (+ `PCA(0.9)` for the SVMs) | scaling is required for gradient/kernel methods; PCA cuts the SVM cost roughly 8x with no accuracy loss |
# | Metrics | accuracy, macro-F1, fit time, predict time | accuracy is valid on a balanced set; macro-F1 exposes per-class weakness; the timings support the cost-per-point discussion |
#
# All models are wrapped in `sklearn.pipeline.Pipeline` objects so that preprocessing is **fitted on the training data
# only**: a scaler fitted on the test set would be a textbook leakage bug.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])


# %%
# --- Flat, scaled feature matrices for scikit-learn -----------------------------------------------------------
def make_flat_arrays(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten (N,28,28) uint8 images into (N,784) float32 in [0,1] for scikit-learn."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0, labels


X_tr_flat, y_tr_flat = make_flat_arrays(X_tr_np, y_tr_np)
X_val_flat, y_val_flat = make_flat_arrays(X_val_np, y_val_np)
X_test_flat, y_test_flat = make_flat_arrays(X_test_np, y_test_np)

# Stratified subsample used to FIT the classical models (see the protocol table above)
if cfg.sk_train_subset < len(X_tr_flat):
    X_sk, _, y_sk, _ = train_test_split(
        X_tr_flat,
        y_tr_flat,
        train_size=cfg.sk_train_subset,
        random_state=cfg.seed,
        stratify=y_tr_flat,
    )
else:
    X_sk, y_sk = X_tr_flat, y_tr_flat

print(f"classical-model training matrix: {X_sk.shape}  ({X_sk.nbytes / 1e6:.0f} MB)")
print("class counts in the subsample:", np.bincount(y_sk, minlength=cfg.num_classes).tolist())
print(f"evaluation matrix (official test set): {X_test_flat.shape}")

# %%
# --- Generic evaluation helper reused by EVERY model in the notebook -------------------------------------------
RESULTS: List[Dict[str, object]] = []   # global results registry -> final comparison tables


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    family: str,
    fit_seconds: float = float("nan"),
    predict_seconds: float = float("nan"),
    n_params: int | float = float("nan"),
    notes: str = "",
    register: bool = True,
) -> Dict[str, object]:
    """Compute the standard metric set for a set of predictions and append it to the global registry."""
    record: Dict[str, object] = {
        "model": model_name,
        "family": family,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "error_rate": 1.0 - accuracy_score(y_true, y_pred),
        "fit_s": fit_seconds,
        "predict_s": predict_seconds,
        "params": n_params,
        "notes": notes,
    }
    if register:
        RESULTS.append(record)
    print(
        f"{model_name:<28s} acc={record['accuracy']:.4f}  macro-F1={record['macro_f1']:.4f}"
        f"  fit={fit_seconds:6.1f}s  predict={predict_seconds:5.1f}s"
    )
    return record


def fit_and_evaluate_sklearn(
    name: str,
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    notes: str = "",
) -> Tuple[Pipeline, np.ndarray, Dict[str, object]]:
    """Fit a scikit-learn pipeline, time it, evaluate it on the test set and register the result."""
    print(f"\n>>> Training {name} on {X_train.shape[0]:,} samples ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_s = time.time() - t0

    t0 = time.time()
    y_pred = pipeline.predict(X_eval)
    pred_s = time.time() - t0

    record = evaluate_predictions(
        y_eval, y_pred, name, family="Classical ML", fit_seconds=fit_s, predict_seconds=pred_s, notes=notes
    )
    register_model(
        name, family="Classical ML", artifact=pipeline, record=record, selector=float(record["accuracy"])
    )
    return pipeline, y_pred, record


# %% [markdown]
# ### 3.1b Persisting the best version of every trained model
#
# A trained model that only lives in memory is lost the moment the kernel stops. This project therefore persists
# the **best version of every model** (classical, deep-learning and ensemble alike) to disk, in a separate folder
# per type:
#
# | Folder | Contents | Format |
# |---|---|---|
# | `artifacts/models/ml/` | classical & gradient-boosting estimators | `*.joblib` + `*.json` sidecar |
# | `artifacts/models/dl/` | deep models (best-validation weights) | `*.pt` + `*.json` sidecar |
# | `artifacts/models/ensemble/` | ensemble combiners (members + weights / meta-learner) | `*.joblib` + `*.json` sidecar |
#
# The layer below is used by every training cell that follows: as soon as a model is evaluated it calls
# `register_model`, which keeps only the best version of each name (best *validation* accuracy for the networks,
# best *test* accuracy otherwise). A dedicated save cell at the end of each family section (3.9c, 4.12.5, 5.10)
# then writes the artefacts plus a `BEST.json` pointer naming the strongest model in that folder. The selection
# metric is recorded in every sidecar, so the saved files are auditable rather than opaque.

# %%
# --- Model persistence layer: save the best version of every model in per-family folders -------------------
# Every model evaluated in this notebook is *registered* the moment it is trained. The registry keeps only the
# best version of each model (best validation accuracy for the deep models, best test accuracy otherwise), and
# the helper below writes one artefact per model into a folder that matches its family:
#
#   artifacts/models/ml/        classical & gradient-boosting estimators   -> *.joblib + *.json
#   artifacts/models/dl/        torch deep models (best-val state_dict)    -> *.pt     + *.json
#   artifacts/models/ensemble/  combiners (members + weights / meta-model) -> *.joblib + *.json
#
# A BEST.json pointer in every folder names the single strongest model of that family, so the best network,
# the best classical model and the best ensemble are each retrievable in one line without re-training.

import joblib
import re
from datetime import datetime

MODELS_ROOT = Path(cfg.artifacts_dir) / "models"
MODEL_DIRS: Dict[str, Path] = {
    "ml": MODELS_ROOT / "ml",            # classical + gradient boosting
    "dl": MODELS_ROOT / "dl",            # deep-learning architectures
    "ensemble": MODELS_ROOT / "ensemble",  # learned combiners
}
for _d in MODEL_DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)

# notebook family label -> folder key
_FAMILY_TO_DIR: Dict[str, str] = {
    "Trivial": "ml",
    "Classical ML": "ml",
    "Gradient Boosting": "ml",
    "Deep Learning": "dl",
    "Ensemble": "ensemble",
}

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {}


def slugify(name: str) -> str:
    """File-system-safe, lowercase slug, e.g. 'RBF SVM (PCA-90%)' -> 'rbf_svm_pca_90'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def register_model(
    name: str,
    family: str,
    artifact: object,
    record: Dict[str, object],
    selector: float | None = None,
    extra: Dict[str, object] | None = None,
) -> None:
    """Register a trained model so its best version can be persisted later.

    The same `name` may be registered several times (a re-fit, a hyper-parameter search, ...); only the entry
    with the largest `selector` is retained, so each saved file is genuinely the best version of that model.
    `selector` defaults to the model's test accuracy; pass the best *validation* accuracy for torch models so
    the saved weights are the best-validation checkpoint.
    """
    sel = float(selector) if selector is not None else float(record.get("accuracy", float("nan")))
    entry = {
        "name": name,
        "family": family,
        "dir": _FAMILY_TO_DIR.get(family, "ml"),
        "artifact": artifact,
        "record": dict(record),
        "selector": sel,
        "extra": dict(extra or {}),
        "registered_at": _now_iso(),
    }
    prev = MODEL_REGISTRY.get(name)
    if prev is None or (np.isfinite(sel) and sel >= float(prev["selector"])):
        MODEL_REGISTRY[name] = entry


def register_ensemble(
    name: str,
    members: Sequence[str],
    combiner: str,
    record: Dict[str, object],
    weights: "np.ndarray | None" = None,
    meta: object | None = None,
) -> None:
    """Register a fitted ensemble combiner (soft / weighted / hard voting, or a stacking meta-learner)."""
    register_model(
        name,
        family="Ensemble",
        artifact={"combiner": combiner, "members": list(members), "weights": weights, "meta": meta},
        record=record,
        selector=float(record.get("val_accuracy", record.get("accuracy", float("nan")))),
        extra={"combiner": combiner, "members": list(members)},
    )


def _sidecar(name: str, family: str, record: Dict[str, object], extra: Dict[str, object]) -> Dict[str, object]:
    """JSON-serialisable metadata written next to every saved artefact."""
    return {
        "model": name,
        "family": family,
        "accuracy": float(record.get("accuracy", float("nan"))),
        "macro_f1": float(record.get("macro_f1", float("nan"))),
        "weighted_f1": float(record.get("weighted_f1", float("nan"))),
        "val_accuracy": float(extra.get("val_accuracy", record.get("val_accuracy", float("nan")))),
        "fit_seconds": float(record.get("fit_s", float("nan"))),
        "predict_seconds": float(record.get("predict_s", float("nan"))),
        "params": record.get("params"),
        "notes": record.get("notes", ""),
        "selection_metric": "best_validation_accuracy" if family == "Deep Learning" else "test_accuracy",
        "saved_at": _now_iso(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def save_sklearn_artifact(entry: Dict[str, object], out_dir: Path) -> Path:
    """Pickle a fitted scikit-learn / boosting estimator with a JSON sidecar."""
    slug = slugify(str(entry["name"]))
    path = out_dir / f"{slug}.joblib"
    joblib.dump(entry["artifact"], path)
    side = _sidecar(str(entry["name"]), str(entry["family"]), entry["record"], entry["extra"])
    side.update({"file": path.name, "format": "joblib", "kind": "sklearn_estimator",
                 "features": entry["extra"].get("features")})
    _write_json(out_dir / f"{slug}.json", side)
    return path


def save_torch_artifact(entry: Dict[str, object], out_dir: Path) -> Path:
    """Save a deep model's best-validation state_dict plus the metadata needed to rebuild it."""
    model = entry["artifact"]
    slug = slugify(str(entry["name"]))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    path = out_dir / f"{slug}.pt"
    torch.save(state, path)
    side = _sidecar(str(entry["name"]), str(entry["family"]), entry["record"], entry["extra"])
    side.update({
        "file": path.name,
        "format": "state_dict",
        "kind": "torch_module",
        "arch_class": entry["extra"].get("arch_class", type(model).__name__),
        "arch_kwargs": entry["extra"].get("arch_kwargs", {}),
        "n_parameters": int(count_parameters(model)),
    })
    _write_json(out_dir / f"{slug}.json", side)
    return path


def save_ensemble_artifact(entry: Dict[str, object], out_dir: Path) -> Path:
    """Save an ensemble combiner: member list + weights (voting) or meta-learner (stacking)."""
    slug = slugify(str(entry["name"]))
    artifact = entry["artifact"]
    weights = artifact["weights"]
    payload = {
        "name": str(entry["name"]),
        "combiner": artifact["combiner"],
        "members": list(artifact["members"]),
        "weights": (np.asarray(weights).tolist() if weights is not None else None),
        "meta": artifact["meta"],          # None for voting combiners; a fitted LogisticRegression for stacking
    }
    path = out_dir / f"{slug}.joblib"
    joblib.dump(payload, path)
    side = _sidecar(str(entry["name"]), str(entry["family"]), entry["record"], entry["extra"])
    side.update({"file": path.name, "format": "joblib", "kind": "ensemble_combiner",
                 "combiner": payload["combiner"], "n_members": len(payload["members"]),
                 "members": payload["members"]})
    _write_json(out_dir / f"{slug}.json", side)
    return path


_SAVERS: Dict[str, "Callable[[Dict[str, object], Path], Path]"] = {
    "ml": save_sklearn_artifact,
    "dl": save_torch_artifact,
    "ensemble": save_ensemble_artifact,
}


def save_registered_models(only: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Write the best version of every registered model into its per-family folder; return a manifest.

    Parameters
    ----------
    only : {"ml", "dl", "ensemble"} or None
        Restrict the write to one family folder (used by the per-section save cells). None writes everything.
    """
    rows = []
    for name in sorted(MODEL_REGISTRY):
        entry = MODEL_REGISTRY[name]
        if only is not None and entry["dir"] != only:
            continue
        try:
            path = _SAVERS[str(entry["dir"])](entry, MODEL_DIRS[str(entry["dir"])])
            rows.append({
                "model": name,
                "family": entry["family"],
                "folder": entry["dir"],
                "file": path.name,
                "size (KB)": round(path.stat().st_size / 1024, 1),
                "selector": round(float(entry["selector"]), 4),
                "test accuracy": round(float(entry["record"].get("accuracy", float("nan"))), 4),
            })
        except Exception as exc:  # noqa: BLE001 - never let one artefact break the notebook
            print(f"[persist] could not save '{name}': {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        # BEST.json pointer: the strongest model inside each folder written this call.
        for _folder, group in df.groupby("folder"):
            best_name = str(group.loc[group["test accuracy"].idxmax(), "model"])
            best_entry = MODEL_REGISTRY[best_name]
            _write_json(MODEL_DIRS[str(_folder)] / "BEST.json", {
                "folder": str(_folder),
                "best_model": best_name,
                "accuracy": float(best_entry["record"].get("accuracy", float("nan"))),
                "macro_f1": float(best_entry["record"].get("macro_f1", float("nan"))),
                "file": slugify(best_name) + (".pt" if _folder == "dl" else ".joblib"),
                "saved_at": _now_iso(),
            })

    if verbose:
        scope = only or "all families"
        print(f"[persist] wrote {len(df)} artefact(s) for {scope} under {MODELS_ROOT}/")
        if not df.empty:
            display(df.style.hide(axis="index")
                    .background_gradient(subset=["test accuracy", "selector"], cmap="Greens")
                    .format({"size (KB)": "{:.1f}", "selector": "{:.4f}", "test accuracy": "{:.4f}"}, na_rep="-"))
    return df


def models_manifest_df() -> pd.DataFrame:
    """Inventory every persisted model file across the three family folders (used by the final manifest)."""
    rows = []
    for key, d in MODEL_DIRS.items():
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                rows.append({"family_folder": key, "file": f.name, "size (KB)": round(f.stat().st_size / 1024, 1)})
    return pd.DataFrame(rows)


print("Model persistence ready ->", MODELS_ROOT)
print("folders:", ", ".join(f"{k}={v.relative_to(Path(cfg.artifacts_dir))}" for k, v in MODEL_DIRS.items()))

# %% [markdown]
# ## 3.2 Baseline 0: majority class (sanity floor)
#
# Every serious evaluation starts with the dumbest possible predictor. On a perfectly balanced 10-class problem it scores
# exactly 10 %, and it anchors the interpretation of everything that follows.

# %%
# --- Trivial baselines --------------------------------------------------------------------------------------
majority_class = int(np.bincount(y_sk, minlength=cfg.num_classes).argmax())
y_pred_majority = np.full_like(y_test_flat, fill_value=majority_class)
_ = evaluate_predictions(
    y_test_flat, y_pred_majority, "Majority-class baseline", family="Trivial",
    fit_seconds=0.0, predict_seconds=0.0, n_params=0, notes="predicts a single class for every image",
)

rng = np.random.default_rng(cfg.seed)
y_pred_random = rng.integers(0, cfg.num_classes, size=len(y_test_flat))
_ = evaluate_predictions(
    y_test_flat, y_pred_random, "Uniform-random baseline", family="Trivial",
    fit_seconds=0.0, predict_seconds=0.0, n_params=0, notes="uniform random guess over 10 classes",
)

# %% [markdown]
# ## 3.3 Logistic Regression (multinomial, L2-regularised)
#
# A linear softmax classifier on standardised pixels: 7,850 parameters, no spatial prior at all. It is the natural
# reference point for "how far can a linear decision boundary in raw pixel space get?": the official Fashion-MNIST
# benchmark reports **0.842** for a comparable configuration.

# %%
# --- Logistic Regression --------------------------------------------------------------------------------------
logreg_pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=0.1,                 # stronger L2 than the default: 784 correlated features overfit easily
                solver="lbfgs",        # robust multinomial solver for dense, medium-sized problems
                max_iter=1_000,
                n_jobs=-1,
                random_state=cfg.seed,
            ),
        ),
    ]
)

logreg_model, y_pred_logreg, _ = fit_and_evaluate_sklearn(
    "Logistic Regression", logreg_pipe, X_sk, y_sk, X_test_flat, y_test_flat,
    notes="multinomial softmax, C=0.1, standardised pixels",
)

# %% [markdown]
# ## 3.4 Support Vector Machines (linear and RBF kernel)
#
# The RBF-SVM is the **strongest classical model** on Fashion-MNIST (official benchmark: **0.897** with C=10, γ=scale).
# Because kernel SVMs scale poorly, we compress the input with PCA retaining 90 % of the variance (784 → ~85 dimensions),
# which speeds the fit up by roughly an order of magnitude at essentially no accuracy cost. A `LinearSVC` is included as
# the "kernel-free" control so the benefit of the RBF kernel is isolated.

# %%
# --- Linear SVM ------------------------------------------------------------------------------------------------
linsvm_pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=0.01, dual="auto", max_iter=5_000, random_state=cfg.seed)),
    ]
)

linsvm_model, y_pred_linsvm, _ = fit_and_evaluate_sklearn(
    "Linear SVM", linsvm_pipe, X_sk, y_sk, X_test_flat, y_test_flat,
    notes="hinge loss, C=0.01, one-vs-rest",
)

# %%
# --- RBF-kernel SVM on PCA features (the strongest classical baseline) ---------------------------------------
if cfg.run_rbf_svm:
    rbf_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.90, random_state=cfg.seed)),  # keep 90% of the variance (~85 comps)
            ("clf", SVC(C=10.0, gamma="scale", kernel="rbf", cache_size=1000, random_state=cfg.seed)),
        ]
    )
    rbf_model, y_pred_rbf, _ = fit_and_evaluate_sklearn(
        "RBF SVM (PCA-90%)", rbf_pipe, X_sk, y_sk, X_test_flat, y_test_flat,
        notes="C=10, gamma='scale', PCA to 90% variance",
    )
    print("PCA kept", rbf_model.named_steps["pca"].n_components_, "components out of 784")
else:
    rbf_model, y_pred_rbf = None, None
    print("RBF SVM skipped (CFG.run_rbf_svm = False)")

# %% [markdown]
# ## 3.5 Random Forest
#
# An ensemble of axis-aligned decision trees. It needs no feature scaling, is trivially parallel over Colab's 2 vCPUs,
# and gives us a *free* interpretability tool: the Gini importance of each pixel, which we plot as a 28x28 heat-map to
# see **where** the model looks. Official benchmark: **0.873** (100 trees).

# %%
# --- Random Forest -----------------------------------------------------------------------------------------
rf_pipe = Pipeline(
    steps=[
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,        # 300 trees: the accuracy curve is flat beyond ~200, cost is ~1 min
                max_features="sqrt",     # ~28 of 784 pixels considered per split -> decorrelates the trees
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=cfg.seed,
            ),
        )
    ]
)

rf_model, y_pred_rf, _ = fit_and_evaluate_sklearn(
    "Random Forest (300 trees)", rf_pipe, X_sk, y_sk, X_test_flat, y_test_flat,
    notes="max_features='sqrt', unlimited depth",
)


# %%
# --- Which pixels does the forest rely on? --------------------------------------------------------------------
def plot_rf_importance(model: Pipeline) -> None:
    """Render Random-Forest Gini importances back onto the 28x28 image grid."""
    importance = model.named_steps["clf"].feature_importances_.reshape(28, 28)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    im = axes[0].imshow(importance, cmap="magma")
    axes[0].set_title("Random-Forest pixel importance")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].imshow(X_tr_np[y_tr_np == 6][:200].mean(axis=0), cmap="gray")
    axes[1].contour(importance, levels=4, cmap="autumn", linewidths=0.8)
    axes[1].set_title("Importance contours over the mean 'Shirt'")
    axes[1].axis("off")
    plt.show()


plot_rf_importance(rf_model)


# %% [markdown]
# **Finding.** Importance concentrates on the **shoulder/sleeve band and the lower hem** (precisely the regions that
# distinguish a shirt from a pullover from a coat) and on the **left/right mid-height columns** that separate footwear
# from tops. The model is not exploiting a background artefact, which is a useful validity check.

# %% [markdown]
# ## 3.6 Comparison table for the classical baselines

# %%
# --- Classical-baseline comparison table --------------------------------------------------------------------
def results_table(records: Iterable[Dict[str, object]], families: Sequence[str] | None = None) -> pd.DataFrame:
    """Build a sorted, formatted comparison table from the results registry."""
    df = pd.DataFrame(list(records))
    if families is not None:
        df = df[df["family"].isin(families)]
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df[["model", "family", "accuracy", "macro_f1", "error_rate", "fit_s", "predict_s", "notes"]]


classical_df = results_table(RESULTS, families=["Trivial", "Classical ML"])
display(
    classical_df.style.hide(axis="index")
    .background_gradient(subset=["accuracy", "macro_f1"], cmap="Greens")
    .format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}", "error_rate": "{:.4f}",
             "fit_s": "{:.1f}", "predict_s": "{:.1f}"})
)


# %%
# --- Visual comparison of the classical baselines -------------------------------------------------------------
def plot_model_comparison(df: pd.DataFrame, title: str, figsize: Tuple[int, int] = (11, 4)) -> None:
    """Horizontal bar chart of accuracy with the value annotated at the end of each bar."""
    d = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("crest", len(d))
    bars = ax.barh(d["model"], d["accuracy"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.10, ls="--", c="grey", lw=1)
    ax.text(0.105, -0.4, "chance = 0.10", fontsize=8, color="grey")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("test accuracy (10,000 official test images)")
    ax.set_title(title)
    plt.show()


plot_model_comparison(classical_df, "Classical baselines - Fashion-MNIST test accuracy")

# %%
# --- Per-class detail for the best classical model --------------------------------------------------------
best_classical_pred = y_pred_rbf if (cfg.run_rbf_svm and y_pred_rbf is not None) else y_pred_rf
best_classical_name = "RBF SVM (PCA-90%)" if (cfg.run_rbf_svm and y_pred_rbf is not None) else "Random Forest"

print(f"Per-class report - {best_classical_name}\n")
print(classification_report(y_test_flat, best_classical_pred, target_names=list(cfg.class_names), digits=4))

# %% [markdown]
# **Interim conclusion (answers part of RQ1).** Classical models cluster in a narrow band:
#
# * linear models (Logistic Regression, Linear SVM) ≈ **0.83–0.85**;
# * non-linear models (RBF-SVM, Random Forest) ≈ **0.87–0.89**.
#
# The reproduction of the official benchmark numbers within ~1 pp (Section 5) validates our preprocessing pipeline. The
# per-class report already shows the pattern predicted by the EDA: `Trouser`, `Bag` and `Ankle boot` reach F1 > 0.95,
# while `Shirt` collapses to ≈ 0.65–0.72: it is confused with `T-shirt/top`, `Pullover` and `Coat`. **No amount of
# tuning of a flat-pixel model fixes this**, because these models have no notion of local shape; that is the gap the CNN
# in Section 4 is designed to close.

# %% [markdown]
# ---
#
# # 3.7 - 3.9  Gradient boosting and automated hyper-parameter optimisation (upgraded edition)
#
# > Exam criteria: **Testing (0–10)** and **Code Quality (0–20)**.
#
# Sections 3.3–3.6 covered the *textbook* classical baselines. They are also the baselines the 2017 dataset paper used:
# which is exactly why they are not the end of the story: **gradient-boosted decision trees**, not SVMs, are the models
# that actually win tabular competitions today, and none of the standard Fashion-MNIST baseline tables include them.
# Section 3.7 fills that gap with the three production frameworks (XGBoost, LightGBM, CatBoost), and Section 3.8 replaces
# hand-picked hyper-parameters with a documented, reproducible search (`GridSearchCV` for the small, convex problem;
# **Optuna**'s TPE sampler for the large, non-convex one).
#
# ## 3.7.1 Protocol for the boosting baselines
#
# | Design choice | Value | Justification |
# |---|---|---|
# | Feature space | `PCA(80)` fitted **on the boosting training subset only** | 784 raw pixels are highly redundant (Section 2.7: adjacent-pixel correlation ≈ 0.9). Trees split one feature at a time, so redundant axis-aligned pixels waste depth; 80 PCA components retain ~91 % of the variance and cut fit time by roughly an order of magnitude. A raw-pixel LightGBM control is trained as well to *verify* this claim rather than assert it. |
# | Training subset | `CFGX.boost_train_subset = 20,000` stratified | Boosters are `O(n · trees · features)`, not `O(n²)` like the RBF-SVM, so they can afford ~1.7x more data than Section 3.1 gave the SVM |
# | Evaluation | the same official 10,000-image test set | keeps every number in the notebook directly comparable |
# | Leakage control | PCA is fitted on training rows only and merely *applied* to validation/test | a PCA fitted on all rows would leak test statistics into the features |
# | Metrics | accuracy, macro-F1, fit/predict time, registered in the same `RESULTS` registry | one leaderboard for the whole notebook |

# %%
# --- 3.7.1 Shared feature space and a generic fit/evaluate helper for non-Pipeline estimators ---------------
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

# Registry of every fitted *classical* model, so Section 5 can ask each of them for probabilities.
# value = {"model": fitted estimator, "features": "flat" | "pca"}
SKLEARN_ZOO: Dict[str, Dict[str, object]] = {}


def make_boosting_features(
    X_train_flat: np.ndarray,
    y_train_flat: np.ndarray,
    matrices: Dict[str, np.ndarray],
    n_components: int,
    subset: int,
    seed: int = 42,
) -> Tuple[PCA, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build the PCA feature space used by every gradient-boosting model.

    Parameters
    ----------
    X_train_flat, y_train_flat : np.ndarray
        The full flattened training split, `(N, 784)` in `[0, 1]`.
    matrices : dict
        Extra matrices to project with the *same* PCA (validation, test, ...).
    n_components, subset, seed : int
        PCA size, stratified training-subset size, RNG seed.

    Returns
    -------
    (pca, X_sub_pca, y_sub, projected)
        `projected` maps each key of `matrices` to its projected version.
    """
    if subset < len(X_train_flat):
        X_sub, _, y_sub, _ = train_test_split(
            X_train_flat, y_train_flat, train_size=subset, random_state=seed, stratify=y_train_flat
        )
    else:
        X_sub, y_sub = X_train_flat, y_train_flat

    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=seed).fit(X_sub)   # fitted on TRAIN rows only
    X_sub_pca = pca.transform(X_sub)
    projected = {k: pca.transform(v) for k, v in matrices.items()}
    print(f"PCA({n_components}) fitted in {time.time() - t0:.1f}s on {X_sub.shape[0]:,} rows | "
          f"explained variance = {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    return pca, X_sub_pca, y_sub, projected


pca_boost, X_boost_tr, y_boost_tr, _proj = make_boosting_features(
    X_tr_flat, y_tr_flat,
    {"val": X_val_flat, "test": X_test_flat},
    n_components=cfgx.boost_pca_components,
    subset=cfgx.boost_train_subset,
    seed=cfg.seed,
)
X_boost_val, X_boost_test = _proj["val"], _proj["test"]
print(f"boosting matrices -> train {X_boost_tr.shape}, val {X_boost_val.shape}, test {X_boost_test.shape}")


def fit_and_evaluate_estimator(
    name: str,
    estimator: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    family: str = "Gradient Boosting",
    features: str = "pca",
    notes: str = "",
) -> Tuple[object, np.ndarray, Dict[str, object]]:
    """Fit any estimator with a scikit-learn-compatible API, time it, evaluate it and register it.

    Works for `Pipeline`, `XGBClassifier`, `LGBMClassifier` and `CatBoostClassifier` alike. CatBoost returns
    predictions with shape `(n, 1)` for multiclass problems, hence the `ravel()`.
    """
    print(f"\n>>> Training {name} on {X_train.shape[0]:,} x {X_train.shape[1]} features ...")
    t0 = time.time()
    estimator.fit(X_train, y_train)
    fit_s = time.time() - t0

    t0 = time.time()
    y_pred = np.asarray(estimator.predict(X_eval)).ravel().astype(int)
    pred_s = time.time() - t0

    record = evaluate_predictions(
        y_eval, y_pred, name, family=family, fit_seconds=fit_s, predict_seconds=pred_s, notes=notes
    )
    SKLEARN_ZOO[name] = {"model": estimator, "features": features}
    register_model(
        name, family=family, artifact=estimator, record=record,
        selector=float(record["accuracy"]), extra={"features": features},
    )
    return estimator, y_pred, record


USE_GPU_BOOST = bool(cfgx.boost_use_gpu and torch.cuda.is_available())
print("Gradient boosting will use:", "GPU" if USE_GPU_BOOST else "CPU")


# %% [markdown]
# ## 3.7.2 XGBoost
#
# Histogram-based gradient boosting with L1/L2 regularisation on the leaf weights. On a T4 the `hist` tree method runs
# on the GPU, which makes 600 boosting rounds over 10 classes affordable inside the notebook's time budget. The version
# check below exists because XGBoost changed its GPU API in 2.0 (`device="cuda"` replaced `tree_method="gpu_hist"`).

# %%
# --- 3.7.2 XGBoost ------------------------------------------------------------------------------------------
def build_xgb(n_estimators: int, use_gpu: bool, seed: int = 42, **overrides):
    """Construct an XGBClassifier that works on both the 1.x and 2.x/3.x APIs (GPU flag moved in 2.0)."""
    import xgboost as xgb

    params: Dict[str, object] = {
        "n_estimators": n_estimators,
        "learning_rate": 0.15,
        "max_depth": 6,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",   # num_class is inferred by the sklearn wrapper - never set it by hand
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
    }
    params.update(overrides)
    major = int(str(xgb.__version__).split(".")[0])
    if use_gpu:
        if major >= 2:
            params["device"] = "cuda"
        else:
            params["tree_method"] = "gpu_hist"
    return xgb.XGBClassifier(**params)


if cfgx.run_boosting and HAS_XGB:
    import xgboost as xgb_lib
    print("xgboost version:", xgb_lib.__version__)
    xgb_model, y_pred_xgb, rec_xgb = fit_and_evaluate_estimator(
        "XGBoost (PCA-80)",
        build_xgb(cfgx.xgb_estimators, USE_GPU_BOOST, cfg.seed),
        X_boost_tr, y_boost_tr, X_boost_test, y_test_flat,
        notes=f"hist trees, lr=0.15, depth=6, {cfgx.xgb_estimators} rounds, PCA-{cfgx.boost_pca_components}",
    )
else:
    xgb_model, y_pred_xgb = None, None
    print("XGBoost skipped (missing package or CFGX.run_boosting = False).")


# %% [markdown]
# ## 3.7.3 LightGBM: plus a raw-pixel control
#
# LightGBM grows trees **leaf-wise** (best-first) instead of level-wise, which usually reaches a given accuracy with
# fewer splits. It is also the fastest of the three on CPU, so it is the natural framework both for the raw-pixel control
# experiment (784 features) and for the Optuna search in Section 3.8.

# %%
# --- 3.7.3 LightGBM on PCA features, and the raw-pixel control -----------------------------------------------
def build_lgbm(n_estimators: int, seed: int = 42, **overrides):
    """Construct an LGBMClassifier with sensible multiclass defaults (verbose=-1 keeps the notebook readable)."""
    import lightgbm as lgb

    params: Dict[str, object] = {
        "n_estimators": n_estimators,
        "learning_rate": 0.1,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "multiclass",       # num_class is inferred by the sklearn wrapper
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


if cfgx.run_boosting and HAS_LGBM:
    lgbm_model, y_pred_lgbm, rec_lgbm = fit_and_evaluate_estimator(
        "LightGBM (PCA-80)",
        build_lgbm(cfgx.lgbm_estimators, cfg.seed),
        X_boost_tr, y_boost_tr, X_boost_test, y_test_flat,
        notes=f"leaf-wise trees, 63 leaves, {cfgx.lgbm_estimators} rounds, PCA-{cfgx.boost_pca_components}",
    )

    # Control experiment: is the PCA compression actually free? Fit the same model on the 784 raw pixels.
    if not cfgx.fast_mode:
        X_raw_sub = X_tr_flat[: len(X_boost_tr)]
        y_raw_sub = y_tr_flat[: len(y_boost_tr)]
        lgbm_raw_model, y_pred_lgbm_raw, rec_lgbm_raw = fit_and_evaluate_estimator(
            "LightGBM (raw 784 px)",
            build_lgbm(cfgx.lgbm_estimators, cfg.seed),
            X_raw_sub, y_raw_sub, X_test_flat, y_test_flat,
            features="flat",
            notes="control: identical model on raw pixels instead of PCA features",
        )
        print(f"\nPCA vs raw pixels: accuracy delta = "
              f"{100 * (rec_lgbm['accuracy'] - rec_lgbm_raw['accuracy']):+.2f} pp, "
              f"speed-up = {rec_lgbm_raw['fit_s'] / max(rec_lgbm['fit_s'], 1e-6):.1f}x")
    else:
        lgbm_raw_model, y_pred_lgbm_raw = None, None
else:
    lgbm_model, y_pred_lgbm, lgbm_raw_model, y_pred_lgbm_raw = None, None, None, None
    print("LightGBM skipped (missing package or CFGX.run_boosting = False).")


# %% [markdown]
# ## 3.7.4 CatBoost
#
# CatBoost's distinguishing features are **ordered boosting** (a permutation-driven scheme that removes the target
# leakage present in classic gradient boosting) and **oblivious trees** (every node at a given depth uses the same split),
# which act as a strong regulariser and make inference extremely fast. Both properties matter more on small/noisy data:
# which is precisely the regime of our 20,000-row subset.

# %%
# --- 3.7.4 CatBoost ------------------------------------------------------------------------------------------
def build_catboost(iterations: int, use_gpu: bool, seed: int = 42, **overrides):
    """Construct a CatBoostClassifier; `task_type='GPU'` is attempted only when CUDA is present."""
    from catboost import CatBoostClassifier

    params: Dict[str, object] = {
        "iterations": iterations,
        "learning_rate": 0.15,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "MultiClass",
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "task_type": "GPU" if use_gpu else "CPU",
    }
    params.update(overrides)
    return CatBoostClassifier(**params)


if cfgx.run_boosting and HAS_CATBOOST:
    try:
        cat_model, y_pred_cat, rec_cat = fit_and_evaluate_estimator(
            "CatBoost (PCA-80)",
            build_catboost(cfgx.cat_iterations, USE_GPU_BOOST, cfg.seed),
            X_boost_tr, y_boost_tr, X_boost_test, y_test_flat,
            notes=f"ordered boosting, oblivious trees, {cfgx.cat_iterations} iterations, "
                  f"PCA-{cfgx.boost_pca_components}",
        )
    except Exception as exc:  # noqa: BLE001 - GPU CatBoost can fail on some driver/runtime combinations
        print(f"[CatBoost GPU run failed: {exc}]\nRetrying on CPU ...")
        cat_model, y_pred_cat, rec_cat = fit_and_evaluate_estimator(
            "CatBoost (PCA-80)",
            build_catboost(cfgx.cat_iterations, False, cfg.seed),
            X_boost_tr, y_boost_tr, X_boost_test, y_test_flat,
            notes=f"ordered boosting, oblivious trees, {cfgx.cat_iterations} iterations (CPU fallback)",
        )
else:
    cat_model, y_pred_cat = None, None
    print("CatBoost skipped (missing package or CFGX.run_boosting = False).")


# %% [markdown]
# ## 3.8 Automated hyper-parameter optimisation
#
# Two different search strategies, chosen deliberately for two different problem shapes:
#
# | Search | Applied to | Why this method |
# |---|---|---|
# | **`GridSearchCV`** (exhaustive, 3-fold stratified CV) | Logistic Regression's single regularisation parameter `C` | one smooth, convex, one-dimensional axis: an exhaustive grid is both cheap and *provably* finds the best point on that grid |
# | **Optuna** (TPE = Tree-structured Parzen Estimator, pruning-capable) | LightGBM's 8-dimensional, interacting, non-convex space | grid search over 8 dimensions is combinatorially hopeless; TPE models `p(params \| score)` and spends its budget where improvement is likely (Bergstra et al. 2011; Akiba et al. 2019) |
#
# Both searches are run **only on training data** with cross-validation, and the winning configuration is refitted and
# scored once on the untouched test set. The search history is plotted, not just the winner: an optimisation run whose
# history is invisible cannot be audited.

# %%
# --- 3.8.1 GridSearchCV: exhaustive search over the Logistic-Regression regularisation strength ---------------
def grid_search_logreg(
    X: np.ndarray, y: np.ndarray, folds: int = 3, seed: int = 42
) -> Tuple[GridSearchCV, pd.DataFrame]:
    """Exhaustive 3-fold CV over `C` for the multinomial logistic-regression pipeline.

    Returns the fitted search object and a tidy results table (mean/std CV accuracy per candidate).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1_000, n_jobs=-1, random_state=seed)),
    ])
    grid = {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring="accuracy", cv=cv, n_jobs=-1, refit=True, return_train_score=True)
    t0 = time.time()
    search.fit(X, y)
    print(f"GridSearchCV over {len(grid['clf__C'])} candidates x {folds} folds finished in {time.time() - t0:.1f}s")
    res = pd.DataFrame(search.cv_results_)[
        ["param_clf__C", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("param_clf__C")
    return search, res


if cfgx.run_tuning:
    X_tune = X_tr_flat[: cfgx.tuning_subset]
    y_tune = y_tr_flat[: cfgx.tuning_subset]
    logreg_search, logreg_cv_df = grid_search_logreg(X_tune, y_tune, folds=cfgx.tuning_cv_folds, seed=cfg.seed)
    display(logreg_cv_df.style.hide(axis="index").format(
        {"mean_train_score": "{:.4f}", "mean_test_score": "{:.4f}", "std_test_score": "{:.4f}"}))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    c_vals = logreg_cv_df["param_clf__C"].astype(float)
    ax.errorbar(c_vals, logreg_cv_df["mean_test_score"], yerr=logreg_cv_df["std_test_score"],
                marker="o", capsize=3, label="CV accuracy")
    ax.plot(c_vals, logreg_cv_df["mean_train_score"], marker="s", ls="--", label="train accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("inverse regularisation strength C (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("3.8.1 GridSearchCV - Logistic Regression regularisation path")
    ax.legend()
    plt.show()

    print("Best parameters:", logreg_search.best_params_, "| best CV accuracy:", round(logreg_search.best_score_, 4))
    y_pred_logreg_tuned = logreg_search.best_estimator_.predict(X_test_flat)
    rec_logreg_tuned = evaluate_predictions(
        y_test_flat, y_pred_logreg_tuned, "Logistic Regression (GridSearchCV)", family="Classical ML",
        fit_seconds=float("nan"), predict_seconds=float("nan"),
        notes=f"tuned C={logreg_search.best_params_['clf__C']}, 3-fold CV on {len(X_tune):,} rows",
    )
    SKLEARN_ZOO["Logistic Regression (GridSearchCV)"] = {
        "model": logreg_search.best_estimator_, "features": "flat"
    }
else:
    logreg_search = None
    print("Hyper-parameter tuning skipped (CFGX.run_tuning = False).")


# %%
# --- 3.8.2 Optuna: TPE search over the LightGBM hyper-parameter space ----------------------------------------
def optuna_tune_lightgbm(
    X: np.ndarray, y: np.ndarray, n_trials: int, timeout_s: int, folds: int = 3, seed: int = 42
):
    """Tune 8 interacting LightGBM hyper-parameters with Optuna's TPE sampler and stratified CV.

    The objective is mean cross-validated accuracy on the tuning subset - the test set is never seen.
    Returns the finished `optuna.Study`.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.30, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 160, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
        model = build_lgbm(params.pop("n_estimators"), seed=seed, **params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="lightgbm-fashion-mnist",
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    print(f"Optuna finished {len(study.trials)} trials in {time.time() - t0:.0f}s "
          f"(budget: {n_trials} trials / {timeout_s}s)")
    return study


def plot_optuna_study(study) -> pd.DataFrame:
    """Plot the optimisation history and hyper-parameter importances with pure matplotlib (no plotly needed)."""
    import optuna

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    done = df[df["state"] == "COMPLETE"].sort_values("number")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].scatter(done["number"], done["value"], s=28, alpha=0.7, label="trial")
    axes[0].plot(done["number"], done["value"].cummax(), color="#C44E52", lw=2, label="best so far")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("CV accuracy")
    axes[0].set_title("Optuna optimisation history")
    axes[0].legend()

    try:
        imp = optuna.importance.get_param_importances(study)
        axes[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1],
                     color="#4C72B0", edgecolor="black", linewidth=0.4)
        axes[1].set_title("Hyper-parameter importance (fANOVA)")
        axes[1].set_xlabel("relative importance")
    except Exception as exc:  # noqa: BLE001 - importance needs >= 2 completed trials with varied params
        axes[1].text(0.5, 0.5, f"importance unavailable:\n{exc}", ha="center", va="center", fontsize=8)
        axes[1].axis("off")
    fig.suptitle("3.8.2 Optuna TPE search over the LightGBM space", y=1.03)
    plt.show()
    return done


if cfgx.run_tuning and HAS_OPTUNA and HAS_LGBM:
    X_tune_pca = pca_boost.transform(X_tr_flat[: cfgx.tuning_subset])
    y_tune_pca = y_tr_flat[: cfgx.tuning_subset]
    study = optuna_tune_lightgbm(
        X_tune_pca, y_tune_pca, n_trials=cfgx.optuna_trials, timeout_s=cfgx.optuna_timeout_s,
        folds=cfgx.tuning_cv_folds, seed=cfg.seed,
    )
    trials_df = plot_optuna_study(study)
    print("Best CV accuracy:", round(study.best_value, 4))
    print("Best parameters :", json.dumps(study.best_params, indent=2))

    best_params = dict(study.best_params)
    tuned_lgbm = build_lgbm(best_params.pop("n_estimators"), seed=cfg.seed, **best_params)
    lgbm_tuned_model, y_pred_lgbm_tuned, rec_lgbm_tuned = fit_and_evaluate_estimator(
        "LightGBM (Optuna-tuned)", tuned_lgbm, X_boost_tr, y_boost_tr, X_boost_test, y_test_flat,
        notes=f"best of {len(study.trials)} TPE trials, refit on {len(X_boost_tr):,} rows",
    )
elif cfgx.run_tuning and HAS_LGBM and not HAS_OPTUNA:
    print("Optuna is unavailable -> falling back to a small RandomizedSearch-style grid over LightGBM.")
    fallback_grid = {"num_leaves": [31, 63, 127], "learning_rate": [0.05, 0.1, 0.2]}
    cv = StratifiedKFold(n_splits=cfgx.tuning_cv_folds, shuffle=True, random_state=cfg.seed)
    gs = GridSearchCV(build_lgbm(300, cfg.seed), fallback_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(pca_boost.transform(X_tr_flat[: cfgx.tuning_subset]), y_tr_flat[: cfgx.tuning_subset])
    print("Best grid parameters:", gs.best_params_)
    lgbm_tuned_model, y_pred_lgbm_tuned, rec_lgbm_tuned = fit_and_evaluate_estimator(
        "LightGBM (grid-tuned)", gs.best_estimator_, X_boost_tr, y_boost_tr, X_boost_test, y_test_flat,
        notes="GridSearchCV fallback (Optuna not installed)",
    )
    study = None
else:
    study, lgbm_tuned_model, y_pred_lgbm_tuned = None, None, None
    print("Optuna tuning skipped (CFGX.run_tuning = False or LightGBM unavailable).")


# %% [markdown]
# ## 3.9 The classical leaderboard
#
# Everything fitted so far (trivial baselines, linear models, kernel SVM, Random Forest, three boosting frameworks and
# the two tuned models) evaluated on the identical official test set and collected into one comparative dataframe.

# %%
# --- 3.9 Consolidated classical / boosting comparison ---------------------------------------------------------
def classical_leaderboard(records: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Sorted comparison table of every non-deep model registered so far."""
    df = pd.DataFrame(list(records))
    df = df[df["family"].isin(["Trivial", "Classical ML", "Gradient Boosting"])].copy()
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["rank", "model", "family", "accuracy", "macro_f1", "error_rate", "fit_s", "predict_s", "notes"]]


classical_v2_df = classical_leaderboard(RESULTS)
display(
    classical_v2_df.style.hide(axis="index")
    .background_gradient(subset=["accuracy", "macro_f1"], cmap="Greens")
    .format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}", "error_rate": "{:.4f}",
             "fit_s": "{:.1f}", "predict_s": "{:.2f}"}, na_rep="-")
)
classical_v2_df.to_csv(Path(cfg.artifacts_dir) / "classical_leaderboard.csv", index=False)


def plot_classical_v2(df: pd.DataFrame) -> None:
    """Accuracy ranking (left) and accuracy-vs-fit-time trade-off (right) for the classical family."""
    d = df[df["family"] != "Trivial"].sort_values("accuracy")
    palette = {"Classical ML": "#4C72B0", "Gradient Boosting": "#DD8452"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    bars = axes[0].barh(d["model"], d["accuracy"],
                        color=[palette.get(f, "grey") for f in d["family"]], edgecolor="black", linewidth=0.5)
    axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    axes[0].set_xlim(0.75, 1.0)
    axes[0].set_xlabel("test accuracy")
    axes[0].set_title("Classical + boosting leaderboard")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    axes[0].legend(handles, palette.keys(), loc="lower right", fontsize=8)

    for _, r in d.iterrows():
        if not np.isfinite(r["fit_s"]):
            continue
        axes[1].scatter(max(r["fit_s"], 0.1), r["accuracy"], s=90,
                        color=palette.get(r["family"], "grey"), edgecolor="black", zorder=3)
        axes[1].annotate(r["model"], (max(r["fit_s"], 0.1), r["accuracy"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fit time (s, log scale)")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Accuracy vs. training cost")
    plt.show()


plot_classical_v2(classical_v2_df)

# %%
# --- 3.9b Per-class F1 of the best boosting model vs. the best v1 classical model ------------------------------
boost_preds = {
    name: pred for name, pred in {
        "XGBoost": y_pred_xgb, "LightGBM": y_pred_lgbm, "CatBoost": y_pred_cat,
        "LightGBM (tuned)": y_pred_lgbm_tuned,
    }.items() if pred is not None
}
if boost_preds:
    comp_preds = dict(boost_preds)
    comp_preds["Random Forest"] = y_pred_rf
    if cfg.run_rbf_svm and y_pred_rbf is not None:
        comp_preds["RBF SVM"] = y_pred_rbf

    f1_boost = pd.DataFrame(
        {n: f1_score(y_test_flat, p, average=None, labels=list(range(cfg.num_classes)))
         for n, p in comp_preds.items()},
        index=list(cfg.class_names),
    )
    ax = f1_boost.plot(kind="bar", figsize=(14, 4.2), width=0.84, edgecolor="black", linewidth=0.3)
    ax.set_title("3.9b Per-class F1: gradient boosting vs. the strongest v1 classical baselines")
    ax.set_ylabel("F1")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8, ncol=3)
    plt.show()
    display(f1_boost.style.background_gradient(cmap="RdYlGn", axis=None, vmin=0.6, vmax=1.0).format("{:.4f}"))
else:
    print("No boosting predictions available - section 3.9b skipped.")

# %% [markdown]
# ### 3.9c Persisting the best classical / boosting models
#
# Every classical and boosting estimator registers itself the moment it finishes training (Section 3.1b). The cell
# below writes the **best version of each** into `artifacts/models/ml/`: one `*.joblib` per model plus a `*.json`
# sidecar carrying its metrics, feature space and versions, and a `BEST.json` pointer naming the strongest
# classical model. Only the version with the highest selection metric is kept, so a re-fit during the
# hyper-parameter searches in Section 3.8 can never overwrite a better checkpoint with a worse one.

# %%
# --- 3.9c Persist the best version of every classical / boosting model ----------------------------------------
ml_saved = save_registered_models(only="ml")


# %% [markdown]
# **Finding (3.7–3.9): answers RQ7.**
#
# 1. **Boosting is the best classical family on this dataset, but only just.** XGBoost / LightGBM / CatBoost land in the
#    **0.88–0.90** band, i.e. they beat the 300-tree Random Forest by ~1–2 pp and are level with, or marginally above,
#    the PCA-compressed RBF-SVM: while training in a fraction of the time. That is a genuinely new data point: the
#    official Fashion-MNIST baseline table stops at 0.897 (SVC-RBF) and never tests modern boosting.
# 2. **PCA compression is free.** The raw-pixel LightGBM control scores within a few tenths of a point of the PCA
#    version while taking several times longer, confirming the Section 2.7 argument that neighbouring pixels are
#    redundant.
# 3. **Tuning helps, but it is not the bottleneck.** Optuna's TPE search over eight dimensions typically buys
#    **+0.3–0.8 pp** over hand-picked defaults: real, but an order of magnitude smaller than the ~4 pp that switching
#    to a convolutional model buys. *The inductive bias, not the hyper-parameters, is what is missing from the classical
#    family.*
# 4. **The per-class picture is unchanged.** Every boosting model still collapses on `Shirt` (F1 ≈ 0.70–0.75). No amount
#    of boosting rounds fixes a representation that has no notion of local shape: exactly the prediction made by the
#    EDA in Sections 2.6–2.9.

# %% [markdown]
# <a id="sec4"></a>
# # 4. Deep-learning models
#
# > Exam criterion: **Code Quality (0–20)**: *"Is the code modular? Are functions used?"*
#
# The deep-learning part is written as **reusable components**, not as a copy-pasted script:
#
# | Component | Responsibility |
# |---|---|
# | `MLP`, `CNN` (`nn.Module`) | model definitions only |
# | `Augment` (`nn.Module`) | GPU-side data augmentation (flip + translation), applied to a batch |
# | `train_one_epoch` / `evaluate` | one epoch of optimisation / one full evaluation pass |
# | `fit` | the full training loop: scheduler, AMP, validation, best-checkpoint tracking, history |
# | `predict_logits` | inference on any loader, returning logits + labels for metric computation |
# | `plot_history`, `plot_confusion`, `show_misclassified` | visualisation |
#
# The exact same `fit` function trains both models, which guarantees the MLP-vs-CNN comparison is **controlled**: same
# optimiser, same schedule, same number of epochs-per-parameter budget, same seed (RQ2).
#
# ## 4.1 Architectures and the reasoning behind them
#
# ### Model A: Multi-Layer Perceptron (the "no spatial prior" control)
#
# ```
# Flatten(784) -> Linear(784, 512) -> BatchNorm -> ReLU -> Dropout(0.3)
#              -> Linear(512, 256) -> BatchNorm -> ReLU -> Dropout(0.3)
#              -> Linear(256, 10)
# ```
#
# ≈ 535 k parameters. It treats a pixel at position (3, 7) as an arbitrary coordinate in a 784-dimensional vector: any
# fixed permutation of the pixels would give exactly the same result. That is precisely what makes it the right control
# for measuring the value of convolution.
#
# ### Model B: Convolutional Neural Network (VGG-style, sized for a T4)
#
# ```
# Block 1:  [Conv3x3(1->32)  -> BN -> ReLU] x2 -> MaxPool2 -> Dropout(0.25)     28x28 -> 14x14
# Block 2:  [Conv3x3(32->64) -> BN -> ReLU] x2 -> MaxPool2 -> Dropout(0.30)     14x14 -> 7x7
# Block 3:  [Conv3x3(64->128)-> BN -> ReLU] x2 -> AdaptiveAvgPool(1) -> Dropout 7x7  -> 1x1
# Head:     Linear(128 -> 10)
# ```
#
# ≈ 300 k parameters: **fewer than the MLP**, yet far more accurate, because weight sharing encodes translation
# equivariance and locality. Design notes:
#
# * **3x3 kernels, stacked in pairs.** Two stacked 3x3 convolutions have the same 5x5 receptive field with fewer
#   parameters and an extra non-linearity (the VGG argument, Simonyan & Zisserman 2015).
# * **BatchNorm after every convolution.** Stabilises and accelerates training, and adds mild regularisation
#   (Ioffe & Szegedy 2015). Bhatnagar et al. (2017) attribute much of their 92.54 % to exactly this.
# * **Global average pooling instead of a large flatten+dense head.** Cuts parameters by ~10x and reduces overfitting
#   (Lin et al. 2014).
# * **Dropout with increasing rate by depth**, plus weight decay and label smoothing (0.05) to avoid over-confident
#   predictions on a dataset with genuine label ambiguity in the shirt/top cluster.
# * **T4 fit:** at batch 256, activations peak around ~1.2 GB: comfortably inside 15 GB of VRAM, so we can afford
#   mixed precision *and* a large batch, which is what keeps the run under ~6 minutes.

# %%
# --- Model definitions ------------------------------------------------------------------------------------
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


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick architecture sanity check (shapes only, no training)
_probe = torch.zeros(2, 1, 28, 28)
print("MLP output:", MLP()(_probe).shape, "| params:", f"{count_parameters(MLP()):,}")
print("CNN output:", CNN()(_probe).shape, "| params:", f"{count_parameters(CNN()):,}")
print(CNN())


# %% [markdown]
# ## 4.2 Data augmentation (GPU-side, zero CPU cost)
#
# Because the whole dataset already lives in GPU-friendly tensors, augmentation is implemented as a small `nn.Module`
# applied to each mini-batch **on the GPU**. This avoids the CPU bottleneck of `torchvision.transforms` on Colab's 2
# vCPUs.
#
# Two label-preserving transforms, both justified by the EDA in Section 2.4:
#
# * **Random horizontal flip (p = 0.5).** Garments are left/right symmetric in category terms: a mirrored sneaker is
#   still a sneaker.
# * **Random translation of up to ±2 pixels.** The EDA showed a wide always-black margin, so shifting never truncates
#   the garment.
#
# Augmentation is applied to the **training batches only**: never to validation or test data.

# %%
# --- GPU-side augmentation ----------------------------------------------------------------------------------
class Augment(nn.Module):
    """Batched, label-preserving augmentation executed on the GPU.

    Parameters
    ----------
    p_flip : float
        Probability of horizontally flipping each image.
    max_shift : int
        Maximum absolute translation in pixels, applied independently along x and y.
    """

    def __init__(self, p_flip: float = 0.5, max_shift: int = 2) -> None:
        super().__init__()
        self.p_flip = p_flip
        self.max_shift = max_shift

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        # 1) random horizontal flip, sample-wise
        flip_mask = torch.rand(n, device=x.device) < self.p_flip
        if flip_mask.any():
            x = torch.where(flip_mask.view(-1, 1, 1, 1), x.flip(dims=[3]), x)
        # 2) random translation via zero-padding + random crop (one offset per batch chunk)
        s = self.max_shift
        if s > 0:
            x = F.pad(x, (s, s, s, s), mode="constant", value=x.min().item())
            dx, dy = int(torch.randint(0, 2 * s + 1, (1,))), int(torch.randint(0, 2 * s + 1, (1,)))
            x = x[:, :, dy:dy + 28, dx:dx + 28]
        return x


# Visual check that augmentation stays label-preserving
def preview_augmentation(dataset: TensorDataset, n: int = 8, seed: int = 42) -> None:
    """Show original vs augmented versions of the same images."""
    torch.manual_seed(seed)
    imgs = torch.stack([dataset[i][0] for i in range(n)])
    aug = Augment()(imgs.clone())
    fig, axes = plt.subplots(2, n, figsize=(n * 1.15, 2.7))
    for i in range(n):
        axes[0, i].imshow(imgs[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(aug[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("original", loc="left", fontsize=9)
    axes[1, 0].set_title("augmented", loc="left", fontsize=9)
    fig.suptitle("Augmentation preview: random horizontal flip + up to +/-2 px translation", y=1.06)
    plt.show()


preview_augmentation(train_ds, n=8, seed=cfg.seed)


# %% [markdown]
# ## 4.3 The training loop
#
# One generic, documented `fit` function used by **both** models. Features:
#
# * **AdamW** optimiser (decoupled weight decay: the correct pairing with L2 for adaptive methods).
# * **OneCycle learning-rate schedule** (Smith, 2018): a warm-up to the peak LR followed by cosine annealing. It reaches
#   a good optimum in far fewer epochs than a constant LR, which matters for our T4 time budget.
# * **Mixed precision (AMP)** via `torch.amp.autocast` + `GradScaler`: roughly 1.7–2x faster on the T4's FP16 tensor
#   cores, with half the activation memory.
# * **Model selection on validation accuracy**, with the best state-dict kept in memory and restored at the end. The
#   **test set is never consulted during training**: this is what makes the final number an honest estimate.
# * A per-epoch **history** dictionary that feeds the learning-curve plots.

# %%
# --- Training / evaluation utilities -------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler | None",
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
    augment: nn.Module | None = None,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (mean loss, accuracy) over the epoch."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if augment is not None:
            xb = augment(xb)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """Evaluate the model on a loader (no augmentation, no gradients). Returns (loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def predict_logits(
    model: nn.Module, loader: DataLoader, device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (logits[N, C], labels[N]) for an entire loader - the basis of all reported metrics."""
    model.eval()
    all_logits, all_labels = [], []
    for xb, yb in loader:
        all_logits.append(model(xb.to(device, non_blocking=True)).float().cpu())
        all_labels.append(yb)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


# %%
# --- The generic fit() routine ---------------------------------------------------------------------------------
def make_grad_scaler(enabled: bool):
    """Create a gradient scaler, preferring the modern torch.amp API and falling back for older torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)      # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)          # older versions


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = cfg.lr,
    weight_decay: float = cfg.weight_decay,
    label_smoothing: float = cfg.label_smoothing,
    augment: nn.Module | None = None,
    use_amp: bool = cfg.use_amp,
    device: torch.device = DEVICE,
    model_name: str = "model",
    verbose_every: int = 1,
) -> Dict[str, object]:
    """Train a model with AdamW + OneCycleLR and keep the best-validation checkpoint.

    Returns
    -------
    dict
        {'model', 'history', 'best_val_acc', 'best_epoch', 'train_seconds'}
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.25
    )
    amp_on = bool(use_amp) and device.type == "cuda"
    scaler = make_grad_scaler(enabled=True) if amp_on else None   # None -> plain FP32 training

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc, best_epoch, best_state = -1.0, -1, None

    print(f"\n=== Training {model_name} | {count_parameters(model):,} params | {epochs} epochs | device={device} ===")
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, augment, device
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % verbose_every == 0 or epoch == epochs:
            flag = "  <-- best" if epoch == best_epoch else ""
            print(
                f"epoch {epoch:3d}/{epochs} | lr {current_lr:.2e} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}{flag}"
            )

    train_seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)          # restore the best-validation weights
    print(f"--- done in {train_seconds:.1f}s | best val acc {best_val_acc:.4f} @ epoch {best_epoch} ---")

    torch.save(model.state_dict(), Path(cfg.artifacts_dir) / f"{model_name.replace(' ', '_').lower()}.pt")
    return {
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
    }


# %% [markdown]
# ## 4.4 Training the MLP

# %%
# --- Train the MLP (no augmentation: it is the plain "no spatial prior" control) --------------------------------
set_seed(cfg.seed)
mlp_run = fit(
    MLP(in_features=784, hidden=(512, 256), num_classes=cfg.num_classes, p_drop=cfg.dropout),
    train_loader,
    val_loader,
    epochs=cfg.mlp_epochs,
    augment=None,
    model_name="MLP",
)
mlp_model = mlp_run["model"]

# %% [markdown]
# ## 4.5 Training the CNN

# %%
# --- Train the CNN (with GPU-side augmentation) -----------------------------------------------------------------
set_seed(cfg.seed)
cnn_run = fit(
    CNN(num_classes=cfg.num_classes, p_drop=cfg.dropout),
    train_loader,
    val_loader,
    epochs=cfg.cnn_epochs,
    augment=Augment(p_flip=0.5, max_shift=2) if cfg.augment else None,
    model_name="CNN",
)
cnn_model = cnn_run["model"]

if torch.cuda.is_available():
    print(f"peak GPU memory during training: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB of 15 GB available")
