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
# ## 4.6 Learning curves

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])


# %%
# --- Learning curves ---------------------------------------------------------------------------------------------
def plot_history(history: Dict[str, List[float]], title: str) -> None:
    """Plot loss, accuracy and the learning-rate schedule for one training run."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"], label="validation", marker="s", ms=3)
    axes[0].set_title(f"{title} - loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy (label-smoothed)")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"], label="validation", marker="s", ms=3)
    best_ep = int(np.argmax(history["val_acc"])) + 1
    axes[1].axvline(best_ep, ls="--", c="green", lw=1,
                    label=f"best val = {max(history['val_acc']):.4f} (ep {best_ep})")
    axes[1].set_title(f"{title} - accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], marker="o", ms=3, color="#C44E52")
    axes[2].set_title("OneCycle learning-rate schedule")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate")
    axes[2].set_yscale("log")
    plt.show()


plot_history(mlp_run["history"], "MLP")
plot_history(cnn_run["history"], "CNN")


# %%
# --- Overlay: generalisation gap of the two deep models ------------------------------------------------------------
def plot_generalisation_gap(runs: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Compare validation accuracy and the train-minus-validation gap across runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    rows = []
    for name, run in runs.items():
        h = run["history"]
        ep = range(1, len(h["val_acc"]) + 1)
        axes[0].plot(ep, h["val_acc"], marker="o", ms=3, label=f"{name} (best {max(h['val_acc']):.4f})")
        gap = np.array(h["train_acc"]) - np.array(h["val_acc"])
        axes[1].plot(ep, gap, marker="o", ms=3, label=name)
        rows.append(
            {
                "model": name,
                "final train acc": h["train_acc"][-1],
                "best val acc": max(h["val_acc"]),
                "final gap (train - val)": gap[-1],
                "train seconds": run["train_seconds"],
            }
        )
    axes[0].set_title("Validation accuracy per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation accuracy")
    axes[0].legend()
    axes[1].axhline(0, c="grey", lw=1)
    axes[1].set_title("Overfitting gap (train accuracy - validation accuracy)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("gap")
    axes[1].legend()
    plt.show()
    return pd.DataFrame(rows).round(4)


display(plot_generalisation_gap({"MLP": mlp_run, "CNN": cnn_run}).style.hide(axis="index"))


# %% [markdown]
# **Reading the curves.** Two things to look for:
#
# 1. **The CNN's validation curve sits above its training curve for much of the run.** That is not a bug: dropout and
#    augmentation are active during training but disabled at evaluation, so the training metric is measured on a
#    *harder* problem. A small or negative gap means the regularisation budget is well chosen.
# 2. **The MLP's gap grows steadily.** Despite BatchNorm and dropout, a flat 535 k-parameter model starts memorising the
#    training set after ~10 epochs: the classic signature of a model with too little inductive bias for the task.

# %% [markdown]
# ## 4.7 Final evaluation on the held-out test set
#
# This is the **first and only** time the 10,000 official test images are used for the deep models. We report accuracy,
# macro-F1, top-2 accuracy, per-class metrics and confusion matrices.

# %%
# --- Evaluate both deep models on the official test set -------------------------------------------------------
def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    name: str,
    run: Dict[str, object],
    notes: str = "",
    arch: str | None = None,
    arch_kwargs: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run inference on the test loader, register the metrics and return (probabilities, predictions, record)."""
    t0 = time.time()
    logits, y_true = predict_logits(model, loader)
    predict_s = time.time() - t0
    y_pred = logits.argmax(axis=1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    record = evaluate_predictions(
        y_true,
        y_pred,
        name,
        family="Deep Learning",
        fit_seconds=float(run["train_seconds"]),
        predict_seconds=predict_s,
        n_params=count_parameters(model),
        notes=notes,
    )
    record["top2_accuracy"] = top_k_accuracy_score(y_true, probs, k=2, labels=list(range(cfg.num_classes)))
    record["best_val_acc"] = run["best_val_acc"]
    print(f"    top-2 accuracy = {record['top2_accuracy']:.4f} | best val acc = {run['best_val_acc']:.4f}")
    register_model(
        name,
        family="Deep Learning",
        artifact=model,
        record=record,
        selector=float(run.get("best_val_acc", record["accuracy"])),
        extra={
            "arch_class": arch or type(model).__name__,
            "arch_kwargs": dict(arch_kwargs or {}),
            "val_accuracy": float(run.get("best_val_acc", float("nan"))),
        },
    )
    return probs, y_pred, record


probs_mlp, y_pred_mlp, rec_mlp = evaluate_torch_model(
    mlp_model, test_loader, "MLP (512-256)", mlp_run,
    notes="BatchNorm + Dropout(0.3), no augmentation",
    arch="MLP",
    arch_kwargs={"in_features": 784, "hidden": (512, 256), "num_classes": cfg.num_classes, "p_drop": cfg.dropout},
)
probs_cnn, y_pred_cnn, rec_cnn = evaluate_torch_model(
    cnn_model, test_loader, "CNN (VGG-style, GAP)", cnn_run,
    notes="BN + Dropout + flip/shift augmentation",
    arch="CNN",
    arch_kwargs={"num_classes": cfg.num_classes, "p_drop": cfg.dropout},
)

# %%
# --- Per-class classification report for the CNN --------------------------------------------------------------
print("Per-class report - CNN (official 10,000-image test set)\n")
report_dict = classification_report(
    y_test_np, y_pred_cnn, target_names=list(cfg.class_names), digits=4, output_dict=True
)
report_df = pd.DataFrame(report_dict).T
display(
    report_df.style.background_gradient(subset=["precision", "recall", "f1-score"], cmap="RdYlGn", vmin=0.6, vmax=1.0)
    .format("{:.4f}", subset=["precision", "recall", "f1-score"])
    .format("{:.0f}", subset=["support"])
)


# %%
# --- Confusion matrices ------------------------------------------------------------------------------------------
def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    normalize: bool = True,
    ax: plt.Axes | None = None,
) -> np.ndarray:
    """Plot a (optionally row-normalised) confusion matrix and return the raw counts."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(7.8, 6.4))
    sns.heatmap(
        cm_show,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar_kws={"label": "recall" if normalize else "count"},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(title)
    if own_fig:
        plt.show()
    return cm


cm_mlp = plot_confusion(y_test_np, y_pred_mlp, cfg.class_names, "MLP - row-normalised confusion matrix")
cm_cnn = plot_confusion(y_test_np, y_pred_cnn, cfg.class_names, "CNN - row-normalised confusion matrix")


# %%
# --- Where exactly does the CNN lose accuracy? ------------------------------------------------------------------
def top_confusions(cm: np.ndarray, class_names: Sequence[str], k: int = 10) -> pd.DataFrame:
    """List the k largest off-diagonal entries of a confusion matrix."""
    rows = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true": class_names[i],
                        "predicted": class_names[j],
                        "count": int(cm[i, j]),
                        "% of true class": 100.0 * cm[i, j] / cm[i].sum(),
                    }
                )
    return pd.DataFrame(rows).nlargest(k, "count").reset_index(drop=True)


conf_df = top_confusions(cm_cnn, cfg.class_names, k=10)
display(conf_df.style.hide(axis="index").format({"% of true class": "{:.1f}%"}))

total_err = int((y_pred_cnn != y_test_np).sum())
upper_body = {0, 2, 4, 6}  # T-shirt/top, Pullover, Coat, Shirt
mask_ub = np.isin(y_test_np, list(upper_body)) & np.isin(y_pred_cnn, list(upper_body)) & (y_pred_cnn != y_test_np)
print(f"\nTotal CNN test errors: {total_err} / {len(y_test_np)}  ({100 * total_err / len(y_test_np):.2f}%)")
print(f"Errors *inside* the T-shirt/Pullover/Coat/Shirt cluster: {int(mask_ub.sum())} "
      f"({100 * mask_ub.sum() / max(total_err, 1):.1f}% of all errors)")


# %%
# --- Inspect the hardest misclassified images -----------------------------------------------------------------
def show_misclassified(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    n: int = 16,
    most_confident: bool = True,
) -> None:
    """Display misclassified images, sorted by the model's confidence in its wrong answer."""
    wrong = np.flatnonzero(y_pred != y_true)
    if len(wrong) == 0:
        print("No misclassified images.")
        return
    conf = probs[wrong, y_pred[wrong]]
    order = wrong[np.argsort(-conf)] if most_confident else wrong[np.argsort(conf)]
    sel = order[:n]

    cols = 8
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.9))
    for ax, i in zip(np.array(axes).ravel(), sel):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(
            f"true: {class_names[y_true[i]]}\npred: {class_names[y_pred[i]]} ({probs[i, y_pred[i]]:.2f})",
            fontsize=6.5,
        )
        ax.axis("off")
    for ax in np.array(axes).ravel()[len(sel):]:
        ax.axis("off")
    kind = "most confident" if most_confident else "least confident"
    fig.suptitle(f"CNN misclassifications ({kind} mistakes)", y=1.02)
    plt.show()


show_misclassified(X_test_np, y_test_np, y_pred_cnn, probs_cnn, cfg.class_names, n=16, most_confident=True)


# %% [markdown]
# **Finding (answers RQ3).** Inspecting the confident mistakes is the most informative diagnostic in the whole
# notebook. The great majority of them are *shirt-like* garments where a human annotator would plausibly disagree with
# the ground-truth label: a long-sleeved top labelled `Shirt` but visually identical to `Pullover`, or a `Coat` whose
# open front has been flattened by the 28x28 downsampling.
#
# This supports the conclusion that a substantial part of the remaining ~6 % error is **irreducible label ambiguity**
# rather than a modelling deficiency: which is consistent with the fact that the published state of the art on
# Fashion-MNIST saturates in the 95–96.5 % range even with Wide-ResNets and heavy augmentation, while MNIST reaches
# 99.7 %.

# %% [markdown]
# ---
#
# # 4.9 - 4.12  Modern architectures and a production-grade training loop (upgraded edition)
#
# > Exam criteria: **Code Quality (0–20)** and **Testing (0–10)**.
#
# The v1 `fit()` in Section 4.3 is deliberately minimal: AdamW + OneCycle + AMP + best-checkpoint-in-memory. That is
# enough for a 25-epoch CNN, but it is *not* what a production training script looks like. Section 4.9 upgrades it to
# `fit_v2()`, and Sections 4.10–4.11 add the two architectures the v1 study was missing:
#
# | Addition | What it brings |
# |---|---|
# | `EarlyStopping` | stops when validation accuracy has not improved by `min_delta` for `patience` epochs: saves GPU time and prevents late-schedule overfitting |
# | Pluggable schedulers | `onecycle` (per-batch), `cosine` (per-epoch), `plateau` (metric-driven): selected per architecture, because transformers and CNNs do not want the same schedule |
# | Disk checkpointing | best weights **and** a resumable checkpoint (weights + optimiser + scheduler + epoch + history) written to `artifacts/` |
# | Gradient clipping | `clip_grad_norm_`: mandatory for stable transformer training |
# | Deterministic epoch timing + history | every run returns a complete, plottable record |
# | **Residual CNN** (Section 4.10) | tests whether skip connections add anything over the plain VGG-style stack: the exact upgrade Bhatnagar et al. (2017) reported as their best model |
# | **Vision Transformer** (Section 4.11) | a pure-attention architecture with *no* convolutional prior, custom-sized for 28x28 grayscale inputs (RQ8) |
#
# `fit()` from Section 4.3 is **left untouched** so the v1 MLP/CNN numbers remain reproducible; `fit_v2()` is a strict
# superset used for the new models.

# %%
# --- 4.9.1 Early stopping and checkpoint utilities -------------------------------------------------------------
class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement greater than `min_delta` before stopping.
    min_delta : float
        Minimum change that counts as an improvement (guards against numerical noise).
    mode : {"max", "min"}
        Whether the monitored metric should be maximised (accuracy) or minimised (loss).
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = -float("inf") if mode == "max" else float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Update the internal state with the metric of `epoch`. Returns True if this epoch is a new best."""
        improved = (
            metric > self.best + self.min_delta if self.mode == "max" else metric < self.best - self.min_delta
        )
        if improved:
            self.best, self.best_epoch, self.counter = metric, epoch, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
    history: Dict[str, List[float]], extra: Dict[str, object] | None = None,
) -> None:
    """Write a fully resumable checkpoint (weights + optimiser state + epoch + history + metadata)."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> Dict:
    """Restore a checkpoint written by `save_checkpoint` (weights always, optimiser state optionally)."""
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def build_scheduler(
    name: str, optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int, max_lr: float, pct_start: float
) -> Tuple[object, str]:
    """Create a learning-rate scheduler and report how it must be stepped.

    Returns
    -------
    (scheduler, cadence) where cadence is one of {"batch", "epoch", "plateau"}.
    """
    name = name.lower()
    if name == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start
        )
        return sched, "batch"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), "epoch"
    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2),
            "plateau",
        )
    raise ValueError(f"unknown scheduler '{name}' (expected onecycle | cosine | plateau)")


# %%
# --- 4.9.2 fit_v2: the production training loop ----------------------------------------------------------------
def fit_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = cfg.lr,
    weight_decay: float = cfg.weight_decay,
    label_smoothing: float = cfg.label_smoothing,
    scheduler_name: str = "onecycle",
    pct_start: float = 0.25,
    augment: nn.Module | None = None,
    use_amp: bool = cfg.use_amp,
    grad_clip: float | None = 1.0,
    patience: int = 8,
    min_delta: float = 1e-4,
    device: torch.device = DEVICE,
    model_name: str = "model",
    verbose_every: int = 1,
) -> Dict[str, object]:
    """Train a model with AdamW, a pluggable LR schedule, AMP, gradient clipping, early stopping and checkpointing.

    Compared with the v1 `fit()` this adds: scheduler choice, gradient clipping, early stopping, on-disk
    checkpoints (best + resumable last) and per-epoch wall-clock timing.

    Returns
    -------
    dict
        `{'model', 'history', 'best_val_acc', 'best_epoch', 'train_seconds', 'stopped_early',
          'epochs_run', 'checkpoint'}` - the same keys the v1 `fit()` returns, plus three new ones, so every
        downstream plotting/evaluation helper keeps working unchanged.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler, cadence = build_scheduler(
        scheduler_name, optimizer, epochs, len(train_loader), lr, pct_start
    )
    amp_on = bool(use_amp) and device.type == "cuda"
    scaler = make_grad_scaler(enabled=True) if amp_on else None

    stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode="max")
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_seconds": []
    }
    slug = model_name.replace(" ", "_").replace("/", "-").lower()
    best_path = Path(cfg.artifacts_dir) / f"{slug}_best.pt"
    last_path = Path(cfg.artifacts_dir) / f"{slug}_last.ckpt"
    best_state = None

    print(f"\n=== Training {model_name} | {count_parameters(model):,} params | up to {epochs} epochs "
          f"| scheduler={scheduler_name} | AMP={amp_on} | device={device} ===")
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- one training epoch (inlined so that gradient clipping can sit between backward and step) ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if augment is not None:
                xb = augment(xb)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
                logits = model(xb)
                loss = criterion(logits, yb)
            if amp_on:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)                      # unscale before clipping, or the norm is wrong
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if cadence == "batch":
                scheduler.step()
            running_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        tr_loss, tr_acc = running_loss / total, correct / total

        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if cadence == "epoch":
            scheduler.step()
        elif cadence == "plateau":
            scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)
        history["epoch_seconds"].append(time.time() - t_epoch)

        improved = stopper.step(va_acc, epoch)
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_path)                       # best weights only (small, portable)
        save_checkpoint(last_path, model, optimizer, epoch, history, {"model_name": model_name})

        if epoch % verbose_every == 0 or epoch == epochs or stopper.should_stop:
            flag = "  <-- best" if improved else f"  (no improvement for {stopper.counter}/{patience})"
            print(f"epoch {epoch:3d}/{epochs} | lr {current_lr:.2e} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
                  f" | {history['epoch_seconds'][-1]:.1f}s{flag}")

        if stopper.should_stop:
            print(f"--- early stopping triggered at epoch {epoch} "
                  f"(best {stopper.best:.4f} @ epoch {stopper.best_epoch}) ---")
            break

    train_seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)                           # always return the best model, not the last
    print(f"--- {model_name}: {train_seconds:.1f}s total | best val acc {stopper.best:.4f} "
          f"@ epoch {stopper.best_epoch} | checkpoints: {best_path.name}, {last_path.name} ---")

    return {
        "model": model,
        "history": history,
        "best_val_acc": stopper.best,
        "best_epoch": stopper.best_epoch,
        "train_seconds": train_seconds,
        "stopped_early": stopper.should_stop,
        "epochs_run": len(history["val_acc"]),
        "checkpoint": str(best_path),
    }


# %% [markdown]
# ## 4.10 A residual CNN (basic post-activation blocks, ~0.7 M parameters)
#
# The v1 CNN is a plain VGG-style stack. Residual connections (He et al., 2016) change the optimisation problem: each
# block learns a *residual* `F(x)` added to an identity path, so gradients reach early layers unattenuated and depth
# stops hurting. Bhatnagar et al. (2017) reported exactly this upgrade as their best Fashion-MNIST model (0.9254), which
# makes it the most direct architectural comparison available to us.
#
# ```
# stem   : Conv3x3(1 -> w) -> BN -> ReLU                                        28x28
# stage 1: [ResBlock(w  -> w )] x2                                              28x28
# stage 2: [ResBlock(w  -> 2w, stride 2)] + [ResBlock(2w -> 2w)]                14x14
# stage 3: [ResBlock(2w -> 4w, stride 2)] + [ResBlock(4w -> 4w)]                 7x7
# head   : GlobalAvgPool -> Dropout -> Linear(4w -> 10)
# ```
#
# With `w = 32` this is ~0.7 M parameters: still tiny by modern standards, and still under a minute per epoch on a T4.

# %%
# --- 4.10 Residual CNN -----------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Basic two-convolution residual block with BatchNorm and an optional projection shortcut.

    out = ReLU( BN(Conv(ReLU(BN(Conv(x))))) + shortcut(x) )

    The shortcut is the identity when the shape is unchanged, and a 1x1 strided convolution otherwise (option B
    in He et al. 2016).
    """

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

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last convolutional feature map (used by Grad-CAM in Section 6)."""
        return self.stage3(self.stage2(self.stage1(self.stem(x))))


_probe = torch.zeros(2, 1, 28, 28)
_resnet_probe = ResNetSmall(num_classes=cfg.num_classes, width=cfgx.resnet_width, p_drop=cfg.dropout)
print("ResNetSmall output:", _resnet_probe(_probe).shape, "| params:", f"{count_parameters(_resnet_probe):,}")
print("last feature map  :", _resnet_probe.feature_maps(_probe).shape)


# %% [markdown]
# ## 4.11 A Vision Transformer for 28x28 grayscale images
#
# An off-the-shelf ViT-Base expects 224x224 RGB inputs, 86 M parameters and 14 M pre-training images (Dosovitskiy et al.,
# 2021). None of that applies here, so the architecture is **re-derived for this dataset** rather than copied:
#
# | Design decision | Value | Reason |
# |---|---|---|
# | Patch size | **7x7** → 4x4 = **16 tokens** | 28 is divisible by 7; 16 tokens keeps attention cheap while each token still covers a semantically meaningful garment region (a shoulder, a cuff, a heel) |
# | Embedding dim | 128 | 4 heads x 32 dims per head; deliberately small, because 54k images cannot support a wide transformer |
# | Depth / heads | 6 blocks / 4 heads | ≈ 0.8 M parameters: the same order of magnitude as the residual CNN (≈ 0.7 M), so the comparison isolates *inductive bias* rather than capacity |
# | Position encoding | **learnable**, 17 x 128 (16 patches + CLS) | with only 16 positions there is nothing to gain from sinusoids |
# | Pre-norm blocks + GELU MLP (ratio 2.0) | | pre-norm is what makes deep transformers trainable without a long warm-up |
# | Stochastic depth (DropPath) linearly 0 → 0.1 | | the standard ViT regulariser; matters a lot in the small-data regime |
# | CLS token for classification | | keeps the head a single `Linear(192, 10)` |
# | Training | lower peak LR (`CFGX.vit_lr = 1e-3`), gradient clipping at 1.0, more epochs, same augmentation | transformers have no locality prior, so they need more epochs and a gentler optimiser to reach the same point |
#
# **This is the honest test of RQ8:** a transformer with no convolutional prior, trained from scratch on 54k tiny images:
# the regime where transformers are widely reported to lose to CNNs.

# %%
# --- 4.11 Vision Transformer ------------------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Split a 28x28 image into non-overlapping patches and linearly project each one to `dim`.

    Implemented as a strided convolution, which is mathematically identical to 'flatten each patch and apply a
    shared linear layer' but faster.
    """

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
    """Standard multi-head self-attention with a fused QKV projection.

    `return_attention=True` also returns the (B, heads, N, N) attention map, which Section 6.6 visualises.
    """

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
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, heads, N, N)
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
        dpr = torch.linspace(0, drop_path, depth).tolist()      # linearly increasing stochastic depth
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
        """Embed patches, prepend the CLS token and add positional embeddings."""
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])                    # classify from the CLS token

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Attention matrices of every block, `(B, heads, N+1, N+1)` - used for attention roll-out in Section 6.6."""
        self.eval()
        maps: List[torch.Tensor] = []
        h = self.forward_tokens(x)
        for blk in self.blocks:
            normed = blk.norm1(h)
            out, attn = blk.attn(normed, return_attention=True)
            maps.append(attn.detach())
            h = h + out
            h = h + blk.mlp(blk.norm2(h))
        return maps


_vit_probe = VisionTransformer(
    patch=cfgx.vit_patch, num_classes=cfg.num_classes, dim=cfgx.vit_dim, depth=cfgx.vit_depth,
    heads=cfgx.vit_heads, mlp_ratio=cfgx.vit_mlp_ratio, drop=cfgx.vit_dropout,
)
print("ViT output      :", _vit_probe(_probe).shape, "| params:", f"{count_parameters(_vit_probe):,}")
print("tokens per image:", _vit_probe.patch_embed.n_patches, "patches + 1 CLS")
print("attention maps  :", [tuple(a.shape) for a in _vit_probe.attention_maps(_probe)][:1], "... x",
      cfgx.vit_depth, "blocks")

# %% [markdown]
# ## 4.12 Training and evaluating the two new architectures
#
# Both models are trained with `fit_v2` on the identical loaders, the identical augmentation and the identical seed as
# the v1 CNN: only the architecture, the peak learning rate and the schedule differ, so the comparison stays
# controlled (this is the same discipline used for the MLP-vs-CNN comparison in Section 4.4–4.5).

# %%
# --- 4.12.1 Train the residual CNN -------------------------------------------------------------------------------
if cfgx.run_resnet:
    set_seed(cfg.seed)
    resnet_run = fit_v2(
        ResNetSmall(num_classes=cfg.num_classes, width=cfgx.resnet_width, p_drop=cfg.dropout),
        train_loader, val_loader,
        epochs=cfgx.resnet_epochs,
        lr=cfg.lr,
        scheduler_name="onecycle",
        augment=Augment(p_flip=0.5, max_shift=2) if cfg.augment else None,
        patience=cfgx.early_stopping_patience,
        min_delta=cfgx.min_delta,
        model_name="ResNet-small",
    )
    resnet_model = resnet_run["model"]
    probs_resnet, y_pred_resnet, rec_resnet = evaluate_torch_model(
        resnet_model, test_loader, "ResNet-small (residual CNN)", resnet_run,
        notes=f"3 stages x 2 residual blocks, width={cfgx.resnet_width}, flip/shift augmentation",
        arch="ResNetSmall",
        arch_kwargs={"num_classes": cfg.num_classes, "width": cfgx.resnet_width, "p_drop": cfg.dropout},
    )
else:
    resnet_run, resnet_model, probs_resnet, y_pred_resnet = None, None, None, None
    print("Residual CNN skipped (CFGX.run_resnet = False).")

# %%
# --- 4.12.2 Train the Vision Transformer -------------------------------------------------------------------------
if cfgx.run_vit:
    set_seed(cfg.seed)
    vit_run = fit_v2(
        VisionTransformer(
            patch=cfgx.vit_patch, num_classes=cfg.num_classes, dim=cfgx.vit_dim, depth=cfgx.vit_depth,
            heads=cfgx.vit_heads, mlp_ratio=cfgx.vit_mlp_ratio, drop=cfgx.vit_dropout, drop_path=0.1,
        ),
        train_loader, val_loader,
        epochs=cfgx.vit_epochs,
        lr=cfgx.vit_lr,                 # transformers need a gentler peak LR than the CNNs
        weight_decay=0.05,              # ... and much stronger weight decay (standard ViT recipe)
        scheduler_name="onecycle",
        pct_start=0.30,                 # longer warm-up: attention is unstable in the first epochs
        augment=Augment(p_flip=0.5, max_shift=2) if cfg.augment else None,
        grad_clip=1.0,
        patience=cfgx.early_stopping_patience,
        min_delta=cfgx.min_delta,
        model_name="ViT-tiny",
    )
    vit_model = vit_run["model"]
    probs_vit, y_pred_vit, rec_vit = evaluate_torch_model(
        vit_model, test_loader, "ViT-tiny (16 patches)", vit_run,
        notes=f"patch={cfgx.vit_patch}, dim={cfgx.vit_dim}, depth={cfgx.vit_depth}, heads={cfgx.vit_heads}",
        arch="VisionTransformer",
        arch_kwargs={
            "patch": cfgx.vit_patch, "num_classes": cfg.num_classes, "dim": cfgx.vit_dim,
            "depth": cfgx.vit_depth, "heads": cfgx.vit_heads,
            "mlp_ratio": cfgx.vit_mlp_ratio, "drop": cfgx.vit_dropout,
        },
    )
else:
    vit_run, vit_model, probs_vit, y_pred_vit = None, None, None, None
    print("Vision Transformer skipped (CFGX.run_vit = False).")

# %%
# --- 4.12.3 All four deep models side by side ---------------------------------------------------------------------
# Registry of every trained torch model, consumed by Sections 5 (ensembles), 6 (explainability) and 7 (tests).
TORCH_ZOO: Dict[str, nn.Module] = {"MLP (512-256)": mlp_model, "CNN (VGG-style, GAP)": cnn_model}
DEEP_RUNS: Dict[str, Dict[str, object]] = {"MLP": mlp_run, "CNN": cnn_run}
DEEP_PROBS: Dict[str, np.ndarray] = {"MLP (512-256)": probs_mlp, "CNN (VGG-style, GAP)": probs_cnn}
DEEP_PREDS: Dict[str, np.ndarray] = {"MLP (512-256)": y_pred_mlp, "CNN (VGG-style, GAP)": y_pred_cnn}
RUN_TO_MODEL: Dict[str, nn.Module] = {"MLP": mlp_model, "CNN": cnn_model}

if resnet_model is not None:
    TORCH_ZOO["ResNet-small (residual CNN)"] = resnet_model
    DEEP_RUNS["ResNet"] = resnet_run
    RUN_TO_MODEL["ResNet"] = resnet_model
    DEEP_PROBS["ResNet-small (residual CNN)"] = probs_resnet
    DEEP_PREDS["ResNet-small (residual CNN)"] = y_pred_resnet
if vit_model is not None:
    TORCH_ZOO["ViT-tiny (16 patches)"] = vit_model
    DEEP_RUNS["ViT"] = vit_run
    RUN_TO_MODEL["ViT"] = vit_model
    DEEP_PROBS["ViT-tiny (16 patches)"] = probs_vit
    DEEP_PREDS["ViT-tiny (16 patches)"] = y_pred_vit

for name, run in DEEP_RUNS.items():
    plot_history(run["history"], name)

gap_df = plot_generalisation_gap(DEEP_RUNS)
display(gap_df.style.hide(axis="index"))

deep_summary = pd.DataFrame([
    {
        "model": name,
        "parameters": count_parameters(RUN_TO_MODEL[name]),
        "epochs run": run.get("epochs_run", len(run["history"]["val_acc"])),
        "stopped early": run.get("stopped_early", False),
        "best val acc": run["best_val_acc"],
        "train seconds": run["train_seconds"],
        "sec / epoch": run["train_seconds"] / max(len(run["history"]["val_acc"]), 1),
    }
    for name, run in DEEP_RUNS.items()
])
display(deep_summary.style.hide(axis="index").format(
    {"parameters": "{:,.0f}", "best val acc": "{:.4f}", "train seconds": "{:.1f}", "sec / epoch": "{:.1f}"},
    na_rep="-"))

# %% [markdown]
# ### 4.12.5 Persisting the best deep models
#
# The same mechanism saves the **best-validation checkpoint** of each deep architecture to `artifacts/models/dl/`
# as a `*.pt` state-dict, with a sidecar recording the architecture class, the constructor arguments needed to
# rebuild it, the parameter count and the test metrics. A `BEST.json` pointer names the single best deep model, so
# the strongest network is retrievable in one line without re-training.

# %%
# --- 4.12.5 Persist the best version of every deep model ------------------------------------------------------
dl_saved = save_registered_models(only="dl")

# %%
# --- 4.12.4 Confusion matrices and error overlap of the new architectures -------------------------------------
new_preds = {k: v for k, v in DEEP_PREDS.items() if v is not None and "MLP" not in k}
n_new = len(new_preds)
if n_new:
    fig, axes = plt.subplots(1, n_new, figsize=(7.2 * n_new, 5.6))
    for ax, (name, pred) in zip(np.atleast_1d(axes), new_preds.items()):
        plot_confusion(y_test_np, pred, cfg.class_names, f"{name}\nrow-normalised confusion", ax=ax)
    plt.show()


def error_overlap_matrix(pred_map: Dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    """Jaccard overlap between the *error sets* of the models - the key diagnostic for ensembling potential.

    Two models that make the same mistakes cannot help each other; low overlap means the errors are complementary
    and a soft-voting ensemble should improve on both.
    """
    names = list(pred_map)
    err = {n: set(np.flatnonzero(pred_map[n] != y_true).tolist()) for n in names}
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            union = len(err[a] | err[b])
            mat[i, j] = len(err[a] & err[b]) / union if union else 1.0
    df = pd.DataFrame(mat, index=names, columns=names)
    plt.figure(figsize=(1.35 * len(names) + 3.5, 1.05 * len(names) + 2.6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": "Jaccard overlap of error sets"}, annot_kws={"size": 8})
    plt.title("4.12.4 Do the models make the SAME mistakes?")
    plt.show()
    return df.round(3)


overlap_source = {k: v for k, v in DEEP_PREDS.items() if v is not None}
if cfg.run_rbf_svm and y_pred_rbf is not None:
    overlap_source["RBF SVM"] = y_pred_rbf
if y_pred_cat is not None:
    overlap_source["CatBoost"] = y_pred_cat
elif y_pred_lgbm is not None:
    overlap_source["LightGBM"] = y_pred_lgbm

error_overlap_df = error_overlap_matrix(overlap_source, y_test_np)
print("Mean off-diagonal error overlap:",
      round(float(error_overlap_df.values[~np.eye(len(error_overlap_df), dtype=bool)].mean()), 3),
      "\n(values well below 1.0 mean the models fail on different images -> ensembling in Section 5 should help)")
