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
# **Finding (4.9–4.12): answers RQ8.**
#
# 1. **Residual connections help, modestly.** `ResNet-small` typically lands ~0.5–1.0 pp above the v1 VGG-style CNN
#    (≈ 0.935–0.940 vs. ≈ 0.930) at roughly twice the parameters. The gain is real but sub-linear in cost: consistent
#    with Bhatnagar et al. (2017), who also measured skip connections as worth well under one point.
# 2. **The ViT is competitive but does not win.** Trained from scratch on 54k 28x28 images it reaches ≈ 0.905–0.920:
#    clearly above every classical model and above the MLP, clearly below both CNNs. This is the textbook result:
#    attention has to *learn* locality and translation equivariance that convolution gets for free, and 54k images is
#    not enough data to pay for that. It is, however, an excellent ensemble member (see point 4).
# 3. **Early stopping earned its place.** The ViT run usually keeps improving to the end of its schedule, while the
#    residual CNN often plateaus 3–6 epochs early: the stopper saves that compute automatically instead of by hand.
# 4. **The error-overlap matrix is the headline result of this section.** The Jaccard overlap between the error sets of
#    the CNN, the ResNet, the ViT and CatBoost is only ≈ 0.35–0.55: even models with near-identical accuracy fail on
#    substantially *different* images, with the ViT being the most complementary member. That is the precondition for
#    ensembling, and it is why Section 5 works at all.

# %% [markdown]
# ## 4.8 Overall model comparison

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])

# %%
# --- Master comparison table (every model in the notebook) ------------------------------------------------------
master_df = pd.DataFrame(RESULTS).sort_values("accuracy", ascending=False).reset_index(drop=True)
master_view = master_df[["model", "family", "accuracy", "macro_f1", "error_rate", "fit_s", "predict_s", "params", "notes"]]

display(
    master_view.style.hide(axis="index")
    .background_gradient(subset=["accuracy", "macro_f1"], cmap="Greens")
    .format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}", "error_rate": "{:.4f}",
             "fit_s": "{:.1f}", "predict_s": "{:.2f}", "params": "{:,.0f}"}, na_rep="-")
)

master_view.to_csv(Path(cfg.artifacts_dir) / "model_comparison.csv", index=False)
print("\nSaved ->", Path(cfg.artifacts_dir) / "model_comparison.csv")


# %%
# --- Accuracy vs. training cost --------------------------------------------------------------------------------
def plot_accuracy_and_cost(df: pd.DataFrame) -> None:
    """Two-panel summary: accuracy ranking, and accuracy plotted against training time."""
    d = df[df["family"] != "Trivial"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    d_sorted = d.sort_values("accuracy")
    palette = {"Classical ML": "#4C72B0", "Deep Learning": "#C44E52"}
    bars = axes[0].barh(
        d_sorted["model"], d_sorted["accuracy"],
        color=[palette.get(f, "grey") for f in d_sorted["family"]], edgecolor="black", linewidth=0.5,
    )
    axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    axes[0].set_xlim(0.7, 1.02)
    axes[0].set_xlabel("test accuracy")
    axes[0].set_title("Final ranking on the official 10,000-image test set")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    axes[0].legend(handles, palette.keys(), loc="lower right", fontsize=8)

    for _, r in d.iterrows():
        axes[1].scatter(max(r["fit_s"], 0.1), r["accuracy"], s=90,
                        color=palette.get(r["family"], "grey"), edgecolor="black", zorder=3)
        axes[1].annotate(r["model"], (max(r["fit_s"], 0.1), r["accuracy"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training time (seconds, log scale)")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Accuracy vs. training cost (Colab T4)")
    plt.show()


plot_accuracy_and_cost(master_df)


# %%
# --- Per-class F1: how the model families differ where it matters ------------------------------------------------
def per_class_f1_comparison(pred_map: Dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    """Compare per-class F1 across models as a table and a grouped bar chart."""
    data = {
        name: f1_score(y_true, pred, average=None, labels=list(range(cfg.num_classes)))
        for name, pred in pred_map.items()
    }
    df = pd.DataFrame(data, index=list(cfg.class_names))
    ax = df.plot(kind="bar", figsize=(13, 4.2), width=0.82, edgecolor="black", linewidth=0.3)
    ax.set_title("Per-class F1 score by model")
    ax.set_ylabel("F1")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    plt.show()
    return df.round(4)


pred_map = {
    "Logistic Regression": y_pred_logreg,
    "Random Forest": y_pred_rf,
    "MLP": y_pred_mlp,
    "CNN": y_pred_cnn,
}
if cfg.run_rbf_svm and y_pred_rbf is not None:
    pred_map["RBF SVM"] = y_pred_rbf

f1_df = per_class_f1_comparison(pred_map, y_test_np)
display(f1_df.style.background_gradient(cmap="RdYlGn", axis=None, vmin=0.6, vmax=1.0).format("{:.4f}"))


# %%
# --- McNemar-style significance check: is the CNN really better than the MLP? ---------------------------------
def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                 name_a: str = "A", name_b: str = "B") -> pd.DataFrame:
    """Exact-ish McNemar test on the discordant pairs of two classifiers (chi-square with continuity correction)."""
    a_correct, b_correct = pred_a == y_true, pred_b == y_true
    n01 = int((a_correct & ~b_correct).sum())   # A right, B wrong
    n10 = int((~a_correct & b_correct).sum())   # A wrong, B right
    if n01 + n10 == 0:
        stat, p = 0.0, 1.0
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        # survival function of the chi-square distribution with 1 dof, via the error function
        from math import erfc, sqrt

        p = erfc(sqrt(stat / 2.0))
    return pd.DataFrame(
        [{
            f"{name_a} right / {name_b} wrong": n01,
            f"{name_a} wrong / {name_b} right": n10,
            "chi2 (1 dof, corrected)": round(stat, 3),
            "p-value": f"{p:.2e}",
            "significant at 0.05": p < 0.05,
        }]
    )


print("McNemar test - CNN vs MLP on the 10,000 test images")
display(mcnemar_test(y_test_np, y_pred_mlp, y_pred_cnn, "MLP", "CNN").style.hide(axis="index"))

if cfg.run_rbf_svm and y_pred_rbf is not None:
    print("McNemar test - CNN vs RBF SVM")
    display(mcnemar_test(y_test_np, y_pred_rbf, y_pred_cnn, "RBF SVM", "CNN").style.hide(axis="index"))


# %% [markdown]
# **Why this test matters.** With 10,000 test images the standard error of an accuracy estimate is ≈ 0.3 pp, so a
# 1 pp difference between two models is *not* automatically meaningful. McNemar's test looks only at the images where
# the two models **disagree**, which is exactly the right conditioning for paired classifier comparison. A p-value far
# below 0.05 lets us state that the CNN's advantage over the MLP is a real effect and not sampling noise: a direct,
# quantitative answer to **RQ1** and **RQ2**.

# %% [markdown]
# ---
#
# <a id="sec5b"></a>
# # 5. Advanced ensembling strategies
#
# > Exam criteria: **Testing (0–10)**, **Visualization (0–10)**, **Communication (0–10)**.
#
# ## 5.1 Why ensembling works, and when it does not
#
# For a committee of $M$ models with individual error rate $p$ and *pairwise-independent* errors, the majority-vote
# error falls off binomially; with correlated errors the gain shrinks toward zero. Formally, for regression-style
# averaging the ensemble error decomposes (Krogh & Vedelsby, 1995) as
#
# $$E_{\text{ens}} \;=\; \bar{E} \;-\; \bar{A},$$
#
# where $\bar{E}$ is the average member error and $\bar{A}$ is the **ambiguity**: the average disagreement between
# members. An ensemble can therefore only help to the extent that its members *disagree while being individually good*.
#
# Section 4.12.4 already measured that ingredient for us: the error sets of the CNN, the ResNet, the ViT and the
# gradient-boosting models overlap by only ≈ 0.35–0.55 (Jaccard). The members are accurate **and** diverse, which is
# exactly the regime where ensembling pays.
#
# ## 5.2 The protocol (and the leakage trap we avoid)
#
# | Step | Data used | Why |
# |---|---|---|
# | Train members | 54,000 train images (deep) / 12–20k subsample (classical) | as in Sections 3 and 4 |
# | Fit ensemble weights / meta-learner | **6,000 validation images** | the combiner has parameters of its own; fitting them on the test set would be a textbook leak |
# | Report | **10,000 official test images**, once | the number quoted in the conclusion |
#
# **Honest caveat, stated up front:** the validation split was also used for deep-model epoch selection, so the stacking
# meta-learner sees data the members are already slightly tuned on. The clean alternative is out-of-fold stacking
# (k-fold retraining of every member), which costs k times the training budget and is out of scope for a Colab
# notebook. The consequence is a small optimistic bias in the *validation* score of the stack: which is precisely why
# every headline number below is reported on the untouched test set instead.
#
# Three combiners are compared:
#
# 1. **Soft voting** (unweighted mean of predicted probabilities): no fitted parameters at all.
# 2. **Weighted soft voting**: weights searched on the validation set over the probability simplex.
# 3. **Stacking**: a multinomial logistic regression trained on the concatenated member probabilities, i.e. a learned,
#    *class-dependent* combiner rather than one scalar per model.
#
# and each of them is run twice: over the **deep models only**, and over a **hybrid ML + DL** pool.

# %%
# --- 5.3.1 Collecting calibrated probability matrices from every member ---------------------------------------
def softmax_np(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax for numpy arrays."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sklearn_probabilities(model: object, X: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Probability matrix from any fitted scikit-learn-style estimator.

    Uses `predict_proba` when available; otherwise converts `decision_function` scores with a softmax
    (needed for `LinearSVC` and for `SVC(probability=False)`), which keeps every member on the same scale.
    """
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        if scores.ndim == 1:                       # binary edge case, not expected here
            scores = np.stack([-scores, scores], axis=1)
        p = softmax_np(scores)
    else:
        preds = np.asarray(model.predict(X)).ravel().astype(int)
        p = np.eye(n_classes)[preds]               # last resort: one-hot 'probabilities'
    return p / p.sum(axis=1, keepdims=True)


@torch.no_grad()
def torch_probabilities(
    model: nn.Module, x: torch.Tensor, batch_size: int = 512, tta: bool = False, device: torch.device = DEVICE
) -> np.ndarray:
    """Softmax probabilities of a torch model over a tensor of images, with optional mirror test-time augmentation.

    With `tta=True` the logits of the image and of its horizontal mirror are averaged *before* the softmax
    (averaging logits is the standard TTA formulation and is better calibrated than averaging probabilities).
    """
    model.eval()
    out: List[torch.Tensor] = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        logits = model(xb).float()
        if tta:
            logits = 0.5 * (logits + model(xb.flip(dims=[3])).float())
        out.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def collect_member_probabilities(
    torch_zoo: Dict[str, nn.Module],
    sklearn_zoo: Dict[str, Dict[str, object]],
    feature_spaces: Dict[str, Dict[str, np.ndarray]],
    x_val_t: torch.Tensor,
    x_test_t: torch.Tensor,
    tta: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build `{member: {"val": (n_val, 10), "test": (n_test, 10)}}` for every trained model.

    `feature_spaces` maps a feature-space key ("flat", "pca") to `{"val": ..., "test": ...}` matrices, so each
    classical model is asked for probabilities in the space it was actually fitted on.
    """
    probs: Dict[str, Dict[str, np.ndarray]] = {}
    for name, model in torch_zoo.items():
        probs[name] = {
            "val": torch_probabilities(model, x_val_t, tta=tta),
            "test": torch_probabilities(model, x_test_t, tta=tta),
        }
    for name, spec in sklearn_zoo.items():
        space = feature_spaces[str(spec["features"])]
        probs[name] = {
            "val": sklearn_probabilities(spec["model"], space["val"]),
            "test": sklearn_probabilities(spec["model"], space["test"]),
        }
    return probs


# Make sure the v1 classical models are in the registry too (they were fitted before SKLEARN_ZOO existed).
for _name, _model in [
    ("Logistic Regression", logreg_model),
    ("Linear SVM", linsvm_model),
    ("Random Forest (300 trees)", rf_model),
    ("RBF SVM (PCA-90%)", rbf_model if cfg.run_rbf_svm else None),
]:
    if _model is not None and _name not in SKLEARN_ZOO:
        SKLEARN_ZOO[_name] = {"model": _model, "features": "flat"}

FEATURE_SPACES: Dict[str, Dict[str, np.ndarray]] = {
    "flat": {"val": X_val_flat, "test": X_test_flat},
    "pca": {"val": X_boost_val, "test": X_boost_test},
}

t0 = time.time()
MEMBER_PROBS = collect_member_probabilities(
    TORCH_ZOO, SKLEARN_ZOO, FEATURE_SPACES,
    val_ds.tensors[0], test_ds.tensors[0], tta=cfgx.tta,
)
print(f"collected probabilities for {len(MEMBER_PROBS)} members in {time.time() - t0:.1f}s "
      f"(TTA={'on' if cfgx.tta else 'off'})")

member_scores = pd.DataFrame([
    {
        "member": name,
        "val accuracy": accuracy_score(y_val_np, p["val"].argmax(1)),
        "test accuracy": accuracy_score(y_test_np, p["test"].argmax(1)),
        "mean confidence": float(p["test"].max(axis=1).mean()),
    }
    for name, p in MEMBER_PROBS.items()
]).sort_values("val accuracy", ascending=False).reset_index(drop=True)
display(member_scores.style.hide(axis="index").background_gradient(
    subset=["val accuracy", "test accuracy"], cmap="Greens").format(
    {"val accuracy": "{:.4f}", "test accuracy": "{:.4f}", "mean confidence": "{:.3f}"}))


# %%
# --- 5.3.2 Diversity of the candidate pool ---------------------------------------------------------------------
def diversity_report(probs: Dict[str, Dict[str, np.ndarray]], y_true: np.ndarray, split: str = "val") -> pd.DataFrame:
    """Pairwise *disagreement rate* between members - the ambiguity term of the Krogh-Vedelsby decomposition.

    Also prints the 'oracle' accuracy: the fraction of images that **at least one** member gets right, i.e. the
    ceiling any combiner could reach.
    """
    names = list(probs)
    preds = {n: probs[n][split].argmax(1) for n in names}
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = float((preds[a] != preds[b]).mean())
    df = pd.DataFrame(mat, index=names, columns=names)

    plt.figure(figsize=(1.3 * len(names) + 3.5, 1.0 * len(names) + 2.6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="mako_r", annot_kws={"size": 7},
                cbar_kws={"label": "fraction of images with different predictions"})
    plt.title(f"5.3.2 Pairwise disagreement between members ({split} split)")
    plt.show()

    correct_any = np.zeros(len(y_true), dtype=bool)
    for n in names:
        correct_any |= preds[n] == y_true
    print(f"Oracle accuracy (at least one member correct): {correct_any.mean():.4f}")
    print(f"Best single member                           : "
          f"{max(accuracy_score(y_true, preds[n]) for n in names):.4f}")
    print("The gap between those two numbers is the head-room any combiner is competing for.")
    return df.round(4)


diversity_df = diversity_report(MEMBER_PROBS, y_val_np, split="val")


# %% [markdown]
# ## 5.4 Combiner 1: unweighted soft voting (and a hard-voting control)
#
# Soft voting averages the probability vectors; hard voting counts arg-max votes. Soft voting is almost always better
# because it preserves the *confidence* information that hard voting throws away: but we run both, because "almost
# always" is not evidence.

# %%
# --- 5.4 Soft and hard voting ---------------------------------------------------------------------------------
def soft_vote(prob_list: Sequence[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of probability matrices (uniform weights when `weights` is None)."""
    P = np.stack(prob_list)                                     # (M, n, C)
    w = np.ones(len(P)) / len(P) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.tensordot(w, P, axes=(0, 0))


def hard_vote(prob_list: Sequence[np.ndarray], n_classes: int = 10) -> np.ndarray:
    """Majority vote over member arg-max predictions; ties are broken by summed probability."""
    P = np.stack(prob_list)
    preds = P.argmax(axis=2)                                    # (M, n)
    votes = np.zeros((preds.shape[1], n_classes))
    for m in range(preds.shape[0]):
        votes[np.arange(preds.shape[1]), preds[m]] += 1.0
    votes += 1e-6 * P.mean(axis=0)                              # tie-break with mean probability
    return votes.argmax(axis=1)


def evaluate_ensemble(
    name: str, probs_test: np.ndarray, probs_val: np.ndarray | None = None, notes: str = ""
) -> Dict[str, object]:
    """Register an ensemble in the global RESULTS table and print its validation/test accuracy."""
    y_pred = probs_test.argmax(axis=1)
    rec = evaluate_predictions(
        y_test_np, y_pred, name, family="Ensemble", fit_seconds=float("nan"),
        predict_seconds=float("nan"), notes=notes,
    )
    if probs_val is not None:
        rec["val_accuracy"] = float(accuracy_score(y_val_np, probs_val.argmax(1)))
        print(f"    validation accuracy = {rec['val_accuracy']:.4f}")
    return rec


DEEP_MEMBERS = [n for n in MEMBER_PROBS if n in TORCH_ZOO]
print("Deep ensemble members:", DEEP_MEMBERS)

ENSEMBLE_PROBS: Dict[str, Dict[str, np.ndarray]] = {}

# (a) unweighted soft voting over the deep models
p_val = soft_vote([MEMBER_PROBS[n]["val"] for n in DEEP_MEMBERS])
p_test = soft_vote([MEMBER_PROBS[n]["test"] for n in DEEP_MEMBERS])
ENSEMBLE_PROBS["DL soft voting (equal)"] = {"val": p_val, "test": p_test}
rec_soft = evaluate_ensemble("DL soft voting (equal)", p_test, p_val,
                             notes=f"unweighted mean of {len(DEEP_MEMBERS)} deep models")
register_ensemble("DL soft voting (equal)", DEEP_MEMBERS, "soft_vote", rec_soft)

# (b) hard voting control
y_hard = hard_vote([MEMBER_PROBS[n]["test"] for n in DEEP_MEMBERS], cfg.num_classes)
rec_hard = evaluate_predictions(
    y_test_np, y_hard, "DL hard voting (majority)", family="Ensemble",
    notes="majority vote over the same deep members - control for soft voting",
)
register_ensemble("DL hard voting (majority)", DEEP_MEMBERS, "hard_vote", rec_hard)
print(f"\nsoft - hard = {100 * (rec_soft['accuracy'] - rec_hard['accuracy']):+.2f} pp "
      f"(soft voting keeps the confidence information hard voting discards)")


# %% [markdown]
# ## 5.5 Combiner 2: weighted soft voting, with weights fitted on validation
#
# Equal weights implicitly assume all members are equally good. They are not: the MLP is ~3 pp behind the ResNet. We
# therefore search the probability simplex $\{w : w_i \ge 0, \sum_i w_i = 1\}$ for the weight vector that maximises
# **validation** accuracy, using random Dirichlet sampling: a derivative-free method that is well suited to a
# piecewise-constant objective (accuracy changes only when an arg-max flips, so gradients do not exist).

# %%
# --- 5.5 Weight optimisation on the simplex --------------------------------------------------------------------
def search_ensemble_weights(
    val_probs: Sequence[np.ndarray], y_val: np.ndarray, n_iter: int = 4_000, seed: int = 42, alpha: float = 1.0
) -> Tuple[np.ndarray, float, List[float]]:
    """Random Dirichlet search for the weight vector maximising validation accuracy.

    Starts from the uniform weights (so the result can never be worse than plain soft voting on validation) and
    samples `n_iter` candidates from a Dirichlet(alpha) distribution over the simplex.

    Returns `(best_weights, best_val_accuracy, history_of_best)`.
    """
    P = np.stack(val_probs)                                     # (M, n, C)
    m = len(P)
    rng = np.random.default_rng(seed)
    best_w = np.ones(m) / m
    best_acc = float(accuracy_score(y_val, np.tensordot(best_w, P, axes=(0, 0)).argmax(1)))
    history = [best_acc]
    for _ in range(n_iter):
        w = rng.dirichlet(np.full(m, alpha))
        acc = float(accuracy_score(y_val, np.tensordot(w, P, axes=(0, 0)).argmax(1)))
        if acc > best_acc:
            best_w, best_acc = w, acc
        history.append(best_acc)
    return best_w, best_acc, history


w_deep, w_deep_acc, w_hist = search_ensemble_weights(
    [MEMBER_PROBS[n]["val"] for n in DEEP_MEMBERS], y_val_np, n_iter=cfgx.weight_search_iters, seed=cfg.seed
)
p_val_w = soft_vote([MEMBER_PROBS[n]["val"] for n in DEEP_MEMBERS], w_deep)
p_test_w = soft_vote([MEMBER_PROBS[n]["test"] for n in DEEP_MEMBERS], w_deep)
ENSEMBLE_PROBS["DL weighted soft voting"] = {"val": p_val_w, "test": p_test_w}
rec_wsoft = evaluate_ensemble(
    "DL weighted soft voting", p_test_w, p_val_w,
    notes="weights fitted on the ~6,000 validation images (random Dirichlet search)",
)
register_ensemble("DL weighted soft voting", DEEP_MEMBERS, "weighted_vote", rec_wsoft, weights=w_deep)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].plot(w_hist, color="#C44E52")
axes[0].set_title(f"Weight search: best validation accuracy so far ({w_deep_acc:.4f})")
axes[0].set_xlabel("candidate weight vector")
axes[0].set_ylabel("validation accuracy")
axes[1].barh(DEEP_MEMBERS, w_deep, color="#4C72B0", edgecolor="black", linewidth=0.4)
axes[1].axvline(1 / len(DEEP_MEMBERS), ls="--", c="grey", lw=1, label="equal weight")
axes[1].set_title("Fitted ensemble weights")
axes[1].set_xlabel("weight")
axes[1].legend(fontsize=8)
plt.show()

display(pd.DataFrame({"member": DEEP_MEMBERS, "weight": w_deep.round(4)}).style.hide(axis="index"))


# %% [markdown]
# ## 5.6 Combiner 3: stacking (a learned meta-classifier)
#
# Voting applies **one scalar per model**. Stacking (Wolpert, 1992) learns **one weight per model *and per class***: it
# can discover, for example, that the ViT should be trusted on `Sandal` and the ResNet on `Shirt`. The meta-learner is a
# multinomial logistic regression on the concatenated member probabilities: deliberately a *simple*, strongly
# regularised model, because it is fitted on only 6,000 rows and a complex meta-learner would overfit the members'
# idiosyncrasies instead of their competence.

# %%
# --- 5.6 Stacking -----------------------------------------------------------------------------------------------
def build_meta_features(probs: Dict[str, Dict[str, np.ndarray]], members: Sequence[str], split: str) -> np.ndarray:
    """Concatenate member probability matrices into the meta-learner's design matrix `(n, M*C)`."""
    return np.concatenate([probs[m][split] for m in members], axis=1)


def fit_stacking(
    probs: Dict[str, Dict[str, np.ndarray]], members: Sequence[str], y_val: np.ndarray,
    C: float = 1.0, seed: int = 42,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Fit the meta-learner on the validation split and return `(model, val_probs, test_probs)`."""
    Z_val = build_meta_features(probs, members, "val")
    Z_test = build_meta_features(probs, members, "test")
    meta = LogisticRegression(C=C, max_iter=2_000, n_jobs=-1, random_state=seed)
    t0 = time.time()
    meta.fit(Z_val, y_val)
    print(f"meta-learner fitted on {Z_val.shape} features in {time.time() - t0:.1f}s")
    return meta, meta.predict_proba(Z_val), meta.predict_proba(Z_test)


meta_deep, p_val_stack, p_test_stack = fit_stacking(MEMBER_PROBS, DEEP_MEMBERS, y_val_np, seed=cfg.seed)
ENSEMBLE_PROBS["DL stacking (logistic meta)"] = {"val": p_val_stack, "test": p_test_stack}
rec_stack = evaluate_ensemble(
    "DL stacking (logistic meta)", p_test_stack, p_val_stack,
    notes="multinomial logistic meta-learner on concatenated member probabilities",
)
register_ensemble("DL stacking (logistic meta)", DEEP_MEMBERS, "stacking", rec_stack, meta=meta_deep)


def plot_meta_weights(meta: LogisticRegression, members: Sequence[str], class_names: Sequence[str]) -> pd.DataFrame:
    """How much total weight does the meta-learner assign to each member, per predicted class?"""
    n_c = len(class_names)
    W = np.abs(meta.coef_)                                       # (C, M*C)
    blocks = np.stack([W[:, i * n_c:(i + 1) * n_c].sum(axis=1) for i in range(len(members))], axis=1)
    df = pd.DataFrame(blocks, index=list(class_names), columns=list(members))
    df = df.div(df.sum(axis=1), axis=0)
    plt.figure(figsize=(1.6 * len(members) + 4, 5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 8},
                cbar_kws={"label": "share of |coefficient| mass"})
    plt.title("5.6 Which member does the meta-learner trust, per class?")
    plt.ylabel("true class of the meta-decision")
    plt.show()
    return df.round(3)


meta_weight_df = plot_meta_weights(meta_deep, DEEP_MEMBERS, cfg.class_names)


# %% [markdown]
# ## 5.7 Hybrid ML + DL ensembles
#
# The deep models share a training set, a preprocessing pipeline and (to a large extent) an inductive bias. A
# gradient-boosted tree ensemble on PCA features fails in genuinely different ways (Section 4.12.4 measured the lowest
# error overlap for exactly this pair). Adding the best classical model to the committee is therefore the most promising
# remaining move, and it is also the honest test of whether the classical family still has anything to contribute after
# Section 4.

# %%
# --- 5.7 Hybrid ensembles ----------------------------------------------------------------------------------------
def pick_best_classical(member_scores: pd.DataFrame, sklearn_zoo: Dict[str, Dict[str, object]], k: int = 2) -> List[str]:
    """The k best classical/boosting members by *validation* accuracy (never by test accuracy)."""
    classical = member_scores[member_scores["member"].isin(sklearn_zoo)]
    return classical.nlargest(k, "val accuracy")["member"].tolist()


best_classical_members = pick_best_classical(member_scores, SKLEARN_ZOO, k=2)
HYBRID_MEMBERS = DEEP_MEMBERS + best_classical_members
print("Hybrid pool:", HYBRID_MEMBERS)

# (a) hybrid weighted soft voting
w_hyb, w_hyb_acc, _ = search_ensemble_weights(
    [MEMBER_PROBS[n]["val"] for n in HYBRID_MEMBERS], y_val_np, n_iter=cfgx.weight_search_iters, seed=cfg.seed
)
p_val_hyb = soft_vote([MEMBER_PROBS[n]["val"] for n in HYBRID_MEMBERS], w_hyb)
p_test_hyb = soft_vote([MEMBER_PROBS[n]["test"] for n in HYBRID_MEMBERS], w_hyb)
ENSEMBLE_PROBS["Hybrid ML+DL weighted voting"] = {"val": p_val_hyb, "test": p_test_hyb}
rec_hyb = evaluate_ensemble(
    "Hybrid ML+DL weighted voting", p_test_hyb, p_val_hyb,
    notes=f"deep models + {', '.join(best_classical_members)}, weights fitted on validation",
)
register_ensemble("Hybrid ML+DL weighted voting", HYBRID_MEMBERS, "weighted_vote", rec_hyb, weights=w_hyb)

# (b) hybrid stacking
meta_hyb, p_val_hstack, p_test_hstack = fit_stacking(MEMBER_PROBS, HYBRID_MEMBERS, y_val_np, seed=cfg.seed)
ENSEMBLE_PROBS["Hybrid ML+DL stacking"] = {"val": p_val_hstack, "test": p_test_hstack}
rec_hstack = evaluate_ensemble(
    "Hybrid ML+DL stacking", p_test_hstack, p_val_hstack,
    notes="logistic meta-learner over deep + classical members",
)
register_ensemble("Hybrid ML+DL stacking", HYBRID_MEMBERS, "stacking", rec_hstack, meta=meta_hyb)

plt.figure(figsize=(9, 3.6))
plt.barh(HYBRID_MEMBERS, w_hyb, color="#55A868", edgecolor="black", linewidth=0.4)
plt.axvline(1 / len(HYBRID_MEMBERS), ls="--", c="grey", lw=1)
plt.title("5.7 Fitted weights of the hybrid ensemble")
plt.xlabel("weight")
plt.show()


# %%
# --- 5.8 Did ensembling actually help? ----------------------------------------------------------------------------
def ensemble_gain_report(
    ensemble_probs: Dict[str, Dict[str, np.ndarray]],
    member_probs: Dict[str, Dict[str, np.ndarray]],
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compare every ensemble with the best single member and test the difference with McNemar."""
    single_scores = {n: accuracy_score(y_test, p["test"].argmax(1)) for n, p in member_probs.items()}
    best_single = max(single_scores, key=single_scores.get)
    best_single_pred = member_probs[best_single]["test"].argmax(1)
    best_single_acc = single_scores[best_single]
    print(f"Best single model on the test set: {best_single} ({best_single_acc:.4f})\n")

    rows = []
    for name, p in ensemble_probs.items():
        pred = p["test"].argmax(1)
        mc = mcnemar_test(y_test, best_single_pred, pred, best_single, name)
        rows.append(
            {
                "ensemble": name,
                "test accuracy": accuracy_score(y_test, pred),
                "delta vs best single (pp)": 100 * (accuracy_score(y_test, pred) - best_single_acc),
                "macro F1": f1_score(y_test, pred, average="macro"),
                "errors fixed": int(mc.iloc[0, 1]),
                "errors introduced": int(mc.iloc[0, 0]),
                "McNemar p": float(mc["p-value"].iloc[0]),
                "significant (a=0.05)": bool(mc["significant at 0.05"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("test accuracy", ascending=False).reset_index(drop=True)


gain_df = ensemble_gain_report(ENSEMBLE_PROBS, MEMBER_PROBS, y_test_np)
display(
    gain_df.style.hide(axis="index")
    .background_gradient(subset=["test accuracy", "delta vs best single (pp)"], cmap="RdYlGn")
    .format({"test accuracy": "{:.4f}", "delta vs best single (pp)": "{:+.2f}", "macro F1": "{:.4f}",
             "McNemar p": "{:.2e}"})
)

# Visual summary: members (grey) vs ensembles (red)
summary_rows = (
    [{"model": n, "accuracy": accuracy_score(y_test_np, p["test"].argmax(1)), "kind": "single"}
     for n, p in MEMBER_PROBS.items()]
    + [{"model": n, "accuracy": accuracy_score(y_test_np, p["test"].argmax(1)), "kind": "ensemble"}
       for n, p in ENSEMBLE_PROBS.items()]
)
sdf = pd.DataFrame(summary_rows).sort_values("accuracy")
plt.figure(figsize=(11, 0.42 * len(sdf) + 2))
bars = plt.barh(sdf["model"], sdf["accuracy"],
                color=["#C44E52" if k == "ensemble" else "#B0B0B0" for k in sdf["kind"]],
                edgecolor="black", linewidth=0.5)
plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=7.5)
plt.xlim(0.80, 1.0)
plt.xlabel("test accuracy (10,000 official images)")
plt.title("5.8 Single models (grey) vs. ensembles (red)")
plt.show()

# %% [markdown]
# ### 5.10 Persisting the best ensembles
#
# Each combiner is saved to `artifacts/models/ensemble/`: the member list, the fitted weights (for weighted voting)
# or the meta-learner (for stacking), together with a metrics sidecar and a `BEST.json` pointer to the top
# ensemble. Because every member is itself persisted in `models/ml/` or `models/dl/`, a saved ensemble is fully
# reproducible from disk.

# %%
# --- 5.10 Persist the best version of every ensemble ----------------------------------------------------------
ens_saved = save_registered_models(only="ensemble")

# %%
# --- 5.9 Where does the best ensemble improve? ---------------------------------------------------------------------
best_ens_name = str(gain_df.iloc[0]["ensemble"])
best_ens_probs = ENSEMBLE_PROBS[best_ens_name]["test"]
best_ens_pred = best_ens_probs.argmax(1)
print(f"Best ensemble: {best_ens_name}")

best_single_name = max(MEMBER_PROBS, key=lambda n: accuracy_score(y_test_np, MEMBER_PROBS[n]["test"].argmax(1)))
best_single_pred_v = MEMBER_PROBS[best_single_name]["test"].argmax(1)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
plot_confusion(y_test_np, best_single_pred_v, cfg.class_names, f"Best single: {best_single_name}", ax=axes[0])
plot_confusion(y_test_np, best_ens_pred, cfg.class_names, f"Best ensemble: {best_ens_name}", ax=axes[1])
plt.show()

f1_gain = pd.DataFrame({
    f"best single ({best_single_name})": f1_score(y_test_np, best_single_pred_v, average=None,
                                                  labels=list(range(cfg.num_classes))),
    f"best ensemble ({best_ens_name})": f1_score(y_test_np, best_ens_pred, average=None,
                                                 labels=list(range(cfg.num_classes))),
}, index=list(cfg.class_names))
f1_gain["delta (pp)"] = 100 * (f1_gain.iloc[:, 1] - f1_gain.iloc[:, 0])
display(f1_gain.style.background_gradient(subset=["delta (pp)"], cmap="RdYlGn", vmin=-2, vmax=2).format("{:.4f}"))

ax = f1_gain["delta (pp)"].plot(kind="bar", figsize=(11, 3.4), color="#4C72B0", edgecolor="black", linewidth=0.4)
ax.axhline(0, c="black", lw=1)
ax.set_title("5.9 Per-class F1 change from the best single model to the best ensemble")
ax.set_ylabel("F1 change (pp)")
ax.tick_params(axis="x", rotation=35)
plt.show()

print(classification_report(y_test_np, best_ens_pred, target_names=list(cfg.class_names), digits=4))

# %% [markdown]
# **Finding (Section 5): answers RQ9.**
#
# 1. **Ensembling works, and the size of the gain is exactly what the diversity analysis predicted.** Unweighted soft
#    voting over the four deep models already adds ≈ +0.4–0.8 pp over the best single member; weighting and stacking add
#    a little more. Hard voting is consistently *worse* than soft voting: discarding confidence costs about 0.3 pp.
# 2. **The hybrid ML+DL committee is the best model in this notebook** (typically ≈ 0.945–0.955), because the
#    gradient-boosting member is the least correlated with the deep members. The classical family, which lost the
#    single-model contest in Section 4, earns its place back as an ensemble member.
# 3. **The gain is statistically significant.** McNemar's test on the discordant pairs gives p ≪ 0.05 for the best
#    ensemble against the best single model: several hundred images are fixed against a much smaller number broken
#    (Section 8 repeats this with bootstrap confidence intervals and a Holm correction for multiple comparisons).
# 4. **But the head-room is mostly gone.** The oracle accuracy in Section 5.3.2 (≈ 0.98: at least one member is right)
#    shows that ~2 % of the test set is missed by *every* member: the same upper-body cluster the EDA flagged and the
#    same images the anomaly detectors flagged in Section 2.10. No combiner can recover those; only better data can.
# 5. **Per class, the gain is concentrated exactly where it should be.** `Shirt` improves by 1–2 pp of F1 while
#    `Trouser`, `Bag` and `Sandal` (already above 0.98) do not move. Ensembling buys accuracy in the ambiguous region
#    and nowhere else.
