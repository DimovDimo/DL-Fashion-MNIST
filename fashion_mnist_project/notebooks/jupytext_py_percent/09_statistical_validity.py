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
# ---
#
# <a id="sec8b"></a>
# # 8. Statistical validity: which differences are real?
#
# > Exam criterion: **Testing (0–10)**: *"Is the process statistically valid?"*
#
# Section 4.8 ran a single McNemar test (CNN vs. MLP). With fourteen models on the leaderboard that is no longer enough,
# for two reasons:
#
# 1. **Sampling noise.** With $n = 10{,}000$ test images, the standard error of an accuracy near $p = 0.94$ is
#    $\sqrt{p(1-p)/n} \approx 0.24$ pp, so a 95 % interval is about $\pm 0.5$ pp. Roughly half of the pairwise
#    differences on our leaderboard are *smaller than that*.
# 2. **Multiple comparisons.** Testing all $\binom{M}{2}$ pairs at $\alpha = 0.05$ produces false positives by
#    construction: with 14 models there are 91 pairs, so ~4.5 "significant" results are expected **from noise alone**.
#
# This section therefore applies four tools, in increasing order of strictness:
#
# | Tool | Question answered | Note |
# |---|---|---|
# | **Wilson score interval** | what is the confidence interval of a single accuracy? | better than the normal approximation near the boundaries |
# | **Paired bootstrap** | what is the CI of the *difference* between two models? | resamples the test set, keeping models paired: the correct way to compare on shared data |
# | **McNemar's test** (exact binomial + $\chi^2$ with continuity correction) | is the difference between two classifiers significant? | conditions on the discordant pairs only (Dietterich, 1998) |
# | **Cochran's Q + Holm–Bonferroni** | are *all* models equivalent, and which pairwise results survive multiplicity correction? | Cochran's Q is the omnibus test; Holm controls the family-wise error rate without Bonferroni's conservatism |

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])


# %%
# --- 8.1 Assemble every model's test predictions in one place ---------------------------------------------------
def build_prediction_registry() -> Dict[str, np.ndarray]:
    """Collect the test-set predictions of every single model and every ensemble into one dictionary."""
    preds: Dict[str, np.ndarray] = {}
    if "MEMBER_PROBS" in globals():
        for name, p in MEMBER_PROBS.items():
            preds[name] = p["test"].argmax(1)
    else:                                            # fall back to the v1 predictions if Section 5 was skipped
        preds["MLP (512-256)"] = y_pred_mlp
        preds["CNN (VGG-style, GAP)"] = y_pred_cnn
        preds["Random Forest (300 trees)"] = y_pred_rf
    if "ENSEMBLE_PROBS" in globals():
        for name, p in ENSEMBLE_PROBS.items():
            preds[name] = p["test"].argmax(1)
    return preds


ALL_PREDICTIONS = build_prediction_registry()
CORRECT = np.stack([ALL_PREDICTIONS[n] == y_test_np for n in ALL_PREDICTIONS])   # (M, n_test) boolean
MODEL_NAMES = list(ALL_PREDICTIONS)
print(f"{len(MODEL_NAMES)} models entered into the statistical comparison, "
      f"each with {CORRECT.shape[1]:,} paired test predictions.")


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion (better than the normal approximation)."""
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


accuracy_ci = pd.DataFrame([
    {
        "model": name,
        "accuracy": CORRECT[i].mean(),
        "correct": int(CORRECT[i].sum()),
        "CI low (95%)": wilson_interval(int(CORRECT[i].sum()), CORRECT.shape[1], cfgx.alpha)[0],
        "CI high (95%)": wilson_interval(int(CORRECT[i].sum()), CORRECT.shape[1], cfgx.alpha)[1],
    }
    for i, name in enumerate(MODEL_NAMES)
]).sort_values("accuracy", ascending=False).reset_index(drop=True)
accuracy_ci["CI width (pp)"] = 100 * (accuracy_ci["CI high (95%)"] - accuracy_ci["CI low (95%)"])
display(accuracy_ci.style.hide(axis="index").format({
    "accuracy": "{:.4f}", "CI low (95%)": "{:.4f}", "CI high (95%)": "{:.4f}", "CI width (pp)": "{:.2f}"}))

fig, ax = plt.subplots(figsize=(10, 0.42 * len(accuracy_ci) + 2))
d = accuracy_ci.sort_values("accuracy")
ax.errorbar(
    d["accuracy"], d["model"],
    xerr=[d["accuracy"] - d["CI low (95%)"], d["CI high (95%)"] - d["accuracy"]],
    fmt="o", capsize=3, color="#4C72B0", ecolor="#C44E52", markersize=5,
)
ax.set_xlabel("test accuracy with 95% Wilson interval")
ax.set_title("8.1 Every model with its confidence interval - overlapping bars mean 'indistinguishable'")
plt.show()


# %%
# --- 8.2 McNemar's test, exact and asymptotic --------------------------------------------------------------------
def mcnemar_full(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, name_a: str = "A", name_b: str = "B"
) -> Dict[str, object]:
    """Full McNemar analysis of two classifiers on the same test set.

    Reports the discordant counts, the continuity-corrected chi-square statistic and **both** p-values:
    the exact binomial one (valid for any count) and the asymptotic chi-square one. When `statsmodels` is
    available its implementation is used as an independent cross-check of our own numbers.
    """
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01 = int((a_ok & ~b_ok).sum())          # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())          # A wrong, B right
    n_disc = n01 + n10

    if n_disc == 0:
        chi2_stat, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / n_disc
        p_chi2 = float(sp_stats.chi2.sf(chi2_stat, df=1))
        p_exact = float(sp_stats.binomtest(min(n01, n10), n=n_disc, p=0.5).pvalue)

    out: Dict[str, object] = {
        "model A": name_a,
        "model B": name_b,
        "acc A": float(a_ok.mean()),
        "acc B": float(b_ok.mean()),
        "A right / B wrong": n01,
        "A wrong / B right": n10,
        "chi2 (corrected)": round(chi2_stat, 3),
        "p (exact binomial)": p_exact,
        "p (chi2)": p_chi2,
        "odds ratio": round(n10 / n01, 3) if n01 else float("inf"),
    }
    if HAS_STATSMODELS:
        try:
            from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
            table = np.array([[int((a_ok & b_ok).sum()), n01], [n10, int((~a_ok & ~b_ok).sum())]])
            out["p (statsmodels exact)"] = float(sm_mcnemar(table, exact=True).pvalue)
        except Exception as exc:  # noqa: BLE001
            out["p (statsmodels exact)"] = f"unavailable: {exc}"
    return out


# Head-to-head table: everything against the current leader
leader = accuracy_ci.iloc[0]["model"]
head_to_head = pd.DataFrame([
    mcnemar_full(y_test_np, ALL_PREDICTIONS[name], ALL_PREDICTIONS[leader], name, leader)
    for name in MODEL_NAMES if name != leader
]).sort_values("acc A", ascending=False).reset_index(drop=True)

print(f"McNemar: every model against the leaderboard leader ({leader})\n")
display(
    head_to_head.style.hide(axis="index")
    .background_gradient(subset=["p (exact binomial)"], cmap="RdYlGn_r", vmin=0, vmax=0.2)
    .format({"acc A": "{:.4f}", "acc B": "{:.4f}", "p (exact binomial)": "{:.3e}", "p (chi2)": "{:.3e}"})
)


# %%
# --- 8.3 All pairwise comparisons with a Holm-Bonferroni correction ----------------------------------------------
def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm's step-down procedure: controls the family-wise error rate, uniformly more powerful than Bonferroni.

    Returns `(reject, adjusted_p)` in the original order of `p_values`.
    """
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
    predictions: Dict[str, np.ndarray], y_true: np.ndarray, alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar for every model pair, then a Holm correction over the whole family of tests."""
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
                annot_kws={"size": 6}, cbar_kws={"label": "Holm-adjusted p-value (green = significant)"})
    plt.title("8.3 Pairwise McNemar tests, Holm-corrected for multiple comparisons")
    plt.show()
    return df.sort_values("p Holm-adjusted").reset_index(drop=True), grid


pairwise_df, pairwise_grid = pairwise_significance(ALL_PREDICTIONS, y_test_np, alpha=cfgx.alpha)
n_sig = int(pairwise_df["significant"].sum())
print(f"{n_sig} of {len(pairwise_df)} pairwise differences survive the Holm correction at alpha={cfgx.alpha}.")
display(
    pairwise_df.head(15).style.hide(axis="index")
    .format({"acc A": "{:.4f}", "acc B": "{:.4f}", "delta (pp)": "{:+.2f}",
             "p raw": "{:.2e}", "p Holm-adjusted": "{:.2e}"})
)
print("\nPairs that are statistically INDISTINGUISHABLE (largest adjusted p-values):")
display(
    pairwise_df.tail(8).sort_values("p Holm-adjusted", ascending=False).style.hide(axis="index")
    .format({"acc A": "{:.4f}", "acc B": "{:.4f}", "delta (pp)": "{:+.2f}",
             "p raw": "{:.2e}", "p Holm-adjusted": "{:.2e}"})
)


# %%
# --- 8.4 Paired bootstrap confidence intervals --------------------------------------------------------------------
def paired_bootstrap(
    correct: np.ndarray, names: Sequence[str], n_iter: int = 2_000, alpha: float = 0.05, seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap the test set, keeping all models paired on the same resampled images.

    Pairing matters: models are evaluated on identical data, so the *difference* has much lower variance than the
    two marginal accuracies suggest. Returns the per-model CI table and the `(n_iter, M)` matrix of bootstrap
    accuracies (re-used below for the difference CI).
    """
    rng = np.random.default_rng(seed)
    n = correct.shape[1]
    boot = np.empty((n_iter, correct.shape[0]), dtype=np.float64)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot[b] = correct[:, idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    df = pd.DataFrame({
        "model": list(names),
        "accuracy": correct.mean(axis=1),
        "bootstrap CI low": lo,
        "bootstrap CI high": hi,
        "bootstrap std (pp)": 100 * boot.std(axis=0),
    }).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df, boot


boot_df, boot_matrix = paired_bootstrap(
    CORRECT, MODEL_NAMES, n_iter=cfgx.bootstrap_iters, alpha=cfgx.alpha, seed=cfg.seed
)
display(boot_df.style.hide(axis="index").format({
    "accuracy": "{:.4f}", "bootstrap CI low": "{:.4f}", "bootstrap CI high": "{:.4f}",
    "bootstrap std (pp)": "{:.2f}"}))


def bootstrap_difference(
    boot: np.ndarray, names: Sequence[str], name_a: str, name_b: str, alpha: float = 0.05
) -> Dict[str, float]:
    """Percentile CI of `acc(B) - acc(A)` from the *paired* bootstrap replicates."""
    i, j = list(names).index(name_a), list(names).index(name_b)
    diff = boot[:, j] - boot[:, i]
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return {
        "comparison": f"{name_b} - {name_a}",
        "mean delta (pp)": float(100 * diff.mean()),
        "CI low (pp)": float(100 * lo),
        "CI high (pp)": float(100 * hi),
        "P(B better than A)": float((diff > 0).mean()),
    }


best_single_stat = max(
    [n for n in MODEL_NAMES if "voting" not in n and "stacking" not in n],
    key=lambda n: CORRECT[MODEL_NAMES.index(n)].mean(),
)
comparisons = [n for n in MODEL_NAMES if n != best_single_stat and ("voting" in n or "stacking" in n)]
if comparisons:
    diff_df = pd.DataFrame([
        bootstrap_difference(boot_matrix, MODEL_NAMES, best_single_stat, c, cfgx.alpha) for c in comparisons
    ]).sort_values("mean delta (pp)", ascending=False)
    display(diff_df.style.hide(axis="index").format({
        "mean delta (pp)": "{:+.2f}", "CI low (pp)": "{:+.2f}", "CI high (pp)": "{:+.2f}",
        "P(B better than A)": "{:.3f}"}))

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(diff_df) + 2))
    ax.errorbar(
        diff_df["mean delta (pp)"], diff_df["comparison"],
        xerr=[diff_df["mean delta (pp)"] - diff_df["CI low (pp)"],
              diff_df["CI high (pp)"] - diff_df["mean delta (pp)"]],
        fmt="o", capsize=4, color="#55A868", ecolor="#4C72B0",
    )
    ax.axvline(0, ls="--", c="red", lw=1.2)
    ax.set_xlabel("accuracy gain over the best single model (pp, 95% paired-bootstrap CI)")
    ax.set_title("8.4 Ensemble gains - intervals that exclude 0 are real gains")
    plt.show()


# %%
# --- 8.5 Cochran's Q: are all models equivalent? -------------------------------------------------------------------
def cochrans_q(correct: np.ndarray) -> Dict[str, float]:
    """Cochran's Q omnibus test for k paired binary classifiers.

    H0: all k models have the same success probability. Q is chi-square distributed with k-1 degrees of freedom.
    It is the natural generalisation of McNemar (which is the k = 2 case) and is the correct *first* test to run
    before looking at any individual pair.
    """
    k, n = correct.shape
    col_totals = correct.sum(axis=1).astype(float)      # successes per model
    row_totals = correct.sum(axis=0).astype(float)      # models correct per image
    total = col_totals.sum()
    numerator = (k - 1) * (k * (col_totals ** 2).sum() - total ** 2)
    denominator = k * total - (row_totals ** 2).sum()
    q = float(numerator / denominator) if denominator else 0.0
    p = float(sp_stats.chi2.sf(q, df=k - 1))
    return {"k models": k, "n images": n, "Cochran Q": round(q, 2), "df": k - 1, "p-value": p}


q_result = cochrans_q(CORRECT)
display(pd.Series(q_result, name="Cochran's Q (all models)").to_frame())
print("Interpretation:", "the models are NOT all equivalent (reject H0)"
      if q_result["p-value"] < cfgx.alpha else "no evidence that the models differ")

# The same test restricted to the top-5 models: are the leaders distinguishable from each other?
top5 = accuracy_ci.head(5)["model"].tolist()
q_top5 = cochrans_q(np.stack([ALL_PREDICTIONS[n] == y_test_np for n in top5]))
display(pd.Series(q_top5, name=f"Cochran's Q (top 5: {', '.join(top5)})").to_frame())
print("Interpretation:", "even the top-5 differ significantly" if q_top5["p-value"] < cfgx.alpha
      else "the top-5 models are statistically indistinguishable from one another")


# %%
# --- 8.6 The final leaderboard -------------------------------------------------------------------------------------
def final_leaderboard(records: Sequence[Dict[str, object]], ci_table: pd.DataFrame) -> pd.DataFrame:
    """Merge the results registry with the confidence intervals into the notebook's headline table."""
    df = pd.DataFrame(list(records)).drop_duplicates(subset=["model"], keep="last")
    df = df.merge(ci_table[["model", "CI low (95%)", "CI high (95%)"]], on="model", how="left")
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df[["rank", "model", "family", "accuracy", "CI low (95%)", "CI high (95%)",
               "macro_f1", "error_rate", "fit_s", "params", "notes"]]


leaderboard = final_leaderboard(RESULTS, accuracy_ci)
display(
    leaderboard.style.hide(axis="index")
    .background_gradient(subset=["accuracy", "macro_f1"], cmap="Greens")
    .format({"accuracy": "{:.4f}", "CI low (95%)": "{:.4f}", "CI high (95%)": "{:.4f}",
             "macro_f1": "{:.4f}", "error_rate": "{:.4f}", "fit_s": "{:.1f}", "params": "{:,.0f}"}, na_rep="-")
)
leaderboard.to_csv(Path(cfg.artifacts_dir) / "final_leaderboard.csv", index=False)
print("Saved ->", Path(cfg.artifacts_dir) / "final_leaderboard.csv")

palette = {"Trivial": "#B0B0B0", "Classical ML": "#4C72B0", "Gradient Boosting": "#DD8452",
           "Deep Learning": "#C44E52", "Ensemble": "#55A868"}
d = leaderboard[leaderboard["family"] != "Trivial"].sort_values("accuracy")
plt.figure(figsize=(12, 0.4 * len(d) + 2.5))
bars = plt.barh(d["model"], d["accuracy"], color=[palette.get(f, "grey") for f in d["family"]],
                edgecolor="black", linewidth=0.5)
plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=7.5)
plt.xlim(0.78, 1.0)
plt.xlabel("test accuracy on the official 10,000-image test set")
plt.title("8.6 Final leaderboard - every model in the notebook, coloured by family")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in palette.items() if f in set(d["family"])]
labels = [f for f in palette if f in set(d["family"])]
plt.legend(handles, labels, loc="lower right", fontsize=8)
plt.show()

# %% [markdown]
# **Finding (Section 8).**
#
# 1. **Cochran's Q over all models is astronomically significant** (p ≈ 0), so the null hypothesis "all these models are
#    the same" is dead on arrival. Restricted to the **top five**, however, Q is usually *still* significant but driven
#    almost entirely by the gap between the ensembles and the single models: the top three ensembles are typically
#    indistinguishable from one another.
# 2. **After Holm correction, roughly two thirds of the pairwise differences remain significant.** The ones that do not
#    are exactly the pairs we would *expect* to be inseparable: XGBoost vs. LightGBM vs. CatBoost; the weighted-voting
#    ensemble vs. the stacking ensemble; the two convolutional networks when their accuracies land within ~0.4 pp.
#    Reporting those as "improvements" would be exactly the mistake this section exists to prevent.
# 3. **The paired bootstrap is more informative than the marginal intervals.** The Wilson intervals of the CNN and the
#    ResNet overlap heavily, yet the *paired* difference CI excludes zero: because the models agree on ~93 % of the
#    images and only the discordant ones carry information. This is the single most common statistical error in model
#    comparison, and the reason McNemar exists.
# 4. **Practical rule adopted for the whole notebook:** a difference below ~0.5 pp on this test set is reported as
#    "indistinguishable" unless the paired test says otherwise, and every headline claim in Sections 5, 9 and 10 is
#    backed by a p-value in the tables above.
