"""
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
