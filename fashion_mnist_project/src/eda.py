"""
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
        ax.set_title(f"PC{k+1}\n{pca.explained_variance_ratio_[k]*100:.1f}% var", fontsize=8)
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
        ax.set_title(f"c{i}: {class_names[dom]}\n({cont_norm[i, dom]*100:.0f}% pure, n={int(cont[i].sum())})", fontsize=7)
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
        axes[0, j].set_title(f"{class_names[labels[i_out]]}\n{scores[i_out]:.3f}", fontsize=6)
        axes[0, j].axis("off")
        axes[1, j].imshow(images[i_in], cmap="gray")
        axes[1, j].set_title(f"{class_names[labels[i_in]]}\n{scores[i_in]:.3f}", fontsize=6)
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
        axes[0, j].set_title(f"{class_names[labels[i_bad]]}\n{err[i_bad]:.3f}", fontsize=6)
        axes[1, j].imshow(recon[i_bad, 0], cmap="gray")
        axes[2, j].imshow(images_u8[i_good], cmap="gray")
        axes[2, j].set_title(f"{class_names[labels[i_good]]}\n{err[i_good]:.3f}", fontsize=6)
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
