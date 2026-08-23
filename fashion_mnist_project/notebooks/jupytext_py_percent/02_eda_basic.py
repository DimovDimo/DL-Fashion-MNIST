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
# <a id="sec2"></a>
# # 2. Exploratory Data Analysis (EDA)
#
# > Exam criterion: **Visualization (0–10)**: *"Are the visualisations correct and clear?"*
#
# Before modelling we build an evidence-based picture of the data. Every plot below answers a specific question that
# later informs a modelling decision:
#
# | # | Question | Plot | Modelling consequence |
# |---|---|---|---|
# | 2.1 | Is the dataset balanced? | class-distribution bars | plain accuracy is a valid headline metric; no class weighting needed |
# | 2.2 | What do the images look like? | sample grid per class | 28x28 is tiny: a small CNN suffices; no need for ImageNet-scale nets |
# | 2.3 | How bright / sparse are the images? | pixel histogram, per-class intensity, ink coverage | justifies mean/std normalisation; shows footwear ≈ sparse, coats ≈ dense |
# | 2.4 | Where is the information located? | mean image per class, pixel-variance map | borders are almost always black -> small translations are safe augmentation |
# | 2.5 | Which classes are intrinsically confusable? | class-mean correlation heat-map, PCA scatter | predicts the Shirt/T-shirt/Pullover/Coat confusion seen later in Section 4 |

# %% [markdown]
# ## 2.1 Class distribution

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])


# %%
# --- Class distribution ---------------------------------------------------------------------------------
def plot_class_distribution(
    label_sets: Dict[str, np.ndarray], class_names: Sequence[str], figsize: Tuple[int, int] = (12, 4)
) -> pd.DataFrame:
    """Bar-plot the class distribution of several splits side by side and return the counts table."""
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


counts_df = plot_class_distribution(
    {"train": y_tr_np, "validation": y_val_np, "test": y_test_np}, cfg.class_names
)

percent_df = (counts_df / counts_df.sum() * 100).round(2)
percent_df.columns = [f"{c} (%)" for c in percent_df.columns]
display(pd.concat([counts_df, percent_df], axis=1))


# %% [markdown]
# **Finding.** Every class holds exactly 10 % of every split (6,000 / 600 / 1,000 images). Consequences:
#
# * the **majority-class baseline is 10 %**, and any model must beat that by a wide margin to be interesting;
# * **accuracy is an unbiased, interpretable headline metric**: there is no imbalance to hide behind it;
# * we still report **macro-F1** and per-class recall, because a *balanced* dataset can still produce *unbalanced errors*
#   (as we will see: `Shirt` is far harder than `Trouser`).

# %% [markdown]
# ## 2.2 What the images actually look like

# %%
# --- Sample grid: N random examples per class ------------------------------------------------------------
def plot_samples_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    n_per_class: int = 8,
    seed: int = 42,
) -> None:
    """Show a grid with one row per class and `n_per_class` random examples per row."""
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


plot_samples_per_class(X_tr_np, y_tr_np, cfg.class_names, n_per_class=8, seed=cfg.seed)


# %% [markdown]
# **Finding.** The images are silhouettes: bright garment on a black background, centred, scale-normalised and
# uniformly oriented. Texture and print details are largely destroyed by the 28x28 downsampling: which is exactly why
# `Shirt` vs `T-shirt/top` vs `Pullover` vs `Coat` is hard even for a human annotator: at this resolution the four
# classes differ mostly by sleeve length and a few contour pixels.

# %% [markdown]
# ## 2.3 Pixel-intensity statistics

# %%
# --- Global pixel-intensity distribution -----------------------------------------------------------------
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

    stats = pd.Series(
        {
            "mean": flat.mean(),
            "std": flat.std(),
            "median": float(np.median(flat)),
            "min": flat.min(),
            "max": flat.max(),
            "% exactly 0 (background)": 100.0 * (flat == 0).mean(),
            "% saturated at 255": 100.0 * (flat == 255).mean(),
            "mean of non-zero pixels": nonzero.mean(),
        },
        name="pixel statistics (raw 0-255)",
    ).round(3)
    return stats


display(pixel_intensity_overview(X_tr_np, sample_size=4_000, seed=cfg.seed).to_frame())


# %%
# --- Per-class brightness and "ink coverage" --------------------------------------------------------------
def per_class_intensity_table(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Mean intensity, std and fraction of non-background pixels for each class."""
    flat = images.reshape(len(images), -1).astype(np.float32)
    rows = []
    for c, name in enumerate(class_names):
        sub = flat[labels == c]
        rows.append(
            {
                "class": name,
                "mean intensity": sub.mean(),
                "std intensity": sub.std(),
                "ink coverage % (pixels > 20)": 100.0 * (sub > 20).mean(),
                "mean brightness of garment": sub[sub > 0].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("class").round(2)


intensity_df = per_class_intensity_table(X_tr_np, y_tr_np, cfg.class_names)
display(intensity_df.style.background_gradient(cmap="Blues", axis=0).format("{:.2f}"))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
intensity_df["mean intensity"].plot(kind="barh", ax=axes[0], color="#4C72B0", edgecolor="black", linewidth=0.4)
axes[0].set_title("Mean pixel intensity per class")
axes[0].set_xlabel("intensity [0-255]")
intensity_df["ink coverage % (pixels > 20)"].plot(
    kind="barh", ax=axes[1], color="#55A868", edgecolor="black", linewidth=0.4
)
axes[1].set_title("Ink coverage: % of pixels brighter than 20")
axes[1].set_xlabel("% of image area")
plt.show()


# %% [markdown]
# **Finding.** Roughly half of all pixels are exact background zeros, and the classes split into two intensity
# regimes: bulky garments (`Pullover`, `Coat`, `Dress`) cover 45–60 % of the frame, while footwear (`Sandal`, `Sneaker`)
# covers barely 25 %. Two practical consequences:
#
# 1. **Normalisation matters.** Raw features have mean ≈ 73 and std ≈ 90 on a 0–255 scale; feeding that to a linear model
#    or a neural net slows optimisation. We standardise with the training mean/std (Section 1.6): this alone typically
#    makes logistic regression converge several times faster.
# 2. **Area is a real signal.** A trivially simple feature (ink coverage) already separates footwear from coats, which is
#    why even Logistic Regression reaches ~84 %: a useful sanity check on the deep-learning gains.

# %% [markdown]
# ## 2.4 Where the information lives: mean images and variance map

# %%
# --- Class-mean images ("class templates") and the global pixel-variance map --------------------------------
def plot_class_means_and_variance(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> np.ndarray:
    """Plot the average image of every class plus the per-pixel std map. Returns the class-mean matrix (10, 784)."""
    flat = images.reshape(len(images), -1).astype(np.float32)
    means = np.stack([flat[labels == c].mean(axis=0) for c in range(len(class_names))])

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for c, ax in enumerate(axes.ravel()):
        ax.imshow(means[c].reshape(28, 28), cmap="viridis")
        ax.set_title(f"{c}: {class_names[c]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Mean image per class - the 'template' each classifier is implicitly matching", y=1.0)
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


class_means = plot_class_means_and_variance(X_tr_np, y_tr_np, cfg.class_names)


# %% [markdown]
# **Finding.** The corners and the outer border are (nearly) always black, and the variance map shows that almost all
# discriminative signal sits in a central ~20x20 region. Two decisions follow:
#
# * **Data augmentation by ±2-pixel translation is safe**: shifting the garment inside the black margin produces a
#   perfectly plausible image and cannot push content out of frame.
# * **Horizontal flipping is label-preserving** for all ten garment categories (a mirrored sneaker is still a sneaker),
#   unlike MNIST where flipping a digit destroys it. This is exactly why augmentation is worth much more here than on
#   MNIST.
# * A handful of pixels are constant zero across the whole training set; they contribute nothing but are harmless
#   (`StandardScaler` in the sklearn pipelines is created with default settings and handles zero-variance columns by
#   leaving them at zero).

# %% [markdown]
# ## 2.5 Which classes are intrinsically confusable?

# %%
# --- Similarity between class templates ---------------------------------------------------------------------
def plot_class_similarity(class_means: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    """Correlation heat-map between class-mean images: a cheap predictor of future confusions."""
    corr = np.corrcoef(class_means)
    corr_df = pd.DataFrame(corr, index=list(class_names), columns=list(class_names))
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": "Pearson correlation of class-mean images"}, annot_kws={"size": 7})
    plt.title("Similarity between class templates")
    plt.show()
    return corr_df


corr_df = plot_class_similarity(class_means, cfg.class_names)

# Report the most similar (and therefore most confusable) class pairs
pairs = [
    (cfg.class_names[i], cfg.class_names[j], corr_df.iat[i, j])
    for i in range(cfg.num_classes)
    for j in range(i + 1, cfg.num_classes)
]
top_pairs = pd.DataFrame(pairs, columns=["class A", "class B", "correlation"]).nlargest(8, "correlation")
display(top_pairs.reset_index(drop=True).style.hide(axis="index").format({"correlation": "{:.3f}"}))


# %%
# --- 2-D PCA projection: how linearly separable is the raw pixel space? ----------------------------------------
def plot_pca_scatter(
    images: np.ndarray, labels: np.ndarray, class_names: Sequence[str], n_samples: int = 6_000, seed: int = 42
) -> PCA:
    """Fit a 2-component PCA on raw pixels and scatter-plot the classes; also show the explained-variance curve."""
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
    axes[0].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0] * 100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1] * 100:.1f}% var)")
    axes[0].legend(markerscale=2, fontsize=7, ncol=2, loc="best")

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    axes[1].axhline(0.90, ls="--", c="red", lw=1, label="90% variance")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("number of principal components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].legend()
    plt.show()

    n90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"{n90} principal components explain 90% of the pixel variance (out of 784 raw dimensions).")
    return pca_full


pca_model = plot_pca_scatter(X_tr_np, y_tr_np, cfg.class_names, n_samples=6_000, seed=cfg.seed)

# %% [markdown]
# **Finding: and the central hypothesis of this project.** The template-correlation heat-map and the PCA scatter tell
# the same story:
#
# * **Footwear** (`Sandal`, `Sneaker`, `Ankle boot`), `Trouser` and `Bag` occupy well-separated regions of pixel space →
#   even a linear model should classify them well.
# * The **upper-body cluster** (`T-shirt/top`, `Pullover`, `Coat`, `Shirt`) collapses into one overlapping blob, with
#   template correlations above 0.9. **This cluster will dominate the error budget of every model in this notebook**:
#   we verify this explicitly with the confusion matrix in Section 4.6.
# * ~85 components already capture 90 % of the variance, i.e. the intrinsic dimensionality is far below 784. This is why
#   classical models are viable at all, and it also motivates the PCA-compressed SVM pipeline in Section 3.
