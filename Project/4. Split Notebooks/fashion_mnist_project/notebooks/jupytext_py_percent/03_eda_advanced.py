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
# # 2.6 - 2.10  Advanced EDA (upgraded edition)
#
# > Exam criteria: **Visualization (0–10)** and **Data Gathering / Cleaning / Formatting (0–10)**.
#
# Sections 2.1–2.5 established the *first-order* facts: the dataset is balanced, the images are centred silhouettes,
# half the pixels are background, and the four upper-body classes share nearly identical templates. The upgraded EDA
# below goes after the *second-order* structure that actually drives modelling decisions:
#
# | # | Question | Method | Modelling consequence |
# |---|---|---|---|
# | 2.6 | How is intensity distributed **within** each class, not just on average? | per-class histograms, ECDFs, violin plots, moment table, KS distances | shows that intensity alone separates footwear from tops but never separates the upper-body cluster |
# | 2.7 | Which pixels carry class information, and how redundant are neighbouring pixels? | class-mean / class-variance images, Fisher discriminability map, pixel-correlation heat-maps | justifies convolution (strong local correlation ⇒ weight sharing) and PCA compression for the classical models |
# | 2.8 | What does the data manifold look like? | PCA (2D/3D), tuned t-SNE, tuned UMAP, trustworthiness + kNN probes | quantifies how much of the class structure is *linearly* accessible vs. non-linear |
# | 2.9 | Can the taxonomy be recovered **without labels**? | k-means over a grid of k, silhouette / ARI / NMI, contingency heat-map | reveals that unsupervised structure merges exactly the classes the classifiers later confuse |
# | 2.10 | Are there anomalies / probable mislabels? | Isolation Forest **and** a convolutional autoencoder, plus agreement analysis | tells us whether the residual error is noise in the data rather than a modelling failure |
#
# Every routine below is a documented, self-contained function that takes arrays and returns a table or a figure, so it
# can be re-used on any other 28x28 grayscale dataset without modification.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])

# %%
# --- Section-local imports and the shared EDA working sample ---------------------------------------------
# These imports are deliberately kept next to the section that uses them (the original Section 1.2 import
# block is left untouched so that every v1 result stays reproducible).
import inspect

from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


def make_eda_sample(
    images: np.ndarray,
    labels: np.ndarray,
    n: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a **stratified** working sample so that every heavy EDA routine sees the same data.

    Parameters
    ----------
    images : np.ndarray
        `(N, 28, 28)` uint8 images.
    labels : np.ndarray
        `(N,)` integer labels.
    n : int
        Total sample size (split evenly across the classes).
    seed : int
        RNG seed.

    Returns
    -------
    (images_sample, labels_sample, flat_sample)
        The last element is `(n, 784)` float32 scaled to `[0, 1]` - the format every sklearn routine wants.
    """
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


X_eda, y_eda, F_eda = make_eda_sample(X_tr_np, y_tr_np, cfgx.eda_sample, seed=cfg.seed)
print(f"EDA working sample: {X_eda.shape} images, flat matrix {F_eda.shape} "
      f"({F_eda.nbytes / 1e6:.1f} MB), class counts {np.bincount(y_eda, minlength=cfg.num_classes).tolist()}")


# %% [markdown]
# ## 2.6 Pixel-intensity distributions, global and per class
#
# Section 2.3 reported *means*. A mean hides the shape of the distribution, and the shape is what tells us whether a
# threshold-style feature can work at all. Fashion-MNIST intensities are strongly **bimodal**: a huge spike at 0
# (background) and a broad garment mode between roughly 60 and 220. We therefore analyse the two regimes separately:
# the background spike would otherwise dominate every statistic.

# %%
# --- 2.6.1 Global intensity distribution: raw, foreground-only, and on the log-count scale ------------------
def plot_global_intensity(flat01: np.ndarray, bins: int = 64) -> pd.Series:
    """Four complementary views of the global pixel-intensity distribution.

    Returns a Series of distribution moments (mean/std/skew/kurtosis, background and saturation share).
    """
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
    fig.suptitle("2.6.1 Global pixel-intensity distribution", y=1.04)
    plt.show()

    return pd.Series(
        {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "skewness": float(sp_stats.skew(vals)),
            "excess kurtosis": float(sp_stats.kurtosis(vals)),
            "% background (== 0)": float(100.0 * (vals == 0).mean()),
            "% saturated (== 255)": float(100.0 * (vals == 255).mean()),
            "foreground mean": float(fg.mean()),
            "foreground std": float(fg.std()),
            "foreground median": float(np.median(fg)),
        },
        name="global intensity statistics",
    ).round(3)


display(plot_global_intensity(F_eda).to_frame())


# %%
# --- 2.6.2 Per-class intensity distributions -----------------------------------------------------------------
def plot_per_class_intensity(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str], bins: int = 48
) -> pd.DataFrame:
    """Per-class histograms, ECDFs and a violin plot of *foreground* intensity + a moment table.

    Foreground-only statistics are used because the background spike at 0 is identical for every class and would
    swamp any real difference.
    """
    n_classes = len(class_names)
    palette = sns.color_palette("tab10", n_classes)

    # (a) small-multiples: one histogram per class, foreground pixels only
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.4), sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        v = (flat01[labels == c] * 255.0).reshape(-1)
        ax.hist(v[v > 0], bins=bins, color=palette[c], density=True, edgecolor="black", linewidth=0.2)
        ax.set_title(f"{c}: {class_names[c]}", fontsize=9)
        ax.set_xlabel("intensity")
    fig.suptitle("2.6.2a Foreground-intensity density per class", y=1.02)
    plt.show()

    # (b) overlaid ECDFs - the cleanest way to see stochastic dominance between classes
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

    # (c) violin of the per-IMAGE mean intensity (one point per image, not per pixel)
    per_image_df = pd.DataFrame(
        {"class": [class_names[c] for c in labels], "mean intensity": flat01.mean(axis=1) * 255.0}
    )
    try:                       # seaborn >= 0.13 wants an explicit hue to accept a palette
        sns.violinplot(
            data=per_image_df, x="class", y="mean intensity", hue="class", ax=axes[1],
            palette=palette, inner="quartile", cut=0, order=list(class_names), legend=False,
        )
    except TypeError:          # seaborn < 0.13 has no `legend` argument
        sns.violinplot(
            data=per_image_df, x="class", y="mean intensity", ax=axes[1],
            palette=palette, inner="quartile", cut=0, order=list(class_names),
        )
    axes[1].set_title("Distribution of the per-image mean intensity")
    axes[1].set_ylabel("mean intensity of the image")
    axes[1].tick_params(axis="x", rotation=40)
    plt.show()

    rows = []
    for c in range(n_classes):
        allv = (flat01[labels == c] * 255.0).reshape(-1)
        fg = allv[allv > 0]
        rows.append(
            {
                "class": class_names[c],
                "image-mean": float(flat01[labels == c].mean() * 255),
                "fg mean": float(fg.mean()),
                "fg std": float(fg.std()),
                "fg skew": float(sp_stats.skew(fg)),
                "% background": float(100.0 * (allv == 0).mean()),
                "ink coverage % (>20)": float(100.0 * (allv > 20).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("class").round(2)


intensity_v2_df = plot_per_class_intensity(F_eda, y_eda, cfg.class_names)
display(intensity_v2_df.style.background_gradient(cmap="Blues", axis=0).format("{:.2f}"))


# %%
# --- 2.6.3 How different are two classes' intensity distributions? Kolmogorov-Smirnov distances --------------
def intensity_ks_matrix(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str], max_pixels: int = 20_000, seed: int = 42
) -> pd.DataFrame:
    """Pairwise two-sample KS statistic between the per-image mean-intensity distributions of the classes.

    The KS statistic is the maximum vertical gap between two ECDFs: 0 = identical, 1 = disjoint. It gives a
    *distribution-level* (not mean-level) measure of how far apart two classes are on this single feature.
    """
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
                annot_kws={"size": 7}, cbar_kws={"label": "KS distance (0 = identical distributions)"})
    plt.title("2.6.3 KS distance between per-image mean-intensity distributions")
    plt.show()
    return df.round(3)


ks_df = intensity_ks_matrix(F_eda, y_eda, cfg.class_names, seed=cfg.seed)
closest = [
    (cfg.class_names[i], cfg.class_names[j], ks_df.iat[i, j])
    for i in range(cfg.num_classes) for j in range(i + 1, cfg.num_classes)
]
print("Class pairs that brightness alone cannot separate (smallest KS distance):")
display(
    pd.DataFrame(closest, columns=["class A", "class B", "KS distance"])
    .nsmallest(6, "KS distance").reset_index(drop=True).style.hide(axis="index").format({"KS distance": "{:.3f}"})
)


# %% [markdown]
# **Finding (2.6).** Three facts that shape everything downstream:
#
# 1. The global distribution is **extremely non-Gaussian**: a ~50 % point mass at zero plus a left-skewed garment mode.
#    Standardising with a single mean/std (Section 1.6) is still the right thing to do for optimisation, but it does
#    *not* Gaussianise the input; this is one reason tree-based models (which are scale-free) remain competitive with
#    linear models here.
# 2. **Brightness is a genuine but weak feature.** Footwear vs. coats is nearly separable on the per-image mean alone
#    (KS ≈ 0.8–0.9), which is why even a linear model reaches ~0.84.
# 3. **Brightness is useless exactly where it matters.** `Shirt` vs. `T-shirt/top` vs. `Pullover` vs. `Coat` have KS
#    distances of only ~0.1–0.3 on this feature: their intensity distributions almost coincide. Any model that hopes to
#    separate them must use **shape**, not brightness: the concrete, measurable justification for a convolutional (or
#    attention-based) architecture.

# %% [markdown]
# ## 2.7 Image metrics: class means, class variances, discriminability and pixel correlation
#
# Section 2.4 showed the class means. Here we add the three things that a mean image cannot show:
#
# * the **per-class variance image**: where within a class the images disagree (sleeves, hems, heels);
# * the **Fisher discriminability map**: the ratio of between-class to within-class variance for every pixel, i.e. an
#   analytic, model-free answer to *"which pixels can possibly carry class information?"*;
# * the **pixel-correlation structure**: how redundant neighbouring pixels are. Strong short-range correlation is the
#   mathematical justification for weight sharing (convolution) and for PCA compression.

# %%
# --- 2.7.1 Class means, class standard deviations and deviation-from-global maps ----------------------------
def class_mean_variance_panels(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot, for each class, the mean image, the per-pixel std image and the (mean - global mean) map.

    Returns
    -------
    (means, stds) : each of shape (n_classes, 784)
    """
    n_classes = len(class_names)
    means = np.stack([flat01[labels == c].mean(axis=0) for c in range(n_classes)])
    stds = np.stack([flat01[labels == c].std(axis=0) for c in range(n_classes)])
    global_mean = flat01.mean(axis=0)

    for title, mats, cmap, kw in [
        ("Class MEAN images", means, "viridis", {}),
        ("Class per-pixel STD images (within-class variability)", stds, "magma", {}),
        ("Class mean MINUS global mean (what makes this class special)",
         means - global_mean, "coolwarm", {"vmin": -0.45, "vmax": 0.45}),
    ]:
        fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
        for c, ax in enumerate(axes.ravel()):
            im = ax.imshow(mats[c].reshape(28, 28), cmap=cmap, **kw)
            ax.set_title(f"{c}: {class_names[c]}", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        fig.suptitle(f"2.7.1 {title}", y=1.01)
        plt.show()
    return means, stds


class_means_v2, class_stds_v2 = class_mean_variance_panels(F_eda, y_eda, cfg.class_names)


# %%
# --- 2.7.2 Fisher discriminability map: between-class variance / within-class variance ----------------------
def fisher_discriminability(
    flat01: np.ndarray, labels: np.ndarray, n_classes: int = 10, eps: float = 1e-8
) -> np.ndarray:
    """Per-pixel Fisher ratio  Var_between / Var_within  (a model-free 'usefulness' score for every pixel).

    High values mark pixels whose value depends strongly on the class relative to how much it fluctuates
    inside a class - exactly the pixels any classifier should be using.
    """
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


fisher_map = fisher_discriminability(F_eda, y_eda, cfg.num_classes)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
im0 = axes[0].imshow(fisher_map.reshape(28, 28), cmap="inferno")
axes[0].set_title("Fisher ratio per pixel (between/within)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

top_mask = fisher_map >= np.quantile(fisher_map, 0.90)
axes[1].imshow(F_eda.mean(axis=0).reshape(28, 28), cmap="gray")
axes[1].contour(top_mask.reshape(28, 28), levels=[0.5], colors="cyan", linewidths=1.2)
axes[1].set_title("Top-10% most discriminative pixels\n(over the global mean image)")
axes[1].axis("off")

axes[2].hist(fisher_map, bins=50, color="#C44E52", edgecolor="black", linewidth=0.3)
axes[2].set_yscale("log")
axes[2].set_title("Distribution of the Fisher ratio")
axes[2].set_xlabel("Fisher ratio")
axes[2].set_ylabel("pixel count (log)")
plt.show()

print(f"Pixels with Fisher ratio > 0.5 : {(fisher_map > 0.5).sum():3d} / 784")
print(f"Pixels with Fisher ratio < 0.01: {(fisher_map < 0.01).sum():3d} / 784  (effectively uninformative)")


# %%
# --- 2.7.3 Pixel-correlation structure -----------------------------------------------------------------------
def pixel_correlation_analysis(
    flat01: np.ndarray, grid: int = 14, max_rows: int = 4_000, seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Correlation structure of the pixel features.

    Two views:
      1. the full 784x784 correlation matrix (rendered without annotations - it is a *texture*, not a table);
      2. the correlation between `grid` x `grid` block-averaged pixels, which is legible and shows the same structure;
      3. correlation as a function of spatial distance between pixels - the quantity that justifies convolution.

    Returns the full correlation matrix and the distance-decay table.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(flat01), size=min(max_rows, len(flat01)), replace=False)
    X = flat01[rows]
    X = X + rng.normal(0, 1e-6, X.shape).astype(np.float32)   # break exact-zero columns so corrcoef is defined
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)

    # block-average the IMAGES onto a coarse grid, then correlate -> a legible (grid^2 x grid^2) matrix
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

    # correlation of every pixel with the centre pixel, shown back on the image grid
    centre = 14 * 28 + 14
    im1 = axes[1].imshow(corr[centre].reshape(28, 28), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Correlation of every pixel with the centre pixel (14,14)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # correlation vs. euclidean distance between pixel positions
    yy, xx = np.mgrid[0:28, 0:28]
    pos = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(784, k=1)
    dist_bins = np.round(d[iu]).astype(int)
    corr_vals = corr[iu]
    decay = (
        pd.DataFrame({"distance (px)": dist_bins, "correlation": corr_vals})
        .groupby("distance (px)")["correlation"].agg(["mean", "std", "count"])
        .reset_index()
    )
    axes[2].plot(decay["distance (px)"], decay["mean"], marker="o", ms=3, color="#4C72B0")
    axes[2].fill_between(decay["distance (px)"], decay["mean"] - decay["std"],
                         decay["mean"] + decay["std"], alpha=0.2, color="#4C72B0")
    axes[2].axhline(0, c="grey", lw=1)
    axes[2].set_title("Mean pixel correlation vs. spatial distance")
    axes[2].set_xlabel("euclidean distance between pixels (px)")
    axes[2].set_ylabel("Pearson correlation")
    fig.suptitle("2.7.3 Pixel-correlation structure", y=1.03)
    plt.show()

    plt.figure(figsize=(8.5, 7))
    sns.heatmap(pd.DataFrame(coarse).round(2), cmap="RdBu_r", vmin=-1, vmax=1, annot=False,
                cbar_kws={"label": "block-averaged correlation"})
    plt.title(f"2.7.3b Correlation between {grid}x{grid} block-averaged pixels "
              f"({grid * grid} features - same structure, legible)")
    plt.xlabel("coarse pixel block")
    plt.ylabel("coarse pixel block")
    plt.show()
    return corr, decay.round(3)


pixel_corr, corr_decay = pixel_correlation_analysis(F_eda, grid=cfgx.corr_grid, seed=cfg.seed)
print("Average correlation between directly adjacent pixels (distance 1):",
      float(corr_decay.loc[corr_decay["distance (px)"] == 1, "mean"].iloc[0]))
print("Average correlation at distance 10 px                          :",
      float(corr_decay.loc[corr_decay["distance (px)"] == 10, "mean"].iloc[0]))


# %% [markdown]
# **Finding (2.7).**
#
# * The **within-class std images** peak exactly where garments differ *within* a category (sleeve ends, hemlines and
#   shoe heels) which is why simple template matching (nearest class mean) tops out in the low 70 % range.
# * The **Fisher map** shows that only ~250 of 784 pixels carry a meaningful between/within variance ratio; the border
#   ring is analytically useless. This is the model-free version of the Random-Forest importance map in Section 3.5, and
#   the two agree: a nice cross-validation of both.
# * **Adjacent pixels correlate at ≈ 0.9 and the correlation decays smoothly with distance.** That is precisely the
#   statistical property convolution is built to exploit: if neighbouring inputs are near-duplicates, a shared local
#   filter is a far better-conditioned estimator than 784 independent weights. It is also why ~85 principal components
#   retain 90 % of the variance and why the PCA-compressed SVM in Section 3.4 loses nothing.

# %% [markdown]
# ## 2.8 Dimensionality reduction: PCA, t-SNE and UMAP (2D and 3D, with tuning)
#
# A 2D scatter plot is *not* evidence by itself: t-SNE and UMAP can manufacture clusters that are artefacts of their
# hyper-parameters. We therefore do three things that a casual EDA usually skips:
#
# 1. **Tune** the key hyper-parameter of each method (t-SNE `perplexity`, UMAP `n_neighbors`) and show the results
#    side by side, so the reader can see which structures are stable across settings and which are not.
# 2. Pre-reduce with **PCA-50** before t-SNE/UMAP. This is standard practice: it removes pixel noise, makes the
#    neighbour search far cheaper, and is what the original t-SNE paper recommends.
# 3. **Quantify** each embedding with two numbers instead of eyeballing it:
#    * `trustworthiness` ∈ [0, 1]: how well local neighbourhoods of the 50-D space survive the projection;
#    * **kNN accuracy inside the embedding**: how much class information is still linearly/locally accessible after
#      projecting to 2 dimensions.

# %%
# --- 2.8.1 PCA in 2D and 3D, with the explained-variance spectrum -------------------------------------------
def pca_embedding_analysis(
    flat01: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    n_components: int = 50, seed: int = 42, show_3d: bool = True,
) -> Tuple[PCA, np.ndarray]:
    """Fit a PCA, plot the 2D and 3D projections, the spectrum, and the leading eigen-garments.

    Returns the fitted PCA and the `(n, n_components)` embedding, which is re-used by t-SNE, UMAP,
    clustering and the Isolation Forest so that every method sees the identical feature space.
    """
    pca = PCA(n_components=n_components, random_state=seed).fit(flat01)
    Z = pca.transform(flat01)
    palette = sns.color_palette("tab10", len(class_names))

    fig = plt.figure(figsize=(16, 4.6))
    ax0 = fig.add_subplot(1, 3, 1)
    for c in range(len(class_names)):
        m = labels == c
        ax0.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.55, color=palette[c], label=class_names[c])
    ax0.set_title("PCA - components 1 & 2")
    ax0.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax0.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax0.legend(fontsize=6, ncol=2, markerscale=2)

    ax1 = fig.add_subplot(1, 3, 2)
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
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
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
    else:
        ax2.axis("off")
    fig.suptitle("2.8.1 Principal-component structure of raw pixel space", y=1.03)
    plt.show()

    # the leading principal components rendered as images ("eigen-garments")
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.8))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(pca.components_[k].reshape(28, 28), cmap="RdBu_r")
        ax.set_title(f"PC{k + 1}\n{pca.explained_variance_ratio_[k] * 100:.1f}% var", fontsize=8)
        ax.axis("off")
    fig.suptitle("2.8.1b The first 12 principal components as images ('eigen-garments')", y=1.02)
    plt.show()
    return pca, Z


pca50, Z_pca50 = pca_embedding_analysis(
    F_eda, y_eda, cfg.class_names, n_components=cfgx.pca_pre_components,
    seed=cfg.seed, show_3d=cfgx.run_3d_embeddings,
)


# %%
# --- 2.8.2 Quantitative quality of an embedding --------------------------------------------------------------
def embedding_quality(
    X_high: np.ndarray, X_low: np.ndarray, labels: np.ndarray, k: int = 10, seed: int = 42
) -> Dict[str, float]:
    """Score a low-dimensional embedding with two complementary numbers.

    * `trustworthiness` - fraction of the k nearest neighbours in the high-dimensional space that are still
      neighbours after projection (1.0 = perfect local structure preservation).
    * `knn_accuracy` - 5-fold-style hold-out accuracy of a k-NN classifier fitted *inside* the embedding; a direct
      measure of how much class information the 2D picture actually contains.
    """
    n = len(X_low)
    tw = float(trustworthiness(X_high, X_low, n_neighbors=k))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_low[perm[:cut]], labels[perm[:cut]])
    acc = float(knn.score(X_low[perm[cut:]], labels[perm[cut:]]))
    return {"trustworthiness": round(tw, 4), "knn_accuracy_in_embedding": round(acc, 4)}


EMBEDDINGS: Dict[str, np.ndarray] = {}          # name -> (n, 2) embedding, filled in by the cells below
EMBED_SCORES: List[Dict[str, object]] = []      # name -> quality metrics


def register_embedding(name: str, Z2: np.ndarray, X_high: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Store an embedding and its quality metrics in the two global registries."""
    EMBEDDINGS[name] = Z2
    rec = {"embedding": name, **embedding_quality(X_high, Z2, labels)}
    EMBED_SCORES.append(rec)
    print(f"{name:<28s} trustworthiness={rec['trustworthiness']:.4f}  "
          f"kNN-acc in 2D={rec['knn_accuracy_in_embedding']:.4f}")
    return rec


# Subsample once: t-SNE and UMAP are the expensive parts of this section.
_rng = np.random.default_rng(cfg.seed)
_sub = _rng.choice(len(Z_pca50), size=min(cfgx.embed_sample, len(Z_pca50)), replace=False)
Z_high = Z_pca50[_sub]              # (m, 50) PCA features - the shared input space
y_embed = y_eda[_sub]
X_embed_imgs = X_eda[_sub]
print(f"manifold-learning subsample: {Z_high.shape}")

_ = register_embedding("PCA (2D)", Z_high[:, :2], Z_high, y_embed)


# %%
# --- 2.8.3 t-SNE with a perplexity sweep ----------------------------------------------------------------------
def make_tsne(n_components: int, perplexity: float, n_iter: int, seed: int) -> TSNE:
    """Construct a TSNE object in a way that works across scikit-learn versions.

    scikit-learn renamed `n_iter` to `max_iter` in 1.5 (the old name was removed in 1.7), so the argument name is
    resolved by inspecting the constructor signature instead of pinning a version.
    """
    params = inspect.signature(TSNE.__init__).parameters
    iter_kw = "max_iter" if "max_iter" in params else "n_iter"
    kwargs = {
        "n_components": n_components,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
        iter_kw: n_iter,
    }
    return TSNE(**kwargs)


def tsne_perplexity_sweep(
    Z_high: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    perplexities: Sequence[int], n_iter: int, seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Run t-SNE for several perplexities and plot them side by side.

    Perplexity is roughly 'how many neighbours each point tries to keep'. Small values expose fine local structure
    (and hallucinate small clusters); large values emphasise global layout. Structures that survive the whole sweep
    are the ones worth interpreting.
    """
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
        ax.set_title(f"t-SNE, perplexity={p}  ({time.time() - t0:.0f}s)")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][-1].legend(fontsize=6, ncol=2, markerscale=2, loc="best")
    fig.suptitle("2.8.3 t-SNE perplexity sweep (PCA-50 input)", y=1.02)
    plt.show()
    return out


tsne_runs = tsne_perplexity_sweep(
    Z_high, y_embed, cfg.class_names, cfgx.tsne_perplexities, cfgx.tsne_iter, seed=cfg.seed
)
best_perp = max(
    tsne_runs, key=lambda p: embedding_quality(Z_high, tsne_runs[p], y_embed)["knn_accuracy_in_embedding"]
)
print(f"\nBest perplexity by kNN accuracy inside the embedding: {best_perp}")
_ = register_embedding(f"t-SNE (perp={best_perp})", tsne_runs[best_perp], Z_high, y_embed)


# %%
# --- 2.8.4 3D t-SNE (optional, controlled by CFGX.run_3d_embeddings) -----------------------------------------
def plot_embedding_3d(Z3: np.ndarray, labels: np.ndarray, class_names: Sequence[str], title: str) -> None:
    """Render a 3-component embedding from two viewing angles (a single angle is easy to over-read)."""
    palette = sns.color_palette("tab10", len(class_names))
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim) in enumerate([(20, 45), (20, 135)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for c in range(len(class_names)):
            m = labels == c
            ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=4, alpha=0.55, color=palette[c], label=class_names[c])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title}  (elev={elev}, azim={azim})", fontsize=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if k == 1:
            ax.legend(fontsize=6, ncol=2, markerscale=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.show()


if cfgx.run_3d_embeddings:
    t0 = time.time()
    Z3_tsne = make_tsne(3, float(best_perp), cfgx.tsne_iter, cfg.seed).fit_transform(Z_high)
    print(f"3D t-SNE finished in {time.time() - t0:.0f}s")
    plot_embedding_3d(Z3_tsne, y_embed, cfg.class_names, f"3D t-SNE (perplexity={best_perp})")
else:
    Z3_tsne = None
    print("3D embeddings disabled (CFGX.run_3d_embeddings = False).")


# %%
# --- 2.8.5 UMAP with an n_neighbors sweep (2D) and an optional 3D view ---------------------------------------
def umap_sweep(
    Z_high: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    neighbor_grid: Sequence[int], min_dist: float, seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Run UMAP for several `n_neighbors` values. Small values -> local detail, large values -> global topology."""
    import umap  # imported lazily: the section is skipped entirely when the package is absent

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
        ax.set_title(f"UMAP, n_neighbors={k}, min_dist={min_dist}  ({time.time() - t0:.0f}s)")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][-1].legend(fontsize=6, ncol=2, markerscale=2, loc="best")
    fig.suptitle("2.8.5 UMAP neighbourhood-size sweep (PCA-50 input)", y=1.02)
    plt.show()
    return out


if HAS_UMAP:
    umap_runs = umap_sweep(Z_high, y_embed, cfg.class_names, cfgx.umap_neighbors, cfgx.umap_min_dist, cfg.seed)
    best_nn = max(
        umap_runs, key=lambda k: embedding_quality(Z_high, umap_runs[k], y_embed)["knn_accuracy_in_embedding"]
    )
    print(f"\nBest n_neighbors by kNN accuracy inside the embedding: {best_nn}")
    _ = register_embedding(f"UMAP (n_neighbors={best_nn})", umap_runs[best_nn], Z_high, y_embed)

    if cfgx.run_3d_embeddings:
        import umap as _umap
        Z3_umap = _umap.UMAP(
            n_components=3, n_neighbors=int(best_nn), min_dist=cfgx.umap_min_dist, random_state=cfg.seed
        ).fit_transform(Z_high)
        plot_embedding_3d(Z3_umap, y_embed, cfg.class_names, f"3D UMAP (n_neighbors={best_nn})")
    else:
        Z3_umap = None
else:
    umap_runs, best_nn, Z3_umap = {}, None, None
    print("umap-learn is not installed -> UMAP skipped. PCA and t-SNE above already answer RQ5;\n"
          "install with `pip install umap-learn` to enable this cell.")

# %%
# --- 2.8.6 Which projection preserves the most structure? ------------------------------------------------------
embed_score_df = pd.DataFrame(EMBED_SCORES).sort_values("knn_accuracy_in_embedding", ascending=False)
display(
    embed_score_df.style.hide(axis="index")
    .background_gradient(subset=["trustworthiness", "knn_accuracy_in_embedding"], cmap="Greens")
    .format({"trustworthiness": "{:.4f}", "knn_accuracy_in_embedding": "{:.4f}"})
)

ax = embed_score_df.set_index("embedding")[["trustworthiness", "knn_accuracy_in_embedding"]].plot(
    kind="barh", figsize=(10, 3.4), edgecolor="black", linewidth=0.4
)
ax.set_xlim(0, 1.05)
ax.set_title("2.8.6 Embedding quality: local-structure preservation vs. class information retained in 2D")
ax.set_xlabel("score")
plt.show()


# %% [markdown]
# **Finding (2.8): answers the first half of RQ5.**
#
# * **PCA-2D** keeps only ~45–50 % kNN accuracy: two linear directions are simply not enough, though the footwear /
#   clothing split is already visible along PC1 (which is essentially "ink coverage", exactly the feature Section 2.6
#   isolated).
# * **t-SNE and UMAP recover 5–7 well-separated islands, not 10.** `Trouser`, `Bag` and each footwear type form their
#   own island, but `T-shirt/top`, `Pullover`, `Coat` and `Shirt` merge into a single continent with no internal
#   boundary: at every perplexity and every neighbourhood size we tried. **The structure is a property of the data,
#   not of the hyper-parameters**, which is the point of running the sweep.
# * kNN accuracy inside the 2D UMAP/t-SNE embedding reaches ~0.75–0.85, versus ~0.93 for the CNN on the raw images:
#   a 2-dimensional summary of Fashion-MNIST loses roughly 10 accuracy points: useful to know before anyone proposes
#   "just cluster the embeddings" as a production solution.

# %% [markdown]
# ## 2.9 Unsupervised clustering: can the taxonomy be recovered without labels?
#
# If the ten official categories were natural clusters in pixel space, k-means with k = 10 would recover them and the
# Adjusted Rand Index (ARI) would be high. Testing this is the cleanest possible check of the central hypothesis from
# Section 2.5, because it uses **no label information at all** during fitting.
#
# Metrics used:
#
# * **Silhouette** (label-free): cluster compactness vs. separation, used to choose k honestly.
# * **ARI** and **NMI** (label-aware, evaluation only): agreement between the discovered partition and the ground truth,
#   corrected for chance.

# %%
# --- 2.9.1 k-means over a grid of k --------------------------------------------------------------------------
def kmeans_grid(
    Z: np.ndarray, labels: np.ndarray, k_grid: Sequence[int], seed: int = 42
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """Fit k-means for several k and score each partition with silhouette (label-free), ARI and NMI (label-aware).

    Returns the score table and the cluster assignments for each k.
    """
    rows, assignments = [], {}
    for k in k_grid:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(Z)
        lab = km.labels_
        assignments[int(k)] = lab
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(Z, lab, sample_size=min(3_000, len(Z)), random_state=seed)),
                "ARI vs. true labels": float(adjusted_rand_score(labels, lab)),
                "NMI vs. true labels": float(normalized_mutual_info_score(labels, lab)),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(df["k"], df["inertia"], marker="o")
    axes[0].set_title("Elbow curve (inertia)")
    axes[0].set_xlabel("k")
    axes[1].plot(df["k"], df["silhouette"], marker="o", color="#DD8452")
    axes[1].set_title("Silhouette (no labels used)")
    axes[1].set_xlabel("k")
    axes[2].plot(df["k"], df["ARI vs. true labels"], marker="o", label="ARI")
    axes[2].plot(df["k"], df["NMI vs. true labels"], marker="s", label="NMI")
    axes[2].axvline(10, ls="--", c="grey", lw=1)
    axes[2].text(10.1, axes[2].get_ylim()[0], "true k = 10", fontsize=8, color="grey")
    axes[2].set_title("Agreement with the official taxonomy")
    axes[2].set_xlabel("k")
    axes[2].legend()
    fig.suptitle("2.9.1 k-means over a grid of cluster counts (PCA-50 features)", y=1.04)
    plt.show()
    return df.round(4), assignments


_rng = np.random.default_rng(cfg.seed)
_cs = _rng.choice(len(Z_pca50), size=min(cfgx.cluster_sample, len(Z_pca50)), replace=False)
Z_cluster, y_cluster, X_cluster_imgs = Z_pca50[_cs], y_eda[_cs], X_eda[_cs]

cluster_df, cluster_assign = kmeans_grid(Z_cluster, y_cluster, cfgx.cluster_k_grid, seed=cfg.seed)
display(cluster_df.style.hide(axis="index").background_gradient(
    subset=["silhouette", "ARI vs. true labels", "NMI vs. true labels"], cmap="Greens"))


# %%
# --- 2.9.2 What do the k=10 clusters actually contain? ---------------------------------------------------------
def cluster_composition(
    cluster_labels: np.ndarray, true_labels: np.ndarray, images: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Contingency heat-map (cluster x true class) plus the mean image of every discovered cluster."""
    k = int(cluster_labels.max()) + 1
    cont = np.zeros((k, len(class_names)))
    for i in range(k):
        cont[i] = np.bincount(true_labels[cluster_labels == i], minlength=len(class_names))
    cont_norm = cont / np.maximum(cont.sum(axis=1, keepdims=True), 1)
    df = pd.DataFrame(cont_norm, columns=list(class_names), index=[f"cluster {i}" for i in range(k)])

    plt.figure(figsize=(10, 5.6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 7},
                cbar_kws={"label": "share of the cluster"})
    plt.title("2.9.2 Composition of each k-means cluster (rows sum to 1)")
    plt.xlabel("true class")
    plt.show()

    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(1.5 * k, 4.4))
    for i, ax in enumerate(np.array(axes).ravel()[:k]):
        ax.imshow(images[cluster_labels == i].mean(axis=0), cmap="gray")
        dom = int(np.argmax(cont[i]))
        ax.set_title(f"c{i}: {class_names[dom]}\n({cont_norm[i, dom] * 100:.0f}% pure, n={int(cont[i].sum())})",
                     fontsize=7)
        ax.axis("off")
    for ax in np.array(axes).ravel()[k:]:
        ax.axis("off")
    fig.suptitle("2.9.2b Mean image of every discovered cluster", y=1.03)
    plt.show()
    return df.round(3)


if 10 in cluster_assign:
    comp_df = cluster_composition(cluster_assign[10], y_cluster, X_cluster_imgs, cfg.class_names)
    purity = float(np.mean(comp_df.values.max(axis=1)))
    print(f"Mean cluster purity at k=10: {purity:.3f}  "
          f"(1.0 would mean every cluster contains exactly one class)")


# %% [markdown]
# **Finding (2.9): answers the second half of RQ5.** k-means on PCA-50 features peaks at an **ARI of only ≈ 0.35–0.42**,
# and importantly the silhouette score does *not* have a maximum at k = 10. The discovered clusters split the
# *easy* classes further (two sneaker clusters: high-top vs. low-top) while **merging the entire upper-body group into
# one or two clusters**. The unsupervised geometry of pixel space therefore encodes "silhouette shape", not "garment
# category": the label taxonomy is a semantic overlay that only supervision can recover. This is the strongest possible
# form of the Section 2.5 prediction, and it is confirmed later by every confusion matrix in the notebook.

# %% [markdown]
# ## 2.10 Outlier / anomaly detection: Isolation Forest and a convolutional autoencoder
#
# "Fashion-MNIST is clean" is an assumption, and Section 1.5 only verified *structural* integrity (shapes, ranges,
# duplicates, leakage). Here we look for **semantic** anomalies: images that are unusual for their class, and therefore
# candidates for label noise or genuinely rare products. Two independent detectors are used precisely so that their
# **agreement** can be measured:
#
# | Detector | Notion of "anomalous" | Blind spot |
# |---|---|---|
# | **Isolation Forest** on PCA-50 features | few random axis-aligned splits are enough to isolate the point | linear feature space; ignores spatial structure |
# | **Convolutional autoencoder** reconstruction error | the network cannot compress and rebuild the image | can flag merely *high-frequency* images rather than semantically odd ones |
#
# If two methods with different blind spots flag the same images, those images are very likely to be real problems.

# %%
# --- 2.10.1 Isolation Forest on the PCA embedding ---------------------------------------------------------------
def isolation_forest_outliers(
    Z: np.ndarray, labels: np.ndarray, images: np.ndarray, class_names: Sequence[str],
    contamination: float = 0.01, n_estimators: int = 300, seed: int = 42, n_show: int = 10,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit an Isolation Forest and visualise the most anomalous images plus the per-class anomaly rate.

    Returns
    -------
    (scores, per_class_table)
        `scores` is the anomaly score (lower = more anomalous, sklearn convention).
    """
    iso = IsolationForest(
        n_estimators=n_estimators, contamination=contamination, random_state=seed, n_jobs=-1
    ).fit(Z)
    scores = iso.score_samples(Z)
    flag = iso.predict(Z) == -1

    order = np.argsort(scores)          # most anomalous first
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
    fig.suptitle("2.10.1 Isolation Forest - top row: most anomalous, bottom row: most typical", y=1.06)
    plt.show()

    per_class = pd.DataFrame(
        {
            "class": list(class_names),
            "anomaly rate %": [100.0 * flag[labels == c].mean() for c in range(len(class_names))],
            "mean anomaly score": [float(scores[labels == c].mean()) for c in range(len(class_names))],
        }
    ).set_index("class").round(3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    per_class["anomaly rate %"].plot(kind="barh", ax=axes[0], color="#C44E52", edgecolor="black", linewidth=0.4)
    axes[0].axvline(100 * contamination, ls="--", c="grey", lw=1)
    axes[0].set_title(f"Share of images flagged per class (global target = {100 * contamination:.1f}%)")
    axes[1].hist(scores, bins=60, color="#4C72B0", edgecolor="black", linewidth=0.3)
    axes[1].axvline(np.quantile(scores, contamination), ls="--", c="red", lw=1.2, label="decision threshold")
    axes[1].set_title("Distribution of anomaly scores")
    axes[1].set_xlabel("score (lower = more anomalous)")
    axes[1].legend()
    plt.show()
    return scores, per_class


iso_scores, iso_per_class = isolation_forest_outliers(
    Z_pca50, y_eda, X_eda, cfg.class_names,
    contamination=cfgx.iforest_contamination, n_estimators=cfgx.iforest_estimators, seed=cfg.seed,
)
display(iso_per_class.style.background_gradient(cmap="Reds", subset=["anomaly rate %"]).format("{:.3f}"))


# %%
# --- 2.10.2 A small convolutional autoencoder as a second, independent detector -------------------------------
class ConvAutoencoder(nn.Module):
    """Compact conv autoencoder for 28x28x1 images: 784 -> `latent` -> 784.

    Encoder : Conv(1->16,s2) -> Conv(16->32,s2) -> Flatten -> Linear(32*7*7 -> latent)
    Decoder : Linear(latent -> 32*7*7) -> ConvT(32->16,s2) -> ConvT(16->1,s2) -> identity output

    The bottleneck forces the network to learn the *typical* garment manifold; images it cannot rebuild are, by
    construction, images unlike the training distribution.
    """

    def __init__(self, latent: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 28 -> 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent),
        )
        self.decoder_fc = nn.Sequential(nn.Linear(latent, 32 * 7 * 7), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),  # 7 -> 14
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),                                             # 14 -> 28
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        h = self.decoder_fc(z).view(-1, 32, 7, 7)
        return self.decoder(h)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent code (used below to visualise the learned manifold)."""
        return self.encoder(x)


def train_autoencoder(
    x_train: torch.Tensor, epochs: int, batch_size: int, latent: int,
    lr: float = 2e-3, device: torch.device = DEVICE, seed: int = 42,
) -> Tuple[ConvAutoencoder, List[float]]:
    """Train the autoencoder with MSE reconstruction loss. Returns the model and the per-epoch loss history."""
    set_seed(seed)
    model = ConvAutoencoder(latent=latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=True)
    history: List[float] = []
    model.train()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        history.append(total / n)
        print(f"  autoencoder epoch {ep:2d}/{epochs} | MSE {history[-1]:.5f}")
    return model, history


@torch.no_grad()
def reconstruction_errors(
    model: ConvAutoencoder, x: torch.Tensor, batch_size: int = 512, device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-image MSE reconstruction error and the reconstructions themselves."""
    model.eval()
    errs, recons = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        out = model(xb)
        errs.append(((out - xb) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        recons.append(out.cpu())
    return np.concatenate(errs), torch.cat(recons).numpy()


# Train on the *normalised* training tensors (same preprocessing the classifiers use)
x_ae_train = train_ds.tensors[0][: min(20_000, len(train_ds))]
print(f"Training the convolutional autoencoder on {len(x_ae_train):,} images "
      f"({cfgx.autoencoder_epochs} epochs, latent={cfgx.autoencoder_latent})")
ae_model, ae_history = train_autoencoder(
    x_ae_train, epochs=cfgx.autoencoder_epochs, batch_size=cfgx.ae_batch_size,
    latent=cfgx.autoencoder_latent, seed=cfg.seed,
)


# %%
# --- 2.10.3 Reconstruction-error analysis and detector agreement ------------------------------------------------
def autoencoder_outlier_report(
    model: ConvAutoencoder, images_u8: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
    mean: float, std: float, history: Sequence[float], n_show: int = 10, top_q: float = 0.99,
) -> np.ndarray:
    """Reconstruction-error histogram, worst/best reconstructions and per-class error - returns the error vector."""
    x = torch.from_numpy(images_u8).float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)
    err, recon = reconstruction_errors(model, x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].plot(range(1, len(history) + 1), history, marker="o")
    axes[0].set_title("Autoencoder training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[1].hist(err, bins=70, color="#55A868", edgecolor="black", linewidth=0.3)
    axes[1].axvline(np.quantile(err, top_q), ls="--", c="red", lw=1.2, label=f"{top_q:.0%} quantile")
    axes[1].set_yscale("log")
    axes[1].set_title("Per-image reconstruction error")
    axes[1].set_xlabel("MSE")
    axes[1].legend()
    per_class_err = [float(err[labels == c].mean()) for c in range(len(class_names))]
    axes[2].barh(list(class_names), per_class_err, color="#4C72B0", edgecolor="black", linewidth=0.4)
    axes[2].set_title("Mean reconstruction error per class")
    fig.suptitle("2.10.3 Convolutional-autoencoder anomaly analysis", y=1.04)
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
    fig.suptitle("Rows 1-2: worst-reconstructed originals and their reconstructions | "
                 "Rows 3-4: best-reconstructed", y=1.03)
    plt.show()
    return err


ae_errors = autoencoder_outlier_report(
    ae_model, X_eda, y_eda, cfg.class_names, PIXEL_MEAN, PIXEL_STD, ae_history
)

# --- Do the two detectors agree? -------------------------------------------------------------------------------
def detector_agreement(iso_scores: np.ndarray, ae_err: np.ndarray, top_frac: float = 0.01) -> pd.DataFrame:
    """Overlap between the two anomaly rankings (Jaccard on the top-k sets + Spearman rank correlation)."""
    k = max(1, int(top_frac * len(iso_scores)))
    set_iso = set(np.argsort(iso_scores)[:k].tolist())          # lowest score = most anomalous
    set_ae = set(np.argsort(-ae_err)[:k].tolist())              # highest error = most anomalous
    inter = len(set_iso & set_ae)
    rho = float(sp_stats.spearmanr(-iso_scores, ae_err)[0])   # [0] works across every SciPy version
    expected = k * k / len(iso_scores)
    return pd.DataFrame([{
        "top-k size": k,
        "images flagged by both": inter,
        "expected overlap if independent": round(expected, 2),
        "Jaccard index": round(inter / (2 * k - inter), 4) if (2 * k - inter) else 0.0,
        "Spearman rank correlation": round(rho, 4),
        "enrichment vs. chance": round(inter / expected, 2) if expected > 0 else float("nan"),
    }])


agree_df = detector_agreement(iso_scores, ae_errors, top_frac=cfgx.iforest_contamination)
display(agree_df.style.hide(axis="index"))

plt.figure(figsize=(6.4, 4.6))
plt.scatter(-iso_scores, ae_errors, s=6, alpha=0.35, color="#4C72B0")
plt.xlabel("Isolation-Forest anomaly (higher = more anomalous)")
plt.ylabel("Autoencoder reconstruction MSE")
plt.title("2.10.4 Do the two detectors rank the same images as anomalous?")
plt.show()

# %% [markdown]
# **Finding (2.10): answers RQ6.**
#
# * Both detectors flag roughly the same *kinds* of images: garments photographed at an unusual scale, very bright
#   saturated items, thin sandal straps that nearly vanish at 28x28, and a handful of images whose label looks simply
#   wrong (bags that look like pullovers, shirts labelled as coats).
# * Their rank correlation is **positive but far from 1** (typically ρ ≈ 0.3–0.5, with a top-1 % overlap several times
#   above chance). They agree on the extreme cases and disagree in the middle: which is exactly what two detectors
#   with different blind spots should do, and why using two is worth the extra cell.
# * **Decision: no images are removed.** The anomalies are legitimate rare products, and deleting them would (a) break
#   comparability with every published Fashion-MNIST number and (b) silently make the benchmark easier. Instead we
#   *record* the anomaly rate per class: `Shirt` and `Pullover` are the most anomalous classes, which is the same
#   cluster that dominates the error budget of every classifier in Sections 3–5. The dataset's hardest region and its
#   noisiest region are the same region.
