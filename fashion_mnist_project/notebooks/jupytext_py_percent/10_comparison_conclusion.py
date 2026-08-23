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
# <a id="sec9"></a>
# # 9. Comparison with previous research
#
# > Exam criterion: **Previous Research (0–10)**: *"Were external sources consulted? Are at least two sources cited?
# > Are the results compared with previous work?"*
#
# Four independent published sources are used below (full bibliographic details in Section 7). Two of them are the
# "required minimum"; the other two put our numbers into a wider context.
#
# ## 9.1 Source 1: Xiao, Rasul & Vollgraf (2017), *"Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine
# Learning Algorithms"* (arXiv:1708.07747)
#
# This is the **dataset paper**, and the origin of the classical-baseline table that ships with the official
# [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) repository. The authors ran a large
# grid of scikit-learn classifiers with default-ish hyper-parameters on the full 60k/10k split. Their headline results:
#
# | Classifier (their configuration) | Their test accuracy |
# |---|---|
# | LogisticRegression | 0.842 |
# | Linear SVC (`C=1`, `loss=hinge`) | 0.836 |
# | KNeighborsClassifier (`k=5`) | 0.854 |
# | MLPClassifier (one hidden layer, 100 units, `relu`, `adam`) | 0.871 |
# | RandomForestClassifier (`n_estimators=100`) | 0.873 |
# | **SVC (RBF, `C=10`, `gamma=scale`)** | **0.897**: best classical result |
# | *(the same paper's MNIST column)* | most methods > 0.97 |
#
# Their central argument is methodological: on MNIST, all of these methods crowd into a narrow 0.96–0.98 band, so the
# benchmark cannot rank algorithms; on Fashion-MNIST the same methods spread over 0.51–0.90, restoring the benchmark's
# discriminative power.
#
# **Comparison with our Section 3.** We reproduce this ordering exactly (linear ≈ 0.84, ensemble ≈ 0.87, RBF-SVM ≈ 0.89) even though we fit on a 12,000-image stratified subsample rather than all 54,000. The ~0.5–1.5 pp shortfall is the
# expected price of the smaller training set, and its small size confirms the learning curve on this dataset is nearly
# flat past ~10k samples. Reproducing a published baseline within ~1 pp is our main evidence that the preprocessing
# pipeline is correct and that no leakage inflates our numbers.
#
# ## 9.2 Source 2: Bhatnagar, Ghosal & Kolekar (2017), *"Classification of Fashion Article Images using Convolutional
# Neural Networks"*, ICIIP 2017 (DOI 10.1109/ICIIP.2017.8313740)
#
# The most-cited early CNN study on this dataset. They compare three architectures and report:
#
# | Their model | Their test accuracy |
# |---|---|
# | CNN with 2 conv layers (baseline) | ≈ 0.9161 |
# | CNN2 + **Batch Normalization** | ≈ 0.9227 |
# | **CNN2 + Batch Normalization + residual skip connections** | **0.9254** |
#
# Their conclusions, which directly shaped the design in Section 4.1:
#
# 1. Batch normalisation alone buys roughly **+0.7 pp** over an otherwise identical CNN and markedly accelerates
#    convergence: this is why every convolution in our network is followed by `BatchNorm2d`.
# 2. Depth beyond a couple of blocks yields diminishing returns at 28x28 resolution; capacity is better spent on
#    regularisation than on layers.
#
# **Comparison with our Section 4.** Our CNN is in the same family (two-to-three conv blocks + BatchNorm) but adds
# global average pooling, dropout, label smoothing, a OneCycle schedule and flip/translate augmentation. It typically
# lands in the **0.925–0.935** range: i.e. *at or slightly above* their 0.9254, with a comparable parameter budget and
# ~5 minutes of T4 training time. The extra points come from **regularisation and schedule, not from capacity**, which
# is a concrete, falsifiable answer to RQ2.
#
# ## 9.3 Source 3: Zhong, Zheng, Kang, Li & Yang (2020), *"Random Erasing Data Augmentation"*, AAAI
#
# Reported on the official Fashion-MNIST benchmark board: **WRN-28-10 + Random Erasing = 96.35 %** top-1 accuracy
# (the plain WRN-28-10 with standard crops/flips reaches ≈ 95.99 %). This is effectively the practical ceiling for
# single-model results on this dataset.
#
# **Comparison with ours.** They use a network with ~36.5 million parameters (roughly **120x** ours) trained for
# hundreds of epochs on far larger hardware, and gain ~3 pp over our 300 k-parameter CNN. In terms of accuracy per
# FLOP, the small CNN is dramatically more efficient; in terms of raw accuracy, scale still wins. This trade-off is
# the honest framing for any e-commerce deployment decision (Section 6).
#
# ## 9.4 Source 4: Simonyan & Zisserman (2015), *VGG*; Ioffe & Szegedy (2015), *Batch Normalization*;
# Lin, Chen & Yan (2014), *Network in Network*
#
# These are the **methodological** sources behind our architecture rather than Fashion-MNIST results: stacked 3x3
# convolutions (VGG), BatchNorm, and global average pooling instead of large dense heads (NiN). Each is cited at the
# point of use in Section 4.1.
#
# ## 9.5 Consolidated comparison table
#
# The cell below places our measured results side by side with the published numbers. It reads the actual values
# computed earlier in this notebook, so the table is generated, never hard-coded for our own models.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])

# %%
# --- Comparison against published results -------------------------------------------------------------------
LITERATURE = [
    # (model, accuracy, source, year, note)
    ("LogisticRegression", 0.8420, "Xiao et al. (2017), official benchmark", 2017, "full 60k training set"),
    ("Linear SVC", 0.8360, "Xiao et al. (2017), official benchmark", 2017, "C=1, hinge loss"),
    ("KNeighbors (k=5)", 0.8540, "Xiao et al. (2017), official benchmark", 2017, "L2 distance on raw pixels"),
    ("MLPClassifier (100 hidden)", 0.8710, "Xiao et al. (2017), official benchmark", 2017, "single hidden layer"),
    ("RandomForest (100 trees)", 0.8730, "Xiao et al. (2017), official benchmark", 2017, "max_features='sqrt'"),
    ("SVC (RBF, C=10)", 0.8970, "Xiao et al. (2017), official benchmark", 2017, "best classical result"),
    ("CNN2 (2 conv layers)", 0.9161, "Bhatnagar, Ghosal & Kolekar (2017)", 2017, "ICIIP 2017"),
    ("CNN2 + BatchNorm", 0.9227, "Bhatnagar, Ghosal & Kolekar (2017)", 2017, "BN after each conv"),
    ("CNN2 + BatchNorm + skip", 0.9254, "Bhatnagar, Ghosal & Kolekar (2017)", 2017, "their best model"),
    ("WRN-28-10 (std. augmentation)", 0.9599, "Zhong et al. (2020) / benchmark board", 2020, "~36.5M parameters"),
    ("WRN-28-10 + Random Erasing", 0.9635, "Zhong et al. (2020), AAAI", 2020, "near state of the art"),
    ("Human-level estimate", 0.8350, "Zalando crowd study, reported in benchmarks", 2017, "single-annotator, indicative"),
]

lit_df = pd.DataFrame(LITERATURE, columns=["model", "accuracy", "source", "year", "note"])
lit_df["origin"] = "published"

ours_df = (
    pd.DataFrame(RESULTS)
    .query("family != 'Trivial'")[["model", "accuracy", "notes"]]
    .rename(columns={"notes": "note"})
    .assign(source="This notebook (Colab T4)", year=2026, origin="ours")
)

comparison_df = (
    pd.concat([lit_df, ours_df], ignore_index=True)
    .sort_values("accuracy", ascending=False)
    .reset_index(drop=True)
)

display(
    comparison_df[["model", "accuracy", "origin", "source", "year", "note"]]
    .style.hide(axis="index")
    .apply(lambda s: ["background-color: #fff3cd" if v == "ours" else "" for v in comparison_df["origin"]], axis=0)
    .format({"accuracy": "{:.4f}"})
)


# %%
# --- Visual: our results against the published landscape -------------------------------------------------------
def plot_literature_comparison(df: pd.DataFrame) -> None:
    """Horizontal bar chart contrasting our models (highlighted) with published results."""
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


plot_literature_comparison(comparison_df)


# %%
# --- Head-to-head deltas against the two primary sources ----------------------------------------------------
def delta_table(our_records: List[Dict[str, object]]) -> pd.DataFrame:
    """Explicit like-for-like comparison of each of our models with its closest published counterpart."""
    lookup = {r["model"]: r["accuracy"] for r in our_records}
    pairs = [
        ("Logistic Regression", "LogisticRegression (Xiao et al. 2017)", 0.8420),
        ("Linear SVM", "Linear SVC (Xiao et al. 2017)", 0.8360),
        ("RBF SVM (PCA-90%)", "SVC RBF (Xiao et al. 2017)", 0.8970),
        ("Random Forest (300 trees)", "RandomForest 100 trees (Xiao et al. 2017)", 0.8730),
        ("MLP (512-256)", "MLPClassifier 100 hidden (Xiao et al. 2017)", 0.8710),
        ("CNN (VGG-style, GAP)", "CNN2+BN+skip (Bhatnagar et al. 2017)", 0.9254),
    ]
    rows = []
    for ours_name, ref_name, ref_acc in pairs:
        if ours_name not in lookup:
            continue
        ours_acc = float(lookup[ours_name])
        rows.append(
            {
                "our model": ours_name,
                "our accuracy": ours_acc,
                "published counterpart": ref_name,
                "published accuracy": ref_acc,
                "delta (pp)": 100 * (ours_acc - ref_acc),
                "verdict": "above published" if ours_acc >= ref_acc else "below published",
            }
        )
    return pd.DataFrame(rows)


delta_df = delta_table(RESULTS)
display(
    delta_df.style.hide(axis="index")
    .background_gradient(subset=["delta (pp)"], cmap="RdYlGn", vmin=-3, vmax=3)
    .format({"our accuracy": "{:.4f}", "published accuracy": "{:.4f}", "delta (pp)": "{:+.2f}"})
)

print(
    "\nInterpretation guide: with n = 10,000 test images the standard error of an accuracy near 0.93 is\n"
    "about 0.26 pp, so a 95% confidence interval spans roughly +/-0.5 pp. Deltas inside that band are\n"
    "statistically indistinguishable from the published number; only larger deltas warrant a claim."
)

# %% [markdown]
# ## 9.5b Additional sources for the components added in the upgraded edition
#
# The four sources above cover the classical baselines and the CNN. The v2 components (boosting, transformer,
# ensembles, explainability, statistics) rest on their own literature, and each was used to make a concrete design
# decision rather than merely cited:
#
# | # | Source | What we took from it | Where it is used |
# |---|---|---|---|
# | 5 | **Chen & Guestrin (2016)**, *XGBoost: A Scalable Tree Boosting System*, KDD | regularised, histogram-based boosting; the `hist` tree method that makes 600 rounds affordable | 3.7.2 |
# | 6 | **Ke et al. (2017)**, *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS | leaf-wise growth and GOSS/EFB; the framework used for the Optuna search | 3.7.3, 3.8.2 |
# | 7 | **Prokhorenkova et al. (2018)**, *CatBoost: unbiased boosting with categorical features*, NeurIPS | ordered boosting removes the target leakage of classic GBDT; oblivious trees as a regulariser | 3.7.4 |
# | 8 | **Akiba et al. (2019)**, *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD (with **Bergstra et al., 2011** for TPE) | define-by-run TPE search; fANOVA importance for auditing the search | 3.8.2 |
# | 9 | **Dosovitskiy et al. (2021)**, *An Image is Worth 16x16 Words (ViT)*, ICLR | patch embedding + CLS token + learnable positions; and the explicit warning that ViTs underperform CNNs without large-scale pre-training: the hypothesis we test at 28x28 | 4.11 |
# | 10 | **Touvron et al. (2021)**, *Training data-efficient image transformers (DeiT)*, ICML | the small-data ViT recipe: strong augmentation, stochastic depth, high weight decay, gradient clipping | 4.11, 4.12.2 |
# | 11 | **Wolpert (1992)**, *Stacked Generalization*, Neural Networks; **Krogh & Vedelsby (1995)**, NIPS | the stacking construction, and the error = mean-error − ambiguity decomposition that predicts when ensembling helps | 5.1, 5.6 |
# | 12 | **Selvaraju et al. (2017)** Grad-CAM (ICCV); **Sundararajan et al. (2017)** Integrated Gradients (ICML); **Lundberg & Lee (2017)** SHAP (NeurIPS); **Ribeiro et al. (2016)** LIME (KDD); **Abnar & Zuidema (2020)** attention roll-out (ACL) | the five attribution methods, and their known failure modes | 6.2 – 6.7 |
# | 13 | **Dietterich (1998)**, *Approximate Statistical Tests…*, Neural Computation; **Demšar (2006)**, JMLR | McNemar for paired classifier comparison; the discipline of correcting for multiple comparisons | 8.2 – 8.5 |
# | 14 | **Tanveer, Khan & Kang (2021)**, *Fine-Tuning DARTS for Image Classification* | a NAS-discovered architecture reported at **96.91 %** on Fashion-MNIST: the strongest published single-model number we are aware of, and the practical ceiling quoted below | 9.5c |
#
# **A note on how these numbers are used.** Every published accuracy in the tables below is quoted *as reported by its
# authors* on the official 10,000-image test set. They are not re-run here, so they are subject to the usual caveats:
# different training-set sizes, different numbers of runs, and (in the NAS case) a search budget several orders of
# magnitude larger than this notebook's.

# %%
# --- 9.5c Our v2 models against the published landscape ---------------------------------------------------------
LITERATURE_V2 = [
    # (model, accuracy, source, year, note)
    ("CNN2 + BatchNorm + skip", 0.9254, "Bhatnagar, Ghosal & Kolekar (2017)", 2017,
     "closest published counterpart of our ResNet-small"),
    ("WRN-28-10 (standard augmentation)", 0.9599, "Zhong et al. (2020) / benchmark board", 2020,
     "~36.5M parameters"),
    ("WRN-28-10 + Random Erasing", 0.9635, "Zhong et al. (2020), AAAI", 2020, "near state of the art"),
    ("DARTS-based NAS architecture", 0.9691, "Tanveer, Khan & Kang (2021)", 2021,
     "architecture search; strongest published single model we are aware of"),
    ("SVC (RBF, C=10)", 0.8970, "Xiao et al. (2017), official benchmark", 2017,
     "best *classical* result in the dataset paper - no boosting was tested"),
    ("Human-level estimate", 0.8350, "Zalando crowd study, reported in benchmarks", 2017,
     "single annotator, indicative only"),
]

lit2_df = pd.DataFrame(LITERATURE_V2, columns=["model", "accuracy", "source", "year", "note"])
lit2_df["origin"] = "published"

ours_v2 = (
    pd.DataFrame(RESULTS)
    .drop_duplicates(subset=["model"], keep="last")
    .query("family in ['Classical ML', 'Gradient Boosting', 'Deep Learning', 'Ensemble']")
    [["model", "accuracy", "family", "notes"]]
    .rename(columns={"notes": "note", "family": "source"})
    .assign(year=2026, origin="ours")
)

landscape = (
    pd.concat([lit2_df, ours_v2], ignore_index=True)
    .sort_values("accuracy", ascending=False)
    .reset_index(drop=True)
)

plt.figure(figsize=(12, 0.38 * len(landscape) + 2.5))
colors = ["#C44E52" if o == "ours" else "#B0B0B0" for o in landscape.sort_values("accuracy")["origin"]]
d = landscape.sort_values("accuracy")
bars = plt.barh(d["model"] + "  [" + d["origin"] + "]", d["accuracy"], color=colors,
                edgecolor="black", linewidth=0.5)
plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
plt.xlim(0.78, 1.0)
plt.axvline(0.8350, ls=":", c="green", lw=1.2)
plt.text(0.837, 0.2, "human-level estimate", color="green", fontsize=8, rotation=90, va="bottom")
plt.axvline(0.9691, ls=":", c="purple", lw=1.2)
plt.text(0.9705, 0.2, "published best (NAS)", color="purple", fontsize=8, rotation=90, va="bottom")
plt.xlabel("test accuracy on the official 10,000-image Fashion-MNIST test set")
plt.title("9.5c This notebook (red) vs. the published landscape (grey)")
plt.show()

# Like-for-like deltas for the v2 models
def delta_table_v2(records: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Compare each v2 model with the closest published counterpart."""
    lookup = {r["model"]: r["accuracy"] for r in records}
    pairs = [
        ("ResNet-small (residual CNN)", "CNN2 + BN + skip (Bhatnagar et al. 2017)", 0.9254),
        ("ViT-tiny (16 patches)", "CNN2 + BN + skip (Bhatnagar et al. 2017)", 0.9254),
        ("XGBoost (PCA-80)", "SVC RBF - best classical (Xiao et al. 2017)", 0.8970),
        ("LightGBM (PCA-80)", "SVC RBF - best classical (Xiao et al. 2017)", 0.8970),
        ("CatBoost (PCA-80)", "SVC RBF - best classical (Xiao et al. 2017)", 0.8970),
        ("LightGBM (Optuna-tuned)", "SVC RBF - best classical (Xiao et al. 2017)", 0.8970),
        ("Hybrid ML+DL stacking", "WRN-28-10 + Random Erasing (Zhong et al. 2020)", 0.9635),
        ("Hybrid ML+DL weighted voting", "WRN-28-10 + Random Erasing (Zhong et al. 2020)", 0.9635),
    ]
    rows = []
    for ours_name, ref_name, ref_acc in pairs:
        if ours_name not in lookup:
            continue
        acc = float(lookup[ours_name])
        rows.append({
            "our model": ours_name,
            "our accuracy": acc,
            "published counterpart": ref_name,
            "published accuracy": ref_acc,
            "delta (pp)": 100 * (acc - ref_acc),
            "verdict": "above published" if acc >= ref_acc else "below published",
        })
    return pd.DataFrame(rows)


delta_v2 = delta_table_v2(list(pd.DataFrame(RESULTS).drop_duplicates(subset=["model"], keep="last")
                               .to_dict("records")))
display(
    delta_v2.style.hide(axis="index")
    .background_gradient(subset=["delta (pp)"], cmap="RdYlGn", vmin=-5, vmax=5)
    .format({"our accuracy": "{:.4f}", "published accuracy": "{:.4f}", "delta (pp)": "{:+.2f}"})
)

# %% [markdown]
# ### 9.5d What the extended comparison adds
#
# 1. **We fill a genuine gap in the published baseline table.** Xiao et al. (2017) never tested gradient boosting; our
#    XGBoost / LightGBM / CatBoost results (≈ 0.88–0.90) show that modern GBDTs match (but do **not** beat) the
#    RBF-SVM that the dataset paper crowned as the best classical model. That is a small but real contribution of this
#    notebook, and it is a negative result worth recording.
# 2. **Our residual CNN reproduces and slightly exceeds Bhatnagar et al. (2017)** with a comparable parameter budget,
#    confirming their finding that BatchNorm + skip connections are the two changes that matter at this scale.
# 3. **Our ViT trails both CNNs**, exactly as Dosovitskiy et al. (2021) predict for the no-pre-training, small-data
#    regime. Reproducing a *predicted failure* is as much a validation of the pipeline as reproducing a success.
# 4. **Our best hybrid ensemble (≈ 0.95) sits between the published WRN-28-10 (0.9599) and everything below it**, while
#    using roughly two orders of magnitude fewer parameters and ~1 GPU-hour instead of a full training run of a
#    36.5 M-parameter wide ResNet. Against the NAS-discovered 0.9691, we remain ~2 pp behind: the honest price of not
#    running an architecture search.

# %% [markdown]
# ### 9.6 What the comparison tells us
#
# 1. **Our classical baselines reproduce the official benchmark** (all within ~1.5 pp, despite using 22 % of the
#    training data). This validates the data pipeline end to end and is the strongest available evidence that the
#    deep-learning numbers reported here are trustworthy.
# 2. **Our CNN matches or slightly exceeds Bhatnagar et al. (2017)** with a comparable parameter budget. The gain comes
#    from modern *training* practice (OneCycle, AdamW, label smoothing, augmentation, GAP), not from a bigger model:
#    consistent with the broader finding in the literature that training recipe often matters more than architecture at
#    this scale.
# 3. **A ~3 pp gap to the WRN-28-10 state of the art remains**, and closing it costs roughly two orders of magnitude
#    more parameters and compute. For a real catalogue-tagging system, the small CNN is very likely the better
#    engineering choice; the WRN is the better choice only when the last 3 pp are worth the bill.
# 4. **Everything beats the ~0.835 single-annotator human estimate**, which is a useful reminder that "super-human" on a
#    benchmark mostly means "better than a tired human labelling 28x28 thumbnails": not that the problem is solved.

# %% [markdown]
# <a id="sec10"></a>
# # 10. Final discussion and communication
#
# > Exam criterion: **Communication (0–10)**: *"Is the story told clearly? Is the reasoning easy to follow?"*
#
# ## 10.1 The story in one paragraph
#
# We set out to classify 28x28 grayscale garment images into ten categories, and to establish **how much of the task is
# solved by which kind of model**. Exploratory analysis showed a perfectly balanced dataset with a hard, visually
# overlapping upper-body cluster (`T-shirt/top`, `Pullover`, `Coat`, `Shirt`) and easy, well-separated footwear and
# `Bag` classes. Classical models confirmed that prediction quantitatively: linear methods reach ≈ 0.84, an RBF-SVM
# ≈ 0.89, but all of them collapse on `Shirt`. A 535 k-parameter MLP adds little over the SVM: showing that *capacity
# without a spatial prior* is not the missing ingredient. A 300 k-parameter CNN with BatchNorm, dropout, label smoothing
# and geometric augmentation reaches ≈ 0.93, matching a published ICIIP 2017 CNN, and its residual errors are
# concentrated almost entirely in the upper-body cluster, where inspection of the confident mistakes suggests genuine
# label ambiguity rather than model failure.
#
# ## 10.2 What worked well
#
# | # | What | Evidence |
# |---|---|---|
# | 1 | **Convolution over flat pixels.** The CNN beat the MLP with ~45 % *fewer* parameters. | McNemar p ≪ 0.05 (Section 4.8) |
# | 2 | **Regularisation stack** (BatchNorm + dropout + weight decay + label smoothing + augmentation). | The CNN's train/validation gap stays near zero while the MLP's widens (Section 4.6) |
# | 3 | **Augmentation choice grounded in EDA.** Flip and ±2 px shift were chosen *because* the variance map showed a wide black margin and left/right symmetry. | Section 2.4 → Section 4.2 |
# | 4 | **OneCycle + AdamW + AMP.** Near-converged results in ~25 epochs / ~5 minutes on a T4, using ~1.2 GB of 15 GB VRAM. | Section 4.5 timing output |
# | 5 | **Honest evaluation protocol.** Three-way split, leakage hash check, test set used once, McNemar significance testing. | Sections 1.5, 1.6, 4.8 |
# | 6 | **PCA-compressed RBF-SVM.** Made the strongest classical baseline affordable (minutes instead of hours) with negligible accuracy loss. | Section 3.4 |
#
# ## 10.3 What did *not* work as well / honest limitations of this study
#
# * **Classical models were fitted on a 12,000-image subsample.** This is a deliberate compute trade-off, and it costs
#   an estimated 0.5–1.5 pp relative to the full-data published numbers. Anyone reproducing the benchmark exactly should
#   set `CFG.sk_train_subset = 54_000` and expect an RBF-SVM run measured in hours on Colab's 2 vCPUs.
# * **No systematic hyper-parameter search.** Learning rate, dropout and depth were chosen from established practice
#   rather than a validation sweep. A modest random search over `lr`, `dropout` and channel widths would plausibly add
#   0.5–1 pp; it was omitted to keep the notebook within the runtime budget.
# * **Single seed.** Every number is one run. Ideally we would report mean ± std over 3–5 seeds, since run-to-run
#   variation on this dataset is roughly ±0.2–0.3 pp: the same order as some of the differences we discuss.
# * **No test-time augmentation or ensembling.** Both are known to add ~0.5–1 pp here, but they would blur the clean
#   single-model comparison that this project is about.
#
# ## 10.4 Limitations of the Fashion-MNIST dataset itself
#
# This is important context, and it is the honest counterweight to any accuracy number in this notebook:
#
# 1. **Resolution destroys the information humans use.** At 28x28 grayscale, fabric texture, print, buttons and colour (the very cues that separate a shirt from a pullover) are gone. Part of the residual error is therefore
#    *information-theoretically irreducible*, not a modelling failure.
# 2. **Label ambiguity.** `Shirt` overlaps semantically with `T-shirt/top`, `Pullover` and `Coat`. The confident
#    misclassifications in Section 4.7 include images where the ground-truth label is arguably wrong. With ~10 %
#    ambiguity in one class, a 100 % ceiling does not exist.
# 3. **Curated, unrealistically clean distribution.** Every image is centred, scale-normalised, background-removed and
#    single-object. Real catalogue and user-generated photographs contain multiple garments, models wearing them,
#    occlusion, shadows, varied backgrounds and arbitrary poses. Accuracy here **does not transfer** to that setting.
# 4. **No colour.** Colour is one of the strongest signals in real fashion retrieval and is entirely absent.
# 5. **Closed set of 10 coarse categories.** A production taxonomy has hundreds to thousands of fine-grained labels,
#    is hierarchical, and needs an "unknown / other" option. Fashion-MNIST cannot exercise any of that.
# 6. **Benchmark saturation.** With the state of the art at ~96.5 % and cheap CNNs at ~93 %, the remaining headroom is
#    small and increasingly dominated by label noise, so the dataset is now better as a *teaching and prototyping*
#    instrument than as a research frontier.
#
# ## 10.5 Future improvements, in order of expected return on effort
#
# | Priority | Improvement | Expected gain | Cost |
# |---|---|---|---|
# | 1 | **Random Erasing / Cutout augmentation** (Zhong et al. 2020) | +0.5–1.0 pp | ~10 lines, no extra training time |
# | 2 | **Test-time augmentation** (average logits over the image and its mirror) | +0.3–0.7 pp | 2x inference cost only |
# | 3 | **Wider / deeper backbone** (WRN-16-4 or a small ResNet) | +1–2 pp | ~15–25 min on a T4 |
# | 4 | **Ensemble of 3–5 CNNs with different seeds** | +0.5–1.0 pp | linear in the number of models |
# | 5 | **Mixup / CutMix** | +0.3–0.8 pp | small; needs a slightly longer schedule |
# | 6 | **Hyper-parameter search** (Optuna over lr / dropout / width) | +0.5–1.0 pp | 20–50 short runs |
# | 7 | **Two-stage hierarchical classifier**: first {footwear, bag, trouser, upper-body}, then a specialist head for the upper-body cluster | targets the actual error mass | moderate; a genuinely interesting research direction |
# | 8 | **Multi-seed reporting with confidence intervals** | no accuracy gain, but far higher scientific credibility | 3–5x runtime |
#
# ## 10.6 Conclusion of the original study
#
# Convolutional inductive bias, not raw capacity, is what solves Fashion-MNIST: a 300 k-parameter CNN outperforms a
# larger MLP and the best classical baseline by a statistically significant margin, matches a published ICIIP CNN, and
# does so in five minutes on a single T4. The remaining ~7 % error is concentrated in one semantically ambiguous cluster
# of garment types, where the dataset's own labels are unreliable. For an e-commerce catalogue application, the practical
# lesson is that the next unit of effort is far better spent on **higher-resolution, colour input and a cleaner label
# taxonomy** than on a bigger network.

# %% [markdown]
# ## 10.7 What the upgraded edition adds to the story
#
# Section 10.5 of the original study listed eight prioritised improvements. Six of them have now been implemented and
# measured, which turns a list of speculations into a table of results:
#
# | v1 prediction | What we actually measured in v2 | Verdict |
# |---|---|---|
# | Test-time augmentation: +0.3–0.7 pp | mirror-TTA is enabled for every ensemble member (Section 5.3.1) | **confirmed**, at the low end |
# | Wider / deeper backbone: +1–2 pp | `ResNet-small` gains ≈ +0.5–1.0 pp over the v1 CNN (Section 4.12) | **partially confirmed**: the estimate was optimistic |
# | Ensemble of several models: +0.5–1.0 pp | deep soft voting +0.4–0.8 pp; hybrid ML+DL stacking +1.0–1.7 pp over the best single model (Section 5.8) | **confirmed and exceeded** by the hybrid variant |
# | Hyper-parameter search: +0.5–1.0 pp | Optuna over 8 LightGBM dimensions: +0.3–0.8 pp (Section 3.8.2) | **confirmed**, but only for the classical family |
# | Multi-seed reporting with confidence intervals | replaced by something stronger: Wilson intervals, a *paired* bootstrap and Holm-corrected McNemar tests over all pairs (Section 8) | **done differently, and better** |
# | Two-stage hierarchical classifier for the upper-body cluster | not implemented; the explainability analysis (Section 6.8) suggests why it would help *and* why its ceiling is low | still open |
#
# ### 10.7.1 A correction to the v1 protocol
#
# The v1 integrity report (Section 1.5) *detected* ten byte-identical train/test images but did not act on them, so
# every number in the original study was computed on a marginally contaminated test set. Section 1.5b now removes
# those rows from the **training** side before any model is fitted. The measurable effect is negligible (10 of
# 10,000 test images, i.e. at most 0.1 pp, well inside the ±0.5 pp noise band), but the methodological effect is not:
# the test set is once again a set of images no model has ever seen, and the claim can now be *verified* rather than
# assumed: `test_leakage_removed` re-scans the two splits byte-by-byte every time the notebook runs.
#
# ### 10.7.2 The three genuinely new conclusions
#
# 1. **Gradient boosting closes the classical gap but does not cross it (RQ7).** XGBoost / LightGBM / CatBoost reach
#    0.88–0.90: better than Random Forest, level with the RBF-SVM the dataset paper considered the classical ceiling,
#    and still ~4 pp behind a small CNN. Tuning moves this by less than a point. The missing ingredient is the
#    representation, not the optimiser.
# 2. **A from-scratch Vision Transformer is a strong *ensemble member* rather than a strong *model* (RQ8).** It is 1–2 pp
#    behind the CNNs on its own, but its errors are the least correlated with everyone else's (Section 4.12.4), which is
#    what makes the committees in Section 5 work. "Weaker but different" beat "stronger but redundant" here: a concrete,
#    measured instance of the Krogh–Vedelsby decomposition.
# 3. **Every analysis converges on the same 2 % of the data.** The unsupervised clustering merges it (Section 2.9), both
#    anomaly detectors flag it (Section 2.10), every model misclassifies it (Sections 3–4), no ensemble recovers it
#    (the oracle bound in Section 5.3.2), and the attribution maps show why: the deciding pixels (collar shape, sleeve
#    termination, fabric texture) are largely destroyed by the 28x28 grayscale encoding (Section 6.8). **The remaining
#    error is a property of the dataset, not of the models.**
#
# ### 10.7.3 Updated conclusion
#
# The original conclusion (*convolutional inductive bias, not raw capacity, is what solves Fashion-MNIST*) survives the
# upgrade intact and is now supported by a much wider sweep: two more model families (boosting, transformers), automated
# tuning, four ensembling strategies and a proper statistical treatment. What the upgrade adds is a **quantified ceiling**:
# with an honest protocol and ~1 GPU-hour, a hybrid ML+DL ensemble reaches ≈ 0.95, about 1 pp below a 36.5 M-parameter
# WRN-28-10 and ~2 pp below a NAS-discovered architecture, while the *oracle* over our own committee sits at ≈ 0.98.
# The last two points are not an engineering problem: they are a **data problem**, and the correct next investment for a
# real catalogue-tagging system is higher-resolution colour images and a cleaner label taxonomy, exactly as the v1 study
# argued, now with the measurements to prove it.

# %% [markdown]
# <a id="sec11"></a>
# # 11. References
#
# ## 11.1 References of the original study
#
# **Primary sources compared against in Section 5**
#
# 1. **Xiao, H., Rasul, K., & Vollgraf, R. (2017).** *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine
#    Learning Algorithms.* arXiv:1708.07747. <https://arxiv.org/abs/1708.07747>: dataset paper and the source of the
#    official classical-baseline table (LogisticRegression 0.842, RandomForest 0.873, SVC-RBF 0.897).
#    Benchmark board: <http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/>
# 2. **Bhatnagar, S., Ghosal, D., & Kolekar, M. H. (2017).** *Classification of Fashion Article Images using
#    Convolutional Neural Networks.* 4th International Conference on Image Information Processing (ICIIP), 1–6.
#    DOI: [10.1109/ICIIP.2017.8313740](https://doi.org/10.1109/ICIIP.2017.8313740): CNN2 + BatchNorm + skip
#    connections, **92.54 %**.
# 3. **Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020).** *Random Erasing Data Augmentation.* AAAI 2020.
#    arXiv:1708.04896: WRN-28-10 + Random Erasing, **96.35 %** on Fashion-MNIST.
#
# **Methodological sources for the architecture and training recipe**
#
# 4. **Simonyan, K., & Zisserman, A. (2015).** *Very Deep Convolutional Networks for Large-Scale Image Recognition
#    (VGG).* ICLR. arXiv:1409.1556: the stacked-3x3-convolution design used in Section 4.1.
# 5. **Ioffe, S., & Szegedy, C. (2015).** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal
#    Covariate Shift.* ICML. arXiv:1502.03167.
# 6. **Lin, M., Chen, Q., & Yan, S. (2014).** *Network In Network.* ICLR. arXiv:1312.4400: global average pooling
#    instead of a large dense head.
# 7. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** *Dropout: A Simple Way
#    to Prevent Neural Networks from Overfitting.* JMLR 15(56), 1929–1958.
# 8. **Loshchilov, I., & Hutter, F. (2019).** *Decoupled Weight Decay Regularization (AdamW).* ICLR. arXiv:1711.05101.
# 9. **Smith, L. N. (2018).** *A Disciplined Approach to Neural Network Hyper-Parameters: Part 1: Learning Rate, Batch
#    Size, Momentum, and Weight Decay (the "1cycle" policy).* arXiv:1803.09820.
# 10. **Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).** *Rethinking the Inception Architecture
#     for Computer Vision*: origin of label smoothing. CVPR.
# 11. **Dietterich, T. G. (1998).** *Approximate Statistical Tests for Comparing Supervised Classification Learning
#     Algorithms.* Neural Computation 10(7), 1895–1923: justification for using McNemar's test in Section 4.8.
#
# **Software and data**
#
# 12. **Paszke, A., et al. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
# 13. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
# 14. **Kaggle dataset mirror:** `zalando-research/fashionmnist`:
#     <https://www.kaggle.com/datasets/zalando-research/fashionmnist>, downloaded here with `kagglehub`.

# %% [markdown]
# ## 11.2 Additional references for the upgraded edition
#
# **Gradient boosting and hyper-parameter optimisation**
#
# 15. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* KDD '16, 785–794.
#     arXiv:1603.02754.
# 16. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).** *LightGBM: A Highly
#     Efficient Gradient Boosting Decision Tree.* NeurIPS 30.
# 17. **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).** *CatBoost: unbiased boosting
#     with categorical features.* NeurIPS 31. arXiv:1706.09516.
# 18. **Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).** *Optuna: A Next-generation Hyperparameter
#     Optimization Framework.* KDD '19. arXiv:1907.10902.
# 19. **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).** *Algorithms for Hyper-Parameter Optimization.*
#     NeurIPS 24: the TPE sampler Optuna uses by default.
# 20. **Bergstra, J., & Bengio, Y. (2012).** *Random Search for Hyper-Parameter Optimization.* JMLR 13, 281–305.
#
# **Architectures**
#
# 21. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* CVPR.
#     arXiv:1512.03385: the residual block used in Section 4.10.
# 22. **Dosovitskiy, A., et al. (2021).** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*
#     ICLR. arXiv:2010.11929: the ViT architecture adapted in Section 4.11.
# 23. **Vaswani, A., et al. (2017).** *Attention Is All You Need.* NeurIPS 30. arXiv:1706.03762.
# 24. **Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021).** *Training data-efficient
#     image transformers & distillation through attention (DeiT).* ICML. arXiv:2012.12877: the small-data ViT recipe.
# 25. **Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. (2016).** *Deep Networks with Stochastic Depth.* ECCV:
#     the DropPath regulariser used in the ViT blocks.
# 26. **Tanveer, M. S., Khan, M. U. K., & Kang, C. M. (2021).** *Fine-Tuning DARTS for Image Classification.* ICPR:
#     reports **96.91 %** on Fashion-MNIST, the strongest published single-model result cited in Section 9.5c.
#
# **Ensembling**
#
# 27. **Wolpert, D. H. (1992).** *Stacked Generalization.* Neural Networks 5(2), 241–259.
# 28. **Krogh, A., & Vedelsby, J. (1995).** *Neural Network Ensembles, Cross Validation, and Active Learning.* NIPS 7:
#     the error/ambiguity decomposition quoted in Section 5.1.
# 29. **Breiman, L. (1996).** *Bagging Predictors.* Machine Learning 24(2), 123–140.
# 30. **Dietterich, T. G. (2000).** *Ensemble Methods in Machine Learning.* MCS 2000, LNCS 1857, 1–15.
#
# **Explainability**
#
# 31. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** *Grad-CAM: Visual
#     Explanations from Deep Networks via Gradient-based Localization.* ICCV. arXiv:1610.02391.
# 32. **Sundararajan, M., Taly, A., & Yan, Q. (2017).** *Axiomatic Attribution for Deep Networks (Integrated
#     Gradients).* ICML. arXiv:1703.01365.
# 33. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions (SHAP).*
#     NeurIPS 30. arXiv:1705.07874.
# 34. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?" Explaining the Predictions of Any
#     Classifier (LIME).* KDD '16. arXiv:1602.04938.
# 35. **Zeiler, M. D., & Fergus, R. (2014).** *Visualizing and Understanding Convolutional Networks.* ECCV:
#     the occlusion-sensitivity analysis of Section 6.4.
# 36. **Abnar, S., & Zuidema, W. (2020).** *Quantifying Attention Flow in Transformers.* ACL: attention roll-out.
#
# **Unsupervised analysis, anomaly detection and statistics**
#
# 37. **van der Maaten, L., & Hinton, G. (2008).** *Visualizing Data using t-SNE.* JMLR 9, 2579–2605.
# 38. **McInnes, L., Healy, J., & Melville, J. (2018).** *UMAP: Uniform Manifold Approximation and Projection for
#     Dimension Reduction.* arXiv:1802.03426.
# 39. **Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).** *Isolation Forest.* ICDM 2008, 413–422.
# 40. **Hinton, G. E., & Salakhutdinov, R. R. (2006).** *Reducing the Dimensionality of Data with Neural Networks.*
#     Science 313(5786), 504–507: the autoencoder used as the second anomaly detector.
# 41. **Hubert, L., & Arabie, P. (1985).** *Comparing Partitions.* Journal of Classification 2, 193–218: the Adjusted
#     Rand Index used in Section 2.9.
# 42. **Rousseeuw, P. J. (1987).** *Silhouettes: a graphical aid to the interpretation and validation of cluster
#     analysis.* J. Comput. Appl. Math. 20, 53–65.
# 43. **Demšar, J. (2006).** *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7, 1–30: the
#     discipline behind Section 8.3.
# 44. **Holm, S. (1979).** *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of
#     Statistics 6(2), 65–70.
# 45. **Efron, B., & Tibshirani, R. (1993).** *An Introduction to the Bootstrap.* Chapman & Hall: the paired bootstrap
#     of Section 8.4.
# 46. **Wilson, E. B. (1927).** *Probable Inference, the Law of Succession, and Statistical Inference.* JASA 22, 209–212.
#
# **Additional software**
#
# 47. **Seabold, S., & Perktold, J. (2010).** *statsmodels: Econometric and statistical modeling with Python.* SciPy.
# 48. **Virtanen, P., et al. (2020).** *SciPy 1.0: fundamental algorithms for scientific computing in Python.*
#     Nature Methods 17, 261–272.

# %% [markdown]
# <a id="sec12"></a>
# # 12. Appendix: reproducibility checklists and exam-criteria map
#
# ## 12.1 Reproducibility checklist (original study)
#
# | Item | Status |
# |---|---|
# | Random seeds fixed for `random`, `numpy`, `torch`, CUDA | ✔ `set_seed(42)` |
# | cuDNN deterministic mode | ✔ enabled in `set_seed` |
# | Every hyper-parameter in one place | ✔ the `CFG` dataclass |
# | Data acquisition scripted (no manual downloads) | ✔ `kagglehub.dataset_download` |
# | Train/validation/test split stratified and fixed by seed | ✔ Section 1.6 |
# | Normalisation statistics computed on the training split only | ✔ Section 1.6 |
# | Train/test leakage explicitly tested | ✔ hash-intersection check, Section 1.5 |
# | Test set used exactly once per model, after selection | ✔ Sections 3 and 4.7 |
# | Model weights and the results table exported | ✔ `artifacts/*.pt`, `artifacts/model_comparison.csv` |
# | Statistical significance of the headline comparison | ✔ McNemar test, Section 4.8 |
#
# ## 12.2 Map from exam criteria to notebook sections (original study)
#
# | Exam criterion (max points) | Where it is addressed |
# |---|---|
# | **Problem Statement (10)** | Section 0: formal task definition, four real-world motivations, RQ1–RQ4, success criteria fixed in advance |
# | **Layout (20)** | Numbered sections 0–12, table of contents with anchors, consistent "code → finding" rhythm, summary tables |
# | **Code Quality (20)** | `CFG` dataclass, typed and docstring'd functions (`load_fashion_csv`, `fit`, `evaluate`, `plot_*`), `nn.Module` classes, sklearn `Pipeline`s, one generic training loop reused by both models, results registry |
# | **Previous Research (10)** | Section 9: four primary sources, side-by-side tables, delta table, discussion of *why* the deltas occur; Section 11: 48 references |
# | **Data Gathering / Cleaning / Formatting (10)** | Section 1: KaggleHub acquisition, CSV structure documented, 11-point integrity report, leakage check, normalisation, stratified split, memory budgeting |
# | **Testing (10)** | Three-way split; 7 models evaluated on the identical official test set; accuracy / macro-F1 / weighted-F1 / top-2 / per-class report; confusion matrices; McNemar significance test |
# | **Visualization (10)** | 15+ figures: class distribution, sample grid, pixel histograms, class means, variance map, template correlation, PCA, RF importance, augmentation preview, learning curves, generalisation gap, confusion matrices, misclassification gallery, accuracy-vs-cost, literature comparison |
# | **Communication (10)** | Every code cell is followed by an explicit **Finding**; Section 10 tells the whole story, including what did *not* work and the dataset's own limitations |
#
# ## 12.3 How to re-run cheaply (v1 settings)
#
# For a quick smoke test (~2 minutes end to end), edit the `CFG` cell:
#
# ```python
# cfg = CFG(
#     sk_train_subset=3_000,   # much smaller classical fit
#     run_rbf_svm=False,       # skip the most expensive baseline
#     mlp_epochs=3,
#     cnn_epochs=3,
# )
# ```
#
# Then `Runtime -> Run all`. Accuracies will be a few points lower, but every cell, plot and table will execute exactly
# as in the full run.
#
# ## 12.4 Environment summary printed for the record

# %%
# --- Final environment / session summary --------------------------------------------------------------------
summary = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
    "device_used": str(DEVICE),
    "seed": cfg.seed,
    "train/val/test sizes": [int(len(train_ds)), int(len(val_ds)), int(len(test_ds))],
    "models_evaluated": int(len(RESULTS)),
    "best_model": str(pd.DataFrame(RESULTS).sort_values("accuracy", ascending=False).iloc[0]["model"]),
    "best_accuracy": float(pd.DataFrame(RESULTS)["accuracy"].max()),
}
print(json.dumps(summary, indent=2))

with open(Path(cfg.artifacts_dir) / "run_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("\nArtifacts written to:", Path(cfg.artifacts_dir).resolve())
for f in sorted(Path(cfg.artifacts_dir).iterdir()):
    print("  -", f.name)


# %% [markdown]
# ## 12.5 Reproducibility checklist for the upgraded edition
#
# | Item | Status |
# |---|---|
# | Every v2 hyper-parameter in one dataclass (`CFGX`) | ✔ Section 1.2b |
# | Train/test leakage detected byte-exactly **and removed** before any model is fitted | ✔ Section 1.5b, verified again by two unit tests |
# | Official 10,000-image test set left untouched so literature comparisons stay like for like | ✔ `CFGX.leakage_policy = 'drop_from_train'` |
# | Single switch for a full smoke test (`CFGX(fast_mode=True)`) | ✔ Section 1.2b + 12.6 |
# | Optional dependencies probed, never assumed | ✔ `AVAILABLE` table, Section 1.2b |
# | Manifold learning tuned and *quantified*, not eyeballed | ✔ trustworthiness + kNN probe, Section 2.8.6 |
# | Anomaly detection cross-validated by two independent methods | ✔ Section 2.10.4 |
# | PCA for the boosting models fitted on training rows only | ✔ Section 3.7.1 (+ unit test) |
# | Hyper-parameter search on training data with CV; winner refit once | ✔ Sections 3.8.1–3.8.2 |
# | Search history plotted, not just the winning configuration | ✔ Section 3.8.2 |
# | Early stopping + resumable checkpoints for every new model | ✔ `fit_v2`, Section 4.9.2 |
# | **Best version of every trained model persisted to a per-type folder** (`models/{ml,dl,ensemble}/`, each with a `BEST.json` pointer) | ✔ Sections 3.1b, 3.9c, 4.12.5, 5.10 (+ unit test) |
# | Ensemble weights / meta-learner fitted on validation, never on test | ✔ Section 5.2 (caveat stated explicitly) |
# | Explainability validated against a model-free reference map | ✔ Section 6.8 |
# | Machine-checked contracts on shapes, dtypes, losses, probabilities | ✔ 27-test suite, Section 7 |
# | Multiple-comparison correction applied to all pairwise tests | ✔ Holm, Section 8.3 |
# | Paired bootstrap for every ensemble-vs-single claim | ✔ Section 8.4 |
# | All tables, checkpoints and the run summary exported to `artifacts/` | ✔ Section 12.8 |
#
# ## 12.6 How to re-run cheaply (v2)
#
# ```python
# # Cell 1.2 (v1 configuration)
# cfg = CFG(sk_train_subset=3_000, run_rbf_svm=False, mlp_epochs=3, cnn_epochs=3)
#
# # Cell 1.2b (v2 configuration)
# cfgx = CFGX(fast_mode=True)      # shrinks EDA samples, Optuna trials, ViT/ResNet epochs, bootstrap iterations
# ```
#
# `Runtime -> Run all` then completes in roughly 8–12 minutes on a T4. Every cell, plot, table and test executes exactly
# as in the full run; only the accuracies are a few points lower.
#
# ## 12.7 Updated map from exam criteria to notebook sections
#
# | Exam criterion (max points) | Where it is addressed (v1 + v2) |
# |---|---|
# | **Problem Statement (10)** | Section 0: formal task, four real-world motivations, RQ1–RQ4, plus RQ5–RQ10 and the extended success criteria in 0.5 |
# | **Layout (20)** | Sections 0–12 with anchors and a table of contents; consistent *question → code → figure → **Finding*** rhythm in every subsection |
# | **Code Quality (20)** | Two configuration dataclasses; every routine a typed, docstring'd, self-contained function; `nn.Module` classes for all six architectures; sklearn `Pipeline`s; two reusable training loops; global registries (`RESULTS`, `SKLEARN_ZOO`, `TORCH_ZOO`, `MEMBER_PROBS`); a 27-test suite; graceful degradation for every optional dependency |
# | **Previous Research (10)** | Section 9: 4 primary sources compared numerically, 10 further sources for the v2 components, delta tables and a landscape plot |
# | **Data Gathering / Cleaning / Formatting (10)** | Section 1 (acquisition, 11-point integrity report, leakage hash check, stratified split, normalisation) + Section 2.10 (anomaly detection and the documented decision *not* to remove outliers) |
# | **Testing (10)** | Three-way split; ~20 models on the identical test set; accuracy / macro-F1 / weighted-F1 / top-2 / per-class; confusion matrices; error-overlap analysis; McNemar, Cochran's Q, Holm correction, Wilson and paired-bootstrap intervals; a 27-assertion unit-test suite |
# | **Visualization (10)** | 45+ figures: distributions, ECDFs, violins, KS matrix, class means/variances, Fisher map, correlation heat-maps, PCA/t-SNE/UMAP in 2D and 3D, clustering diagnostics, anomaly galleries, boosting and tuning plots, learning curves, confusion matrices, weight plots, Grad-CAM/IG/occlusion/SHAP/LIME/attention overlays, forest plots of confidence intervals |
# | **Communication (10)** | Every subsection ends in an explicit **Finding** tied to a research question; Section 10 tells the whole story including negative results, limitations and what we would do next |

# %%
# --- 12.8 Final v2 session summary and artifact manifest -----------------------------------------------------
def build_v2_summary() -> Dict[str, object]:
    """Collect the headline facts of the upgraded run into one JSON-serialisable dictionary."""
    lb = pd.DataFrame(RESULTS).drop_duplicates(subset=["model"], keep="last").sort_values(
        "accuracy", ascending=False
    )
    best = lb.iloc[0]
    families = lb.groupby("family")["accuracy"].max().round(4).to_dict()
    summary: Dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "seed": cfg.seed,
        "fast_mode": cfgx.fast_mode,
        "models_evaluated": int(len(lb)),
        "best_model": str(best["model"]),
        "best_accuracy": float(best["accuracy"]),
        "best_per_family": families,
        "optional_dependencies": AVAILABLE,
        "unit_tests": test_report["status"].value_counts().to_dict() if "test_report" in globals() else {},
        "significant_pairs_after_holm": int(pairwise_df["significant"].sum()) if "pairwise_df" in globals() else None,
        "artifacts_dir": str(Path(cfg.artifacts_dir).resolve()),
        "saved_model_artifacts": int(len(models_manifest_df())) if "models_manifest_df" in globals() else 0,
    }
    return summary


v2_summary = build_v2_summary()
print(json.dumps(v2_summary, indent=2, default=str))

with open(Path(cfg.artifacts_dir) / "run_summary_v2.json", "w") as fh:
    json.dump(v2_summary, fh, indent=2, default=str)

print("\nArtifact manifest:")
manifest = pd.DataFrame([
    {"file": f.name, "size (KB)": round(f.stat().st_size / 1024, 1)}
    for f in sorted(Path(cfg.artifacts_dir).iterdir()) if f.is_file()
])
display(manifest.style.hide(axis="index"))
manifest.to_csv(Path(cfg.artifacts_dir) / "artifact_manifest.csv", index=False)

# v3: best version of every trained model is persisted under models/{ml,dl,ensemble}.
if "models_manifest_df" in globals():
    models_manifest = models_manifest_df()
    if not models_manifest.empty:
        print("\nPersisted models - best version of each, grouped by family folder:")
        display(models_manifest.style.hide(axis="index"))
        models_manifest.to_csv(Path(cfg.artifacts_dir) / "models_manifest.csv", index=False)
        print("\nBest model per family folder (from each BEST.json):")
        for _folder in MODEL_DIRS:
            _best = MODEL_DIRS[_folder] / "BEST.json"
            if _best.exists():
                _info = json.loads(_best.read_text())
                print(f"  {_folder:<8s}: {_info.get('best_model')}  "
                      f"(accuracy={_info.get('accuracy'):.4f}, file={_info.get('file')})")

# %% [markdown]
# ---
#
# ### End of notebook
#
# *Fashion-MNIST: classical baselines vs. deep learning, built for the Google Colab T4 runtime
# (15 GB GPU RAM, 12.7 GB system RAM, 112 GB disk). Dataset acquired with `kagglehub` from
# `zalando-research/fashionmnist`. All results in this notebook are produced by the cells above; nothing is
# hard-coded except the published figures quoted from the literature in Section 5.*
