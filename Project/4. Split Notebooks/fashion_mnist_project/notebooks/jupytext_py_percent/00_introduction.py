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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fashion-MNIST: From Classical Machine Learning Baselines to Convolutional Neural Networks
#
# **Deep Learning Exam Project: end-to-end image-classification study on the Fashion-MNIST dataset**
#
# *Upgraded edition (v2): advanced EDA and manifold learning, gradient-boosting baselines with automated hyper-parameter search, a residual CNN and a Vision Transformer, four ensembling strategies, model explainability, an inline unit-test suite and a full statistical treatment of the results.*
#
# | Item | Value |
# |---|---|
# | Dataset | `zalando-research/fashionmnist` (Kaggle), 70,000 grayscale 28x28 images, 10 balanced classes |
# | Acquisition | `kagglehub.dataset_download(...)` -> `fashion-mnist_train.csv`, `fashion-mnist_test.csv` |
# | Frameworks | PyTorch (deep learning), scikit-learn, XGBoost / LightGBM / CatBoost, Optuna, UMAP, SHAP / LIME, statsmodels / SciPy, pandas / matplotlib / seaborn |
# | Target runtime | Google Colab: NVIDIA **T4** GPU (15 GB VRAM), 12.7 GB system RAM, 112 GB disk |
# | Models | Logistic Regression, Linear SVM, RBF SVM, Random Forest, XGBoost, LightGBM, CatBoost, MLP, CNN, residual CNN, Vision Transformer, plus soft-voting / weighted / stacked and hybrid ML+DL ensembles |
# | Deliverable | This notebook: code + markdown narrative + 45 figures + comparison tables + an inline unit-test suite + exported artifacts |
# | Edition | **v2 (upgraded)**: v1 results are preserved verbatim; every addition is marked "upgraded edition" |
#
# > **How to run:** `Runtime -> Change runtime type -> Hardware accelerator: T4 GPU`, then `Runtime -> Run all`.
# > Total expected wall-clock time on a T4: **~75–110 minutes** for the full v2 study (t-SNE/UMAP, the boosting models, the Optuna search, the ViT and the ensembles dominate); **~8–12 minutes** in smoke-test mode.
# > Every expensive step is controlled by a switch in the `CFG` / `CFGX` configuration cells, so the notebook can also be run in a fast "smoke-test" mode (`cfgx = CFGX(fast_mode=True)`). Every optional third-party library is probed before use, so a missing package skips a section with an explanation instead of raising.
#
# ---
#
# ## Table of contents
#
# **Part I: the original study (v1)**
#
# 0. [Problem statement and motivation](#sec0): task, relevance, RQ1–RQ4 (and RQ5–RQ10 for the upgrade)
# 1. [Loading the dataset](#sec1): KaggleHub download, CSV parsing, integrity checks, tensor conversion, splits
# 2. [Exploratory Data Analysis](#sec2): class distribution, samples, pixel statistics, class templates, PCA
# 3. [Traditional machine-learning baselines](#sec3): Logistic Regression, SVMs, Random Forest
# 4. [Deep-learning models](#sec4): MLP, CNN, training loop, evaluation, confusion matrices, error analysis
#
# **Part II: the upgraded edition (v2)**
#
# 2.6–2.10 [Advanced EDA](#sec2): intensity distributions, image metrics, pixel correlations, t-SNE / UMAP in 2D & 3D,
# clustering, outlier detection with Isolation Forest **and** a convolutional autoencoder  
# 3.7–3.9 [Gradient boosting and automated tuning](#sec3): XGBoost, LightGBM, CatBoost, `GridSearchCV`, Optuna  
# 4.9–4.12 [Modern architectures](#sec4): `fit_v2` (schedulers, early stopping, checkpointing), residual CNN,
# Vision Transformer  
# 5. [Advanced ensembling](#sec5b): soft / weighted / stacked voting, hybrid ML+DL committees  
# 6. [Model explainability](#sec6b): Grad-CAM, Integrated Gradients, occlusion, SHAP, LIME, attention roll-out  
# 7. [Unit tests](#sec7b): tensor shapes, model output dimensions, loss values, probability contracts  
# 8. [Statistical validity](#sec8b): McNemar, Cochran's Q, Holm correction, Wilson and bootstrap intervals  
#
# **Part III: context and conclusions**
#
# 9. [Comparison with previous research](#sec9): published results vs. ours, for both editions
# 10. [Final discussion and communication](#sec10): what worked, limitations, future work
# 11. [References](#sec11): 48 sources
# 12. [Appendix](#sec12): reproducibility checklists, exam-criteria map, session summary

# %% [markdown]
# <a id="sec0"></a>
# # 0. Problem statement and motivation
#
# > Exam criterion: **Problem Statement (0–10)**: *"Is the problem clearly defined? Is it relevant?"*
#
# ## 0.1 The problem, stated formally
#
# Given a grayscale image $x \in \mathbb{R}^{28 \times 28}$ with pixel values in $\{0, 1, \dots, 255\}$, predict the garment
# category $y \in \{0, 1, \dots, 9\}$ that the image depicts. We learn a parametric mapping
# $f_\theta : \mathbb{R}^{784} \rightarrow \Delta^{9}$ (a probability distribution over the ten classes) by minimising the
# multi-class cross-entropy (negative log-likelihood) over a labelled training set of 60,000 examples:
#
# $$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log f_\theta\big(x^{(i)}\big)_{y^{(i)}} \; + \; \lambda\,\Omega(\theta)$$
#
# and we report performance on a **held-out test set of 10,000 images that is never used for model selection**.
#
# **This is a single-label, closed-set, balanced, multi-class image-classification task.** Each image contains exactly one
# garment, photographed on a neutral background, already centred and size-normalised by Zalando's preprocessing pipeline.
#
# The ten classes are:
#
# | Label | Class | Label | Class |
# |---|---|---|---|
# | 0 | T-shirt/top | 5 | Sandal |
# | 1 | Trouser | 6 | Shirt |
# | 2 | Pullover | 7 | Sneaker |
# | 3 | Dress | 8 | Bag |
# | 4 | Coat | 9 | Ankle boot |
#
# ## 0.2 Why this problem is relevant in the real world
#
# 1. **E-commerce catalogue automation.** Zalando (the company that released this dataset) processes millions of product
#    photographs per year. Automatic category tagging drives search, filtering, recommendation and inventory management;
#    a mislabelled item is effectively invisible to customers. Fashion-MNIST is a deliberately small, public proxy for
#    exactly that industrial pipeline.
# 2. **Visual search and recommendation.** "Find me something like this" features require a robust garment-category
#    embedding as a first stage before fine-grained similarity ranking.
# 3. **A serious benchmark replacing MNIST.** Xiao, Rasul and Vollgraf (2017) released Fashion-MNIST precisely because
#    handwritten-digit MNIST is "too easy": classical methods already exceed 97 % and CNNs reach 99.7 %, so the benchmark
#    no longer discriminates between algorithms. Fashion-MNIST is a **drop-in replacement** (same 28x28 grayscale format,
#    same 60k/10k split, same file layout) but substantially harder, mainly because of the visually confusable
#    *T-shirt / Pullover / Coat / Shirt* group.
# 4. **Cost-constrained deployment.** Garment tagging must run cheaply at scale. A central practical question of this
#    project is therefore not only *"what is the highest achievable accuracy?"* but **"what accuracy do we get per unit of
#    compute?"**: we deliberately compare a 5-minute CNN on a single T4 with classical models and with published
#    heavyweight architectures.
#
# ## 0.3 Research questions
#
# * **RQ1**: How much better is a convolutional network than strong classical baselines (Logistic Regression, SVM,
#   Random Forest) on identical, properly preprocessed data?
# * **RQ2**: How much of that gap comes from *convolution* itself and how much from *regularisation* (BatchNorm,
#   Dropout, data augmentation)? The MLP is the control condition: same optimiser, same schedule, no spatial prior.
# * **RQ3**: Which classes remain confusable after training, and is the residual error **irreducible label noise** or a
#   modelling failure? (Analysed with the confusion matrix and inspection of misclassified images.)
# * **RQ4**: How do our numbers compare with the published literature (Section 5), and are the differences statistically
#   meaningful given a 10,000-image test set?
#
# ## 0.4 Success criteria defined *before* running anything
#
# | Criterion | Target |
# |---|---|
# | Classical baseline reproduced within ~1 pp of the official benchmark | Logistic Regression ≈ 0.84, RBF SVM ≈ 0.89, Random Forest ≈ 0.87 |
# | MLP test accuracy | ≥ 0.88 |
# | CNN test accuracy | ≥ 0.92 (competitive with published mid-size CNNs) |
# | Honest protocol | test set touched exactly **once**, after all model selection is complete |
# | Runtime | full notebook < 30 min on a single T4 |

# %% [markdown]
# ## 0.5 Scope of this **upgraded edition** (v2)
#
# The first edition of this notebook answered RQ1–RQ4 with five classical baselines, an MLP and a CNN. This second
# edition keeps **every result and every conclusion of the original study intact** and extends it into a full
# production-grade study: advanced EDA (manifold learning, clustering, anomaly detection), gradient-boosting baselines
# with automated hyper-parameter search, two additional deep architectures (a residual CNN and a Vision Transformer
# tailored to 28x28 grayscale inputs), ensembling, explainability, an inline unit-test suite and a rigorous statistical
# comparison of every model pair.
#
# ### 0.5.1 Additional research questions
#
# * **RQ5**: Does the *global* geometry of pixel space (PCA / t-SNE / UMAP) predict the confusion structure that the
#   trained classifiers actually exhibit, and can unsupervised clustering recover the label taxonomy without labels?
# * **RQ6**: Are there **anomalous / mislabelled** images in Fashion-MNIST, and do two independent detectors
#   (Isolation Forest on a PCA embedding, and a convolutional autoencoder's reconstruction error) agree on which they
#   are?
# * **RQ7**: Do modern **gradient-boosting** frameworks (XGBoost, LightGBM, CatBoost), which dominate tabular
#   benchmarks, close the gap to a CNN when pixels are treated as tabular features? Does automated tuning (Optuna)
#   change the answer?
# * **RQ8**: Does a **Vision Transformer**, which has *no* convolutional inductive bias, remain competitive at
#   28x28 resolution and 54k training images: the regime where transformers are usually said to fail?
# * **RQ9**: How much accuracy does **ensembling** actually buy over the single best model, is that gain
#   *statistically significant* (McNemar / bootstrap), and is a hybrid ML+DL ensemble better than a pure DL ensemble?
# * **RQ10**: Do explainability methods (Grad-CAM, Integrated Gradients, occlusion, SHAP) agree with the EDA about
#   *which pixels* carry the class signal: i.e. is the model right **for the right reasons**?
#
# ### 0.5.2 Additional success criteria (again, fixed before running anything)
#
# | Criterion | Target |
# |---|---|
# | Best gradient-boosting baseline | ≥ 0.88 test accuracy (i.e. beats Random Forest) |
# | Residual CNN | ≥ 0.930 test accuracy |
# | Vision Transformer (from scratch, no pre-training) | ≥ 0.900 test accuracy |
# | Best ensemble | ≥ +0.3 pp over the best single model, **and** McNemar p < 0.05 |
# | Unit-test suite | 100 % of assertions pass before any conclusion is drawn |
# | Explainability | saliency mass concentrated on the garment, not on the background |
# | Reproducibility | one `CFG`/`CFGX` cell controls every switch; a `fast_mode` smoke test runs end to end |
#
# ### 0.5.3 Where each new component lives
#
# | Requirement | Section |
# |---|---|
# | Train/test leakage: byte-exact detection and removal | 1.5b |
# | Pixel-intensity distributions (global + per class) | 2.6 |
# | Image metrics: mean / variance images, pixel correlation heat-maps | 2.7 |
# | Dimensionality reduction: PCA (2D/3D), t-SNE (tuned), UMAP (tuned) | 2.8 |
# | Unsupervised clustering + agreement with the label taxonomy | 2.9 |
# | Outlier detection: Isolation Forest **and** a convolutional autoencoder | 2.10 |
# | XGBoost / LightGBM / CatBoost | 3.7 |
# | Automated hyper-parameter tuning (Optuna, `GridSearchCV` fallback) | 3.8 |
# | Residual CNN, Vision Transformer, upgraded trainer (schedulers, early stopping, checkpointing) | 4.9 – 4.12 |
# | Deep ensembles (soft voting, weighted voting, stacking) and hybrid ML+DL ensembles | 5 |
# | Explainability (Grad-CAM, Integrated Gradients, occlusion, SHAP, LIME) | 6 |
# | Unit tests (tensor shapes, model output dims, loss values) | 7 |
# | Statistical validity (McNemar, bootstrap CIs, Cochran's Q, Holm correction) | 8 |
#
# ### 0.5.4 Runtime budget of the upgraded notebook
#
# | Mode | Switch | Wall-clock on a Colab T4 |
# |---|---|---|
# | Full study | `cfgx = CFGX()` (default) | ≈ 75–110 min |
# | Fast smoke test | `cfgx = CFGX(fast_mode=True)` + the `CFG` overrides in Section 12.3 | ≈ 8–12 min |
#
# Every expensive block is behind a boolean switch, and every optional third-party library is behind an availability
# probe, so **the notebook always runs end to end**: a missing package degrades a section into a printed explanation
# instead of raising.

# %% [markdown]
# <a id="sec1"></a>
# # 1. Loading the dataset
#
# > Exam criterion: **Data Gathering / Cleaning / Formatting (0–10)**: *"How was the data acquired? Is the process
# > statistically valid? How was the data cleaned and formatted?"*
#
# This section is organised as:
#
# 1. **1.1** Environment check (GPU, RAM, disk) and dependency installation.
# 2. **1.2** Imports, global configuration and seeding.
# 3. **1.3** KaggleHub download of `zalando-research/fashionmnist`.
# 4. **1.4** Reading the CSV files with pandas and documenting the file structure.
# 5. **1.5** Integrity and cleaning checks (dtypes, ranges, NaNs, duplicates, label validity).
# 6. **1.6** Conversion to PyTorch tensors, normalisation, train/validation split and `DataLoader` construction.

# %% [markdown]
# ## 1.1 Environment check and dependencies
#
# The notebook is written for the Colab **T4** runtime described in the exam brief:
#
# | Resource | Available | How this project uses it |
# |---|---|---|
# | GPU | NVIDIA T4, 15 GB VRAM | CNN/MLP training with mixed precision (T4 has FP16 tensor cores) |
# | System RAM | 12.7 GB | The full dataset as `float32` is only ~220 MB, so everything is held in RAM: no streaming needed |
# | Disk | 112 GB | Kaggle download ≈ 140 MB (CSV): negligible |
#
# **Memory budget (why this fits comfortably):** 70,000 x 784 pixels x 4 bytes (float32) ≈ **220 MB**; as `uint8` it is
# only 55 MB. Even with the pandas intermediate DataFrames (which are the real peak, ~1.1 GB because pandas parses the CSV
# into `int64`) we stay far below 12.7 GB: we explicitly downcast to `uint8` and free the DataFrames afterwards.
