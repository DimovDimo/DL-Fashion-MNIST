# Fashion-MNIST Model-Family Comparison: OSF-Style Preregistration Record

> Author: **Dimo Dimov**
> 
> *Created as part of the SoftUni Deep Learning Course (July 2026).*

**Format:** [OSF Standard Preregistration](https://osf.io) template, adapted for a computational/secondary-data study
**Study:** *From Classical Baselines to Convolutional Networks, Transformers and Ensembles: A Controlled, Statistically Grounded Study of Fashion-MNIST Classification on a Single Consumer GPU*
**Source artifacts:** `DL-Fashion-MNIST.ipynb` (executable notebook) · `Fashion-MNIST_Scientific_Paper.pdf` (manuscript) · this file

> **What this document is.** This repository's notebook fixes every research question, hyperparameter and success criterion in two audited configuration objects (`CFG`, `CFGX`) and an explicit "success criteria defined *before* running anything" cell, **before** any model is trained: the notebook's own 0.4/0.5.2 and the manuscript's Appendix A are, in substance, a preregistration. This file reformats that a‑priori record into the field-by-field structure of the OSF Standard Preregistration template, so the design commitments are auditable independent of the narrative prose in the paper. It was assembled after the study had already been run (the repository already contains results), by transcribing the configuration values, research questions and success criteria that the source materials themselves state were fixed in advance: it is **not** a timestamped OSF registration with a DOI. Where the plan and the outcome differ, or where a question doesn't map cleanly onto a wet-lab/survey template, that is stated explicitly rather than smoothed over.
>
> A clearly-marked **post-hoc appendix** at the end reports what was actually achieved against each item below, in the spirit of a Registered Report's confirmatory results section.

---

## Table of contents

- [Metadata](#metadata)
- [Study Information](#study-information)
- [Design Plan](#design-plan)
- [Sampling Plan](#sampling-plan)
- [Variables](#variables)
- [Analysis Plan](#analysis-plan)
- [Other](#other)
- [Appendix: Preregistered vs. Achieved (post-hoc)](#appendix-preregistered-vs-achieved-post-hoc)

---

## Metadata

### Title
From Classical Baselines to Convolutional Networks, Transformers and Ensembles: A Controlled, Statistically Grounded Study of Fashion-MNIST Classification on a Single Consumer GPU

### Description
A single-seed, fully-scripted benchmark of fourteen model families (linear models, kernel SVMs, random forests, three gradient-boosting frameworks, an MLP, two CNN variants and a from-scratch Vision Transformer) plus four ensembling strategies on Fashion-MNIST, run end-to-end on one Colab T4 GPU. The study pairs an audited data-cleaning step (byte-exact train/test leakage removal) and an unsupervised geometric characterisation of the class structure with a full statistical treatment (McNemar, Cochran's Q, Holm correction, Wilson and bootstrap intervals) of every pairwise model comparison, and closes with a multi-method explainability audit of what the best models actually key on.

### Contributors
Dimo Dimov (sole author/analyst). Prepared as an exam project for the SoftUni Deep Learning Course (July 2026).

### License
*Not specified in the uploaded source materials.* Add the repository's chosen license here before publishing (e.g. MIT for the notebook/code, CC-BY-4.0 for the manuscript and this document).

### Subject
Machine Learning; Computer Vision; Image Classification; Applied/Computational Statistics; Explainable AI

### Tags
`fashion-mnist` `image-classification` `cnn` `vision-transformer` `gradient-boosting` `ensembling` `explainability` `reproducibility` `statistical-model-comparison` `data-leakage-audit`

---

## Study Information

### Hypotheses

The notebook pre-registers ten research questions (RQ1-RQ10: RQ1-RQ4 in the original edition, RQ5-RQ10 in the upgrade). Below, each is restated as a specific, directional, testable hypothesis, tied to the statistical test and/or numeric target that would confirm or disconfirm it. All targets below are the literal values fixed in the notebook's 0.4 and 0.5.2 cells before any model was fitted (reproduced in the manuscript as Appendix A, criteria A1-A5 and B1-B7).

- **H1 (RQ1: convolution vs. classical baselines).** A ~300k-parameter CNN will beat the strongest classical baseline (RBF-SVM) and every other classical family, reaching **≥ 0.92** test accuracy, with the gap confirmed by McNemar's test at *p* < .05. *(Criterion A3.)*
- **H2 (RQ2: inductive bias, not capacity, drives the gap).** Trained under an *identical* optimiser, LR schedule and seed to the CNN, an MLP with **more** parameters (~535k vs. ~300k) and no spatial prior will still underperform the CNN, reaching only **≥ 0.88** test accuracy: evidence that convolution itself, not raw capacity, explains H1. *(Criterion A2.)*
- **H3 (RQ3: nature of the residual error).** Post-training confusions will concentrate in the four upper-body classes (T-shirt/top, Pullover, Coat, Shirt); this pattern will be independently corroborated by unsupervised clustering (H5), the anomaly detectors (H6), and the labels of the leaked duplicate pairs found during cleaning: jointly supporting an irreducible-label-ambiguity account over a pure modelling-failure account.
- **H4 (RQ4: agreement with the literature and statistical meaningfulness).** Classical baselines will reproduce Xiao, Rasul & Vollgraf's (2017) published figures (LogReg ≈ 0.84, RBF-SVM ≈ 0.89, Random Forest ≈ 0.87) within ~1 percentage point despite using a fraction of the training data; roughly half of the pairwise gaps among top models will fall inside a ±0.5 pp practical-equivalence band once Wilson/paired-bootstrap uncertainty is accounted for. *(Criterion A1.)*
- **H5 (RQ5: unsupervised geometry predicts the confusion structure).** PCA/t-SNE/UMAP embeddings and *k*-means clustering (scored by trustworthiness, embedding k-NN accuracy, ARI/NMI, silhouette) will merge the four upper-body classes into a single structure, and this merge will persist across the full tuning sweep (t-SNE perplexities 5/30/50; UMAP neighbours 5/15/50) rather than appearing at one setting only.
- **H6 (RQ6: two anomaly detectors partially agree).** Isolation Forest and a convolutional-autoencoder reconstruction-error detector, run on the same PCA-50 embedding, will show positive but moderate rank agreement (Spearman ρ roughly in 0.3-0.5) and a top-1% Jaccard overlap several times above the chance expectation, with flagged images concentrated in the classes H3 identifies as hardest.
- **H7 (RQ7: gradient boosting closes but does not cross the classical ceiling).** XGBoost/LightGBM/CatBoost on PCA-80 features will reach **≥ 0.88** test accuracy (beating Random Forest) yet remain at or below the RBF-SVM figure from Xiao et al. (2017); Optuna tuning (25 trials / 600 s budget) will move LightGBM by well under 1 percentage point versus untuned defaults. *(Criterion B1.)*
- **H8 (RQ8: ViT underperforms CNNs at this scale).** A from-scratch, non-pretrained ViT-tiny (16 patch tokens, ~0.8M parameters) will reach **≥ 0.900** test accuracy but remain below both CNN variants, matching the small-data-regime prediction of Dosovitskiy et al. (2021) and Touvron et al. (2021). *(Criterion B3.)*
- **H9 (RQ9: ensembling yields a real, testable gain).** The best combiner (searched over soft/hard/weighted/stacked voting, pure-DL vs. hybrid ML+DL pools) will beat the best single model by **≥ 0.3 pp**, confirmed by McNemar's test at *p* < .05, with the corresponding paired-bootstrap 95% CI on the accuracy difference excluding zero; the hybrid ML+DL pool is expected to outperform the pure-DL pool. *(Criterion B4.)*
- **H10 (RQ10: explanations track genuine garment evidence).** Across Grad-CAM, Integrated Gradients, occlusion, SHAP and LIME, the majority of attribution mass will fall on garment pixels rather than background, and the mean Integrated-Gradients map will correlate positively with a model-free, per-pixel Fisher-discriminability map: i.e. the models are right for defensible reasons, not via a background shortcut. *(Criterion B6.)*

Two further pre-registered commitments are process/infrastructure rather than empirical hypotheses about the world, and are carried in the Analysis Plan and Other sections instead: **B5** (the inline unit-test suite must be 100% green before any conclusion above is treated as trustworthy) and **B7** (one configuration switch, `CFGX(fast_mode=True)`, must reproduce the full pipeline shape end-to-end in minutes).

---

## Design Plan

### Study type
- [ ] Experiment
- [ ] Observational Study
- [ ] Meta-Analysis
- [x] Other

None of OSF's three primary categories is an exact fit: there is no human/animal subject, no randomised treatment assignment, and no synthesis of *other people's* published effect sizes. The closest honest label is: **a computational benchmarking / model-comparison study on a fixed, existing public dataset (secondary data)**, structured as a repeated-measures design in which every "condition" (model family, or ensemble configuration) is fit and scored on the *identical* data splits, so that per-image outcomes can be paired across conditions for McNemar/bootstrap testing.

### Blinding
- [ ] No blinding is involved in this study.
- [ ] Human subjects will not know their treatment group.
- [ ] Personnel interacting with subjects are unaware of treatment. ("double blind")
- [ ] Personnel analyzing the data are unaware of the treatment applied.

None of the human-subject options apply as written (there are no subjects), so no box above is checked as literally true. The substantive analogue used throughout: described in full below: is **held-out discipline**, i.e. keeping the analyst (and every automated tuning procedure) blind to test-set outcomes until model selection is complete.

**Is there any additional blinding in this study?**
Yes, functionally: the official 10,000-image test set is read **exactly once per model, strictly after** all architecture choices, hyperparameters, epoch/checkpoint selection and ensemble weights/meta-learner coefficients are fixed using train/validation data only. Hyperparameter search (`GridSearchCV`, Optuna) is restricted to training-only cross-validation folds and never touches test. This is enforced mechanically, not just promised in prose: dedicated unit tests check label alignment between predictions and the untouched test labels, and one results registry accumulates every reported number so nothing is hand-typed into a table. **One disclosed exception:** the stacking meta-learner's weights and the deep models' epoch-selection checkpoint are both chosen using the *same* validation split (out-of-fold stacking was judged out of scope for the ~1 GPU-hour compute budget), which the design documents as a small, one-directional optimism risk confined to validation: headline numbers are still reported on the untouched test set.

### Study design

**Overall protocol.** A three-way, stratified, seed-fixed split: 90% of the cleaned training file for fitting, 10% for validation (epoch selection, early stopping, ensemble-weight/meta-learner fitting), and the official 10,000-image test set touched once per model after selection is complete. An MLP and a CNN are trained under one shared, identical trainer (`fit()`: AdamW, OneCycleLR, fixed epoch budgets, no early stopping) so that the MLP-vs-CNN comparison (H2) isolates architecture from everything else. A strict-superset trainer (`fit_v2()`: adds early stopping, pluggable schedulers, gradient clipping, resumable checkpoints) trains the residual CNN and the ViT under the same loaders, augmentation and seed.

**1. Data audit and cleaning** *(precedes any model fitting)*. An 11-point integrity report is run on the raw files:

1. Train array shape is (60000, 28, 28)
2. Test array shape is (10000, 28, 28)
3. No NaN values in train (structurally guaranteed by the `uint8` dtype)
4. All pixel values within [0, 255]
5. All labels within [0, 9]
6. Train is perfectly class-balanced (6,000 images/class)
7. Test is perfectly class-balanced (1,000 images/class)
8. Count of exact-duplicate images *inside* train (informational)
9. Count of exact-duplicate images *inside* test (informational)
10. **No image is byte-identical across the train/test boundary** (64-bit hash intersection, then re-verified by a full 784-pixel byte-exact comparison before anything is deleted)
11. No degenerate (constant-value) images

Only check 10 fails on the pristine Kaggle mirror (`zalando-research/fashionmnist`): ten images are byte-identical across the split, some with conflicting labels on their two copies. The cleaning policy (`CFGX.leakage_policy = "drop_from_train"`) removes these ten images from the **training** side only, leaving the official 10,000-image test set: the yardstick for every literature comparison: untouched. Two alternative policies (`drop_from_test`, `keep`) are implemented as switches but rejected by design: dropping from test would change the denominator used by every published comparison; keeping the leak would let high-capacity models memorise test images. A milder, in-training duplication (~43 pairs) is measured but retained by default (removing it would shift class priors and break comparability with published training-set sizes); a stricter opt-in switch (`CFGX.drop_train_duplicates`) exists for anyone who wants it.

**2. Exploratory characterisation** *(hypothesis-generating for H5/H6; see [Exploratory analysis](#exploratory-analysis))*. Per-class intensity distributions and ECDFs; a per-pixel Fisher discriminability ratio; the full 784×784 pixel-correlation matrix; PCA (2D/3D); t-SNE (perplexities {5, 30, 50}, 750 iterations) and UMAP (neighbours {5, 15, 50}, `min_dist`=0.1), both on a PCA-50 pre-reduction, scored by trustworthiness and 2D k-NN accuracy rather than eyeballed; *k*-means over *k* ∈ {5, 8, 10, 12, 15} on PCA-50, scored by silhouette (label-free) and ARI/NMI (label-aware, evaluation-only); Isolation Forest (300 trees, 1% contamination) and a convolutional autoencoder (latent 32, 8 epochs) run on the same PCA-50 embedding, compared by Spearman rank correlation and top-1% Jaccard overlap.

**3. Classical baselines**: always evaluated on the full, untouched 10,000-image test set:

| Model | Key hyperparameters | Fit subsample |
|---|---|---|
| Logistic Regression | multinomial, `lbfgs`, C = 0.1, `StandardScaler` | 12,000 (stratified) |
| Linear SVM | hinge loss, C = 0.01, one-vs-rest | 12,000 (stratified) |
| RBF SVM | C = 10, γ = `scale`, on PCA retaining 90% variance (~85 comps) | 12,000 (stratified) |
| Random Forest | 300 trees, `max_features='sqrt'`, unlimited depth | 12,000 (stratified) |
| Majority-class / uniform-random | trivial calibration baselines |: |

Tuning: 3-fold `GridSearchCV` over C ∈ {0.003 … 1.0} for Logistic Regression only (a smooth, convex, one-dimensional problem).

**4. Gradient boosting and automated tuning:**

| Model | Rounds/iterations | Key hyperparameters | Features | Fit subsample |
|---|---|---|---|---|
| XGBoost | 600 | lr 0.15, depth 6, subsample 0.9, colsample 0.8, λ = 1.0, histogram trees, GPU | PCA-80 | 20,000 (stratified) |
| LightGBM | 800 | 63 leaves, lr 0.1, bagging | PCA-80 (+ raw-784 control) | 20,000 (stratified) |
| CatBoost | 800 | depth 6, lr 0.15, `l2_leaf_reg` 3, ordered boosting | PCA-80 | 20,000 (stratified) |

Tuning: Optuna TPE, 25 trials **or** 600 s wall-clock (whichever comes first), 3-fold stratified CV on an 8,000-image tuning subset, over 8 LightGBM hyperparameters; the winning configuration is refit once on the full 20,000-image subset. A raw-pixel (784-feature) LightGBM control checks whether the PCA-80 compression is actually free.

**5. Deep architectures**, all under seed 42, AdamW, batch 256/512 (train/eval), weight decay 5e-4, label smoothing 0.05, FP16 mixed precision:

| Model | Params | Epochs | Trainer | LR schedule | Augmentation | Extra regularisation | Early stopping |
|---|---|---|---|---|---|---|---|
| MLP (784→512→256→10) | ~535k | 20 (fixed) | `fit()` | OneCycle, peak 3e-3 | none | BatchNorm, Dropout 0.30 | no |
| CNN (VGG-style, 3×[3×3 conv-BN-ReLU]×2, GAP) | ~300k | 25 (fixed) | `fit()` | OneCycle, peak 3e-3 | h-flip p=.5, ±2px shift (GPU, train-only) | BatchNorm, Dropout, GAP head | no |
| ResNet-small (3 stages, 2 basic residual blocks/stage, width 32) | ~0.7M | ≤ 30 | `fit_v2()` | OneCycle, peak 3e-3 | same as CNN | + grad-clip norm 1.0 | yes (patience 8, min Δ 1e-4) |
| ViT-tiny (patch 7×7 → 16 tokens + CLS, dim 128, depth 6, heads 4, MLP ratio 2.0) | ~0.8M | ≤ 35 | `fit_v2()` | Cosine, peak 1e-3 | same as CNN | Dropout 0.10, DropPath 0→0.1 linear, grad-clip 1.0 | yes (patience 8, min Δ 1e-4) |

Augmentation (random horizontal flip + ≤±2px translation) is justified by the EDA finding of a wide, always-black border and left-right garment symmetry, and is applied to training batches only.

**6. Ensembling.** Probability matrices are collected from every trained model, with mirror test-time augmentation (original + horizontal-flip logits averaged pre-softmax) for deep members. Two member pools:

- **Pure-DL pool** (4 members): MLP, CNN, ResNet-small, ViT-tiny.
- **Hybrid ML+DL pool** (6 members): the 4 deep members plus the top-2 *non*-deep models (classical or boosting) by **validation** accuracy: resolved at runtime, not hard-coded, so the exact pair can vary.

| Combiner | Pure-DL pool | Hybrid ML+DL pool |
|---|:---:|:---:|
| Soft voting (equal weight; no fitted parameters) | ✓ |: |
| Hard/majority voting (control; discards confidence) | ✓ |: |
| Weighted soft voting (random Dirichlet search, 4,000 draws, on validation) | ✓ | ✓ |
| Stacking (multinomial logistic meta-learner, C = 1.0, fit on validation) | ✓ | ✓ |

An **oracle** diagnostic (fraction of test images on which *at least one* pool member is correct) is computed for context: it is an upper bound, not a deployable combiner. All combiner parameters are fit on the validation split only, never on test.

**7. Explainability**, applied to one held-out test image per class:

| Method | Applied to | Key settings |
|---|---|---|
| Grad-CAM | CNN, ResNet (last conv feature maps) |: |
| Integrated Gradients | all deep models | 64 Riemann steps, all-black baseline, completeness checked numerically |
| Occlusion sensitivity | CNN, ResNet | 7×7 patch, stride 2, probability-drop measured |
| SHAP (`GradientExplainer`) | deep models | 64 background images |
| LIME | deep models | explicit SLIC super-pixel segmentation (default quickshift collapses a 28×28 thumbnail into one segment) |
| Attention rollout | ViT only | head-averaged, residual-corrected, CLS row |

A quantitative faithfulness check aggregates mean \|Integrated Gradients\| per class and reports (i) the share of attribution mass on garment pixels (intensity > 20) and (ii) the Pearson correlation between that map and the model-free Fisher-discriminability map from step 2.

**8. Verification.** An inline suite of 28 `assert`-based test functions is executed as part of the pipeline itself (not a separate CI job), covering four areas:

- *Data/tensor contracts (11):* `test_raw_arrays_have_expected_shape_and_dtype`, `test_pixel_value_range`, `test_split_sizes_and_stratification`, `test_leakage_removed`, `test_cleaning_preserved_the_official_test_set`, `test_no_train_test_leakage`, `test_train_val_are_disjoint`, `test_tensor_dataset_shapes_and_dtypes`, `test_normalisation_statistics`, `test_dataloader_batch_shapes`, `test_augmentation_contract`
- *Model contracts (6):* `test_model_output_dimensions`, `test_parameter_counts_are_sane`, `test_initial_loss_matches_uniform_prediction`, `test_single_batch_overfitting`, `test_trained_models_beat_chance`, `test_predict_logits_alignment`
- *Ensembling/attribution contracts (6):* `test_probability_matrices_are_distributions`, `test_soft_vote_identities`, `test_ensemble_weights_on_simplex`, `test_gradcam_output_contract`, `test_integrated_gradients_completeness`, `test_metrics_match_manual_computation`
- *Registry/reproducibility (5):* `test_results_registry_is_consistent`, `test_set_seed_is_reproducible`, `test_pca_feature_spaces_are_consistent`, `test_artifacts_were_written`, `test_best_models_persisted_to_per_family_folders`

*(The manuscript's 3.11 narrative says "twenty-seven test functions"; the executed suite, the abstract, and Appendix A/B all report 28/28. This document follows the code: 28 distinct `def test_...` functions are defined in the notebook: as the authoritative count.)*

### Randomization
No subjects are randomly assigned to a treatment condition (there is no equivalent of arms/blocks here), so no randomization scheme is registered in the classic sense. For completeness, every place randomness genuinely enters the pipeline is: `random`, NumPy, Torch (CPU) and Torch CUDA generators are seeded with **42**; cuDNN runs in deterministic mode (`torch.backends.cudnn.deterministic = True`, `benchmark = False`); the train/validation split and every classical/boosting fit subsample are **stratified random** samples drawn under that seed; ensemble weights are searched by **random Dirichlet sampling** (4,000 draws) over the validation set only. Residual non-determinism from CUDA atomic operations is documented as confined to roughly the fourth decimal place of reported metrics.

---

## Sampling Plan

### Existing data
- [ ] Registration prior to creation of data
- [ ] Registration prior to any human observation of the data
- [ ] Registration prior to accessing the data
- [x] Registration prior to analysis of the data
- [ ] Registration following analysis of the data

### Explanation of existing data
Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017) is a long-established, heavily published-on public benchmark, obtained here via `kagglehub` from the mirror `zalando-research/fashionmnist`. As of fixing the research questions and success-criteria targets, **no analysis specific to this pipeline** had been performed: the ten leaked duplicate pairs, the cluster/manifold structure, the anomaly rankings, and every model's accuracy were all unknown quantities at that point, genuinely determined only once the notebook ran. However, in the interest of the same transparency preregistration is meant to enforce, one nuance is disclosed rather than hidden: several numeric targets (criterion A1's classical-baseline ranges, in particular) are explicitly anchored to figures **already published by other researchers** on this same public dataset (Xiao et al., 2017; Bhatnagar et al., 2017; Zhong et al., 2020). That is standard and expected in benchmarking research: the whole point of comparing against known baselines: and is distinct from the kind of private "peeking" at this specific run's own results that preregistration exists to rule out.

### Data collection procedures
Data are acquired programmatically: no manual downloads: via `kagglehub.dataset_download("zalando-research/fashionmnist")`, retrieving `fashion-mnist_train.csv` and `fashion-mnist_test.csv` (~140 MB combined). Each CSV row holds one integer label (0-9) and 784 integer pixel columns (0-255) in row-major flattening of a 28×28 image; garments are rendered as light objects on a black background. Pixels are parsed with pandas and immediately downcast to `uint8` (full dataset ≈ 55 MB in that dtype, ≈ 220 MB as `float32`), comfortably inside the 12.7 GB system RAM of the target Colab T4 runtime; intermediate `int64` pandas DataFrames (the real peak, ≈ 1.1 GB) are freed immediately after downcasting. The 11-point integrity audit and the byte-exact leakage-removal procedure described under [Study design](#study-design) run as the final step of "collection," before any tensor conversion.

### Sample size
| Component | Size | Note |
|---|---|---|
| Raw dataset (pre-cleaning) | 70,000 images | 60,000 train + 10,000 test |
| Post-cleaning split: train (fit) | 53,991 images | stratified 90% of the cleaned 59,990-image training file |
| Post-cleaning split: validation | 5,999 images | stratified 10% of the cleaned training file |
| **Official test set (unchanged)** | **10,000 images** | untouched by cleaning; read once per model |
| Classical-baseline fit subsample | 12,000 images | stratified; always evaluated on the full 10,000-image test set |
| Boosting-model fit subsample | 20,000 images | stratified; PCA-80 features |
| Hyperparameter-tuning subsample | 8,000 images | 3-fold CV inside Optuna/GridSearchCV |
| Heavy-EDA statistics sample | 8,000 images | intensity/Fisher/correlation analyses |
| Manifold-learning sample (t-SNE/UMAP) | 4,000 images | after PCA-50 pre-reduction |
| Clustering sample | 6,000 images | *k*-means on PCA-50 |
| Ensemble weight search | 4,000 Dirichlet draws | over the validation-set probability simplex, not an image count |
| Paired-bootstrap resampling | 2,000 resamples | of the 10,000-image test set, models kept paired |

### Sample size rationale
The overriding constraint is a single Colab T4 GPU with 2 vCPUs and a self-imposed ≈ 1 GPU-hour deep-learning budget (full run ≈ 75-110 minutes; `fast_mode` smoke test ≈ 8-12 minutes). An RBF-SVM is O(*n*²)-O(*n*³) and would take hours on 2 vCPUs at the full 60k scale, while learning curves for the classical models are documented as nearly flat past ~10k images: so a 12,000-image stratified subsample is judged to cost negligible accuracy for a large runtime saving; boosting frameworks scale better and are given 20,000 images. For the confirmatory test-set comparisons, the sample size is not chosen by a power analysis but is fixed by the dataset itself (the official 10,000-image test set); the notebook instead adopts an explicit **precision ruler**: at an accuracy near *p* ≈ 0.93 with *n* = 10,000, SE = √(p(1−p)/n) ≈ 0.26 pp, so 95% intervals span roughly ±0.5 pp: this figure governs how every comparison in the Analysis Plan is interpreted.

### Stopping rule
Not applicable to data collection in the traditional sense: Fashion-MNIST is a fixed-size public release, not sequentially gathered. The closest pre-specified stopping rules govern **iterative model training and search**, and are fixed in advance rather than chosen post hoc: MLP and CNN train for a **fixed** epoch budget (20 and 25 respectively) with no early stopping: deliberately, to keep the controlled H2 comparison as clean as possible; ResNet and ViT use **early stopping** (patience 8 epochs, `min_delta` 1e-4) with up to 30/35 epochs as a ceiling; the Optuna hyperparameter search stops at **25 trials or 600 seconds**, whichever comes first.

---

## Variables

### Manipulated variables
The primary manipulated factor is **model family/configuration**. Fourteen model-family entries appear on the leaderboard: Logistic Regression, Linear SVM, RBF SVM, Random Forest, XGBoost, LightGBM (PCA-80, plus a raw-784-feature control), CatBoost, MLP, CNN, ResNet-small, ViT-tiny, and the trivial majority-class/uniform-random calibration baselines. A second, crossed factor governs the ensembling sub-study: **combiner strategy** (soft / hard / weighted / stacked voting) × **member pool** (pure-DL, 4 members; hybrid ML+DL, 6 members): see the table under [Study design](#study-design) for exactly which of the 8 possible cells were run (6 were; hybrid soft/hard voting were not separately evaluated). A distinct, tightly **controlled** sub-comparison isolates a single manipulated variable: presence vs. absence of convolutional inductive bias (CNN vs. MLP): while holding the optimiser, LR schedule and seed identical across both arms; this is the only part of the study with the character of a genuine controlled experiment rather than an observational comparison across pre-existing model families.

### Measured variables
| Category | Variables |
|---|---|
| Headline performance | top-1 accuracy (with Wilson 95% CI), macro-F1, weighted-F1, top-2 accuracy, per-class precision/recall/F1, row-normalised confusion matrix |
| Cost | fit wall-clock time, prediction wall-clock time, trainable-parameter count |
| Pairwise / ensemble diagnostics | error-overlap (Jaccard) matrix between model pairs, pairwise disagreement rate, oracle accuracy (≥ 1 member correct) |
| Unsupervised geometry (exploratory) | PCA cumulative explained variance, embedding trustworthiness, 2D-embedding k-NN accuracy, silhouette score, Adjusted Rand Index, Normalised Mutual Information |
| Anomaly detection (exploratory) | Isolation Forest anomaly score, autoencoder reconstruction MSE, Spearman rank correlation between the two, top-1% Jaccard overlap vs. chance expectation, per-class anomaly rate |
| Explainability | per-method attribution/saliency map (Grad-CAM, Integrated Gradients, occlusion Δ-probability, SHAP value, LIME weight, attention-rollout weight), share of attribution mass on garment pixels, Pearson correlation with the Fisher-discriminability map, Integrated-Gradients completeness error |
| Data quality (confirmatory input) | count of byte-identical train/test pairs, count of in-train duplicate pairs, count of degenerate images |

### Indices
- **Standard error / precision ruler:** SE = √(p(1−p)/n), evaluated at n = 10,000, used to derive the ±0.5 pp "practically indistinguishable" band referenced throughout the Analysis Plan.
- **macro-F1** = unweighted mean of the 10 per-class F1 scores; **weighted-F1** = the same, weighted by class support (support is exactly equal across classes here, so the two indices are expected to coincide closely by construction).
- **Oracle accuracy** = the fraction of test images correctly classified by *at least one* member of an ensemble pool: an achievable-upper-bound index, not itself a deployable prediction rule.
- **Krogh-Vedelsby ambiguity decomposition:** E_ens = Ē − Ā (mean member error minus inter-member ambiguity): used, not to compute a headline number, but to *interpret* why a diverse hybrid pool outperforms a more accurate-but-redundant pure-DL pool.
- **"Ink coverage"** = the fraction of an image's pixels with intensity > 20: used both as an EDA descriptor and as the same threshold defining "garment pixel" in the explainability faithfulness check.

---

## Analysis Plan

### Statistical models
Four procedures are applied with increasing strictness, exactly as fixed in advance:

1. **Wilson score intervals** (95%) for every single-model accuracy.
2. **Paired bootstrap** (2,000 resamples of the 10,000-image test set, resampling indices with replacement, all models kept paired on identical resampled images) for the distribution of every pairwise accuracy difference; the difference is treated as real only if the resulting percentile CI excludes zero.
3. **McNemar's test** for every model pair, computed two ways: the continuity-corrected χ² statistic and the exact binomial p-value on the discordant-pair 2×2 table: cross-checked against `statsmodels`' implementation.
4. **Cochran's Q** omnibus test across the full leaderboard (and, separately, restricted to the top five models), to test whether all models are equivalent before running ~91+ pairwise tests; followed by **Holm-Bonferroni** step-down correction (α = 0.05 family-wise) applied to the complete family of pairwise McNemar tests.

Model-selection/tuning procedures are pre-specified alongside the confirmatory tests, though they are search procedures rather than hypothesis tests: 3-fold `GridSearchCV` over Logistic Regression's C axis; Optuna TPE search (25 trials / 600 s, 3-fold stratified CV, 8,000-image subset) over 8 LightGBM hyperparameters, winner refit once.

Each hypothesis in [Study Information](#study-information) maps onto a specific test: H1/H2 → McNemar on CNN-vs-RBF-SVM and CNN-vs-MLP; H4 → literature deltas interpreted through Wilson/paired-bootstrap intervals; H7 → boosting-vs-Random-Forest and boosting-vs-RBF-SVM McNemar tests; H9 → McNemar plus paired bootstrap on best-ensemble-vs-best-single-model; H3/H5/H6/H10 are addressed descriptively (see [Exploratory analysis](#exploratory-analysis)) rather than by a single confirmatory test.

### Transformations
Pixel values are scaled to [0, 1] and then standardised using **training-split-only** statistics (mean = 0.286, std = 0.353), applied unchanged to validation and test so no information leaks across splits. Three separate PCA configurations are used, each fit on training rows only: (a) retain 90% of variance (~85 components) for the RBF-SVM; (b) a fixed 80 components for the three boosting frameworks, with a raw-784-feature LightGBM control kept specifically to test whether that compression is actually free; (c) a 50-component pre-reduction ahead of t-SNE/UMAP/clustering/anomaly detection (standard practice: denoises and speeds up neighbour search). Labels require no recoding (already integer class indices 0-9, single-label). GPU-batched data augmentation (random horizontal flip p = 0.5; random translation ≤ ±2 px) is applied to CNN/ResNet/ViT **training** batches only, justified by the EDA finding of a wide always-black border and left-right garment symmetry; the MLP is trained without augmentation. Label smoothing (0.05) is applied to the cross-entropy loss for every deep model.

### Inference criteria
- α = 0.05 throughout; family-wise after Holm correction for the pairwise McNemar family.
- **Paired over marginal:** where a marginal Wilson-interval comparison and the paired McNemar/bootstrap test disagree, the paired, conditional test takes precedence: pre-specified because two models can have overlapping marginal intervals while still differing significantly on the (much smaller, more informative) set of images where they disagree.
- A pairwise gap below **~0.5 percentage points** on the 10,000-image test set is treated as practically indistinguishable **unless** the paired test says otherwise.
- A claimed ensembling gain (H9) is accepted only if **both** conditions hold: the gain is ≥ 0.3 pp over the best single model, **and** McNemar's test on that pair returns *p* < .05. Meeting only one is not sufficient.
- **A pre-registered gate on drawing any conclusion at all:** all 28 assertions in the inline unit-test suite must pass. A failure or error is defined, in advance, to invalidate downstream claims: not to be footnoted and ignored. *(Criterion B5.)*
- Non-significant pairs after Holm correction are reported as "not significantly different," not as directional wins; specific pairs are *expected in advance* to land here (e.g. the three boosting frameworks against each other; weighted voting vs. stacking; two CNN variants within ~0.4 pp of each other): disclosing this expectation up front is itself part of the pre-registered plan.

### Data exclusion
- **Byte-identical train/test pairs (~10):** re-verified byte-exactly (not merely by hash) before deletion; removed from the **training** side only. Two alternative policies (drop from test; keep) are implemented but rejected: see [Study design](#study-design) for the stated rationale for each.
- **In-training duplicate pairs (~43):** retained by default (removing them would shift class priors and break comparability with published training-set sizes); an opt-in stricter switch exists but defaults to off. The number of pairs that straddle the train/validation boundary is measured and reported, not silently absorbed.
- **Flagged anomalies** (Isolation Forest top-1%, autoencoder top-1%): explicitly **not** excluded from any split. They are treated as legitimate rare product photographs rather than errors; their per-class rate is reported as a diagnostic rather than acted on by deletion, because removing them would break comparability with the untouched official benchmark.
- **True outlier/exclusion rule for the confirmatory analyses:** none beyond the leakage removal above: every remaining image, however unusual, is scored by every model.

### Missing data
None. The integrity audit's "No NaN in train" check passes structurally (pixel values are stored as `uint8`, which cannot represent NaN), and no field in either CSV is ever empty. No imputation logic exists anywhere in the pipeline because none is needed.

### Exploratory analysis
This study explicitly separates a confirmatory arm (model-vs-model accuracy comparisons, tested by the McNemar/Cochran's-Q/Holm machinery above) from an exploratory, hypothesis-generating arm, which is **not** subjected to that same battery of tests:

- All unsupervised characterisation: intensity distributions, the Fisher discriminability map, pixel-correlation structure, PCA/t-SNE/UMAP manifold learning, *k*-means clustering, and the two anomaly detectors: is exploratory. It **motivated** H5 and H6 in advance, but the specific patterns it surfaces (the silhouette-maximising *k*, exactly which images the detectors flag, the precise cluster boundaries) are reported descriptively rather than as hypothesis tests with a p-value.
- Post-hoc qualitative inspection of confident misclassifications (e.g. identifying which errors a human annotator might plausibly also make) is interpretive, not statistical.
- The accuracy-per-unit-of-compute framing and the ranked "highest-return future work" list are discussion-level and explicitly not preregistered confirmatory claims.
- Any additional pattern noticed while producing the 45+ figures in the notebook that is not tied to H1-H10 above is, by the same logic, exploratory: it may be reported, but must be labelled as such rather than folded into the confirmatory leaderboard claims.

---

## Other

### Other
- **Reproducibility infrastructure** (relevant to criterion B7 and to auditing this document itself): every tunable value lives in one of two dataclasses, `CFG` (original edition) and `CFGX` (upgrade), under the stated rule "no magic numbers anywhere else in the notebook"; `random`/NumPy/Torch/CUDA are seeded at 42 with deterministic cuDNN; a single switch, `CFGX(fast_mode=True)`, shrinks every expensive block (fewer Optuna trials, fewer epochs, smaller manifold-learning samples) so a reviewer can validate the entire pipeline's *shape* in ≈ 8-12 minutes before committing to the ≈ 75-110-minute full run; every optional third-party dependency is probed before use, so a missing package degrades a section into a printed explanation rather than an exception; every trained model is persisted with a JSON metadata sidecar and a `BEST.json` pointer per model family.
- **Primary literature anchors:** Xiao, Rasul & Vollgraf (2017): the dataset and its official baseline table; Bhatnagar, Ghosal & Kolekar (2017): early CNN results on Fashion-MNIST; Zhong et al. (2020) and Tanveer, Khan & Kang (2021): the published state-of-the-art ceiling (WRN-28-10, DARTS-based NAS) this study calibrates against without attempting to chase. The manuscript's own reference list (48 sources) is the complete bibliography; it is not reproduced here.
- **Relationship to a formal OSF registration:** this file mirrors the *structure* of the OSF Standard Preregistration template for auditability and version control on GitHub. It has not been submitted to OSF and carries no timestamp/DOI.
- **Companion files in this repository:** `DL-Fashion-MNIST.ipynb` (source notebook: `CFG`/`CFGX`, the 28-test suite, and the "success criteria fixed before running anything" cells all live here) and the manuscript (`Fashion-MNIST Scientific Paper.pdf`) this record accompanies.

---

## Appendix: Preregistered vs. Achieved (post-hoc)

*Everything above this line was fixed before the run. Everything below reports what actually happened, added after the fact: clearly separated so the preregistration itself stays uncontaminated by hindsight.*

### Original-edition criteria (A1-A5)

| # | Criterion | Target | Achieved | Status |
|---|---|---|---|---|
| A1 | Classical baselines reproduce the official benchmark | within ~1 pp (LogReg ≈ 0.84, RBF-SVM ≈ 0.89, RF ≈ 0.87) | LogReg 0.838, RF 0.861, RBF-SVM 0.879 (recorded run) | ✅ MET |
| A2 | MLP test accuracy | ≥ 0.88 | 0.911 (recorded run) | ✅ MET |
| A3 | CNN test accuracy | ≥ 0.92 | 0.942 (recorded run) | ✅ MET |
| A4 | Test set touched exactly once per model, after selection |: | Enforced by protocol + unit tests | ✅ MET |
| A5 | Full notebook within budget on a T4 | < 30 min (v1 scope) | met | ✅ MET |

### Upgraded-edition criteria (B1-B7)

| # | Criterion | Target | Achieved | Status |
|---|---|---|---|---|
| B1 | Best gradient-boosting baseline | ≥ 0.88 | 0.88-0.90 range; 0.8995 for the raw-784 LightGBM control (recorded run) | ✅ MET |
| B2 | Residual CNN (ResNet-small) | ≥ 0.930 | 0.935-0.940 typical; 0.9509 recorded run | ✅ MET |
| B3 | Vision Transformer (from scratch) | ≥ 0.900 | 0.905-0.920 typical; 0.8897 recorded run | ✅ MET (typical-range criterion; the single recorded run sits just under it: see note below) |
| B4 | Best ensemble | ≥ +0.3 pp over best single, **and** McNemar *p* < .05 | +0.4 to +1.7 pp, *p* ≪ .05 | ✅ MET |
| B5 | Unit-test suite: 100% pass before any conclusion is drawn | 28/28 | 28/28 | ✅ MET |
| B6 | Explainability: mass concentrated on garment, not background |: | 80-90% of attribution mass on garment pixels | ✅ MET |
| B7 | Reproducibility: one switch runs a full smoke test | 8-12 min | met | ✅ MET |

*Note on B3: the manuscript reports two different numbers for the ViT: a "typical range across development runs" of 0.905-0.920 (4.5) and a specific single-seed-42 recorded run of 0.8897 (4.10's leaderboard chart). Both are reproduced above rather than silently reconciled; the criterion is worded against the typical range, which clears the ≥ 0.900 bar, while the one exact recorded run in the chart falls just under it: a reminder of exactly the kind of single-seed variability the manuscript's own Limitations section (6.1) flags.*

### Headline leaderboard (recorded run, official 10,000-image test set)

| Model | Accuracy | vs. published reference |
|---|---|---|
| Hybrid ML+DL weighted voting (best overall) | 0.9520 | ≈ 1.2 pp below WRN-28-10 + Random Erasing (96.35%, Zhong et al. 2020) |
| ResNet-small (best single model) | 0.9509 |: |
| DL soft voting (equal weights) | 0.9438 |: |
| CNN (VGG-style, GAP) | 0.9420 | vs. Bhatnagar et al.'s CNN+BN+skip, 92.54% |
| MLP (control condition) | 0.9107 |: |
| ViT-tiny | 0.8897 |: |
| RBF SVM (PCA-90%) | 0.8791 | vs. Xiao et al.'s 89.70% (full data) |
| Random Forest (300 trees) | 0.8611 | vs. Xiao et al.'s 87.30% (100 trees) |
| Logistic Regression | 0.8379 | vs. Xiao et al.'s 84.20% |
| Linear SVM | 0.8344 | vs. Xiao et al.'s 83.60% |
| *Published ceiling: DARTS-based NAS (Tanveer et al., 2021)* | *0.9691* | *not attempted here by design: see 5.2/7 accuracy-per-compute discussion* |

*Full per-model figures, confidence intervals and the complete Holm-corrected pairwise table are in `artifacts/model_comparison.csv` and the manuscript's 4.9-4.10, not reproduced in full here.*

### Hypothesis-by-hypothesis outcome

| Hypothesis | Outcome |
|---|---|
| H1 (convolution beats classical) | Confirmed: CNN 0.942 vs. RBF-SVM 0.879, McNemar *p* ≪ .05 |
| H2 (inductive bias, not capacity) | Confirmed: larger MLP (535k params) still underperforms the smaller CNN (300k params) |
| H3 (residual error is upper-body / label ambiguity) | Confirmed by convergent evidence (clustering, anomaly detectors, leaked-pair label conflicts, per-class F1) |
| H4 (agrees with literature, statistically meaningful) | Confirmed within ~1-1.5 pp; roughly half of top-model pairwise gaps fall inside the ±0.5 pp band |
| H5 (geometry predicts confusions) | Confirmed: upper-body merge persists across the full t-SNE/UMAP sweep; ARI ≈ 0.35-0.42 |
| H6 (detectors partially agree) | Confirmed: Spearman ρ ≈ 0.3-0.5, top-1% overlap several× chance |
| H7 (boosting closes but doesn't cross the ceiling) | Confirmed: 0.88-0.90 band, ≤ RBF-SVM |
| H8 (ViT below both CNNs) | Confirmed on both the typical range and the recorded run |
| H9 (ensembling gain is real) | Confirmed: +0.4 to +1.7 pp, McNemar *p* ≪ .05, bootstrap CI excludes zero |
| H10 (explanations track garment evidence) | Confirmed: 80-90% attribution mass on garment; IG-vs-Fisher-map ρ ≈ 0.55-0.75 |

---
*Created as part of the SoftUni Deep Learning Course (July 2026).*