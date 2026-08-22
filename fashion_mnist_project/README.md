# Fashion-MNIST Deep Learning Project (Restructured Edition)

## Project Structure

```
fashion_mnist_project/
├── README.md                          ← this file
├── restructure.py                     ← script that generated this layout
├── DL-Fashion-MNIST_extracted.txt     ← monolithic original notebook as a text extraction.
├── notebooks/                         ← split Jupyter notebooks (run in order)
│   ├── 00_introduction.ipynb          # Problem statement, motivation, RQs
│   ├── 01_data_loading.ipynb          # Environment, imports, download, cleaning, splitting
│   ├── 02_eda_basic.ipynb             # Class distribution, samples, pixel stats, PCA
│   ├── 03_eda_advanced.ipynb          # Intensity, Fisher map, t-SNE/UMAP, clustering, outliers
│   ├── 04_classical_ml.ipynb          # Logistic Regression, SVMs, RF, XGBoost, Optuna
│   ├── 05_deep_learning.ipynb         # MLP, CNN, ResNet-small, Vision Transformer
│   ├── 06_ensembling.ipynb            # Soft voting, weighted voting, stacking, hybrid ML+DL
│   ├── 07_explainability.ipynb        # Grad-CAM, IG, occlusion, SHAP, LIME, attention roll-out
│   ├── 08_unit_tests.ipynb            # 28-assertion test suite
│   ├── 09_statistical_validity.ipynb  # McNemar, Holm, bootstrap, Cochran's Q
│   └── 10_comparison_conclusion.ipynb # Literature comparison, discussion, references, appendix
│
└── src/                               ← reusable Python modules (imported by notebooks)
    ├── __init__.py
    ├── config.py                      # CFG & CFGX dataclasses, set_seed(), DEVICE, optional-dep probes
    ├── data_loading.py                # Kaggle download, CSV parsing, integrity checks, leakage removal,
    │                                  #   tensor conversion, DataLoader factory, flat-array helpers
    ├── eda.py                         # All EDA functions: distributions, intensity, Fisher map,
    │                                  #   PCA/t-SNE/UMAP, clustering, Isolation Forest, autoencoder outliers
    ├── models.py                      # Neural network architectures: MLP, CNN, ResNetSmall,
    │                                  #   VisionTransformer, ConvAutoencoder, Augment, count_parameters()
    ├── training.py                    # Training loops, evaluate_predictions(), predict_logits(),
    │                                  #   torch_probabilities(), reconstruction_errors(), plot_confusion()
    ├── explainability.py              # Grad-CAM, Integrated Gradients, occlusion, SHAP, LIME,
    │                                  #   attention roll-out, attribution_faithfulness()
    ├── statistics_utils.py            # Wilson CI, McNemar, Holm-Bonferroni, paired bootstrap,
    │                                  #   Cochran's Q, final_leaderboard(), literature comparison plots
    └── notebook_tests.py              # SkipTest, run_test_suite(): minimal test runner
```

## How to Run

### Prerequisites

This project is designed for **Google Colab** with a T4 GPU, but works on any machine with:

```
Python ≥ 3.10
torch, torchvision
scikit-learn, xgboost, lightgbm, catboost, optuna
umap-learn, shap, lime, statsmodels
pandas, matplotlib, seaborn, scipy
kagglehub, tqdm
```

### Running the Notebooks

1. Open the notebooks in order (00 → 10) in Jupyter or Google Colab
2. Each notebook automatically adds `src/` to `sys.path` so it can import from the modules
3. The notebooks contain all the original markdown narrative + the executable code cells
4. Functions and classes are defined in `src/*.py` and imported where needed

### Quick Smoke Test

In `01_data_loading.ipynb`, change the configuration to:
```python
cfg = CFG(sk_train_subset=3_000, run_rbf_svm=False, mlp_epochs=3, cnn_epochs=3)
cfgx = CFGX(fast_mode=True)
```
This reduces runtime to ~8-12 minutes on a T4.

## Module Dependency Graph

```
config.py                   (no internal deps)
    ↑
    ├── data_loading.py     (no internal deps)
    ├── models.py           (no internal deps)
    ├── eda.py              (no internal deps)
    ├── statistics_utils.py (no internal deps)
    ├── notebook_tests.py   (no internal deps)
    │
    ├── training.py         → config, models
    └── explainability.py   → config
```

No circular imports. Leaf modules (no internal dependencies) can be imported in any order.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Two configuration dataclasses** (`CFG`, `CFGX`) | v1 results stay bit-for-bit reproducible; v2 additions are cleanly separated |
| **Functions extracted to `.py` modules** | Code is reusable, testable with `pytest`, and notebook cells stay focused on orchestration |
| **Each notebook adds `src/` to `sys.path`** | Notebooks remain self-contained; no `pip install -e .` required |
| **Optional dependencies probed, not assumed** | Missing packages degrade a section into a printed explanation, never an error |
| **Test suite as plain `test_*` functions** | Same code runs in notebooks and under `pytest` |

## Original Notebook

The monolithic original notebook (`DL-Fashion-MNIST.ipynb`, 223 cells, ~416 KB) is preserved
in `DL-Fashion-MNIST_extracted.txt` as a text extraction. All of its content is
distributed across the 11 notebooks and 8 Python modules above with no loss of functionality.
