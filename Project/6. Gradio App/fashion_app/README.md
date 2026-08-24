# Fashion-MNIST Inference Studio: Gradio web app

A Gradio front-end for the models exported by **`DL-Fashion-MNIST.ipynb`**
(the finished project that writes its trained models into `artifacts/models/`).

The app implements exactly the four required behaviours:

| # | Requirement | How it is implemented |
|---|---|---|
| **1** | User specifies the path to the folder with the model artifacts | Text box → *Load models*. Accepts the `artifacts` folder **or** the `artifacts/models` folder (auto-detected). Scans `ml/*.joblib`, `dl/*.pt`, `ensemble/*.joblib` + their JSON sidecars and shows a manifest (family, file, status, the notebook's test accuracy). |
| **2** | User specifies the path to the folder with sample photos | Text box → *Scan folder* (sub-folders included), **or** direct browser upload. jpg / png / bmp / webp / gif / tiff. |
| **3** | Photos are processed into the Fashion-MNIST format and given to the models | Every photo → grayscale → auto-inversion (garment white on black) → contrast normalisation → centre-crop + square pad → resize **28×28** → `uint8`. Deep models then get `(px/255 - 0.2860)/0.3530` (notebook §1.6); sklearn pipelines get the flattened `px/255` 784-vector they were fitted on. **Every usable model** (classical, boosting, MLP/CNN/ResNet/ViT, and the ensembles) predicts, with optional mirror TTA (`cfgx.tta`). |
| **4** | Export processed photos + model predictions as `.zip` | One click → `exports/fashion_mnist_export_<timestamp>.zip` (see contents below), offered as a browser download. |

---

## Run it

```bash
pip install -r requirements.txt
python app.py                 # → http://localhost:7860
```

Then in the UI:

1. **Models folder**: point at the `artifacts` folder exported from the notebook
   (the one containing `models/ml`, `models/dl`, `models/ensemble`) and press *Load models*.
2. **Photos folder**: point at your sample photos (or upload them).
3. Press **Process photos & run all models** – processed 28×28 images + a
   predictions table (class · confidence for every model × every image) appear;
   the radio button under the table shows a per-model top-3 breakdown for any single image.
4. Press **Export … (.zip)**: the zip is served for download.

### Using the real notebook export (from Google Colab)

```python
# in Colab, after the notebook has run:
!zip -r artifacts.zip artifacts
from google.colab import files; files.download("artifacts.zip")
```

Unzip locally and enter the extracted `artifacts` path in the app.

**ZIP contents**

```
processed_28x28/<name>.png        the processed Fashion-MNIST-format images (authentic 28×28)
processed_preview_x8/<name>.png   ×8 nearest-neighbour previews for human viewing
predictions/predictions.csv       long format: image, model, class, confidence, top-3, ms/image
predictions/predictions_wide.csv  image × model matrix ("class (conf %)")
predictions/full_probabilities.csv all 10 class probabilities per (image, model)
predictions/predictions.json      the same, structured, + provenance
README.txt                        preprocessing settings, normalisation constants, model list & accuracies
```

## Supported artifacts (exactly what the notebook writes: 3.1b)

| Folder | Format | Notes |
|---|---|---|
| `models/ml/*.joblib` | sklearn `Pipeline`s (LogReg, SVMs, RF) & boosting estimators + `*.json` sidecar | Pipelines are self-contained (their own scaler/PCA). |
| `models/dl/*.pt` | `state_dict` + sidecar with `arch_class` / `arch_kwargs` | `MLP`, `CNN`, `ResNetSmall`, `VisionTransformer` rebuilt **verbatim** from the notebook classes (`fashion_arch.py`); missing sidecars are inferred from tensor shapes. |
| `models/ensemble/*.joblib` | `{combiner, members, weights, meta}` | `soft_vote`, `hard_vote`, `weighted_vote` (weights renormalised if a member is missing), `stacking` (needs every member + the meta-learner). |

### Two honest limitations of the notebook's export (handled gracefully)

1. **Boosting on PCA features.** `XGBoost (PCA-80)`, `LightGBM (PCA-80)` and the
   Optuna-tuned booster were fitted on `pca_boost.transform(...)` features, and the
   notebook does **not** persist `pca_boost`. The app marks such models *unavailable*
   with an explanation: or uses them automatically if you export the PCA once:
   ```python
   # add to the notebook (e.g. Section 3.7.1) and re-run just that cell:
   import joblib
   joblib.dump(pca_boost, "artifacts/models/ml/pca_boost.joblib")
   ```
2. **CatBoost.** `pip install catboost` on the machine running the app if your
   export contains a CatBoost model (the app explains any skipped model in the table).

## Files

| File | Purpose |
|---|---|
| `app.py` | the Gradio app (UI + loading + preprocessing + inference + zip export) |
| `fashion_arch.py` | the notebook's model classes, copied verbatim, + checkpoint rebuilding |
| `make_demo_artifacts.py` | regenerates the bundled **demo** `artifacts/` (miniature models trained on the official Fashion-MNIST data in the notebook's exact artifact format) and demo `sample_photos/` |
| `smoke_test.py` | headless end-to-end check (load → preprocess → predict → export) |
| `artifacts/`, `sample_photos/` | ready-to-run demo data (included, replace with your own) |

> The bundled artifacts are **demo** models (small subset, few epochs, ~0.70-0.86 test
> accuracy): they exist so the app can be exercised end-to-end. With the real notebook
> export (0.84-0.95) predictions will be correspondingly stronger.
