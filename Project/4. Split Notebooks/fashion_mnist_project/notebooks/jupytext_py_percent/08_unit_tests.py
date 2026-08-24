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
# <a id="sec7b"></a>
# # 7. Unit tests: verifying the pipeline instead of trusting it
#
# > Exam criteria: **Code Quality (0–20)** and **Testing (0–10)**.
#
# Every number in Sections 2–6 rests on assumptions that are easy to get silently wrong: a transposed image, an
# off-by-one in the label encoding, a normalisation applied twice, a model whose head has the wrong number of outputs, a
# probability matrix whose rows do not sum to one. None of these raise an exception: they just quietly produce a worse
# (or, worse still, a *better*-looking) number.
#
# This section runs a compact test suite over the objects that already exist in the notebook. The tests are written as
# plain `test_*` functions with `assert` statements, so the identical file can be executed by `pytest` in CI; here they
# are driven by a small runner that reports a pass/fail/skip table instead of stopping at the first failure.
#
# | Group | What is verified |
# |---|---|
# | **Data** | raw shapes and dtypes, label ranges, proportional class balance, split sizes, and the **byte-exact absence of train/test leakage** after the Section 1.5b cleaning (both the hash check and an exact re-scan) |
# | **Tensors** | `(N, 1, 28, 28)` shape, `float32`, normalised mean ≈ 0 / std ≈ 1, label dtype, loader batch shapes |
# | **Augmentation** | shape preservation, no NaNs, exact identity when the transform is disabled |
# | **Models** | output dimensionality `(B, 10)` for every architecture, finite logits, parameter counts > 0 |
# | **Loss values** | initial cross-entropy ≈ `ln(10) = 2.3026` for an untrained head; loss actually decreases when a single batch is over-fitted |
# | **Inference** | `predict_logits` alignment, probability matrices are valid distributions |
# | **Ensembling** | weights on the simplex, soft-voting identities |
# | **Explainability** | Grad-CAM range, Integrated-Gradients completeness |
# | **Bookkeeping** | metric consistency in the `RESULTS` registry, seeding reproducibility |

# %%
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# Ensure the src directory with extracted modules is on the path
print('sys.path[0]:', sys.path[0])

# %%
# --- 7.1 A minimal test runner ---------------------------------------------------------------------------------
# `Callable` is not part of the v1 typing import block, so it is pulled in here (that cell is left untouched).
from typing import Callable


class SkipTest(Exception):
    """Raised by a test whose prerequisites are absent (e.g. an optional model was not trained)."""


def run_test_suite(tests: Sequence[Callable[[], None]], verbose: bool = True) -> pd.DataFrame:
    """Execute every test function, catching failures so the whole suite always runs to completion.

    Returns a tidy dataframe with one row per test: status (PASS / FAIL / SKIP), message and duration.
    """
    rows = []
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            status, message = "PASS", (fn.__doc__ or "").strip().split("\n")[0]
        except SkipTest as exc:
            status, message = "SKIP", str(exc)
        except AssertionError as exc:
            status, message = "FAIL", f"AssertionError: {exc}"
        except Exception as exc:  # noqa: BLE001 - an unexpected error is still a test failure
            status, message = "ERROR", f"{type(exc).__name__}: {exc}"
        rows.append({"test": fn.__name__, "status": status, "detail": message,
                     "seconds": round(time.time() - t0, 3)})
        if verbose:
            symbol = {"PASS": "PASS ", "SKIP": "SKIP ", "FAIL": "FAIL ", "ERROR": "ERROR"}[status]
            print(f"[{symbol}] {fn.__name__:<46s} {rows[-1]['seconds']:>6.2f}s  {message[:70]}")
    return pd.DataFrame(rows)


# %%
# --- 7.2 Data-level tests ---------------------------------------------------------------------------------------
def test_raw_arrays_have_expected_shape_and_dtype() -> None:
    """Image arrays are (N, 28, 28) uint8 with int labels in [0, 9]; the test set still has 10,000 rows."""
    assert X_train_full_np.shape[1:] == (28, 28), X_train_full_np.shape
    # 60,000 minus the handful of rows dropped by the leakage cleaning in Section 1.5b
    assert 59_900 <= len(X_train_full_np) <= 60_000, len(X_train_full_np)
    assert X_test_np.shape == (10_000, 28, 28), X_test_np.shape
    assert X_train_full_np.dtype == np.uint8 and X_test_np.dtype == np.uint8
    assert y_train_full_np.min() >= 0 and y_train_full_np.max() <= 9
    assert len(np.unique(y_train_full_np)) == cfg.num_classes


def test_pixel_value_range() -> None:
    """Raw pixels stay within [0, 255] and the flattened matrices within [0, 1]."""
    assert int(X_train_full_np.min()) >= 0 and int(X_train_full_np.max()) <= 255
    assert float(X_tr_flat.min()) >= 0.0 and float(X_tr_flat.max()) <= 1.0
    assert float(X_test_flat.min()) >= 0.0 and float(X_test_flat.max()) <= 1.0


def test_split_sizes_and_stratification() -> None:
    """The three splits partition the data exactly and every class keeps its 10% share."""
    n_total = len(X_train_full_np)
    assert len(X_tr_np) + len(X_val_np) == n_total, (len(X_tr_np), len(X_val_np), n_total)
    assert abs(len(X_val_np) - round(cfg.val_fraction * n_total)) <= 1, len(X_val_np)
    assert len(X_test_np) == 10_000, len(X_test_np)
    # the official test set is never modified, so it stays exactly balanced
    assert (np.bincount(y_test_np, minlength=cfg.num_classes) == 1_000).all()
    # train/val: proportional balance (leakage removal can cost a class one or two images)
    for name, y in [("train", y_tr_np), ("val", y_val_np)]:
        share = np.bincount(y, minlength=cfg.num_classes) / len(y)
        assert np.abs(share - 1 / cfg.num_classes).max() < 0.005, \
            f"{name} split drifted from balance: {np.round(100 * share, 2).tolist()}"


def test_leakage_removed() -> None:
    """No training image is byte-identical to any test image (Section 1.5b must have cleaned them)."""
    remaining = find_cross_split_duplicates(X_train_full_np, X_test_np)
    if cfgx.leakage_policy == "keep":
        raise SkipTest("CFGX.leakage_policy = 'keep': leakage was deliberately retained")
    assert not remaining, f"{len(remaining)} train/test duplicate pairs survived the cleaning"


def test_cleaning_preserved_the_official_test_set() -> None:
    """Cleaning removed rows from the training side only - the 10,000-image benchmark set is intact."""
    assert len(X_test_np) == 10_000
    assert int(leak_info["test rows removed"]) == 0 or cfgx.leakage_policy == "drop_from_test"


def test_no_train_test_leakage() -> None:
    """No image appears in both the training and the official test split."""
    shared = np.intersect1d(image_hashes(X_train_full_np), image_hashes(X_test_np))
    assert len(shared) == 0, f"{len(shared)} images leak between train and test"


def test_train_val_are_disjoint() -> None:
    """The validation split shares no image with the training split it was carved out of."""
    shared = np.intersect1d(image_hashes(X_tr_np), image_hashes(X_val_np))
    # Fashion-MNIST contains a few hundred exact-duplicate product photographs inside the 60k training file, so a
    # stratified split can legitimately place two copies on opposite sides. The tolerance documents that fact; a
    # value far above it would mean the split itself is broken.
    assert len(shared) < 0.02 * len(X_val_np), (
        f"{len(shared)} images shared between train and validation - more than the known duplicate rate explains"
    )


# %%
# --- 7.3 Tensor / DataLoader tests -------------------------------------------------------------------------------
def test_tensor_dataset_shapes_and_dtypes() -> None:
    """TensorDatasets hold (N, 1, 28, 28) float32 images and int64 labels.

    The expected counts are taken from the arrays the datasets were built from, so they track the
    leakage-cleaned split sizes (train + val = the cleaned 60k training file; the 10k test set is never
    touched) instead of the pre-cleaning 54,000 / 6,000 figures.
    """
    for name, ds, n_expected in [
        ("train", train_ds, len(X_tr_np)),
        ("val", val_ds, len(X_val_np)),
        ("test", test_ds, len(X_test_np)),
    ]:
        x, y = ds.tensors
        assert x.shape == (n_expected, 1, 28, 28), f"{name}: {tuple(x.shape)} != {(n_expected, 1, 28, 28)}"
        assert x.dtype == torch.float32, f"{name}: {x.dtype}"
        assert y.dtype == torch.int64 and y.shape == (n_expected,), f"{name}: {y.dtype}, {tuple(y.shape)}"
        assert int(y.min()) >= 0 and int(y.max()) <= 9
    # the tensor splits must partition the cleaned training file exactly; the official test set is immutable
    assert len(train_ds) + len(val_ds) == len(X_train_full_np), (len(train_ds), len(val_ds))
    assert len(test_ds) == 10_000, len(test_ds)


def test_normalisation_statistics() -> None:
    """Normalised training tensors have mean ~ 0 and std ~ 1 (train statistics, applied to all splits)."""
    x = train_ds.tensors[0]
    assert abs(float(x.mean())) < 0.02, f"mean = {float(x.mean()):.4f}"
    assert 0.95 < float(x.std()) < 1.05, f"std = {float(x.std()):.4f}"
    # validation/test are normalised with the SAME statistics, so they are close but not exactly 0/1
    assert abs(float(test_ds.tensors[0].mean())) < 0.15
    assert not torch.isnan(x).any(), "NaN in the normalised tensors"


def test_dataloader_batch_shapes() -> None:
    """A training batch is (B, 1, 28, 28) / (B,) with B <= CFG.batch_size."""
    xb, yb = next(iter(train_loader))
    assert xb.ndim == 4 and xb.shape[1:] == (1, 28, 28), tuple(xb.shape)
    assert yb.ndim == 1 and yb.shape[0] == xb.shape[0]
    assert xb.shape[0] <= cfg.batch_size
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64


def test_augmentation_contract() -> None:
    """Augment() preserves shape/dtype, produces no NaNs, and is the identity when disabled."""
    xb = train_ds.tensors[0][:64].clone()
    out = Augment(p_flip=0.5, max_shift=2)(xb)
    assert out.shape == xb.shape and out.dtype == xb.dtype
    assert torch.isfinite(out).all()
    identity = Augment(p_flip=0.0, max_shift=0)(xb.clone())
    assert torch.allclose(identity, xb), "augmentation with p_flip=0, max_shift=0 must be the identity"
    assert not torch.allclose(out, xb), "augmentation with p_flip=0.5 must change at least some images"


# %%
# --- 7.4 Model-level tests ---------------------------------------------------------------------------------------
def _architectures() -> Dict[str, nn.Module]:
    """Freshly constructed instances of every architecture in the notebook (untrained)."""
    models: Dict[str, nn.Module] = {
        "MLP": MLP(num_classes=cfg.num_classes),
        "CNN": CNN(num_classes=cfg.num_classes),
        "ResNetSmall": ResNetSmall(num_classes=cfg.num_classes, width=cfgx.resnet_width),
        "ViT": VisionTransformer(patch=cfgx.vit_patch, num_classes=cfg.num_classes, dim=cfgx.vit_dim,
                                 depth=2, heads=cfgx.vit_heads),
        "ConvAutoencoder": ConvAutoencoder(latent=cfgx.autoencoder_latent),
    }
    return models


def test_model_output_dimensions() -> None:
    """Every classifier maps (B, 1, 28, 28) -> (B, 10) with finite logits; the autoencoder reconstructs the input."""
    x = torch.randn(4, 1, 28, 28)
    for name, model in _architectures().items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        if name == "ConvAutoencoder":
            assert out.shape == x.shape, f"{name}: {tuple(out.shape)}"
        else:
            assert out.shape == (4, cfg.num_classes), f"{name}: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite outputs"


def test_parameter_counts_are_sane() -> None:
    """Parameter counts are positive, and every architecture stays inside its documented budget."""
    budgets = {"MLP": 1_000_000, "CNN": 800_000, "ResNetSmall": 3_000_000, "ViT": 5_000_000,
               "ConvAutoencoder": 500_000}
    for name, model in _architectures().items():
        n = count_parameters(model)
        assert n > 0, f"{name} has no trainable parameters"
        assert n < budgets[name], f"{name} has {n:,} parameters, above the documented budget"


def test_initial_loss_matches_uniform_prediction() -> None:
    """An untrained 10-class head must score cross-entropy ~ ln(10) = 2.3026 (uniform prediction)."""
    torch.manual_seed(0)
    x = torch.randn(256, 1, 28, 28)
    y = torch.randint(0, cfg.num_classes, (256,))
    expected = float(np.log(cfg.num_classes))
    criterion = nn.CrossEntropyLoss()                       # no label smoothing for this reference value
    for name, model in _architectures().items():
        if name == "ConvAutoencoder":
            continue
        model.eval()
        with torch.no_grad():
            loss = float(criterion(model(x), y))
        assert 0.6 * expected < loss < 2.0 * expected, f"{name}: initial loss {loss:.3f}, expected ~{expected:.3f}"


def test_single_batch_overfitting() -> None:
    """Sanity check that gradients flow: 40 steps on one batch must cut the loss by at least half."""
    torch.manual_seed(0)
    model = CNN(num_classes=cfg.num_classes, p_drop=0.0).to(DEVICE).train()
    xb = train_ds.tensors[0][:64].to(DEVICE)
    yb = train_ds.tensors[1][:64].to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = float(criterion(model(xb), yb))
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
    last = float(loss)
    assert last < 0.5 * first, f"loss did not decrease enough: {first:.3f} -> {last:.3f}"


def test_trained_models_beat_chance() -> None:
    """Every trained model in TORCH_ZOO must beat the 10% chance level by a wide margin."""
    if not TORCH_ZOO:
        raise SkipTest("no trained torch models available")
    xb = test_ds.tensors[0][:1_000]
    yb = test_ds.tensors[1][:1_000].numpy()
    for name, model in TORCH_ZOO.items():
        acc = float((torch_probabilities(model, xb).argmax(1) == yb).mean())
        assert acc > 0.70, f"{name} scores only {acc:.3f} on 1,000 test images"


# %%
# --- 7.5 Inference, ensembling and explainability tests -----------------------------------------------------------
def test_predict_logits_alignment() -> None:
    """predict_logits returns (N, 10) logits whose labels match the dataset order exactly."""
    logits, labels = predict_logits(next(iter(TORCH_ZOO.values())), test_loader)
    assert logits.shape == (len(test_ds), cfg.num_classes), logits.shape
    assert np.array_equal(labels, y_test_np), "label order returned by predict_logits does not match the dataset"


def test_probability_matrices_are_distributions() -> None:
    """Every member's probability matrix has the right shape, is non-negative and sums to 1 per row."""
    if "MEMBER_PROBS" not in globals():
        raise SkipTest("ensembling section did not run")
    for name, p in MEMBER_PROBS.items():
        for split, expected_n in [("val", len(y_val_np)), ("test", len(y_test_np))]:
            m = p[split]
            assert m.shape == (expected_n, cfg.num_classes), f"{name}/{split}: {m.shape}"
            assert (m >= -1e-9).all(), f"{name}/{split} contains negative probabilities"
            assert np.allclose(m.sum(axis=1), 1.0, atol=1e-6), f"{name}/{split} rows do not sum to 1"


def test_soft_vote_identities() -> None:
    """soft_vote of a single member is that member; equal weights reproduce the plain mean."""
    if "MEMBER_PROBS" not in globals():
        raise SkipTest("ensembling section did not run")
    names = list(MEMBER_PROBS)[:3]
    single = MEMBER_PROBS[names[0]]["test"]
    assert np.allclose(soft_vote([single]), single, atol=1e-12)
    stacked = [MEMBER_PROBS[n]["test"] for n in names]
    assert np.allclose(soft_vote(stacked), np.mean(stacked, axis=0), atol=1e-12)


def test_ensemble_weights_on_simplex() -> None:
    """Fitted ensemble weights are non-negative and sum to one."""
    if "w_deep" not in globals():
        raise SkipTest("weight search did not run")
    for w in [w_deep] + ([w_hyb] if "w_hyb" in globals() else []):
        assert (np.asarray(w) >= -1e-12).all(), "negative ensemble weight"
        assert abs(float(np.sum(w)) - 1.0) < 1e-6, f"weights sum to {float(np.sum(w)):.6f}"


def test_gradcam_output_contract() -> None:
    """Grad-CAM returns (B, 28, 28) maps normalised to [0, 1]."""
    conv_models = [m for m in TORCH_ZOO.values() if isinstance(m, (CNN, ResNetSmall))]
    if not conv_models:
        raise SkipTest("no convolutional model available")
    cams, targets = grad_cam(conv_models[0], test_ds.tensors[0][:4])
    assert cams.shape == (4, 28, 28), cams.shape
    assert cams.min() >= -1e-6 and cams.max() <= 1.0 + 1e-6
    assert targets.shape == (4,)


def test_integrated_gradients_completeness() -> None:
    """Integrated Gradients satisfies its completeness axiom to within 15% (Riemann discretisation error)."""
    if not TORCH_ZOO:
        raise SkipTest("no trained torch models available")
    model = TORCH_ZOO.get("CNN (VGG-style, GAP)", next(iter(TORCH_ZOO.values())))
    _, diag = integrated_gradients(model, test_ds.tensors[0][:8], steps=64)
    assert diag["mean relative completeness error"] < 0.15, diag


# %%
# --- 7.6 Bookkeeping / reproducibility tests -----------------------------------------------------------------------
def test_results_registry_is_consistent() -> None:
    """Every registered result has accuracy in (0, 1], error_rate = 1 - accuracy and a non-empty model name."""
    assert len(RESULTS) >= 5, f"only {len(RESULTS)} models registered"
    for r in RESULTS:
        assert 0.0 < float(r["accuracy"]) <= 1.0, r
        assert abs(float(r["error_rate"]) - (1.0 - float(r["accuracy"]))) < 1e-9, r
        assert isinstance(r["model"], str) and r["model"], r
        assert 0.0 <= float(r["macro_f1"]) <= 1.0, r


def test_metrics_match_manual_computation() -> None:
    """evaluate_predictions agrees with a hand-computed accuracy on a synthetic example."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])         # 9 of 10 correct
    rec = evaluate_predictions(y_true, y_pred, "unit-test dummy", family="Test", register=False)
    assert abs(float(rec["accuracy"]) - 0.9) < 1e-12, rec["accuracy"]


def test_set_seed_is_reproducible() -> None:
    """set_seed() makes numpy and torch RNG streams reproducible."""
    set_seed(123)
    a_np, a_t = np.random.rand(5), torch.randn(5)
    set_seed(123)
    b_np, b_t = np.random.rand(5), torch.randn(5)
    assert np.allclose(a_np, b_np), "numpy RNG is not reproducible"
    assert torch.allclose(a_t, b_t), "torch RNG is not reproducible"
    set_seed(cfg.seed)                                        # restore the notebook-wide seed


def test_pca_feature_spaces_are_consistent() -> None:
    """The boosting PCA transforms every split into the same dimensionality without NaNs."""
    if "pca_boost" not in globals():
        raise SkipTest("boosting section did not run")
    k = pca_boost.n_components_
    for name, M in [("train", X_boost_tr), ("val", X_boost_val), ("test", X_boost_test)]:
        assert M.shape[1] == k, f"{name}: {M.shape[1]} components, expected {k}"
        assert np.isfinite(M).all(), f"{name} contains non-finite values"
    assert 0.0 < float(pca_boost.explained_variance_ratio_.sum()) <= 1.0


def test_artifacts_were_written() -> None:
    """Model checkpoints and result tables were actually written to the artifacts directory."""
    files = list(Path(cfg.artifacts_dir).glob("*"))
    assert files, "artifacts directory is empty"
    assert any(f.suffix in {".pt", ".ckpt"} for f in files), "no model checkpoint was saved"


def test_best_models_persisted_to_per_family_folders() -> None:
    """The best version of every trained model is saved under models/{ml,dl,ensemble} with a JSON sidecar."""
    if "MODEL_REGISTRY" not in globals() or not MODEL_REGISTRY:
        raise SkipTest("no models were trained and registered")
    assert MODELS_ROOT.exists(), "models/ root was not created"
    manifest = models_manifest_df()
    assert not manifest.empty, "no model artefact files were written to models/{ml,dl,ensemble}"
    for _name, _entry in MODEL_REGISTRY.items():
        if _entry["family"] == "Trivial":
            continue
        _slug = slugify(_name)
        _ext = ".pt" if _entry["dir"] == "dl" else ".joblib"
        _path = MODEL_DIRS[_entry["dir"]] / f"{_slug}{_ext}"
        assert _path.exists(), f"{_name}: best version not persisted at {_path}"
        assert _path.with_suffix(".json").exists(), f"{_name}: sidecar metadata missing"


# %%
# --- 7.7 Run the whole suite -----------------------------------------------------------------------------------
ALL_TESTS: List[Callable[[], None]] = [
    # data
    test_raw_arrays_have_expected_shape_and_dtype,
    test_pixel_value_range,
    test_split_sizes_and_stratification,
    test_no_train_test_leakage,
    test_leakage_removed,
    test_cleaning_preserved_the_official_test_set,
    test_train_val_are_disjoint,
    # tensors
    test_tensor_dataset_shapes_and_dtypes,
    test_normalisation_statistics,
    test_dataloader_batch_shapes,
    test_augmentation_contract,
    # models
    test_model_output_dimensions,
    test_parameter_counts_are_sane,
    test_initial_loss_matches_uniform_prediction,
    test_single_batch_overfitting,
    test_trained_models_beat_chance,
    # inference / ensembling / explainability
    test_predict_logits_alignment,
    test_probability_matrices_are_distributions,
    test_soft_vote_identities,
    test_ensemble_weights_on_simplex,
    test_gradcam_output_contract,
    test_integrated_gradients_completeness,
    # bookkeeping
    test_results_registry_is_consistent,
    test_metrics_match_manual_computation,
    test_set_seed_is_reproducible,
    test_pca_feature_spaces_are_consistent,
    test_artifacts_were_written,
    test_best_models_persisted_to_per_family_folders,
]

print(f"Running {len(ALL_TESTS)} unit tests ...\n")
test_report = run_test_suite(ALL_TESTS)

summary = test_report["status"].value_counts().to_dict()
print("\n" + "=" * 78)
print(f"SUITE SUMMARY: {summary.get('PASS', 0)} passed | {summary.get('SKIP', 0)} skipped | "
      f"{summary.get('FAIL', 0)} failed | {summary.get('ERROR', 0)} errored "
      f"| total {test_report['seconds'].sum():.1f}s")
print("=" * 78)

display(
    test_report.style.hide(axis="index").apply(
        lambda s: [
            "background-color: #d4edda" if v == "PASS" else
            "background-color: #fff3cd" if v == "SKIP" else
            "background-color: #f8d7da" for v in test_report["status"]
        ],
        axis=0,
    )
)
test_report.to_csv(Path(cfg.artifacts_dir) / "unit_test_report.csv", index=False)

# A failing suite must be loud: no conclusion in this notebook is valid if the pipeline is broken.
if summary.get("FAIL", 0) or summary.get("ERROR", 0):
    print("\n*** ATTENTION: at least one test failed - the results above must not be trusted until it is fixed. ***")
else:
    print("\nAll executed tests passed: shapes, dtypes, splits, losses, probabilities and attributions "
          "behave as documented.")

# %% [markdown]
# **Finding (Section 7).** The suite converts implicit assumptions into explicit, machine-checked contracts. Three of
# these tests are the ones that catch real bugs in practice:
#
# * `test_initial_loss_matches_uniform_prediction`: a freshly initialised classifier **must** score ≈ `ln(10)`. A value
#   far from it is the classic signature of a bad initialisation, a wrong number of output units, or logits that were
#   accidentally passed through a softmax before the loss.
# * `test_single_batch_overfitting`: if a model cannot drive the loss down on 64 fixed images, no amount of epochs will
#   help; this catches frozen parameters, detached graphs and zeroed learning rates in ~2 seconds.
# * `test_no_train_test_leakage` / `test_probability_matrices_are_distributions`: the two failure modes that would
#   silently *inflate* the headline numbers rather than degrade them, which makes them the most dangerous ones.
#
# Because the functions are ordinary `test_*` functions with `assert`s, the same code runs unchanged under `pytest`
# (`pytest notebook_tests.py`) if this project is ever promoted from a notebook to a package.
