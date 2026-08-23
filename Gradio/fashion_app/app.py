"""
app.py — Fashion-MNIST Inference Studio (Gradio web app)
================================================================================
A graphical front-end for the models exported by the notebook
`DL-Fashion-MNIST.ipynb` (artifacts/models/{ml,dl,ensemble}).

Workflow (the four judged requirements):
  1. the user points the app at the folder with the exported model artifacts;
  2. the user points the app at a folder with sample photos
     (or uploads the photos directly in the browser);
  3. the app converts every photo to the Fashion-MNIST format
     (grayscale 28x28, garment bright on black background, values 0-255,
     then [0,1] for the sklearn pipelines and (x-mean)/std for the networks)
     and runs EVERY usable exported model on it;
  4. one click exports the processed images + all model predictions as a .zip.

Run locally:
    pip install -r requirements.txt
    python app.py            # opens http://localhost:7860
"""

from __future__ import annotations

import io
import json
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from fashion_arch import (
    CLASS_NAMES,
    NUM_CLASSES,
    PIXEL_MEAN,
    PIXEL_STD,
    build_torch_model,
    normalize_batch,
    torch_probabilities,
)

APP_DIR = Path(__file__).resolve().parent
EXPORT_DIR = APP_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MAX_IMAGES = 300  # safety cap for one batch

DEFAULT_MODELS_DIR = str(APP_DIR / "artifacts") if (APP_DIR / "artifacts" / "models").exists() \
    else str(APP_DIR / "artifacts")
DEFAULT_PHOTOS_DIR = str(APP_DIR / "sample_photos") if (APP_DIR / "sample_photos").exists() else ""


# ================================================================================
# 1. Preprocessing: arbitrary photo -> Fashion-MNIST format (28x28 uint8)
# ================================================================================
def _flatten_on_black(im: Image.Image) -> Image.Image:
    """Composite images with an alpha channel (transparent-background PNGs) onto black."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))  # Fashion-MNIST: black background
        return Image.alpha_composite(bg, rgba).convert("L")
    return im.convert("L")


def _background_is_light(gray_arr: np.ndarray) -> bool:
    """Decide whether the photo's background is light (=> garment is dark => inversion needed).

    Uses the median of the outer 2-pixel border ring — more robust than the global
    mean when the garment fills most of the frame.
    """
    h, w = gray_arr.shape
    if h < 8 or w < 8:
        return bool(gray_arr.mean() > 127)
    border = np.concatenate([
        gray_arr[:2, :].ravel(), gray_arr[-2:, :].ravel(),
        gray_arr[:, :2].ravel(), gray_arr[:, -2:].ravel(),
    ])
    return bool(np.median(border) > 127)


def to_fashion_mnist(
    img: Image.Image,
    invert: str = "auto",          # "auto" | "yes" | "no"
    autocontrast: bool = True,     # stretch intensity range like Zalando's pipeline
    center_crop: bool = True,      # crop to garment bbox + square pad (size normalisation)
) -> np.ndarray:
    """Convert any photo into a Fashion-MNIST-style image: (28, 28) uint8, garment bright on black.

    Mirrors the dataset properties established in notebook Section 1.4 / 2.4:
    grayscale, 28x28, white-ish garment on a black (~0) background, roughly
    centred and size-normalised.
    """
    gray = _flatten_on_black(img)
    arr = np.asarray(gray)

    do_invert = (invert == "yes") or (invert == "auto" and _background_is_light(arr))
    if do_invert:
        gray = ImageOps.invert(gray)
        arr = np.asarray(gray)

    if autocontrast:
        gray = ImageOps.autocontrast(gray, cutoff=1)
        arr = np.asarray(gray)

    if center_crop:
        mask = arr > 20                                   # 'ink' threshold used in the notebook EDA
        if mask.any():
            ys, xs = np.where(mask)
            pad = max(1, int(0.04 * max(ys.max() - ys.min(), xs.max() - xs.min())))
            y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, arr.shape[0])
            x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, arr.shape[1])
            crop = gray.crop((x0, y0, x1, y1))
            # square pad with the background value so aspect ratio is preserved
            w, h = crop.size
            side = max(w, h)
            bg_val = int(np.median(np.asarray(crop)[:2, :].ravel().tolist() +
                                   np.asarray(crop)[-2:, :].ravel().tolist() +
                                   np.asarray(crop)[:, :2].ravel().tolist() +
                                   np.asarray(crop)[:, -2:].ravel().tolist()))
            sq = Image.new("L", (side, side), bg_val)
            sq.paste(crop, ((side - w) // 2, (side - h) // 2))
            gray = sq

    small = gray.resize((28, 28), Image.LANCZOS)
    return np.clip(np.asarray(small), 0, 255).astype(np.uint8)


def array_to_pil(arr: np.ndarray, scale: int = 1) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8), mode="L")
    return im.resize((28 * scale, 28 * scale), Image.NEAREST) if scale > 1 else im


# ================================================================================
# 2. Loading the exported artifacts
# ================================================================================
@dataclass
class ModelEntry:
    name: str
    family: str                       # "Classical ML" / "Gradient Boosting" / "Deep Learning" / "Ensemble"
    kind: str                         # "ml" | "dl" | "ensemble"
    file: str
    status: str = "ok"                # "ok" | "unavailable" | "error"
    note: str = ""
    accuracy: Optional[float] = None
    artifact: object = None
    members: List[str] = field(default_factory=list)
    needs_pca: bool = False


def _read_sidecar(path: Path) -> dict:
    side = path.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except Exception:
            return {}
    return {}


def resolve_models_root(path_str: str) -> Path:
    """Accept '<project>/artifacts', '<...>/artifacts/models' or any equivalent folder."""
    p = Path(path_str.strip().strip('"').strip("'")).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a folder: {p}")

    def has_model_folders(q: Path) -> bool:
        return (q / "models").is_dir() and any((q / "models" / k).is_dir() for k in ("ml", "dl", "ensemble"))

    if has_model_folders(p):                 # e.g. .../artifacts
        return p / "models"
    if any((p / k).is_dir() for k in ("ml", "dl", "ensemble")):   # e.g. .../artifacts/models
        return p
    # last resort: search one/two levels deep
    for q in sorted(p.rglob("models")):
        if q.is_dir() and any((q / k).is_dir() for k in ("ml", "dl", "ensemble")):
            return q
    raise FileNotFoundError(
        f"No 'models/{{ml,dl,ensemble}}' folder structure found under '{p}'. "
        "Expected the artifacts folder exported by DL-Fashion-MNIST.ipynb."
    )


def _load_ml_folder(ml_dir: Path, pca_anywhere: Optional[object]) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for f in sorted(ml_dir.glob("*.joblib")):
        side = _read_sidecar(f)
        name = side.get("model", f.stem)
        acc = side.get("accuracy")
        feats = side.get("features")

        if f.stem.lower().startswith("pca"):   # a persisted PCA preprocessor, not a model
            continue
        try:
            obj = joblib.load(f)
        except Exception as exc:               # e.g. xgboost not installed
            entries.append(ModelEntry(name, side.get("family", "Classical ML"), "ml", f.name,
                                      status="unavailable", note=f"could not unpickle ({exc})",
                                      accuracy=acc))
            continue

        if feats == "pca" and not hasattr(obj, "transform"):
            # raw boosting estimator trained on PCA features: needs the fitted PCA object
            if pca_anywhere is not None:
                entries.append(ModelEntry(name, side.get("family", "Gradient Boosting"), "ml", f.name,
                                          accuracy=acc, artifact=obj, needs_pca=True,
                                          note="fed through the exported PCA preprocessor"))
            else:
                entries.append(ModelEntry(
                    name, side.get("family", "Gradient Boosting"), "ml", f.name, status="unavailable",
                    accuracy=acc,
                    note="trained on PCA-80 features but the fitted PCA was not exported by the notebook "
                         "(add `joblib.dump(pca_boost, 'artifacts/models/ml/pca_boost.joblib')` in the "
                         "notebook and reload here)"))
            continue

        entries.append(ModelEntry(name, side.get("family", "Classical ML"), "ml", f.name,
                                  accuracy=acc, artifact=obj,
                                  note=f"scikit-learn pipeline/estimator ({feats or 'flat'} features)"))
    return entries


def _load_dl_folder(dl_dir: Path) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for f in sorted(dl_dir.glob("*.pt")):
        side = _read_sidecar(f)
        name = side.get("model", f.stem)
        try:
            state = torch.load(f, map_location="cpu", weights_only=True)
            model = build_torch_model(state, arch_class=side.get("arch_class"),
                                      arch_kwargs=side.get("arch_kwargs"))
            entries.append(ModelEntry(name, side.get("family", "Deep Learning"), "dl", f.name,
                                      accuracy=side.get("accuracy"), artifact=model,
                                      note=f"{type(model).__name__}, "
                                           f"{sum(p.numel() for p in model.parameters()):,} params"))
        except Exception as exc:
            entries.append(ModelEntry(name, side.get("family", "Deep Learning"), "dl", f.name,
                                      status="error", note=f"could not rebuild checkpoint ({exc})",
                                      accuracy=side.get("accuracy")))
    return entries


def _load_ensemble_folder(ens_dir: Path, name_lookup: Dict[str, ModelEntry],
                          slug_lookup: Dict[str, ModelEntry]) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for f in sorted(ens_dir.glob("*.joblib")):
        side = _read_sidecar(f)
        name = side.get("model", f.stem)
        try:
            payload = joblib.load(f)
            if not isinstance(payload, dict) or "combiner" not in payload:
                continue
            members = list(payload.get("members", []))
            resolved: List[Optional[ModelEntry]] = []
            for m in members:
                e = name_lookup.get(m) or slug_lookup.get(
                    __import__("re").sub(r"[^a-z0-9]+", "_", m.strip().lower()).strip("_"))
                resolved.append(e if e and e.status == "ok" else None)
            missing = [m for m, e in zip(members, resolved) if e is None]

            combiner = payload.get("combiner")
            if combiner == "stacking" and (missing or payload.get("meta") is None):
                note = "stacking meta-learner needs every member; missing: " + ", ".join(missing) \
                       if missing else "meta-learner missing"
                entries.append(ModelEntry(name, "Ensemble", "ensemble", f.name, status="unavailable",
                                          accuracy=side.get("accuracy"), note=note, members=members))
                continue

            artifact = {"combiner": combiner, "members": members,
                        "weights": payload.get("weights"), "meta": payload.get("meta"),
                        "resolved": [e.name for e in resolved if e is not None]}
            note = f"{combiner} over {len(members)} members"
            if missing:
                note += f" (running with {len(members) - len(missing)}: {', '.join(missing)} unavailable)"
            entries.append(ModelEntry(name, "Ensemble", "ensemble", f.name,
                                      accuracy=side.get("accuracy"), artifact=artifact,
                                      note=note, members=members))
        except Exception as exc:
            entries.append(ModelEntry(name, "Ensemble", "ensemble", f.name, status="error",
                                      note=f"could not load combiner ({exc})", accuracy=side.get("accuracy")))
    return entries


def find_pca(root: Path) -> Optional[object]:
    """Locate a persisted boosting-PCA preprocessor (pca*.joblib), if the notebook exported one."""
    for cand in list((root / "ml").glob("pca*.joblib")) + list(root.glob("pca*.joblib")) \
            + list(root.parent.glob("pca*.joblib")):
        try:
            obj = joblib.load(cand)
            if hasattr(obj, "transform") and hasattr(obj, "n_components_"):
                return obj
        except Exception:
            continue
    return None


def load_models(models_path: str) -> Tuple[pd.DataFrame, str, List[ModelEntry]]:
    """Load every artifact under the given folder. Returns (manifest_df, status_text, entries)."""
    root = resolve_models_root(models_path)

    # the notebook does not export the boosting PCA — but if the user did, use it
    pca_obj = find_pca(root)

    ml = _load_ml_folder(root / "ml", pca_obj) if (root / "ml").is_dir() else []
    dl = _load_dl_folder(root / "dl") if (root / "dl").is_dir() else []
    name_lookup = {e.name: e for e in ml + dl if e.status == "ok"}
    slug_lookup = {__import__("re").sub(r"[^a-z0-9]+", "_", e.name.lower()).strip("_"): e
                   for e in ml + dl if e.status == "ok"}
    ens = _load_ensemble_folder(root / "ensemble", name_lookup, slug_lookup) \
        if (root / "ensemble").is_dir() else []

    entries = ml + dl + ens
    ok = [e for e in entries if e.status == "ok"]
    bad = [e for e in entries if e.status != "ok"]

    best_note = ""
    best_file = root / "dl" / "BEST.json"
    if best_file.exists():
        try:
            best_note = f" | notebook BEST (dl): {json.loads(best_file.read_text()).get('best_model')}"
        except Exception:
            pass

    status = (f"Loaded {len(ok)}/{len(entries)} models from {root}"
              + best_note
              + (f" | skipped: {len(bad)}" if bad else ""))
    if not ok:
        status += "  — no usable model found!"

    df = pd.DataFrame([{
        "model": e.name,
        "family": e.family,
        "folder": e.kind,
        "file": e.file,
        "status": "✅ ready" if e.status == "ok" else "⚠️ " + e.status,
        "test acc (notebook)": (f"{e.accuracy:.4f}" if isinstance(e.accuracy, (int, float)) else "—"),
        "note": e.note,
    } for e in entries])
    return df, status, entries


# ================================================================================
# 3. Inference
# ================================================================================
def _estimator_probabilities(est, X: np.ndarray) -> np.ndarray:
    """(N,10) probabilities from any sklearn-compatible estimator."""
    if hasattr(est, "predict_proba"):
        P = np.asarray(est.predict_proba(X), dtype=np.float64)
    elif hasattr(est, "decision_function"):
        S = np.asarray(est.decision_function(X), dtype=np.float64)
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        P = E / E.sum(axis=1, keepdims=True)
    else:
        preds = np.asarray(est.predict(X)).ravel().astype(int)
        P = np.eye(NUM_CLASSES)[np.clip(preds, 0, NUM_CLASSES - 1)]
    if P.shape[1] != NUM_CLASSES:      # class order safety net
        classes = getattr(est, "classes_", np.arange(P.shape[1]))
        order = np.argsort(classes)
        P = P[:, order]
    return P


def _combine_ensemble(art: dict, member_probs: List[np.ndarray]) -> Tuple[np.ndarray, str]:
    combiner, n = art["combiner"], len(member_probs)
    if combiner == "soft_vote":
        return np.mean(np.stack(member_probs), axis=0), "mean of member probabilities"
    if combiner in ("weighted_vote", "soft_vote_weighted"):
        w = np.asarray(art.get("weights") or [1.0 / n] * n, dtype=np.float64)[:n]
        w = w / w.sum()
        if len(art["members"]) != n:                # renormalise over available members
            w = np.full(n, 1.0 / n)
        return np.tensordot(w, np.stack(member_probs), axes=(0, 0)), "weighted mean (notebook weights)"
    if combiner == "hard_vote":
        P = np.stack(member_probs)
        preds = P.argmax(axis=2)
        votes = np.zeros((P.shape[1], NUM_CLASSES))
        for m in range(preds.shape[0]):
            votes[np.arange(preds.shape[1]), preds[m]] += 1.0
        votes += 1e-6 * P.mean(axis=0)
        V = votes / votes.sum(axis=1, keepdims=True)
        return V, "majority vote (+ mean-prob tie break)"
    if combiner == "stacking":
        X_meta = np.hstack([p for p in member_probs])
        meta = art["meta"]
        if hasattr(meta, "predict_proba"):
            return _estimator_probabilities(meta, X_meta), "logistic meta-learner over member probs"
        preds = np.asarray(meta.predict(X_meta)).ravel().astype(int)
        return np.eye(NUM_CLASSES)[np.clip(preds, 0, NUM_CLASSES - 1)], "meta-learner predictions"
    raise ValueError(f"unknown combiner {combiner}")


@dataclass
class RunResults:
    image_names: List[str]
    processed: List[np.ndarray]                 # (28,28) uint8
    probs: Dict[str, np.ndarray]                # model name -> (N,10)
    timings: Dict[str, float]
    settings: dict


def run_inference(entries: List[ModelEntry], images: Sequence[np.ndarray],
                  tta_mirror: bool) -> RunResults:
    x = torch.from_numpy(np.stack(images)).unsqueeze(1)          # (N,1,28,28) uint8
    x_norm = normalize_batch(x)                                   # deep models
    flat01 = x.float().div(255.0).flatten(1).numpy().astype(np.float32)  # sklearn pipelines

    ok = [e for e in entries if e.status == "ok"]
    singles = [e for e in ok if e.kind in ("ml", "dl")]
    ensembles = [e for e in ok if e.kind == "ensemble"]

    probs: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}

    for e in singles:
        t0 = time.time()
        if e.kind == "dl":
            p = torch_probabilities(e.artifact, x_norm, tta_mirror=tta_mirror).numpy()
        else:
            X = flat01
            if e.needs_pca:
                X = STATE["pca"].transform(flat01)
            p = _estimator_probabilities(e.artifact, X)
        probs[e.name] = p.astype(np.float64)
        timings[e.name] = (time.time() - t0) * 1000 / max(len(images), 1)

    for e in ensembles:
        t0 = time.time()
        art = e.artifact
        member_probs = [probs[m] for m in art["resolved"] if m in probs]
        if not member_probs:
            continue
        p, _ = _combine_ensemble(art, member_probs)
        probs[e.name] = p.astype(np.float64)
        timings[e.name] = (time.time() - t0) * 1000 / max(len(images), 1)

    return RunResults([], list(images), probs, timings, {})


# ================================================================================
# 4. ZIP export (processed photos + predictions)
# ================================================================================
def build_export_zip(res: RunResults, entries: List[ModelEntry], settings: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EXPORT_DIR / f"fashion_mnist_export_{stamp}.zip"

    model_rows = [{k: v for k, v in {
        "model": e.name, "family": e.family, "file": e.file,
        "notebook_test_accuracy": e.accuracy, "status": e.status}.items()}
        for e in entries]

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        # -- processed images ---------------------------------------------------
        for name, arr in zip(res.image_names, res.processed):
            stem = Path(name).stem
            z.writestr(f"processed_28x28/{stem}.png",
                       _png_bytes(array_to_pil(arr)))                       # authentic 28x28
            z.writestr(f"processed_preview_x8/{stem}_224.png",
                       _png_bytes(array_to_pil(arr, scale=8)))              # human-viewable

        # -- predictions --------------------------------------------------------
        long_rows, wide_rows, prob_rows = [], [], []
        for i, name in enumerate(res.image_names):
            wide = {"image": name}
            for model, P in res.probs.items():
                pred = int(P[i].argmax())
                conf = float(P[i].max())
                wide[model] = f"{CLASS_NAMES[pred]} ({conf * 100:.1f}%)"
                top3 = np.argsort(-P[i])[:3]
                long_rows.append({
                    "image": name, "model": model,
                    "predicted_index": pred, "predicted_class": CLASS_NAMES[pred],
                    "confidence": round(conf, 4),
                    "top3": " | ".join(f"{CLASS_NAMES[c]} {P[i][c]:.3f}" for c in top3),
                    "ms_per_image": round(res.timings.get(model, float("nan")), 2),
                })
                row = {"image": name, "model": model}
                row.update({CLASS_NAMES[c]: round(float(P[i][c]), 5) for c in range(NUM_CLASSES)})
                prob_rows.append(row)
            wide_rows.append(wide)

        z.writestr("predictions/predictions.csv", _df_csv(pd.DataFrame(long_rows)))
        z.writestr("predictions/predictions_wide.csv", _df_csv(pd.DataFrame(wide_rows)))
        z.writestr("predictions/full_probabilities.csv", _df_csv(pd.DataFrame(prob_rows)))
        z.writestr("predictions/predictions.json", json.dumps(
            {"settings": settings, "class_names": list(CLASS_NAMES),
             "models": model_rows,
             "per_image": {name: {m: {"predicted_class": CLASS_NAMES[int(P[i].argmax())],
                                      "confidence": float(P[i].max()),
                                      "probabilities": {CLASS_NAMES[c]: round(float(P[i][c]), 5)
                                                        for c in range(NUM_CLASSES)}}
                                  for m, P in res.probs.items()}
                           for i, name in enumerate(res.image_names)}},
            indent=2))

        # -- provenance ---------------------------------------------------------
        z.writestr("README.txt",
                   "Fashion-MNIST inference export\n"
                   f"generated: {datetime.now().isoformat()}\n"
                   f"images: {len(res.processed)}\n"
                   f"models used: {len(res.probs)}\n\n"
                   "Preprocessing applied to every photo (Fashion-MNIST format):\n"
                   f"  grayscale, resize to 28x28 LANCZOS; invert={settings.get('invert')}; "
                   f"autocontrast={settings.get('autocontrast')}; "
                   f"center_crop={settings.get('center_crop')}\n"
                   "  deep-model input : (pixel/255 - 0.2860)/0.3530   (notebook Section 1.6)\n"
                   "  sklearn input    : pixel/255 flattened to 784 features (pipelines scale internally)\n"
                   f"  mirror TTA       : {settings.get('tta')}\n\n"
                   "Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)) + "\n\n"
                   "Models (with the test accuracy reported by the notebook):\n"
                   + "\n".join(f"  - {m['model']} [{m['family']}] acc="
                               f"{m['notebook_test_accuracy'] if m['notebook_test_accuracy'] is not None else '—'}"
                               for m in model_rows) + "\n")
    return out


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _df_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


# ================================================================================
# 5. Gradio UI
# ================================================================================
STATE: Dict[str, object] = {"entries": [], "photo_paths": [], "results": None, "pca": None}


def cb_load_models(models_dir: str):
    df, status, entries = load_models(models_dir)
    try:
        pca = find_pca(resolve_models_root(models_dir))
    except Exception:
        pca = None
    STATE["entries"] = entries
    STATE["pca"] = pca
    return df, status


def cb_scan_photos(photos_dir: str):
    p = Path(photos_dir.strip().strip('"').strip("'")).expanduser()
    if not p.is_dir():
        raise gr.Error(f"Not a folder: {p}")
    paths = [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if not paths:
        raise gr.Error(f"No image files found under {p}")
    paths = paths[:MAX_IMAGES]
    STATE["photo_paths"] = [str(f) for f in paths]
    gallery = [(str(f), f.name) for f in paths]
    return gallery, f"Found {len(paths)} photo(s) in {p}"


def cb_uploads(files: List[str]):
    paths = [f for f in (files or []) if Path(str(f)).suffix.lower() in IMG_EXTS][:MAX_IMAGES]
    STATE["photo_paths"] = [str(f) for f in paths]
    gallery = [(str(f), Path(str(f)).name) for f in paths]
    return gallery, f"{len(paths)} uploaded photo(s) selected"


def cb_run(invert: str, autocontrast: bool, center_crop: bool, tta: bool):
    paths = STATE.get("photo_paths") or []
    entries = STATE.get("entries") or []
    ok_models = [e for e in entries if e.status == "ok"]
    if not paths:
        raise gr.Error("No photos: set a folder and press 'Scan folder' (or upload files) first.")
    if not ok_models:
        raise gr.Error("No loaded models: set the artifacts folder and press 'Load models' first.")

    processed, names = [], []
    for f in paths:
        try:
            img = Image.open(f)
            img.load()
            processed.append(to_fashion_mnist(img, invert=invert, autocontrast=autocontrast,
                                              center_crop=center_crop))
            names.append(Path(f).name)
        except Exception as exc:
            print(f"[skip] {f}: {exc}")

    res = run_inference(entries, processed, tta_mirror=tta)
    res.image_names = names
    res.settings = {"invert": invert, "autocontrast": autocontrast,
                    "center_crop": center_crop, "tta": tta}
    STATE["results"] = res

    # gallery of processed images (x8 nearest-neighbour for visibility)
    gallery = [(array_to_pil(a, scale=8), f"{n} → "
                + (CLASS_NAMES[int(res.probs[_first_ok(ok_models)][i].argmax())]
                   if _first_ok(ok_models) in res.probs else "?"))
               for i, (a, n) in enumerate(zip(processed, names))]

    wide = pd.DataFrame([{"image": n, **{m: f"{CLASS_NAMES[int(P[i].argmax())]} · {P[i].max() * 100:.1f}%"
                                         for m, P in res.probs.items()}}
                         for i, n in enumerate(names)])
    # per-image majority vote across all models (how much the committee agrees)
    maj = []
    for i in range(len(names)):
        preds = [int(P[i].argmax()) for P in res.probs.values()]
        if preds:
            vals, counts = np.unique(preds, return_counts=True)
            maj.append(f"{CLASS_NAMES[vals[counts.argmax()]]} ({counts.max()}/{len(preds)})")
        else:
            maj.append("—")
    wide.insert(1, "🗳 majority vote", maj)

    summary = (f"Processed {len(names)} photo(s) into Fashion-MNIST format (28x28 grayscale) "
               f"and ran {len(res.probs)} model(s). Select an image below for per-model details.")
    return gallery, wide, summary, gr.update(choices=list(names), value=names[0] if names else None)


def _first_ok(entries) -> Optional[str]:
    for e in entries:
        return e.name
    return None


def cb_detail(image_name: str):
    res: RunResults = STATE.get("results")
    if res is None or image_name not in res.image_names:
        return pd.DataFrame([{"info": "run the pipeline first"}])
    i = res.image_names.index(image_name)
    rows = []
    for m, P in res.probs.items():
        top3 = np.argsort(-P[i])[:3]
        rows.append({
            "model": m,
            "prediction": CLASS_NAMES[int(top3[0])],
            "confidence": f"{P[i][top3[0]] * 100:.1f}%",
            "2nd": f"{CLASS_NAMES[top3[1]]} {P[i][top3[1]] * 100:.1f}%",
            "3rd": f"{CLASS_NAMES[top3[2]]} {P[i][top3[2]] * 100:.1f}%",
        })
    return pd.DataFrame(rows)


def cb_export():
    res: RunResults = STATE.get("results")
    entries = STATE.get("entries") or []
    if res is None:
        raise gr.Error("Nothing to export yet: run 'Process & predict' first.")
    path = build_export_zip(res, entries, res.settings)
    return str(path), f"Export written: {path.name} " \
                      f"({path.stat().st_size / 1024:.0f} KB, {len(res.processed)} image(s), " \
                      f"{len(res.probs)} model(s))"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fashion-MNIST Inference Studio") as demo:
        gr.Markdown(
            """
            # Fashion-MNIST Inference Studio
            **Gradio front-end for the models exported by `DL-Fashion-MNIST.ipynb`**
            (`artifacts/models/{ml,dl,ensemble}`: sklearn pipelines, torch checkpoints and ensemble combiners).

            | Step | Action |
            |---|---|
            | **1** | Point at the folder with the exported **model artifacts** and load it |
            | **2** | Point at a folder with **sample photos** (or upload them) |
            | **3** | The app converts each photo to the **Fashion-MNIST format** (grayscale 28×28, garment-on-black) and runs **every model** |
            | **4** | **Export** the processed images + all predictions as a **.zip** |
            """
        )

        with gr.Row():
            # ------------------------- left: controls -------------------------
            with gr.Column(scale=5):
                gr.Markdown("### 1 · Models")
                models_dir = gr.Textbox(
                    value=DEFAULT_MODELS_DIR, label="Folder with model artifacts",
                    info="The `artifacts` folder exported by the notebook (contains `models/ml`, `models/dl`, "
                         "`models/ensemble`) — or the `models` folder itself.",
                    placeholder="e.g. /content/artifacts  or  D:/project/artifacts")
                btn_load = gr.Button("Load models", variant="primary")
                models_status = gr.Textbox(label="Status", interactive=False, lines=2)
                models_df = gr.Dataframe(
                    headers=["model", "family", "folder", "file", "status",
                             "test acc (notebook)", "note"],
                    label="Loaded models", interactive=False, wrap=True)

                gr.Markdown("### 2 · Sample photos")
                photos_dir = gr.Textbox(
                    value=DEFAULT_PHOTOS_DIR, label="Folder with sample photos",
                    info="Every image inside (sub-folders included) will be processed.",
                    placeholder="e.g. /content/sample_photos")
                btn_scan = gr.Button("Scan folder", variant="secondary")
                uploads = gr.Files(label="…or upload photos directly (jpg/png/bmp/webp…)")
                photos_status = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("### 3 · Process & predict")
                with gr.Accordion("Preprocessing options (photo → Fashion-MNIST)", open=True):
                    invert = gr.Radio(
                        ["auto", "yes", "no"], value="auto", label="Invert colours",
                        info="Fashion-MNIST garments are WHITE on a BLACK background; 'auto' inverts "
                             "photos shot on a light background.")
                    autocontrast = gr.Checkbox(value=True, label="Contrast normalisation (autocontrast)")
                    center_crop = gr.Checkbox(
                        value=True, label="Centre-crop to the garment + square pad (size normalisation)")
                    tta = gr.Checkbox(value=False, label="Mirror test-time augmentation (notebook cfgx.tta)")
                btn_run = gr.Button("Process photos & run all models", variant="primary")
                run_status = gr.Textbox(label="Status", interactive=False, lines=3)

                gr.Markdown("### 4 · Export")
                btn_export = gr.Button("Export processed photos + predictions (.zip)", variant="primary")
                export_status = gr.Textbox(label="Status", interactive=False, lines=2)
                export_file = gr.File(label="Download")

            # ------------------------- right: results -------------------------
            with gr.Column(scale=7):
                gr.Markdown("### Source photos")
                gallery_src = gr.Gallery(label="Found photos", columns=6, height=200, object_fit="contain")
                gr.Markdown("### Processed to Fashion-MNIST format (28×28, ×8 preview)")
                gallery_out = gr.Gallery(label="Processed (28×28)", columns=6, height=260, object_fit="contain")
                gr.Markdown("### Predictions (class · confidence)")
                preds_df = gr.Dataframe(label="Every model × every image", interactive=False, wrap=True)
                image_picker = gr.Radio(label="Inspect a single image", choices=[], value=None)
                detail_df = gr.Dataframe(
                    headers=["model", "prediction", "confidence", "2nd", "3rd"],
                    label="Per-model top-3 for the selected image", interactive=False)

        gr.Markdown(
            """
            ---
            **Notes** · Deep models receive `(px/255 - 0.2860)/0.3530` exactly as in notebook Section 1.6;
            sklearn pipelines receive the flattened `px/255` (784) vector they were fitted on. Boosting models
            trained on PCA features additionally require the exported PCA (`pca_boost.joblib`): the app
            explains this in the model table if it is missing. Classes:
            0 T-shirt/top · 1 Trouser · 2 Pullover · 3 Dress · 4 Coat · 5 Sandal · 6 Shirt · 7 Sneaker · 8 Bag · 9 Ankle boot.
            """
        )

        # ---------------------------- wiring ----------------------------
        btn_load.click(cb_load_models, inputs=[models_dir], outputs=[models_df, models_status])
        btn_scan.click(cb_scan_photos, inputs=[photos_dir], outputs=[gallery_src, photos_status])
        uploads.change(cb_uploads, inputs=[uploads], outputs=[gallery_src, photos_status])
        btn_run.click(cb_run,
                      inputs=[invert, autocontrast, center_crop, tta],
                      outputs=[gallery_out, preds_df, run_status, image_picker])
        image_picker.change(cb_detail, inputs=[image_picker], outputs=[detail_df])
        btn_export.click(cb_export, outputs=[export_file, export_status])
    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        allowed_paths=[str(APP_DIR)],
    )
