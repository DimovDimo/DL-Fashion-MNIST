"""Headless smoke test of the app pipeline (no UI)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import app as A
from fashion_arch import CLASS_NAMES

# 1) load models
df, status, entries = A.load_models(str(Path(__file__).parent / "artifacts"))
print("STATUS:", status)
print(df[["model", "family", "status"]].to_string(index=False))
ok = [e for e in entries if e.status == "ok"]
assert len(ok) >= 10, "too few usable models"

# 2) photos
root = Path(__file__).parent / "sample_photos"
paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in A.IMG_EXTS)
print(f"\nphotos: {len(paths)}")
for p in paths:
    print("  ", p.relative_to(root))

# 3) preprocess + predict
imgs, names = [], []
for p in paths:
    im = Image.open(p); im.load()
    imgs.append(A.to_fashion_mnist(im, invert="auto", autocontrast=True, center_crop=True))
    names.append(p.name)

res = A.run_inference(entries, imgs, tta_mirror=False)
res.image_names = names
res.settings = {"invert": "auto", "autocontrast": True, "center_crop": True, "tta": False}

print(f"\n{'image':38s} {'best-model pred':22s} majority")
from collections import Counter
for i, n in enumerate(names):
    preds = {m: int(P[i].argmax()) for m, P in res.probs.items()}
    best = list(res.probs)[0]
    maj = Counter(preds.values()).most_common(1)[0]
    print(f"{n[:38]:38s} {CLASS_NAMES[preds[best]]:22s} {CLASS_NAMES[maj[0]]} ({maj[1]}/{len(preds)})")

# 4) export
zp = A.build_export_zip(res, entries, res.settings)
print("\nZIP:", zp.name, f"{zp.stat().st_size/1024:.0f} KB")
import zipfile
with zipfile.ZipFile(zp) as z:
    for n in z.namelist()[:6]:
        print("  ", n)
    print("   ...", len(z.namelist()), "files total")
print("\nALL GOOD ✅")
