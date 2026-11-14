"""Train a landing-site safety classifier from topology features only.

Labels come from the Site1 and Site4 ROI polygons (human-vetted candidate
landing zones). This model only sees topology (slope, roughness, relief,
curvature, ...) and is used purely to detect physically safe-to-land terrain.
Communications (Earth visibility), water/ice proximity, and points of
interest are deliberately NOT part of this classifier - they're applied as
separate, explicitly-weighted ranking criteria on top of its output in
predict.py, so each factor's influence stays transparent and adjustable
instead of being blended into one opaque probability. Usage:

    python train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import dataset
from rf_lite import RandomForestLite

ROOT = Path(__file__).resolve().parent.parent.parent
WINDOW = 21

SITES = [
    (ROOT / "Site1", ROOT / "Site1" / "Site01_ROIs.shp"),
    (ROOT / "Site4", ROOT / "Site4" / "Site04_ROIs.shp"),
]


def make_model() -> RandomForestLite:
    return RandomForestLite(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=0,
    )


def main():
    print("Loading sites and building topology features (no external data)...")
    sites = [dataset.load_site(d, roi_shp=s, window=WINDOW, include_external=False) for d, s in SITES]
    for s in sites:
        print(f"  {s.name}: valid px={s.valid_mask.sum():,} roi px={s.roi_mask.sum():,}")

    # Leave-one-site-out cross validation: train on one site's labels, test on the other.
    print("\nLeave-one-site-out validation:")
    for i, test_site in enumerate(sites):
        train_sites = [s for j, s in enumerate(sites) if j != i]
        Xtr = np.concatenate([dataset.sample_training_pixels(s)[0] for s in train_sites])
        ytr = np.concatenate([dataset.sample_training_pixels(s)[1] for s in train_sites])
        Xte, yte, _ = dataset.sample_training_pixels(test_site, negative_ratio=50.0)

        model = make_model()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        ap = average_precision_score(yte, proba)
        print(f"  train={[s.name for s in train_sites]} test={test_site.name}: ROC-AUC={auc:.3f} AP={ap:.3f}")

    # Final model: train on all labeled sites combined.
    print("\nTraining final model on Site1 + Site4 combined...")
    Xs, ys = [], []
    for s in sites:
        X, y, _ = dataset.sample_training_pixels(s)
        Xs.append(X)
        ys.append(y)
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    model = make_model()
    model.fit(X_all, y_all)

    import features as feat
    importances = sorted(zip(feat.FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name:20s} {imp:.3f}")

    out_path = ROOT / "python" / "models" / "landing_site_rf.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "window": WINDOW, "feature_names": feat.FEATURE_NAMES}, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
