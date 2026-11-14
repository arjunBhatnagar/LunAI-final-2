"""Built-in classifier: the LOLA terrain Random Forest trained by ``lunai.site_ml``."""
from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from lunai.classification.base import SiteClassification

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "lola" / "terrain_random_forest.joblib"
CLASSES = ("flat", "rolling", "rugged")
# Accepted input names for each model feature (first match wins).
FEATURE_ALIASES = {
    "slope_deg": ("slope_deg", "median_slope_deg", "slope"),
    "roughness_m": ("roughness_m", "median_roughness_m", "roughness", "terrain_roughness"),
    "local_position_m": ("local_position_m", "local_position"),
    "shelter_depth_m": ("shelter_depth_m", "shelter_depth"),
}


@lru_cache(maxsize=1)
def _forest():
    import joblib

    return joblib.load(MODEL_PATH)


def features_from(site: Mapping[str, Any]) -> list[float]:
    """Model inputs from a site mapping; slope and roughness are required, the rest default to 0."""
    values = []
    for feature, aliases in FEATURE_ALIASES.items():
        raw = next((site[name] for name in aliases if site.get(name) not in (None, "")), None)
        if raw is None:
            if feature in ("slope_deg", "roughness_m"):
                raise ValueError(f"needs a value for {feature.replace('_deg', '').replace('_m', '')} ({', '.join(aliases[:2])}).")
            raw = 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{feature.replace('_deg', '').replace('_m', '').replace('_', ' ')} must be a number, got {raw!r}.") from None
        if not math.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")
        if feature == "slope_deg" and not 0 <= value <= 90:
            raise ValueError("slope must be between 0 and 90 degrees.")
        if feature in ("roughness_m", "shelter_depth_m") and value < 0:
            raise ValueError(f"{feature} cannot be negative.")
        values.append(value)
    return values


class LolaRandomForestClassifier:
    """Terrain class (flat / rolling / rugged) from slope, roughness and optional shelter depth.

    Analysed LOLA regions report the share of their sampled pixels in each class.
    Anything else (for example values typed into the website) is run through
    the trained forest directly. The forest was trained on slope/roughness
    threshold labels, so it reproduces those rules rather than independent ground truth.
    """

    name = "LOLA Random Forest (rule-derived labels)"

    def classify(self, site: Mapping[str, Any]) -> SiteClassification:
        shares = site.get("terrain_class_pct_rf")
        if shares:
            scores = {name: float(pct) / 100.0 for name, pct in shares.items()}
            return SiteClassification(
                terrain_class=site.get("terrain_class_rf") or max(scores, key=scores.get),
                class_scores=scores,
                score_label="Share of sampled pixels",
                model=self.name,
                details={"Training samples": f"{int(site.get('terrain_model_samples', 0)):,}"}
                if site.get("terrain_model_samples") else {},
            )
        probabilities = _forest().predict_proba([features_from(site)])[0]
        labels = [CLASSES[int(label)] for label in _forest().classes_]
        scores = {label: float(p) for label, p in zip(labels, probabilities)}
        best = max(scores, key=scores.get)
        return SiteClassification(
            terrain_class=best,
            confidence=scores[best],
            class_scores=scores,
            model=self.name,
        )
