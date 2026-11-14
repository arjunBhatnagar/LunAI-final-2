"""Template for plugging your own lunar site classifier into LunAI.

Run with:
    $env:LUNAI_SITE_CLASSIFIER = "examples.custom_classifier:SlopeRuleClassifier"   # PowerShell
    streamlit run app.py

``classify(site)`` receives one analysed LOLA region. Useful keys include
``median_slope_deg``, ``p90_slope_deg``, ``median_roughness_m``,
``low_slope_pct``, ``shadow_proxy_pct``, ``landable_near_shelter_pct``,
``median_elevation_m``, ``lat``, ``lon``, ``horizon_deg`` (360 values) and
``dem_path`` (the source GeoTIFF, if present, for models that need raw pixels).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from lunai.classification import SiteClassification


class SlopeRuleClassifier:
    name = "Example slope rule"

    def classify(self, site: Mapping[str, Any]) -> SiteClassification:
        slope = float(site["median_slope_deg"])
        # A soft-max over distances to class centres, standing in for real model probabilities.
        centres = {"flat": 3.0, "rolling": 10.0, "rugged": 20.0}
        weights = {name: math.exp(-abs(slope - centre) / 4.0) for name, centre in centres.items()}
        total = sum(weights.values())
        scores = {name: weight / total for name, weight in weights.items()}
        best = max(scores, key=scores.get)
        return SiteClassification(
            terrain_class=best,
            confidence=scores[best],
            class_scores=scores,
            model=self.name,
            # Anything here is listed under the classification in the Site tab.
            details={"Max horizon": f"{max(site.get('horizon_deg') or [0]):.1f}°", "Slope used": f"{slope:.1f}°"},
        )
