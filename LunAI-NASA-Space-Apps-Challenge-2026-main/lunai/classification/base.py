"""The site-classifier contract.

A classifier receives one analysed landing region (a mapping of LOLA summary
fields such as ``median_slope_deg``, ``low_slope_pct``, ``shadow_proxy_pct``,
``lat``/``lon`` and, when available, ``dem_path`` to the source GeoTIFF) and
returns a :class:`SiteClassification`. Anything with ``classify(site)`` works,
as does a plain function. Mapping results with ``terrain_class``,
``confidence`` and ``model`` keys (the old hook format) are still accepted.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SiteClassification:
    terrain_class: str
    confidence: float | None = None
    """Probability (0-1) of ``terrain_class``, if the model has one."""
    class_scores: Mapping[str, float] = field(default_factory=dict)
    """Per-class scores (0-1), shown as a bar in the details panel."""
    score_label: str = "Class probabilities"
    model: str = "custom classifier"
    details: Mapping[str, Any] = field(default_factory=dict)
    """Extra labelled values your model wants to show, e.g. ``{"Boulder density": "low"}``."""
    candidates: Sequence[Mapping[str, Any]] = ()
    """Recommended spots: each needs ``lat``/``lon`` and may add ``score`` (0-1), ``label`` and ``details``."""
    verdict: Mapping[str, str] | None = None
    """Plain-language rating: ``level`` (good / care / risky / unknown), ``title`` and ``text``."""
    map: Mapping[str, Any] | None = None
    """Optional image overlay for the details panel (see ``LandingSiteClassifier``)."""


@runtime_checkable
class SiteClassifier(Protocol):
    name: str

    def classify(self, site: Mapping[str, Any]) -> SiteClassification | Mapping[str, Any]: ...


class FunctionClassifier:
    def __init__(self, function: Callable[[Mapping[str, Any]], Any], name: str | None = None) -> None:
        self.function = function
        self.name = name or getattr(function, "__name__", "custom classifier")

    def classify(self, site: Mapping[str, Any]) -> Any:
        return self.function(site)


def as_classifier(candidate: Any) -> SiteClassifier:
    if callable(getattr(candidate, "classify", None)):
        if not getattr(candidate, "name", None):
            candidate.name = type(candidate).__name__
        return candidate
    if callable(candidate):
        return FunctionClassifier(candidate)
    raise TypeError("A classifier must have a classify(site) method or be a function.")


def classify_sites(sites: Sequence[Mapping[str, Any]], classifier: SiteClassifier) -> list[dict[str, Any]]:
    """Return copies of ``sites`` with a validated ``classification`` entry; inputs are not modified."""
    classified = []
    for index, site in enumerate(sites):
        label = site.get("name", f"site {index}")
        try:
            result = coerce_classification(classifier.classify(dict(site)), classifier.name)
        except Exception as exc:
            raise ValueError(f"Classifier {classifier.name!r} failed for {label}: {exc}") from exc
        classified.append({**site, "classification": classification_to_viewer(result)})
    return classified


def coerce_classification(result: Any, default_model: str) -> SiteClassification:
    if isinstance(result, Mapping):
        result = SiteClassification(
            terrain_class=result.get("terrain_class"),
            confidence=result.get("confidence"),
            class_scores=result.get("class_scores") or {},
            score_label=result.get("score_label", "Class probabilities"),
            model=result.get("model") or default_model,
            details=result.get("details") or {},
            candidates=result.get("candidates") or (),
            verdict=result.get("verdict"),
            map=result.get("map"),
        )
    if not isinstance(result, SiteClassification):
        raise ValueError("classifier output must be a SiteClassification or a mapping.")
    if not isinstance(result.terrain_class, str) or not result.terrain_class.strip():
        raise ValueError("classifier output needs a non-empty terrain_class.")
    if not isinstance(result.model, str) or not result.model.strip():
        raise ValueError("classifier output has an invalid model name.")
    if result.confidence is not None:
        _probability(result.confidence, "confidence")
    for name, score in result.class_scores.items():
        _probability(score, f"class_scores[{name!r}]")
    for index, candidate in enumerate(result.candidates):
        lat, lon = float(candidate["lat"]), float(candidate["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"candidates[{index}] has invalid coordinates.")
        if candidate.get("score") is not None:
            _probability(candidate["score"], f"candidates[{index}].score")
    if result.verdict is not None and result.verdict.get("level") not in ("good", "care", "risky", "unknown"):
        raise ValueError("verdict.level must be good, care, risky or unknown.")
    return result


def classification_to_viewer(result: SiteClassification) -> dict[str, Any]:
    return {
        "terrain_class": result.terrain_class.strip(),
        "confidence": None if result.confidence is None else float(result.confidence),
        "class_scores": {str(k): float(v) for k, v in result.class_scores.items()},
        "score_label": result.score_label,
        "model": result.model.strip(),
        "details": [{"label": str(k), "value": _display(v)} for k, v in result.details.items()],
        "candidates": [
            {
                **{k: v for k, v in candidate.items() if k != "details"},
                "lat": float(candidate["lat"]), "lon": float(candidate["lon"]),
                "details": [{"label": str(k), "value": _display(v)} for k, v in (candidate.get("details") or {}).items()],
            }
            for candidate in result.candidates
        ],
        "verdict": dict(result.verdict) if result.verdict else None,
        "map": dict(result.map) if result.map else None,
    }


def _probability(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return number


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)
