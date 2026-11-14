"""Pluggable lunar site classification.

Write a class with ``name`` and ``classify(site) -> SiteClassification`` and
point ``LUNAI_SITE_CLASSIFIER=module:ClassName`` at it. See
``examples/custom_classifier.py``.
"""
from __future__ import annotations

import os

from lunai.classification.base import (
    FunctionClassifier,
    SiteClassification,
    SiteClassifier,
    as_classifier,
    classify_sites,
    coerce_classification,
)
from lunai.classification.landing_site import LandingSiteClassifier
from lunai.classification.landing_site import available as landing_classifier_available
from lunai.classification.random_forest import LolaRandomForestClassifier
from lunai.plugins import load_object

ENV_VAR = "LUNAI_SITE_CLASSIFIER"
LEGACY_ENV_VAR = "LUNAI_TERRAIN_MODEL"


def load_classifier(specification: str | None = None) -> SiteClassifier:
    """The classifier named by ``specification`` / ``$LUNAI_SITE_CLASSIFIER``.

    Defaults to the landing-site classifier in ``classifier/`` when present,
    otherwise the built-in terrain Random Forest.
    """
    specification = specification or os.environ.get(ENV_VAR) or os.environ.get(LEGACY_ENV_VAR)
    if not specification:
        return LandingSiteClassifier() if landing_classifier_available() else LolaRandomForestClassifier()
    return as_classifier(load_object(specification, ENV_VAR))


__all__ = [
    "FunctionClassifier",
    "LandingSiteClassifier",
    "LolaRandomForestClassifier",
    "SiteClassification",
    "SiteClassifier",
    "as_classifier",
    "classify_sites",
    "coerce_classification",
    "load_classifier",
]
