import json
from pathlib import Path

import pytest

from lunai.classification import (
    LolaRandomForestClassifier,
    SiteClassification,
    as_classifier,
    classify_sites,
    load_classifier,
)

ANALYSIS = json.loads((Path(__file__).resolve().parents[1] / "data" / "lola_site_analysis.json").read_text(encoding="utf-8"))


def test_default_classifier_reports_random_forest_shares():
    sites = classify_sites(ANALYSIS["sites"], LolaRandomForestClassifier())
    for site in sites:
        result = site["classification"]
        assert result["terrain_class"] == site["terrain_class_rf"]
        assert sum(result["class_scores"].values()) == pytest.approx(1.0, abs=1e-6)
        assert result["confidence"] is None


def test_custom_classifier_adds_details_without_mutating_input():
    class Model:
        name = "lunar-v2"

        def classify(self, site):
            return SiteClassification("highland", confidence=0.91, class_scores={"highland": 0.91, "mare": 0.09},
                                      model=self.name, details={"Boulders": "few", "Ice index": 0.123456})

    site = {"id": "site-01", "name": "A", "terrain_class_rf": "rolling"}
    classified = classify_sites([site], Model())[0]["classification"]
    assert "classification" not in site
    assert classified["terrain_class"] == "highland"
    assert classified["details"] == [{"label": "Boulders", "value": "few"}, {"label": "Ice index", "value": "0.123"}]


def test_legacy_function_output_is_accepted():
    classifier = as_classifier(lambda site: {"terrain_class": "flat", "confidence": 0.5, "model": "old"})
    result = classify_sites([{"name": "A"}], classifier)[0]["classification"]
    assert result == {"terrain_class": "flat", "confidence": 0.5, "class_scores": {}, "score_label": "Class probabilities",
                      "model": "old", "details": [], "candidates": [], "verdict": None, "map": None}


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, "SiteClassification or a mapping"),
        ({"terrain_class": ""}, "non-empty terrain_class"),
        ({"terrain_class": "flat", "confidence": 1.1}, "between 0 and 1"),
        ({"terrain_class": "flat", "class_scores": {"flat": -0.1}}, "between 0 and 1"),
    ],
)
def test_invalid_results_are_rejected_with_site_name(result, message):
    with pytest.raises(ValueError, match=f"Site X.*{message}"):
        classify_sites([{"name": "Site X"}], as_classifier(lambda site: result))


def test_example_classifier_loads_from_spec():
    classifier = load_classifier("examples.custom_classifier:SlopeRuleClassifier")
    result = classify_sites(ANALYSIS["sites"][:1], classifier)[0]["classification"]
    assert result["model"] == "Example slope rule"
    assert 0 < result["confidence"] <= 1


def test_default_is_landing_site_classifier_when_installed(monkeypatch):
    monkeypatch.delenv("LUNAI_SITE_CLASSIFIER", raising=False)
    monkeypatch.delenv("LUNAI_TERRAIN_MODEL", raising=False)
    from lunai.classification import LandingSiteClassifier
    from lunai.classification.landing_site import available

    assert isinstance(load_classifier(), LandingSiteClassifier if available() else LolaRandomForestClassifier)
