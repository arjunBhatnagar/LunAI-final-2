import csv

import pytest

from lunai.classification import classify_sites
from lunai.classification import landing_site
from lunai.classification.landing_site import LandingSiteClassifier
from lunai.services import analysis

pytestmark = pytest.mark.skipif(not landing_site.available(), reason="classifier outputs not installed")


def test_regions_show_the_classifiers_own_numbers():
    sites = classify_sites(analysis()["sites"], LandingSiteClassifier())
    for site in sites:
        result = site["classification"]
        folder = landing_site.SITES[site["site_number"]]
        with (landing_site.OUTPUTS / f"{folder}_candidates.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [c["lat"] for c in result["candidates"]] == [float(r["lat_deg"]) for r in rows]
        assert [c["score"] for c in result["candidates"]] == [float(r["composite_score"]) for r in rows]
        assert result["map"]["figure"] == folder
        assert sum(result["class_scores"].values()) == pytest.approx(result["candidates"][0]["score"], abs=2e-3)


def test_sites_without_results_are_rejected():
    with pytest.raises(ValueError, match="no classifier results"):
        LandingSiteClassifier().classify({"site_number": 99})


def test_figure_path_only_allows_known_sites():
    assert landing_site.figure_path("../secrets") is None
    assert landing_site.figure_path("Site2") is None
