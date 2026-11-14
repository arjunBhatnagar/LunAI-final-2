from datetime import datetime
from pathlib import Path

from lunai.data_loader import load_site_catalog
from lunai.site_selection import rank_sites


def test_ranking_returns_scores():
    sites = load_site_catalog(Path(__file__).resolve().parents[1])
    ranked = rank_sites(sites, datetime(2026, 11, 13, 12, 0), 24)
    assert len(ranked) == len(sites)
    assert ranked[0]["score"] >= ranked[-1]["score"]
