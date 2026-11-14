from pathlib import Path

from lunai.data_loader import load_site_catalog


def test_load_site_catalog_returns_candidates():
    sites = load_site_catalog(Path(__file__).resolve().parents[1])
    assert len(sites) >= 3
    assert all("name" in site for site in sites)
