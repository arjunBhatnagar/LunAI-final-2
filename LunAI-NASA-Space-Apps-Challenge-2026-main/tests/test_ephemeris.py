from datetime import datetime

from lunai.ephemeris import compute_ephemeris_for_site


def test_sun_and_earth_angles_are_numeric():
    site = {"lat": -89.7, "lon": -90.0}
    metrics = compute_ephemeris_for_site(site, datetime(2026, 11, 13, 12, 0))
    assert metrics["sun_elevation_deg"] is not None
    assert metrics["earth_elevation_deg"] is not None
    assert -180.0 <= metrics["sun_azimuth_deg"] <= 360.0
