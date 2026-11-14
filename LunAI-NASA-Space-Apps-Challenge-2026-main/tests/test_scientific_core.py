from datetime import datetime

from lunai.data_loader import validate_site_catalog
from lunai.ephemeris import compute_ephemeris_for_site
from lunai.solar_power import available_solar_power_watts
from lunai.terrain import terrain_blocked


def test_site_catalog_validation_keeps_valid_records():
    records = [
        {
            "id": "a",
            "name": "Alpha",
            "lat": -89.0,
            "lon": 25.0,
            "water_ice": 0.8,
            "slope_deg": 4.0,
            "terrain_roughness": 0.6,
            "elevation_m": -100.0,
            "earth_line_of_sight": 0.7,
        },
        {
            "id": "b",
            "name": "Bad",
            "lat": "bad",
            "lon": 10.0,
            "water_ice": 0.8,
            "slope_deg": 4.0,
            "terrain_roughness": 0.6,
            "elevation_m": -100.0,
            "earth_line_of_sight": 0.7,
        },
    ]
    cleaned = validate_site_catalog(records)
    assert len(cleaned) == 1
    assert cleaned[0]["name"] == "Alpha"


def test_ephemeris_returns_feasible_angles():
    site = {"lat": -89.7, "lon": -90.0}
    values = compute_ephemeris_for_site(site, datetime(2026, 11, 13, 12, 0))
    assert -180.0 <= values["sun_azimuth_deg"] <= 360.0
    assert -90.0 <= values["sun_elevation_deg"] <= 90.0
    assert values["sun_distance_km"] > 0
    assert values["earth_distance_km"] > 0


def test_terrain_and_power_metrics_are_consistent():
    site = {
        "lat": -89.7,
        "lon": -90.0,
        "slope_deg": 3.0,
        "terrain_roughness": 0.7,
        "elevation_m": -220.0,
    }
    assert isinstance(terrain_blocked(site, 0.0, 0.0), bool)
    power = available_solar_power_watts(45.0, panel_area_m2=10.0, panel_efficiency=0.3)
    assert power > 0.0
