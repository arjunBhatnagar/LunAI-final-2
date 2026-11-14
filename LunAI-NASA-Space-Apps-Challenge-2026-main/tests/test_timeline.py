from datetime import datetime

from lunai.data_loader import load_site_catalog
from lunai.mission_timeline import generate_timeline


def test_timeline_has_visible_windows_and_power():
    site = load_site_catalog(__file__.rsplit("\\", 1)[0].replace("tests", ""))[0]
    timeline = generate_timeline(site, datetime(2026, 11, 13, 12, 0), 24, step_minutes=60)
    assert timeline["sun_visible_hours"] >= 0
    assert timeline["earth_visible_hours"] >= 0
    assert timeline["max_power_w"] >= 0
