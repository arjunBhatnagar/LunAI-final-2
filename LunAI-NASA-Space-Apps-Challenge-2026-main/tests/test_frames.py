from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lunai.ephemeris import compute_ephemeris_for_site
from lunai.frames import earth_orientation, lat_lon_of, moon_orientation


def test_rotation_matrices_are_orthonormal():
    when = datetime(2026, 11, 14, tzinfo=timezone.utc)
    for matrix in (earth_orientation(when), moon_orientation(when)):
        assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_earth_prime_meridian_faces_away_from_sun_at_midnight_utc():
    # At 00:00 UTC on the March equinox the Sun (RA ~0) is over ~180 deg longitude.
    when = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    sun_direction = np.array([1.0, 0.0, 0.0])
    lat, lon = lat_lon_of(earth_orientation(when).T @ sun_direction)
    assert abs(lat) < 0.5
    assert abs(abs(lon) - 180) < 5


def test_moon_near_side_faces_earth():
    skyfield = pytest.importorskip("skyfield.api")
    from lunai.ephemeris import _skyfield_resources

    try:
        kernel, timescale = _skyfield_resources()
    except FileNotFoundError:
        pytest.skip("de441.bsp not available")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(0, 60, 5):
        when = start + timedelta(days=day)
        t = timescale.from_datetime(when)
        to_earth = kernel["Earth"].at(t).position.km - kernel["Moon"].at(t).position.km
        lat, lon = lat_lon_of(moon_orientation(when).T @ to_earth)
        assert abs(lat) < 8 and abs(lon) < 9  # within optical libration


def test_south_pole_sun_stays_near_horizon():
    for hour in (0, 8, 16):
        values = compute_ephemeris_for_site({"lat": -89.7, "lon": -90.0}, datetime(2026, 11, 13, hour, tzinfo=timezone.utc))
        if values["source"] != "skyfield":
            pytest.skip("DE441 ephemeris not available")
        assert abs(values["sun_elevation_deg"]) < 2.0
        assert abs(values["earth_elevation_deg"]) < 7.5
