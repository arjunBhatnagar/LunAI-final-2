from __future__ import annotations

import math
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from lunai.frames import moon_orientation

MOON_RADIUS_KM = 1737.4


@lru_cache(maxsize=1)
def _skyfield_resources():
    from skyfield.api import load

    kernel_path = Path(__file__).resolve().parents[1] / "de441.bsp"
    if not kernel_path.exists():
        raise FileNotFoundError(kernel_path)
    return load(str(kernel_path)), load.timescale()


def _site_to_local_enu_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    n = np.array([
        -math.sin(lat) * math.cos(lon),
        -math.sin(lat) * math.sin(lon),
        math.cos(lat),
    ], dtype=float)
    u = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ], dtype=float)
    return e, n, u


def _vector_to_azimuth_elevation(vec: np.ndarray, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    e, n, u = _site_to_local_enu_basis(lat_deg, lon_deg)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return 0.0, -90.0
    x = float(vec @ e)
    y = float(vec @ n)
    z = float(vec @ u)
    az = math.degrees(math.atan2(x, y))
    az = (az + 360.0) % 360.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, z / norm))))
    return az, el


def _solar_declination_deg(day_of_year: int) -> float:
    angle = ((day_of_year - 81.0) / 365.0) * 2.0 * math.pi
    return 23.44 * math.sin(angle)


def _solar_azimuth_and_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    day_of_year = dt.timetuple().tm_yday
    declination_deg = _solar_declination_deg(day_of_year)
    declination_rad = math.radians(declination_deg)
    hour_fraction = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle_deg = (hour_fraction - 12.0) * 15.0
    hour_angle_rad = math.radians(hour_angle_deg)

    sin_elev = (
        math.sin(lat_rad) * math.sin(declination_rad)
        + math.cos(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)
    )
    elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    azimuth_rad = math.atan2(
        math.sin(hour_angle_rad),
        math.cos(hour_angle_rad) * math.sin(lat_rad) - math.tan(declination_rad) * math.cos(lat_rad),
    )
    azimuth_deg = math.degrees(azimuth_rad)
    azimuth_deg = (azimuth_deg + 180.0) % 360.0

    return float(azimuth_deg), float(elevation_deg)


def _earth_azimuth_and_elevation(sun_az_deg: float, sun_el_deg: float, dt: datetime) -> tuple[float, float]:
    earth_az = (sun_az_deg + 180.0) % 360.0
    phase = (dt.hour + dt.minute / 60.0) / 24.0 * 2.0 * math.pi
    earth_el = max(-35.0, min(35.0, 18.0 * math.sin(phase + 0.3) + 0.5 * sun_el_deg))
    return float(earth_az), float(earth_el)


@lru_cache(maxsize=8192)
def _cached_skyfield_ephemeris(lat: float, lon: float, dt: datetime) -> Dict[str, float]:
    try:
        kernels, ts = _skyfield_resources()
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        moon = kernels["Moon"]
        earth = kernels["Earth"]
        sun = kernels["Sun"]

        moon_pos = moon.at(t).position.km
        earth_pos = earth.at(t).position.km
        sun_pos = sun.at(t).position.km

        # Rotate ICRF vectors into the Moon's body-fixed frame, where the site lives.
        icrf_to_moon = moon_orientation(dt).T
        surface_vec = np.array([
            math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
            math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
            math.sin(math.radians(lat)),
        ], dtype=float) * MOON_RADIUS_KM

        sun_site_vec = icrf_to_moon @ np.asarray(sun_pos - moon_pos, dtype=float) - surface_vec
        earth_site_vec = icrf_to_moon @ np.asarray(earth_pos - moon_pos, dtype=float) - surface_vec

        sun_az, sun_el = _vector_to_azimuth_elevation(sun_site_vec, lat, lon)
        earth_az, earth_el = _vector_to_azimuth_elevation(earth_site_vec, lat, lon)

        return {
            "sun_azimuth_deg": float(sun_az),
            "sun_elevation_deg": float(sun_el),
            "earth_azimuth_deg": float(earth_az),
            "earth_elevation_deg": float(earth_el),
            "sun_distance_km": float(np.linalg.norm(sun_site_vec)),
            "earth_distance_km": float(np.linalg.norm(earth_site_vec)),
            "source": "skyfield",
        }
    except Exception:
        return {
            "source": "simulated",
        }


def compute_ephemeris_for_site(site: Dict[str, Any], dt: datetime) -> Dict[str, float]:
    skyfield_result = _cached_skyfield_ephemeris(
        float(site.get("lat", 0.0)),
        float(site.get("lon", 0.0)),
        dt,
    )
    if skyfield_result.get("source") == "skyfield":
        return skyfield_result

    sun_az, sun_el = _solar_azimuth_and_elevation(float(site.get("lat", 0.0)), float(site.get("lon", 0.0)), dt)
    earth_az, earth_el = _earth_azimuth_and_elevation(sun_az, sun_el, dt)
    return {
        "sun_azimuth_deg": float(sun_az),
        "sun_elevation_deg": float(sun_el),
        "earth_azimuth_deg": float(earth_az),
        "earth_elevation_deg": float(earth_el),
        "sun_distance_km": 384_400.0,
        "earth_distance_km": 384_400.0,
        "source": "simulated",
    }
