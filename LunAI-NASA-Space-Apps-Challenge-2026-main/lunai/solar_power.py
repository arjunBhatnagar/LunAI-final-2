from __future__ import annotations

import math
from typing import Dict, List

SOLAR_CONSTANT_WM2 = 1361.0


def solar_irradiance_wm2(sun_elevation_deg: float) -> float:
    if sun_elevation_deg <= 0:
        return 0.0
    return SOLAR_CONSTANT_WM2 * max(0.0, math.sin(math.radians(sun_elevation_deg)))


def available_solar_power_watts(
    sun_elevation_deg: float,
    panel_area_m2: float = 10.0,
    panel_efficiency: float = 0.27,
    panel_tilt_deg: float = 20.0,
    terrain_blocked: bool | None = None,
) -> float:
    """Electrical power from a panel that tracks the Sun in azimuth.

    ``panel_tilt_deg`` is the elevation of the panel normal above the horizon.
    With no atmosphere, the panel receives the full solar constant scaled by
    the cosine of the Sun-to-normal angle. ``terrain_blocked`` means the Sun is
    not visible from the site; without that information, a Sun below 0 deg is
    treated as set.
    """
    if terrain_blocked or sun_elevation_deg <= -5.0 or (sun_elevation_deg <= 0 and terrain_blocked is None):
        return 0.0
    incidence = max(0.0, math.cos(math.radians(sun_elevation_deg - panel_tilt_deg)))
    return float(SOLAR_CONSTANT_WM2 * incidence * panel_area_m2 * panel_efficiency)


def integrate_energy_over_timeline(points: List[Dict[str, float]], step_hours: float = 0.25) -> float:
    total_wh = 0.0
    for point in points:
        power_w = float(point.get("power_w", 0.0))
        total_wh += power_w * step_hours * 3600.0 / 1000.0
    return total_wh / 1000.0
