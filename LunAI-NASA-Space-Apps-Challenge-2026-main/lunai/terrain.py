from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np


@lru_cache(maxsize=32)
def _cached_horizon_profile(slope_deg: float, roughness: float, elevation_m: float, azimuth_steps: int) -> tuple[float, ...]:
    """Fallback horizon proxy for sites without a DEM-derived ``horizon_deg`` profile."""
    azimuths = np.linspace(0, 360, azimuth_steps, endpoint=False)
    offset = elevation_m / 1000.0
    profile = []
    for az in azimuths:
        directional_bias = np.cos(np.radians(az - 180.0)) * 0.8
        horizon = offset + slope_deg * max(0.0, directional_bias) + roughness * 1.5
        profile.append(float(horizon))
    return tuple(profile)


def terrain_horizon_profile(site: Dict[str, Any], azimuth_steps: int = 360) -> List[float]:
    measured = site.get("horizon_deg")
    if measured:
        return [float(value) for value in measured]
    return list(
        _cached_horizon_profile(
            float(site.get("slope_deg", 4.0)),
            float(site.get("terrain_roughness", 0.8)),
            float(site.get("elevation_m", 0.0)),
            azimuth_steps,
        )
    )


def terrain_blocked(site: Dict[str, Any], azimuth_deg: float, elevation_deg: float) -> bool:
    """True when terrain hides a body at the given azimuth/elevation.

    Uses the DEM ray-traced ``horizon_deg`` profile when the site has one,
    otherwise a coarse slope/roughness proxy.
    """
    profile = site.get("horizon_deg") or _cached_horizon_profile(
        float(site.get("slope_deg", 4.0)),
        float(site.get("terrain_roughness", 0.8)),
        float(site.get("elevation_m", 0.0)),
        360,
    )
    position = (azimuth_deg % 360.0) / 360.0 * len(profile)
    lower = int(position) % len(profile)
    upper = (lower + 1) % len(profile)
    fraction = position - int(position)
    horizon_elev = float(profile[lower]) * (1 - fraction) + float(profile[upper]) * fraction
    return float(elevation_deg) < horizon_elev
