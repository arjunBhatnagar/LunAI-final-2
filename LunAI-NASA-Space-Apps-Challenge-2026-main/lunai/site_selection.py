from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from lunai.mission_timeline import generate_timeline

DEFAULT_WEIGHTS = {
    "water_ice": 0.30,
    "terrain": 0.20,
    "sun_visibility": 0.20,
    "earth_visibility": 0.15,
    "solar_energy": 0.15,
}


def _normalise(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def score_site(site: Dict[str, Any], start_dt: datetime, duration_hours: float, weights: Dict[str, float] | None = None) -> Dict[str, Any]:
    weights = weights or DEFAULT_WEIGHTS
    timeline = generate_timeline(site, start_dt, duration_hours, step_minutes=30)
    water_score = _normalise(float(site.get("water_ice", 0.0)), 0.0, 1.0)
    terrain_score = 1.0 - _normalise(float(site.get("slope_deg", 0.0)), 0.0, 15.0)
    sun_score = _normalise(timeline["sun_visible_hours"], 0.0, max(duration_hours, 1.0))
    earth_score = _normalise(timeline["earth_visible_hours"], 0.0, max(duration_hours, 1.0))
    energy_score = _normalise(timeline["total_energy_kwh"], 0.0, max(250.0, timeline["total_energy_kwh"]))

    total = (
        water_score * weights.get("water_ice", 0.0)
        + terrain_score * weights.get("terrain", 0.0)
        + sun_score * weights.get("sun_visibility", 0.0)
        + earth_score * weights.get("earth_visibility", 0.0)
        + energy_score * weights.get("solar_energy", 0.0)
    )

    return {
        **site,
        "score": float(total),
        "timeline": timeline,
        "metrics": {
            "water_ice_score": water_score,
            "terrain_score": terrain_score,
            "sun_visibility_score": sun_score,
            "earth_visibility_score": earth_score,
            "energy_score": energy_score,
            "sun_visible_hours": timeline["sun_visible_hours"],
            "earth_visible_hours": timeline["earth_visible_hours"],
            "total_energy_kwh": timeline["total_energy_kwh"],
            "max_power_w": timeline["max_power_w"],
        },
    }


def rank_sites(sites: List[Dict[str, Any]], start_dt: datetime, duration_hours: float, weights: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    ranked = [score_site(site, start_dt, duration_hours, weights) for site in sites]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)
