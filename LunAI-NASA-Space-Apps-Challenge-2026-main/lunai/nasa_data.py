from __future__ import annotations

from typing import Any, Dict

import requests

NASA_API_KEY = "DEMO_KEY"


def _safe_json(url: str) -> Dict[str, Any] | None:
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_hazard_summary() -> Dict[str, Any]:
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date=2026-11-01&end_date=2026-11-07&api_key={NASA_API_KEY}"
    payload = _safe_json(url)
    if not payload or "near_earth_objects" not in payload:
        return {
            "source": "simulated",
            "source_label": "Simulated NASA fallback",
            "note": "NASA feed unavailable; using a simulated near-Earth hazard summary.",
            "hazard_count": 4,
            "closest_object_km": 1_820_000.0,
            "risk_level": "Low",
            "body_names": ["2026 UB4", "2026 YR12", "2026 PF7", "2026 XN3"],
        }

    objects = []
    for day in payload.get("near_earth_objects", {}).values():
        for asteroid in day:
            objects.append({
                "name": asteroid.get("name", "Unknown"),
                "miss_distance_km": asteroid.get("close_approach_data", [{}])[0].get("miss_distance", {}).get("kilometers", 0),
            })
    closest = min((float(obj["miss_distance_km"]) for obj in objects if obj["miss_distance_km"]), default=0.0)
    return {
        "source": "nasa",
        "source_label": "NASA NEOWISE feed",
        "note": "Live near-Earth object data from NASA.",
        "hazard_count": len(objects),
        "closest_object_km": float(closest),
        "risk_level": "Low" if closest > 1_000_000 else "Elevated",
        "body_names": [obj["name"] for obj in objects[:5]],
    }
