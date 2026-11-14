from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from lunai.ephemeris import compute_ephemeris_for_site
from lunai.solar_power import available_solar_power_watts
from lunai.terrain import terrain_blocked


def generate_timeline(site: Dict[str, Any], start_dt: datetime, duration_hours: float, step_minutes: int = 15) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    total_steps = max(1, int(duration_hours * 60 / step_minutes))
    for idx in range(total_steps + 1):
        time = start_dt + timedelta(minutes=idx * step_minutes)
        ephemeris = compute_ephemeris_for_site(site, time)
        blocked_sun = terrain_blocked(site, ephemeris["sun_azimuth_deg"], ephemeris["sun_elevation_deg"])
        blocked_earth = terrain_blocked(site, ephemeris["earth_azimuth_deg"], ephemeris["earth_elevation_deg"])
        if site.get("horizon_deg"):
            # A DEM horizon already includes curvature, so a ridge-top site can see slightly below 0 deg.
            sun_visible, earth_visible = not blocked_sun, not blocked_earth
        else:
            sun_visible = ephemeris["sun_elevation_deg"] > 0 and not blocked_sun
            earth_visible = ephemeris["earth_elevation_deg"] > 0 and not blocked_earth
        power_w = available_solar_power_watts(
            ephemeris["sun_elevation_deg"],
            panel_area_m2=12.0,
            panel_efficiency=0.28,
            panel_tilt_deg=25.0,
            terrain_blocked=not sun_visible,
        )
        points.append({
            "timestamp": time,
            "sun_azimuth_deg": ephemeris["sun_azimuth_deg"],
            "sun_elevation_deg": ephemeris["sun_elevation_deg"],
            "earth_azimuth_deg": ephemeris["earth_azimuth_deg"],
            "earth_elevation_deg": ephemeris["earth_elevation_deg"],
            "sun_visible": bool(sun_visible),
            "earth_visible": bool(earth_visible),
            "terrain_blocked_sun": bool(blocked_sun),
            "terrain_blocked_earth": bool(blocked_earth),
            "power_w": float(power_w),
            "ephemeris_source": ephemeris.get("source", "unknown"),
        })

    # Each sample stands for the interval that follows it, so the final sample closes the window.
    intervals = points[:-1] or points
    sun_visible_hours = sum(1 for p in intervals if p["sun_visible"]) * (step_minutes / 60.0)
    earth_visible_hours = sum(1 for p in intervals if p["earth_visible"]) * (step_minutes / 60.0)
    max_power = max((p["power_w"] for p in points), default=0.0)

    sun_events = []
    prev = points[0]["sun_visible"] if points else False
    for point in points[1:]:
        current = point["sun_visible"]
        if current != prev:
            sun_events.append({"time": point["timestamp"], "type": "sunrise" if current else "sunset", "state": current})
        prev = current

    earth_events = []
    prev = points[0]["earth_visible"] if points else False
    for point in points[1:]:
        current = point["earth_visible"]
        if current != prev:
            earth_events.append({"time": point["timestamp"], "type": "earthrise" if current else "earthset", "state": current})
        prev = current

    return {
        "points": points,
        "step_minutes": step_minutes,
        "duration_hours": duration_hours,
        "sun_visible_hours": float(sun_visible_hours),
        "earth_visible_hours": float(earth_visible_hours),
        "total_energy_kwh": float(sum(p["power_w"] for p in intervals) * step_minutes * 60 / 3_600_000.0),
        "max_power_w": float(max_power),
        "sun_events": sun_events,
        "earth_events": earth_events,
    }


def find_alternative_windows(candidate_sites: List[Dict[str, Any]], start_date: datetime, duration_hours: float, minimum_sun_hours: float, minimum_earth_hours: float, minimum_energy_kwh: float, max_slope_deg: float, search_days: int = 30):
    candidates = []
    for offset in range(search_days):
        date = start_date + timedelta(days=offset)
        for site in candidate_sites:
            timeline = generate_timeline(site, date.replace(hour=12, minute=0, second=0), duration_hours, step_minutes=30)
            if site.get("slope_deg", 0.0) > max_slope_deg:
                continue
            if timeline["sun_visible_hours"] < minimum_sun_hours:
                continue
            if timeline["earth_visible_hours"] < minimum_earth_hours:
                continue
            if timeline["total_energy_kwh"] < minimum_energy_kwh:
                continue
            candidates.append({
                "site": site["name"],
                "date": date,
                "sun_visible_hours": timeline["sun_visible_hours"],
                "earth_visible_hours": timeline["earth_visible_hours"],
                "total_energy_kwh": timeline["total_energy_kwh"],
                "slope_deg": site.get("slope_deg", 0.0),
            })
    return sorted(candidates, key=lambda row: (row["total_energy_kwh"], row["sun_visible_hours"]), reverse=True)
