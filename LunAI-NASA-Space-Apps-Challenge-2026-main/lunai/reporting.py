from __future__ import annotations

from typing import Any, Dict


def build_report(site_results: Dict[str, Any], selected_site: Dict[str, Any], timeline: Dict[str, Any], hazard_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected_site": selected_site.get("name", "Unknown"),
        "site_coordinates": {
            "lat_deg": selected_site.get("lat"),
            "lon_deg": selected_site.get("lon"),
        },
        "solar_visibility_hours": timeline["sun_visible_hours"],
        "earth_visibility_hours": timeline["earth_visible_hours"],
        "solar_energy_kwh": timeline["total_energy_kwh"],
        "terrain": {
            "slope_deg": selected_site.get("slope_deg"),
            "roughness": selected_site.get("terrain_roughness"),
            "elevation_m": selected_site.get("elevation_m"),
        },
        "site_suitability_score": site_results.get("score"),
        "hazard_summary": hazard_summary,
        "data_sources": [
            "local candidate dataset (CSV)",
            "ephemeris via Skyfield de441.bsp",
            "hazard feed: " + hazard_summary.get("source_label", "simulated"),
        ],
    }
