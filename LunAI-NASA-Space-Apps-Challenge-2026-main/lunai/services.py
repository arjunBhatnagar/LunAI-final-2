"""Back-end logic for the Streamlit page and the small HTTP API.

The site only displays results produced elsewhere: the landing-site
classifier's output files, and CSVs uploaded in the browser (or stored in
``data/mission_paths``). Nothing here runs a model on user input.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from lunai.classification import classify_sites, load_classifier
from lunai.missions import load_missions
from lunai.mission_timeline import generate_timeline
from lunai.simulation_data import generate_de441_state, load_simulation_state
from lunai.site_ml import analyze_lola_sites

ROOT = Path(__file__).resolve().parents[1]
LOLA_DIR = ROOT / "data" / "lola"
ANALYSIS_PATH = ROOT / "data" / "lola_site_analysis.json"
DEFAULT_SIMULATION_PATH = ROOT / "data" / "simulation_input.json"
MISSION_PATHS_DIR = ROOT / "data" / "mission_paths"
DE441_SOURCE = "JPL DE441 via Skyfield"


def current_epoch() -> str:
    """The scene epoch: the current UTC hour, so caches refresh hourly."""
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()


def _env(*names: str) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


@lru_cache(maxsize=4)
def _classifier(specification: str | None):
    return load_classifier(specification)


def classifier():
    return _classifier(_env("LUNAI_SITE_CLASSIFIER", "LUNAI_TERRAIN_MODEL"))


@lru_cache(maxsize=1)
def analysis() -> dict[str, Any]:
    if ANALYSIS_PATH.exists():
        return json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    return analyze_lola_sites(LOLA_DIR, ANALYSIS_PATH)


@lru_cache(maxsize=4)
def _simulation(epoch_iso: str, path: str | None) -> dict[str, Any]:
    if path:
        return load_simulation_state(Path(path))
    if DEFAULT_SIMULATION_PATH.is_file():
        return load_simulation_state(DEFAULT_SIMULATION_PATH)
    return generate_de441_state(datetime.fromisoformat(epoch_iso))


def simulation(epoch_iso: str) -> dict[str, Any]:
    return _simulation(epoch_iso, _env("LUNAI_SIMULATION_DATA"))


def _timeline_json(site: Mapping[str, Any], epoch: datetime) -> dict[str, Any]:
    timeline = generate_timeline(dict(site), epoch, duration_hours=24, step_minutes=30)
    return {
        "step_minutes": timeline["step_minutes"],
        "duration_hours": timeline["duration_hours"],
        "sun_visible_hours": timeline["sun_visible_hours"],
        "earth_visible_hours": timeline["earth_visible_hours"],
        "total_energy_kwh": timeline["total_energy_kwh"],
        "points": [
            {**point, "timestamp": point["timestamp"].isoformat().replace("+00:00", "Z")}
            for point in timeline["points"]
        ],
    }


def site_timelines(sites: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> dict[str, Any]:
    """24 h Sun/Earth visibility for the centre of each analysed region (DE441 inputs only)."""
    if state["source"] != DE441_SOURCE:
        return {}
    epoch = datetime.fromisoformat(str(state["epoch_utc"]).replace("Z", "+00:00"))
    epoch = epoch if epoch.tzinfo else epoch.replace(tzinfo=timezone.utc)
    return {
        site["id"]: _timeline_json({
            "name": site["name"], "lat": site["lat"], "lon": site["lon"],
            "slope_deg": site.get("median_slope_deg", 0.0),
            "terrain_roughness": site.get("median_roughness_m", 0.0),
            "elevation_m": site.get("median_elevation_m", 0.0),
            "horizon_deg": site.get("horizon_deg"),
        }, epoch)
        for site in sites
    }


def build_viewer_config(epoch_iso: str) -> dict[str, Any]:
    """Everything the 3D page needs at load time."""
    data = analysis()
    site_classifier = classifier()
    sites = classify_sites(data["sites"], site_classifier)
    state = simulation(epoch_iso)
    return {
        "epoch": epoch_iso,
        "sites": sites,
        "classifier": {"name": site_classifier.name},
        "simulation": state,
        "siteTimelines": site_timelines(sites, state),
        "dataSource": data["source"],
        "horizonNote": data.get("horizon_note", ""),
        "missions": load_missions(),
        "missionPaths": mission_path_ids(),
    }


def mission_path_ids() -> list[str]:
    """Missions with a flight-path CSV in data/mission_paths (named <mission id>.csv)."""
    known = {mission["id"] for mission in load_missions()["missions"]}
    if not MISSION_PATHS_DIR.is_dir():
        return []
    return sorted(path.stem for path in MISSION_PATHS_DIR.glob("*.csv") if path.stem in known)


def mission_path_csv(mission_id: str) -> str:
    if mission_id not in mission_path_ids():
        raise ValueError(f"No flight-path CSV for {mission_id!r}. Put one at data/mission_paths/{mission_id}.csv.")
    return (MISSION_PATHS_DIR / f"{mission_id}.csv").read_text(encoding="utf-8-sig")
