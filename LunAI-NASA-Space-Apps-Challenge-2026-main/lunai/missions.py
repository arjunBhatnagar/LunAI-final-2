"""Catalogue of past and planned lunar missions shown in the viewer's Missions panel."""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

MISSIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "lunar_missions.json"
STATUSES = {"success", "partial", "failed", "planned"}
REQUIRED = ("id", "name", "agency", "type", "status", "summary", "description")


@lru_cache(maxsize=1)
def load_missions(path: Path = MISSIONS_PATH) -> dict[str, Any]:
    """Validated mission catalogue, past missions in date order followed by planned ones."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for mission in data["missions"]:
        missing = [key for key in REQUIRED if not mission.get(key)]
        if missing:
            raise ValueError(f"Mission {mission.get('id', '?')} is missing {', '.join(missing)}.")
        if mission["id"] in seen:
            raise ValueError(f"Duplicate mission id {mission['id']}.")
        seen.add(mission["id"])
        if mission["status"] not in STATUSES:
            raise ValueError(f"Mission {mission['id']} has unknown status {mission['status']!r}.")
        if mission["status"] == "planned":
            if not mission.get("target"):
                raise ValueError(f"Planned mission {mission['id']} needs a target date.")
        else:
            date.fromisoformat(mission["launch"])
        site = mission.get("site")
        if site and not (-90 <= site["lat"] <= 90 and -180 <= site["lon"] <= 180):
            raise ValueError(f"Mission {mission['id']} has an invalid landing site.")
    data["missions"].sort(key=lambda m: (m["status"] == "planned", m.get("launch") or ""))
    return data
