from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SITE_COLUMNS = [
    "id",
    "name",
    "lat",
    "lon",
    "water_ice",
    "slope_deg",
    "terrain_roughness",
    "elevation_m",
    "earth_line_of_sight",
    "notes",
]

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data"
BASE.mkdir(exist_ok=True)

DEFAULT_ROWS = [
    {
        "id": "site_shackleton",
        "name": "Shackleton Rim",
        "lat": -89.7,
        "lon": -90.0,
        "water_ice": 0.74,
        "slope_deg": 3.1,
        "terrain_roughness": 0.7,
        "elevation_m": -220.0,
        "earth_line_of_sight": 0.72,
        "notes": "Nearest permanent-shadow near the south pole.",
    },
    {
        "id": "site_amundsen",
        "name": "Amundsen Crater",
        "lat": -84.2,
        "lon": 55.0,
        "water_ice": 0.62,
        "slope_deg": 4.8,
        "terrain_roughness": 1.2,
        "elevation_m": -310.0,
        "earth_line_of_sight": 0.68,
        "notes": "Promising crater floor with thermal reserves.",
    },
    {
        "id": "site_nobile",
        "name": "Nobile Rim",
        "lat": -86.5,
        "lon": 30.0,
        "water_ice": 0.58,
        "slope_deg": 6.5,
        "terrain_roughness": 1.7,
        "elevation_m": 480.0,
        "earth_line_of_sight": 0.61,
        "notes": "Higher elevation and broad solar access.",
    },
]

SOURCE = BASE / "lunar_south_pole_data.csv"
if not SOURCE.exists():
    pd.DataFrame(DEFAULT_ROWS, columns=SITE_COLUMNS).to_csv(SOURCE, index=False)

filtered = pd.read_csv(SOURCE)
filtered = filtered.loc[:, [c for c in SITE_COLUMNS if c in filtered.columns]]
(BASE / "runtime_sites.json").write_text(json.dumps(filtered.to_dict(orient="records"), indent=2), encoding="utf-8")
print(f"Preprocessed {len(filtered)} candidate sites to {BASE / 'runtime_sites.json'}")
