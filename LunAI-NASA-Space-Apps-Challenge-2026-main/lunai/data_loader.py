from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DEFAULT_CANDIDATES = [
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
        "notes": "High persistent shadow near crater rim with strong ice prospecting potential.",
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
        "notes": "Large impact crater with nearby permanent shadow and strong scientific value.",
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
        "notes": "Elevated rim with broad solar visibility but more rugged terrain and lower ice concentration.",
    },
]

REQUIRED_SITE_COLUMNS = {
    "id",
    "name",
    "lat",
    "lon",
    "water_ice",
    "slope_deg",
    "terrain_roughness",
    "elevation_m",
    "earth_line_of_sight",
}


def _create_default_csv(data_dir: Path) -> Path:
    csv_path = data_dir / "lunar_south_pole_data.csv"
    df = pd.DataFrame(DEFAULT_CANDIDATES)
    df.to_csv(csv_path, index=False)
    return csv_path


def _create_runtime_json(data_dir: Path) -> Path:
    json_path = data_dir / "runtime_sites.json"
    json_path.write_text(json.dumps(DEFAULT_CANDIDATES, indent=2), encoding="utf-8")
    return json_path


def discover_local_data(base_dir: Path | str) -> List[str]:
    base = Path(base_dir)
    dataset_paths = []
    if not base.exists():
        return dataset_paths
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".geojson", ".tif", ".tiff"}:
            dataset_paths.append(str(path.relative_to(base)))
    return dataset_paths


def validate_site_catalog(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        missing = REQUIRED_SITE_COLUMNS - set(rec)
        if missing:
            continue
        out = dict(rec)
        for field in ["water_ice", "slope_deg", "terrain_roughness", "elevation_m", "earth_line_of_sight", "lat", "lon"]:
            try:
                out[field] = float(out[field])
            except (TypeError, ValueError):
                out[field] = float("nan")
        if any(not pd.notna(out[field]) for field in ["lat", "lon", "water_ice", "slope_deg", "terrain_roughness"]):
            continue
        cleaned.append(out)
    return cleaned


def load_site_catalog(base_dir: Path | str) -> List[Dict[str, Any]]:
    base = Path(base_dir)
    data_dir = base / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "lunar_south_pole_data.csv"
    json_path = data_dir / "runtime_sites.json"

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                records = validate_site_catalog(df.to_dict(orient="records"))
                if records:
                    return records
        except Exception:
            pass

    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data:
                records = validate_site_catalog(data)
                if records:
                    return records
        except Exception:
            pass

    _create_default_csv(data_dir)
    _create_runtime_json(data_dir)
    return validate_site_catalog(DEFAULT_CANDIDATES)


def generation_summary(base_dir: Path | str) -> Dict[str, Any]:
    base = Path(base_dir)
    data_dir = base / "data"
    paths = discover_local_data(base)
    summary: Dict[str, Any] = {
        "available_datasets": paths,
        "data_dir": str(data_dir),
        "record_count": 0,
        "missing_values": {},
        "source_label": "local-simulated",
        "validated_records": 0,
        "dataset_quality": "unknown",
    }
    csv_path = data_dir / "lunar_south_pole_data.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            summary["record_count"] = int(len(df))
            summary["missing_values"] = {col: int(df[col].isna().sum()) for col in df.columns}
            summary["validated_records"] = len(validate_site_catalog(df.to_dict(orient="records")))
            summary["dataset_quality"] = "usable" if summary["validated_records"] else "invalid"
            summary["source_label"] = "local dataset"
        except Exception:
            summary["source_label"] = "local-simulated"
    return summary
