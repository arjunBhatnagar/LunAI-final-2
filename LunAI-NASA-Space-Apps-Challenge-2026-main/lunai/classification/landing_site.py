"""Read-only adapter for the landing-site classifier's results in ``classifier/LunAI-Classifier2-main``.

LunAI does not run that classifier. It only displays what the classifier
already wrote to ``python/outputs``:

* ``Site*_candidates.csv``: the top-ranked landing spots with their scores
  (60% topology + 30% Earth visibility + 10% ice proximity, as computed there);
* ``Site*_pois.csv``: nearby named craters;
* ``Site*_prediction.png``: the classifier's own figure of its input layers
  and ranked candidates (served by ``/api/classifier-figure/<site>``).
"""
from __future__ import annotations

import csv
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from lunai.classification.base import SiteClassification

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = Path(os.environ.get("LUNAI_LANDING_CLASSIFIER_DIR", ROOT / "classifier" / "LunAI-Classifier2-main"))
OUTPUTS = CLASSIFIER_DIR / "python" / "outputs"
SITES = {1: "Site1", 4: "Site4", 23: "Site23"}
# The weights the classifier used for its composite score (from its predict.py), used to split the score for display.
WEIGHTS = {"ground safety": 0.60, "Earth in view": 0.30, "near ice": 0.10}
WATER_DISTANCE_CAP_KM = 50.0


def available() -> bool:
    return any((OUTPUTS / f"{name}_candidates.csv").is_file() for name in SITES.values())


def figure_path(name: str) -> Path | None:
    """The classifier's prediction figure for ``Site1`` / ``Site4`` / ``Site23``, if present."""
    if name not in SITES.values():
        return None
    path = OUTPUTS / f"{name}_prediction.png"
    return path if path.is_file() else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_pois(path: Path) -> list[dict[str, str]]:
    """The classifier writes feature types like ``Crater, craters`` unquoted, so rows can have
    extra fields; the surplus belongs to the ``type`` column."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header, records = rows[0], []
    type_at = header.index("type")
    for row in rows[1:]:
        extra = len(row) - len(header)
        if extra > 0:
            row = row[:type_at] + [",".join(row[type_at:type_at + extra + 1]).strip()] + row[type_at + extra + 1:]
        if len(row) == len(header):
            records.append(dict(zip(header, row)))
    return records


def _number(row: Mapping[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def results(site_number: int) -> dict[str, Any] | None:
    """The classifier's output files for one site, parsed (nothing is recalculated)."""
    name = SITES.get(site_number)
    rows = _read_csv(OUTPUTS / f"{name}_candidates.csv") if name else []
    if not rows:
        return None
    candidates = []
    for index, row in enumerate(rows, 1):
        details = {}
        for label, key in (("Ground safety", "topology_prob"), ("Earth in view", "communications_earth_vis"), ("Sunlit", "illumination")):
            if _number(row, key) is not None:
                details[label] = f"{_number(row, key) * 100:.0f}%"
        if _number(row, "water_km") is not None:
            details["Ice trap"] = f"{_number(row, 'water_km'):.2f} km"
        if _number(row, "area_m2") is not None:
            details["Flat area"] = f"{_number(row, 'area_m2') / 1e4:.1f} ha"
        if row.get("nearest_poi"):
            details["Nearest crater"] = f"{row['nearest_poi']} ({_number(row, 'nearest_poi_km') or 0:.1f} km)"
        rank = row.get("rank") or str(index)
        candidates.append({
            "rank": int(rank), "label": f"Spot {rank}", "lat": float(row["lat_deg"]), "lon": float(row["lon_deg"]),
            "score": _number(row, "composite_score"), "topology": _number(row, "topology_prob"),
            "earth_visibility": _number(row, "communications_earth_vis"), "water_km": _number(row, "water_km"),
            "details": details,
        })
    pois = [{"name": row["name"], "type": row["type"].split(",")[0], "lat": float(row["lat"]), "lon": float(row["lon"])}
            for row in _read_pois(OUTPUTS / f"{name}_pois.csv")]
    return {"name": name, "candidates": candidates, "pois": pois}


class LandingSiteClassifier:
    """Shows the classifier's prebaked results for Sites 1, 4 and 23."""

    name = "LunAI landing-site classifier (prebaked results)"

    def classify(self, site: Mapping[str, Any]) -> SiteClassification:
        data = results(int(site.get("site_number") or 0))
        if not data:
            raise ValueError("no classifier results for this site.")
        best = data["candidates"][0]
        score = best["score"] or 0.0
        breakdown = {}
        if None not in (best["topology"], best["earth_visibility"], best["water_km"]):
            water = 1.0 - min(best["water_km"], WATER_DISTANCE_CAP_KM) / WATER_DISTANCE_CAP_KM
            breakdown = {"ground safety": WEIGHTS["ground safety"] * best["topology"],
                         "Earth in view": WEIGHTS["Earth in view"] * best["earth_visibility"],
                         "near ice": WEIGHTS["near ice"] * water}
        level = "good" if score >= 0.75 else "care" if score >= 0.5 else "risky"
        title = {"good": "Good landing spots found", "care": "Possible landing spots found", "risky": "Weak landing spots"}[level]
        count = len(data["candidates"])
        return SiteClassification(
            terrain_class=f"{count} spots ranked",
            class_scores=breakdown,
            score_label=f"Best spot's score, {score:.2f} of 1, split into its parts",
            model=self.name,
            details={"Best score": f"{score:.2f} of 1", **best["details"]},
            candidates=data["candidates"],
            verdict={"level": level, "title": title, "text": f"The classifier ranked {count} landing spots here; the best scores {score:.2f} out of 1."},
            map={"figure": data["name"], "pois": data["pois"]},
        )
