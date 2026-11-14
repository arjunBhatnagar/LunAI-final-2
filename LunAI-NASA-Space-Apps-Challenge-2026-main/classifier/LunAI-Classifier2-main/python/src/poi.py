"""Named points of interest (craters, massifs, ...) near a site, from the
USGS Gazetteer of Planetary Nomenclature (IAU-approved lunar feature names).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import geo

GAZETTEER_PATH = Path(__file__).resolve().parent.parent.parent / "ExternalData" / "moon_south_pole_gazetteer.json"


def load_pois_near_grid(grid: geo.RasterGrid, margin_m: float = 15000.0) -> list[dict]:
    with open(GAZETTEER_PATH) as f:
        feats = json.load(f)

    lons = np.array([f["center_lon"] for f in feats])
    lats = np.array([f["center_lat"] for f in feats])
    xs, ys = geo.lonlat_to_polar_stereographic_south(lons, lats)

    rows, cols = grid.array.shape
    x0, y0 = grid.pixel_to_map(0, 0)
    x1, y1 = grid.pixel_to_map(rows - 1, cols - 1)
    xmin, xmax = min(x0, x1) - margin_m, max(x0, x1) + margin_m
    ymin, ymax = min(y0, y1) - margin_m, max(y0, y1) + margin_m

    pois = []
    for i, f in enumerate(feats):
        if not (xmin <= xs[i] <= xmax and ymin <= ys[i] <= ymax):
            continue
        row, col = grid.map_to_pixel(xs[i], ys[i])
        pois.append(
            {
                "name": f["feature_name"],
                "type": f["feature_type"],
                "diameter_km": f["diameter_km"],
                "lon": f["center_lon"],
                "lat": f["center_lat"],
                "x_m": float(xs[i]),
                "y_m": float(ys[i]),
                "row": float(row),
                "col": float(col),
                "inside_site": bool(0 <= row < rows and 0 <= col < cols),
            }
        )
    return pois
