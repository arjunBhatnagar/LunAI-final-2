"""Terrain patches for the surface view, cut from the local LOLA 5 m elevation models.

Only Sites 1, 4 and 23 have local elevation data (``data/lola/Site*_surface.tif``,
south polar stereographic, metres). For a point inside one of them this returns
a square grid of heights the browser turns into a 3D surface.
"""
from __future__ import annotations

import base64
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOLA_DIR = ROOT / "data" / "lola"
MOON_RADIUS_M = 1_737_400.0
SITES = {"Site 01": "Site01_surface.tif", "Site 04": "Site04_surface.tif", "Site 23": "Site23_surface.tif"}


def to_stereo(lat: float, lon: float) -> tuple[float, float]:
    """South polar stereographic x/y in metres (the projection of the LOLA site rasters)."""
    rho = 2.0 * MOON_RADIUS_M * math.tan(math.radians(90.0 + lat) / 2.0)
    return rho * math.sin(math.radians(lon)), rho * math.cos(math.radians(lon))


@lru_cache(maxsize=4)
def _bounds(path: Path):
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.bounds


def site_for(lat: float, lon: float) -> tuple[str, Path] | None:
    if lat > -80:
        return None
    x, y = to_stereo(lat, lon)
    for name, filename in SITES.items():
        path = LOLA_DIR / filename
        if path.is_file():
            b = _bounds(path)
            if b.left <= x <= b.right and b.bottom <= y <= b.top:
                return name, path
    return None


def terrain_patch(lat: float, lon: float, size_km: float = 8.0, samples: int = 192) -> dict[str, Any] | None:
    """Heights (m) on a ``samples`` x ``samples`` grid covering ``size_km`` around the point, or None."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("latitude/longitude out of range.")
    size_km = float(min(max(size_km, 1.0), 16.0))
    samples = int(min(max(samples, 32), 256))
    found = site_for(lat, lon)
    if not found:
        return None
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    name, path = found
    x, y = to_stereo(lat, lon)
    half = size_km * 500.0
    with rasterio.open(path) as dataset:
        window = from_bounds(x - half, y - half, x + half, y + half, dataset.transform)
        heights = dataset.read(1, window=window, out_shape=(samples, samples), boundless=True,
                               fill_value=np.nan, resampling=Resampling.average).astype(np.float32)
        if dataset.nodata is not None and not math.isnan(dataset.nodata):
            heights[heights == dataset.nodata] = np.nan
    if not np.isfinite(heights).any():
        return None
    return {
        "site": name, "lat": lat, "lon": lon, "x_m": x, "y_m": y, "size_km": size_km, "samples": samples,
        # Row 0 is the north edge of the map grid (+y); columns run along +x.
        "heights_f32_base64": base64.b64encode(heights.astype("<f4").tobytes()).decode("ascii"),
        "min_m": float(np.nanmin(heights)), "max_m": float(np.nanmax(heights)),
        "source": "LOLA 5 m south-pole elevation model (NASA PGDA product 78)",
    }
