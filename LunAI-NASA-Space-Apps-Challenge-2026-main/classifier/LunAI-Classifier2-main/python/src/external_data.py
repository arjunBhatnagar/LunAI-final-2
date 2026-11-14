"""Load and align external NASA datasets (Earth visibility, permanent shadow,
solar illumination, named points of interest) onto a site's 5 m/px grid.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

import geo

EXTERNAL_DIR = Path(__file__).resolve().parent.parent.parent / "ExternalData"

ILLUM_FILES = {
    "illumination": EXTERNAL_DIR / "Illumination" / "AVGVISIB_85S_060M_201608.TIF",
    "earth_visibility": EXTERNAL_DIR / "Illumination" / "AVGVISIB_85S_060M_201608_EARTH.TIF",
    "shadow_fraction": EXTERNAL_DIR / "Illumination" / "LPSR_85S_060M_201608.TIF",
}


def read_scaled_geotiff(path: str | Path) -> geo.RasterGrid:
    """Like geo.read_geotiff, but applies the PGDA GDAL_METADATA SCALE/OFFSET
    and NODATA so the returned array is already in physical units (0-1 fraction).
    """
    tf = tifffile.TiffFile(str(path))
    page = tf.pages[0]
    raw = page.asarray().astype(np.float64)

    nodata = None
    if "GDAL_NODATA" in page.tags:
        try:
            nodata = float(page.tags["GDAL_NODATA"].value)
        except ValueError:
            nodata = None

    scale, offset = 1.0, 0.0
    if "GDAL_METADATA" in page.tags:
        md = page.tags["GDAL_METADATA"].value
        m = re.search(r'role="scale">([^<]+)<', md)
        if m:
            scale = float(m.group(1))
        m = re.search(r'role="offset">([^<]+)<', md)
        if m:
            offset = float(m.group(1))

    if nodata is not None:
        raw[raw == nodata] = np.nan
    arr = (raw * scale + offset).astype(np.float32)

    sx, sy, _ = page.tags["ModelPixelScaleTag"].value
    tp = page.tags["ModelTiepointTag"].value
    return geo.RasterGrid(array=arr, pixel_size=float(sx), origin_x=float(tp[3]), origin_y=float(tp[4]))


def resample_to_grid(source: geo.RasterGrid, target: geo.RasterGrid, order: int = 1) -> np.ndarray:
    """Resample `source` onto `target`'s pixel grid (bilinear by default)."""
    rows, cols = target.array.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    x, y = target.pixel_to_map(rr, cc)
    src_row, src_col = source.map_to_pixel(x, y)
    out = ndimage.map_coordinates(
        np.nan_to_num(source.array, nan=np.nanmean(source.array)),
        [src_row, src_col],
        order=order,
        mode="nearest",
    )
    oob = (
        (src_row < 0)
        | (src_row > source.array.shape[0] - 1)
        | (src_col < 0)
        | (src_col > source.array.shape[1] - 1)
    )
    out = out.astype(np.float32)
    out[oob] = np.nan
    return out


def load_external_bands_for_site(target_grid: geo.RasterGrid) -> dict[str, np.ndarray]:
    """Return {name: array} for illumination/earth_visibility/shadow_fraction,
    resampled onto target_grid (a site's 5 m/px topology grid).
    """
    bands = {}
    for name, path in ILLUM_FILES.items():
        src = read_scaled_geotiff(path)
        bands[name] = resample_to_grid(src, target_grid)
    return bands


_cold_trap_cache: dict | None = None


def _get_cold_trap_distance_field(threshold: float = 0.5):
    """Lazily build (and cache) a Euclidean-distance-to-nearest-cold-trap field
    over the full south-pole LPSR (permanent shadow) mosaic. LPSR is
    effectively binary (fraction is ~0 or ~1), so the exact threshold doesn't
    matter much. Returns (grid, distance_m array).
    """
    global _cold_trap_cache
    if _cold_trap_cache is None:
        grid = read_scaled_geotiff(ILLUM_FILES["shadow_fraction"])
        cold_trap = np.nan_to_num(grid.array, nan=0.0) >= threshold
        # distance_transform_edt gives distance to the nearest False pixel by
        # default, so invert: we want distance to the nearest True (shadowed) pixel.
        dist_px = ndimage.distance_transform_edt(~cold_trap)
        dist_m = dist_px * grid.pixel_size
        _cold_trap_cache = {"grid": grid, "dist_m": dist_m}
    return _cold_trap_cache["grid"], _cold_trap_cache["dist_m"]


def water_distance_km_grid(target_grid: geo.RasterGrid) -> np.ndarray:
    """Full per-pixel distance-to-nearest-cold-trap (km), resampled onto
    target_grid - for visualization / full-map analysis, not just point lookups.
    """
    src_grid, dist_m = _get_cold_trap_distance_field()
    src_km_grid = geo.RasterGrid(
        array=(dist_m / 1000.0).astype(np.float32),
        pixel_size=src_grid.pixel_size,
        origin_x=src_grid.origin_x,
        origin_y=src_grid.origin_y,
    )
    return resample_to_grid(src_km_grid, target_grid)


def distance_to_nearest_water_km(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance in km from map coordinates (x, y) to the nearest permanently
    shadowed ("cold trap") pixel in the full south-pole LPSR mosaic - our proxy
    for where water ice is most likely to survive.
    """
    grid, dist_m = _get_cold_trap_distance_field()
    row, col = grid.map_to_pixel(np.asarray(x), np.asarray(y))
    row = np.clip(np.round(row).astype(int), 0, dist_m.shape[0] - 1)
    col = np.clip(np.round(col).astype(int), 0, dist_m.shape[1] - 1)
    return dist_m[row, col] / 1000.0
