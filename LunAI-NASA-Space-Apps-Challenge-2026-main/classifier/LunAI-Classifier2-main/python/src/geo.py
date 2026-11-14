"""GeoTIFF / shapefile I/O for the LOLA 5mpp site rasters, without GDAL.

The project's rasterio install is blocked by a Windows Application Control
policy on this machine, so rasters are read with tifffile (pure Python) and
georeferencing is handled manually from the GeoTIFF tiepoint/pixel-scale tags.
Shapefiles (ROI polygons) are parsed directly from the ESRI .shp binary format.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

MOON_RADIUS_M = 1737400.0  # matches the GeoTIFF's SPHEROID radius


@dataclass
class RasterGrid:
    array: np.ndarray  # (rows, cols), float32, NaN = nodata
    pixel_size: float  # meters/pixel (assumed square)
    origin_x: float  # map X of pixel (row=0, col=0)
    origin_y: float  # map Y of pixel (row=0, col=0)

    def pixel_to_map(self, row, col):
        x = self.origin_x + (np.asarray(col) + 0.5) * self.pixel_size
        y = self.origin_y - (np.asarray(row) + 0.5) * self.pixel_size
        return x, y

    def map_to_pixel(self, x, y):
        col = (np.asarray(x) - self.origin_x) / self.pixel_size - 0.5
        row = (self.origin_y - np.asarray(y)) / self.pixel_size - 0.5
        return row, col


def read_geotiff(path: str | Path) -> RasterGrid:
    tf = tifffile.TiffFile(str(path))
    page = tf.pages[0]
    arr = page.asarray().astype(np.float32)
    sx, sy, _ = page.tags["ModelPixelScaleTag"].value
    tp = page.tags["ModelTiepointTag"].value  # (i, j, k, x, y, z)
    origin_x, origin_y = tp[3], tp[4]
    if sx != sy:
        raise ValueError(f"{path}: non-square pixels not supported ({sx} x {sy})")
    return RasterGrid(array=arr, pixel_size=float(sx), origin_x=float(origin_x), origin_y=float(origin_y))


def read_shp_polygons(path: str | Path) -> list[list[tuple[float, float]]]:
    """Return one ring (list of (x, y) points) per polygon record in a .shp file.

    Only handles single-part polygons (shape type 5), which is what the
    Site0*_ROIs.shp files contain.
    """
    data = Path(path).read_bytes()
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type != 5:
        raise ValueError(f"{path}: expected polygon shapefile (type 5), got type {shape_type}")

    polygons = []
    pos = 100
    while pos < len(data):
        content_len_words = struct.unpack(">i", data[pos + 4 : pos + 8])[0]
        content_start = pos + 8
        rtype = struct.unpack("<i", data[content_start : content_start + 4])[0]
        if rtype == 0:
            pos = content_start + 4
            continue
        numparts = struct.unpack("<i", data[content_start + 36 : content_start + 40])[0]
        numpoints = struct.unpack("<i", data[content_start + 40 : content_start + 44])[0]
        points_off = content_start + 44 + 4 * numparts
        pts = []
        for i in range(numpoints):
            x, y = struct.unpack("<2d", data[points_off + 16 * i : points_off + 16 * i + 16])
            pts.append((x, y))
        polygons.append(pts)
        pos = content_start + content_len_words * 2
    return polygons


def rasterize_polygon_mask(grid: RasterGrid, polygons: list[list[tuple[float, float]]]) -> np.ndarray:
    """Boolean mask, True where a pixel center falls inside any polygon.

    Uses matplotlib.path for the point-in-polygon test (no GDAL needed).
    """
    from matplotlib.path import Path as MplPath

    rows, cols = grid.array.shape
    mask = np.zeros((rows, cols), dtype=bool)
    for ring in polygons:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        r0, c0 = grid.map_to_pixel(min_x, max_y)
        r1, c1 = grid.map_to_pixel(max_x, min_y)
        r0, r1 = sorted((int(np.floor(r0)) - 1, int(np.ceil(r1)) + 1))
        c0, c1 = sorted((int(np.floor(c0)) - 1, int(np.ceil(c1)) + 1))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, rows), min(c1, cols)
        if r0 >= r1 or c0 >= c1:
            continue
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        x, y = grid.pixel_to_map(rr, cc)
        pts = np.column_stack([x.ravel(), y.ravel()])
        inside = MplPath(ring).contains_points(pts).reshape(rr.shape)
        mask[r0:r1, c0:c1] |= inside
    return mask


def polar_stereographic_south_to_lonlat(x: np.ndarray, y: np.ndarray, radius: float = MOON_RADIUS_M):
    """Inverse south polar stereographic projection (standard parallel = -90 deg).

    Matches the GeoTIFF PROJCS: Stereographic_South_Pole, central meridian 0,
    false easting/northing 0, on a sphere of `radius` meters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.hypot(x, y)
    c = 2.0 * np.arctan2(rho, 2.0 * radius)
    lat = np.degrees(-np.arcsin(np.where(rho == 0, 0.0, np.cos(c))))
    lon = np.degrees(np.arctan2(x, y))  # Snyder south-polar aspect: lambda = lambda0 + atan2(x, y)
    lat = np.where(rho == 0, -90.0, lat)
    return lon, lat


def lonlat_to_polar_stereographic_south(lon: np.ndarray, lat: np.ndarray, radius: float = MOON_RADIUS_M):
    """Forward south polar stereographic projection (inverse of the function above)."""
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.asarray(lat, dtype=np.float64)
    c = np.radians(90.0 + lat)  # colatitude from the south pole
    rho = 2.0 * radius * np.tan(c / 2.0)
    x = rho * np.sin(lon)
    y = rho * np.cos(lon)
    return x, y
