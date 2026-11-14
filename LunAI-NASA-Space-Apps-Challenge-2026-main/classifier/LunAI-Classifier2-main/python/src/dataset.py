"""Load a site's rasters + (optional) ROI labels and build feature/label arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

import external_data as ext
import features as feat
import geo

BAND_SUFFIXES = ("ldec", "slp", "slperr", "surf", "toterr")
EXTERNAL_FEATURE_NAMES = ["illumination", "earth_visibility", "shadow_fraction"]
FULL_FEATURE_NAMES = feat.FEATURE_NAMES + EXTERNAL_FEATURE_NAMES


@dataclass
class SiteData:
    name: str
    grid: geo.RasterGrid  # georeferencing, using the slope band as reference
    feature_stack: np.ndarray  # (n_features, rows, cols)
    valid_mask: np.ndarray  # True where no NaNs in any feature
    roi_mask: np.ndarray | None  # True inside labeled ROI polygons, else None


def load_site(
    site_dir: str | Path, roi_shp: str | Path | None = None, window: int = 21, include_external: bool = True
) -> SiteData:
    site_dir = Path(site_dir)
    name = site_dir.name
    prefix = next(site_dir.glob("*_final_adj_5mpp_slp.tif")).name.replace("_slp.tif", "")

    bands = {}
    grid = None
    for suffix in BAND_SUFFIXES:
        g = geo.read_geotiff(site_dir / f"{prefix}_{suffix}.tif")
        bands[suffix] = g.array
        if suffix == "slp":
            grid = g

    stack = feat.compute_feature_stack(bands, window=window)

    if include_external:
        ext_bands = ext.load_external_bands_for_site(grid)
        ext_stack = np.stack([ext_bands[n] for n in EXTERNAL_FEATURE_NAMES], axis=0)
        stack = np.concatenate([stack, ext_stack], axis=0)

    valid_mask = ~np.isnan(stack).any(axis=0)

    roi_mask = None
    if roi_shp is not None:
        polys = geo.read_shp_polygons(roi_shp)
        roi_mask = geo.rasterize_polygon_mask(grid, polys)

    return SiteData(name=name, grid=grid, feature_stack=stack, valid_mask=valid_mask, roi_mask=roi_mask)


def sample_training_pixels(
    site: SiteData,
    negative_ratio: float = 8.0,
    exclude_buffer_px: int = 6,
    rng: np.random.Generator | None = None,
):
    """Return (X, y) for a site that has roi_mask labels.

    Positives: every valid pixel inside the ROI polygons.
    Negatives: randomly sampled valid pixels outside a dilated buffer around
    the ROI polygons (avoids ambiguous near-boundary pixels), at
    `negative_ratio` times the positive count.
    """
    if site.roi_mask is None:
        raise ValueError(f"{site.name} has no ROI labels")
    rng = rng or np.random.default_rng(0)

    pos_mask = site.roi_mask & site.valid_mask
    excluded = ndimage.binary_dilation(site.roi_mask, iterations=exclude_buffer_px)
    neg_candidates = site.valid_mask & ~excluded

    pos_idx = np.flatnonzero(pos_mask.ravel())
    neg_idx_all = np.flatnonzero(neg_candidates.ravel())
    n_neg = min(len(neg_idx_all), int(len(pos_idx) * negative_ratio))
    neg_idx = rng.choice(neg_idx_all, size=n_neg, replace=False)

    rows, cols = site.feature_stack.shape[1:]
    flat_feats = site.feature_stack.reshape(site.feature_stack.shape[0], -1)

    idx = np.concatenate([pos_idx, neg_idx])
    X = flat_feats[:, idx].T
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    return X, y, idx
