"""Topology feature engineering for landing-site safety classification.

All features are computed as full-grid arrays (same shape as the input
rasters) using scipy.ndimage, then sampled at whatever pixels are needed for
training or prediction. Window size is in pixels; at 5 m/pixel a window of
21 corresponds to a ~105 m diameter neighborhood, in the ballpark of a lander
footprint plus margin.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURE_NAMES = [
    "slope",
    "slope_error",
    "elevation_relief",
    "roughness_std",
    "slope_mean",
    "slope_max",
    "slope_std",
    "curvature",
    "total_error_mean",
    "ldec_mean",
]


def _window_mean(a: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(a, size=size, mode="nearest")


def _window_std(a: np.ndarray, size: int) -> np.ndarray:
    mean = _window_mean(a, size)
    mean_sq = _window_mean(a * a, size)
    var = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(var)


def compute_feature_stack(bands: dict[str, np.ndarray], window: int = 21) -> np.ndarray:
    """bands must contain keys: slp, slperr, surf, toterr, ldec (same shape).

    Returns an array of shape (n_features, rows, cols) in FEATURE_NAMES order.
    NaNs (nodata) propagate through to the output.
    """
    slope = bands["slp"]
    slope_err = bands["slperr"]
    elev = bands["surf"]
    tot_err = bands["toterr"]
    ldec = bands["ldec"]

    nan_mask = np.isnan(slope) | np.isnan(elev)
    elev_filled = np.where(np.isnan(elev), np.nanmean(elev), elev)
    slope_filled = np.where(np.isnan(slope), 0.0, slope)

    relief = ndimage.maximum_filter(elev_filled, size=window, mode="nearest") - ndimage.minimum_filter(
        elev_filled, size=window, mode="nearest"
    )
    roughness = _window_std(elev_filled, window)
    slope_mean = _window_mean(slope_filled, window)
    slope_max = ndimage.maximum_filter(slope_filled, size=window, mode="nearest")
    slope_std = _window_std(slope_filled, window)
    curvature = np.abs(ndimage.laplace(elev_filled))
    tot_err_mean = _window_mean(np.where(np.isnan(tot_err), 0.0, tot_err), window)
    ldec_mean = _window_mean(np.where(np.isnan(ldec), 0.0, ldec), window)

    stack = np.stack(
        [
            slope,
            slope_err,
            relief,
            roughness,
            slope_mean,
            slope_max,
            slope_std,
            curvature,
            tot_err_mean,
            ldec_mean,
        ],
        axis=0,
    )
    stack[:, nan_mask] = np.nan
    return stack
