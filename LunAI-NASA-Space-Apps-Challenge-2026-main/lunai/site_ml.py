from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, maximum_filter, uniform_filter
from sklearn.ensemble import RandomForestClassifier

MOON_RADIUS_M = 1_737_400.0
SITE_PRODUCTS = {
    "01": ("Connecting ridge", "Site01_surface.tif"),
    "04": ("Shackleton rim", "Site04_surface.tif"),
    "23": ("Malapert massif", "Site23_surface.tif"),
}
FEATURE_NAMES = ["slope_deg", "roughness_m", "local_position_m", "shelter_depth_m"]
TERRAIN_CLASSES = np.array(["flat", "rolling", "rugged"])


def _terrain_labels(slope: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    labels = np.full(slope.shape, 1, dtype=np.uint8)
    labels[(slope < 5.0) & (roughness < 2.0)] = 0
    labels[(slope >= 15.0) | (roughness >= 5.0)] = 2
    return labels


def _center_coordinates(lat0: float, lon0: float) -> tuple[float, float]:
    rho = math.hypot(lat0, lon0)
    colatitude = 2.0 * math.atan(rho / (2.0 * MOON_RADIUS_M))
    latitude = -90.0 + math.degrees(colatitude)
    longitude = math.degrees(math.atan2(lat0, lon0))
    return latitude, ((longitude + 180.0) % 360.0) - 180.0


def _terrain_features(dem: np.ndarray, resolution_m: float) -> Dict[str, np.ndarray]:
    valid = np.isfinite(dem)
    fill_value = float(np.nanmedian(dem))
    filled = np.where(valid, dem, fill_value).astype(np.float32)
    gx, gy = np.gradient(filled, resolution_m, resolution_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    mean = uniform_filter(filled, size=5, mode="nearest")
    mean_square = uniform_filter(filled * filled, size=5, mode="nearest")
    roughness = np.sqrt(np.maximum(0.0, mean_square - mean * mean)).astype(np.float32)
    shelter_depth = np.maximum(0.0, maximum_filter(filled, size=7, mode="nearest") - filled)
    local_position = filled - mean

    return {
        "valid": valid,
        "slope_deg": slope,
        "roughness_m": roughness,
        "local_position_m": local_position.astype(np.float32),
        "shelter_depth_m": shelter_depth.astype(np.float32),
    }


OBSERVER_HEIGHT_M = 2.0
HORIZON_NOTE = (
    "Terrain horizon ray-traced from the site-centre LOLA DEM (20 m) with lunar curvature and a 2 m observer; "
    "terrain beyond the DEM tile is not included."
)


def horizon_profile(dem: np.ndarray, resolution_m: float, x_center: float, y_center: float,
                    observer_height_m: float = OBSERVER_HEIGHT_M, azimuth_step_deg: float = 1.0) -> List[float]:
    """Horizon elevation angle (deg) for each azimuth (clockwise from north) seen from the DEM centre.

    ``x_center``/``y_center`` are south-polar stereographic metres, used to find
    geographic north (directly away from the pole) on the grid.
    """
    rows, cols = dem.shape
    center = np.array([(rows - 1) / 2.0, (cols - 1) / 2.0])
    filled = np.where(np.isfinite(dem), dem, -np.inf)
    observer = float(map_coordinates(np.nan_to_num(dem, nan=float(np.nanmedian(dem))), [[center[0]], [center[1]]], order=1)[0])
    observer += observer_height_m
    rho = math.hypot(x_center, y_center) or 1.0
    north = np.array([x_center, y_center]) / rho  # grid x/y direction of increasing latitude
    east = np.array([north[1], -north[0]])
    max_steps = int(min(rows, cols) / 2) - 1
    distances = np.arange(1, max_steps + 1) * resolution_m
    profile = []
    for azimuth in np.arange(0.0, 360.0, azimuth_step_deg):
        a = math.radians(azimuth)
        dx, dy = math.cos(a) * north + math.sin(a) * east
        # Grid x grows with columns and grid y shrinks with rows (north-up raster).
        sample_rows = center[0] - dy * distances / resolution_m
        sample_cols = center[1] + dx * distances / resolution_m
        heights = map_coordinates(filled, [sample_rows, sample_cols], order=1, mode="constant", cval=-np.inf)
        drop = distances**2 / (2.0 * MOON_RADIUS_M)
        angles = np.degrees(np.arctan2(heights - drop - observer, distances))
        finite = angles[np.isfinite(angles)]
        profile.append(round(float(finite.max()) if finite.size else 0.0, 3))
    return profile


def add_horizon_profiles(analysis: Dict[str, Any], data_dir: Path | str) -> Dict[str, Any]:
    """Add ``horizon_deg`` to each analysed site in place (used to update existing analysis JSON)."""
    import rasterio
    from rasterio.enums import Resampling

    for site in analysis["sites"]:
        filename = SITE_PRODUCTS.get(f"{site['site_number']:02d}", (None, None))[1]
        path = Path(data_dir) / filename if filename else None
        if not path or not path.is_file():
            continue
        with rasterio.open(path) as dataset:
            scale = 4
            dem = dataset.read(1, out_shape=(dataset.height // scale, dataset.width // scale),
                               resampling=Resampling.average).astype(np.float32)
            resolution = float(abs(dataset.transform.a)) * scale
            x_center = (dataset.bounds.left + dataset.bounds.right) / 2.0
            y_center = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
            if dataset.nodata is not None:
                dem[dem == dataset.nodata] = np.nan
        site["horizon_deg"] = horizon_profile(dem, resolution, x_center, y_center)
        site["horizon_range_km"] = round(min(dem.shape) / 2 * resolution / 1000.0, 1)
    analysis["horizon_note"] = HORIZON_NOTE
    return analysis


def analyze_lola_sites(data_dir: Path | str, output_path: Path | str | None = None) -> Dict[str, Any]:
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise RuntimeError("Rasterio is required to analyze LOLA GeoTIFFs; install requirements.txt.") from exc

    base = Path(data_dir)
    summaries: List[Dict[str, Any]] = []
    training_features: List[np.ndarray] = []
    training_labels: List[np.ndarray] = []
    site_prediction_features: List[np.ndarray] = []
    rng = np.random.default_rng(42)

    for site_id, (site_name, filename) in SITE_PRODUCTS.items():
        path = base / filename
        if not path.is_file():
            continue
        with rasterio.open(path) as dataset:
            scale = 4
            height = max(1, dataset.height // scale)
            width = max(1, dataset.width // scale)
            dem = dataset.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.average,
            ).astype(np.float32)
            x_center = (dataset.bounds.left + dataset.bounds.right) / 2.0
            y_center = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
            native_resolution = float(abs(dataset.transform.a))
            crs_description = dataset.crs.to_string() if dataset.crs else "unknown lunar polar stereographic"
            nodata = dataset.nodata

        resolution = native_resolution * (dataset.width / width)
        if nodata is not None:
            dem[dem == nodata] = np.nan
        features = _terrain_features(dem, resolution)
        valid = features["valid"]
        row_indices, col_indices = np.where(valid)
        if len(row_indices) < 100:
            continue

        sample_count = min(6000, len(row_indices))
        chosen = rng.choice(len(row_indices), size=sample_count, replace=False)
        rows, cols = row_indices[chosen], col_indices[chosen]
        x_train = np.column_stack([
            features[name][rows, cols] for name in FEATURE_NAMES
        ]).astype(np.float32)
        labels = _terrain_labels(features["slope_deg"][rows, cols], features["roughness_m"][rows, cols])
        finite = np.isfinite(x_train).all(axis=1)
        training_features.append(x_train[finite])
        training_labels.append(labels[finite])
        site_prediction_features.append(x_train[finite])

        shelter = valid & (features["shelter_depth_m"] >= 30.0)
        low_slope = valid & (features["slope_deg"] <= 8.0)
        distance_to_shelter = distance_transform_edt(~shelter, sampling=resolution)
        landable_near_shelter = low_slope & (distance_to_shelter <= 1000.0)
        classes = _terrain_labels(features["slope_deg"][valid], features["roughness_m"][valid])
        lat, lon = _center_coordinates(x_center, y_center)
        summaries.append({
            "id": f"site{site_id}",
            "site_number": int(site_id),
            "name": f"Site {site_id} — {site_name}",
            "lat": lat,
            "lon": lon,
            "area_km2": float(dataset.width * dataset.height * native_resolution**2 / 1_000_000.0),
            "native_resolution_m": native_resolution,
            "analysis_resolution_m": resolution,
            "crs": crs_description,
            "valid_sample_count": int(valid.sum()),
            "median_elevation_m": float(np.nanmedian(dem)),
            "median_slope_deg": float(np.median(features["slope_deg"][valid])),
            "p90_slope_deg": float(np.percentile(features["slope_deg"][valid], 90)),
            "median_roughness_m": float(np.median(features["roughness_m"][valid])),
            "shadow_proxy_pct": float(100.0 * shelter.sum() / valid.sum()),
            "low_slope_pct": float(100.0 * low_slope.sum() / valid.sum()),
            "landable_near_shelter_pct": float(100.0 * landable_near_shelter.sum() / valid.sum()),
            "terrain_class_rule": str(TERRAIN_CLASSES[int(np.bincount(classes).argmax())]),
            "terrain_class_pct_rule": {
                str(TERRAIN_CLASSES[label]): float(100.0 * np.count_nonzero(classes == label) / len(classes))
                for label in range(len(TERRAIN_CLASSES))
            },
            "water_content_pct": None,
            "mineral_density": None,
        })

    if not summaries:
        raise FileNotFoundError(f"No supported Site 01/04/23 LOLA GeoTIFFs found in {base}")
    if len(training_features) < 2:
        raise ValueError("At least two site DEMs are needed for the terrain-class prototype.")

    x_all = np.vstack(training_features)
    y_all = np.concatenate(training_labels)
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(x_all, y_all)

    model_path = base / "terrain_random_forest.joblib"
    joblib.dump(classifier, model_path)
    for summary, prediction_features in zip(summaries, site_prediction_features):
        prediction = classifier.predict(prediction_features)
        summary["terrain_model"] = "rule-derived random-forest surrogate"
        summary["terrain_model_samples"] = int(len(y_all))
        summary["terrain_class_rf"] = str(TERRAIN_CLASSES[int(np.bincount(prediction).argmax())])
        summary["terrain_class_pct_rf"] = {
            str(TERRAIN_CLASSES[label]): float(100.0 * np.count_nonzero(prediction == label) / len(prediction))
            for label in range(len(TERRAIN_CLASSES))
        }

    result = {
        "source": "NASA GSFC PDS Geosciences Node, LOLA_5mpp (product 78)",
        "source_url": "https://pgda.gsfc.nasa.gov/products/78",
        "readme": "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README",
        "coordinate_system": "South polar stereographic X/Y, meters; MOON_ME, DE421",
        "measurement_note": "LOLA provides elevation/topography only. Water content and mineral density are not measured by these DEMs.",
        "shadow_proxy_note": "Share of valid samples at least 30 m below the maximum within a roughly 120 m neighborhood at 20 m analysis resolution; a local topographic shelter proxy, not measured water or proven permanent shadow.",
        "terrain_model_note": "Random Forest is trained on LOLA-derived pixel features using rule-generated terrain-class labels (slope/roughness thresholds); these are not independent ground-truth labels.",
        "classifier": {
            "algorithm": "RandomForestClassifier",
            "feature_names": FEATURE_NAMES,
            "classes": TERRAIN_CLASSES.tolist(),
            "training_samples": int(len(y_all)),
            "training_sites": [int(site_id) for site_id in SITE_PRODUCTS if any(s["site_number"] == int(site_id) for s in summaries)],
            "validation": "Not independently validated; labels are threshold-derived.",
            "model_file": model_path.name,
        },
        "sites": summaries,
    }
    add_horizon_profiles(result, base)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "lola_site_analysis.json"
    if "--horizons-only" in sys.argv and output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        output.write_text(json.dumps(add_horizon_profiles(existing, root / "data" / "lola"), indent=2), encoding="utf-8")
        print(f"Added terrain horizons to {output}")
    else:
        analyze_lola_sites(root / "data" / "lola", output)
        print(f"Wrote LOLA terrain analysis to {output}")
