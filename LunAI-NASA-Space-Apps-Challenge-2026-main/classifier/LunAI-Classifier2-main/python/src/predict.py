"""Score a site and rank candidate landing coordinates by a layered priority:

    1. Topology    (most important) - is the ground physically safe to land on?
                     Detected by the topology-only Random Forest, exactly as
                     before (slope, roughness, relief, etc. in a ~105 m window).
    2. Communications - mean Earth-visibility fraction over the candidate area.
    3. Water/ice distance - km to the nearest permanently-shadowed "cold trap"
                     pixel (our water-ice proxy), closer is better.
    4. Nearest point of interest - informational only, lowest priority, not
                     part of the score (named craters/massifs don't make a
                     spot better or worse to land on).

Topology alone decides WHICH regions are candidates at all (the detection
step is unchanged). Criteria 1-3 are then combined into one weighted
composite score - see WEIGHTS below - that decides the final ranking, with
each weight set to reflect the stated priority order. POI is attached to
each candidate for context but never affects ranking.

Usage:
    python predict.py Site23
    python predict.py Site1   # sanity check against known ROI polygons
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage

import dataset
import external_data as ext
import geo
import poi as poi_mod

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "python" / "models" / "landing_site_rf.joblib"
OUT_DIR = ROOT / "python" / "outputs"

MIN_SAFE_AREA_PX = 150  # ~3750 m^2 at 5 m/px, a rough stand-in for a lander footprint + margin
PROB_THRESHOLD = 0.8  # topology-only probability threshold used to DETECT candidate regions
TOP_N = 10

# Composite ranking weights, in priority order. Must sum to 1.0.
# - topology: raw mean probability from the topology-only model (already 0-1)
# - communications: mean Earth-visibility fraction over the candidate area (already 0-1)
# - water: "goodness" of distance to the nearest cold trap, linearly mapped to
#   0-1 over [0, WATER_DISTANCE_CAP_KM] (closer = higher score)
WEIGHTS = {"topology": 0.60, "communications": 0.30, "water": 0.10}
WATER_DISTANCE_CAP_KM = 50.0


def score_site(site: dataset.SiteData, model, n_topology_features: int, batch_size: int = 500_000) -> np.ndarray:
    """Score every valid pixel with the topology-only model (first
    n_topology_features bands of the feature stack - external bands, if
    present, are ignored here and used only for secondary ranking criteria).
    """
    rows, cols = site.feature_stack.shape[1:]
    flat = site.feature_stack[:n_topology_features].reshape(n_topology_features, -1).T
    proba = np.full(flat.shape[0], np.nan, dtype=np.float32)
    valid_flat = site.valid_mask.ravel()
    valid_idx = np.flatnonzero(valid_flat)
    for start in range(0, len(valid_idx), batch_size):
        chunk = valid_idx[start : start + batch_size]
        proba[chunk] = model.predict_proba(flat[chunk])[:, 1]
    return proba.reshape(rows, cols)


def find_candidates(prob_map: np.ndarray, site: dataset.SiteData):
    """Detect candidate regions from topology alone (unchanged from before),
    then attach communications/water/POI as secondary, non-detecting criteria.
    """
    grid = site.grid
    safe = prob_map >= PROB_THRESHOLD
    labeled, n = ndimage.label(safe)

    illum_idx = dataset.FULL_FEATURE_NAMES.index("illumination")
    earthvis_idx = dataset.FULL_FEATURE_NAMES.index("earth_visibility")
    has_external = site.feature_stack.shape[0] > illum_idx

    candidates = []
    for comp_id in range(1, n + 1):
        comp_mask = labeled == comp_id
        area = comp_mask.sum()
        if area < MIN_SAFE_AREA_PX:
            continue
        mean_prob = float(prob_map[comp_mask].mean())
        rows_idx, cols_idx = np.nonzero(comp_mask)
        best = np.argmax(prob_map[comp_mask])
        row, col = int(rows_idx[best]), int(cols_idx[best])
        x, y = grid.pixel_to_map(row, col)
        lon, lat = geo.polar_stereographic_south_to_lonlat(np.array([x]), np.array([y]))

        c = {
            "area_px": int(area),
            "area_m2": float(area * grid.pixel_size**2),
            "topology_prob": mean_prob,
            "peak_prob": float(prob_map[comp_mask][best]),
            "row": row,
            "col": col,
            "x_m": float(x),
            "y_m": float(y),
            "lon_deg": float(lon[0]),
            "lat_deg": float(lat[0]),
        }
        if has_external:
            c["illumination"] = float(site.feature_stack[illum_idx][comp_mask].mean())
            c["communications"] = float(site.feature_stack[earthvis_idx][comp_mask].mean())
            c["water_km"] = float(ext.distance_to_nearest_water_km(np.array([x]), np.array([y]))[0])
        else:
            c["illumination"] = c["communications"] = c["water_km"] = None
        candidates.append(c)
    return candidates


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """Combine topology / communications / water into one weighted composite
    score (POI is informational only and never affects ranking), then sort.
    """
    for c in candidates:
        topo_score = c["topology_prob"]
        comm_score = c["communications"] if c["communications"] is not None else 0.0
        if c["water_km"] is not None:
            water_score = 1.0 - min(c["water_km"], WATER_DISTANCE_CAP_KM) / WATER_DISTANCE_CAP_KM
        else:
            water_score = 0.0
        c["composite_score"] = (
            WEIGHTS["topology"] * topo_score + WEIGHTS["communications"] * comm_score + WEIGHTS["water"] * water_score
        )
    candidates.sort(key=lambda c: -c["composite_score"])
    return candidates[:TOP_N]


def annotate_nearest_poi(candidates: list[dict], pois: list[dict]):
    for c in candidates:
        if not pois:
            c["nearest_poi"], c["nearest_poi_km"] = None, None
            continue
        dists = [np.hypot(c["x_m"] - p["x_m"], c["y_m"] - p["y_m"]) for p in pois]
        i = int(np.argmin(dists))
        c["nearest_poi"] = pois[i]["name"]
        c["nearest_poi_km"] = dists[i] / 1000.0


def plot_result(
    site: dataset.SiteData,
    prob_map: np.ndarray,
    composite_map: np.ndarray,
    candidates: list[dict],
    pois: list[dict],
    out_path: Path,
):
    import matplotlib.pyplot as plt

    illum_idx = dataset.FULL_FEATURE_NAMES.index("illumination")
    earthvis_idx = dataset.FULL_FEATURE_NAMES.index("earth_visibility")
    slope = site.feature_stack[0]
    illumination = site.feature_stack[illum_idx]
    earth_vis = site.feature_stack[earthvis_idx]
    water_km = ext.water_distance_km_grid(site.grid)

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    panels = [
        (axes[0, 0], slope, "inferno", 0, 30, f"1. Topology input: slope (deg)"),
        (axes[0, 1], prob_map, "viridis", 0, 1, "1. Topology suitability (model output)"),
        (axes[0, 2], illumination, "cividis", 0, 1, "2. Communications input: solar illumination (fraction)"),
        (axes[1, 0], earth_vis, "cividis", 0, 1, "2. Communications: Earth visibility (fraction)"),
        (axes[1, 1], water_km, "Blues_r", 0, np.nanpercentile(water_km, 99), "3. Water: distance to nearest cold trap (km)"),
        (axes[1, 2], composite_map, "plasma", 0, 1, "FINAL: weighted composite (60% topo + 30% comms + 10% water)"),
    ]
    all_axes = [ax for ax, *_ in panels]
    for ax, data, cmap, vmin, vmax, title in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("column (5 m/px)")
        ax.set_ylabel("row (5 m/px)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Final ranked landing positions (top N, by composite score) - shown on every panel
    # so each input layer can be read at exactly the spots the ranking picked.
    for ax in all_axes:
        for rank, c in enumerate(candidates, 1):
            ax.plot(c["col"], c["row"], "r+", markersize=12, markeredgewidth=2)
            ax.annotate(
                str(rank), (c["col"], c["row"]), color="white", fontsize=8, xytext=(4, 4), textcoords="offset points"
            )

    rows, cols = prob_map.shape
    for p in pois:
        if not (0 <= p["row"] < rows and 0 <= p["col"] < cols):
            continue
        for ax in all_axes:
            ax.plot(p["col"], p["row"], marker="*", color="cyan", markersize=10, markeredgecolor="black")
            ax.annotate(p["name"], (p["col"], p["row"]), color="cyan", fontsize=7, xytext=(5, -8), textcoords="offset points")

    fig.suptitle(f"{site.name}: all input layers and final ranked candidates", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <SiteDirName>  (e.g. Site23)")
        sys.exit(1)
    site_name = sys.argv[1]
    site_dir = ROOT / site_name
    roi_shp = None
    for shp in site_dir.glob("*_ROIs.shp"):
        roi_shp = shp

    bundle = joblib.load(MODEL_PATH)
    model, window, model_feature_names = bundle["model"], bundle["window"], bundle["feature_names"]

    print(f"Loading {site_name} (topology + communications + water + POI)...")
    site = dataset.load_site(site_dir, roi_shp=roi_shp, window=window, include_external=True)

    print("Scoring topology (criterion 1/4: detects candidate regions)...")
    prob_map = score_site(site, model, n_topology_features=len(model_feature_names))

    print("Finding candidate landing zones and attaching communications (2/4) + water distance (3/4)...")
    candidates = find_candidates(prob_map, site)
    candidates = rank_candidates(candidates)

    print("Looking up nearby named points of interest (4/4, informational only)...")
    pois = poi_mod.load_pois_near_grid(site.grid)
    annotate_nearest_poi(candidates, pois)
    for p in pois:
        tag = "inside site" if p["inside_site"] else "nearby"
        print(f"  POI ({tag}): {p['name']} [{p['type']}] lat={p['lat']:.3f} lon={p['lon']:.3f}")

    print(
        f"\nTop {len(candidates)} candidate landing sites for {site_name}, ranked by "
        f"{WEIGHTS['topology']:.0%} topology + {WEIGHTS['communications']:.0%} communications + "
        f"{WEIGHTS['water']:.0%} water proximity "
        f"(topology detection threshold={PROB_THRESHOLD}, min area={MIN_SAFE_AREA_PX * site.grid.pixel_size**2:.0f} m^2):"
    )
    for rank, c in enumerate(candidates, 1):
        poi_txt = f"nearest POI: {c['nearest_poi']} ({c['nearest_poi_km']:.1f} km)" if c["nearest_poi"] else "no nearby POI"
        print(
            f"  #{rank}: composite={c['composite_score']:.3f}  "
            f"local_xy=({c['x_m']:.1f}, {c['y_m']:.1f}) m  lon/lat=({c['lon_deg']:.4f}, {c['lat_deg']:.4f})\n"
            f"       1.topology={c['topology_prob']:.3f} (area={c['area_m2']:.0f} m^2)  "
            f"2.communications(earth_vis)={c['communications']:.3f} (illum={c['illumination']:.2f})  "
            f"3.water={c['water_km']:.1f} km  4.{poi_txt}"
        )

    if site.roi_mask is not None:
        in_roi_mean_prob = prob_map[site.roi_mask & site.valid_mask].mean()
        print(f"\n[sanity check] mean predicted topology prob inside known ROI polygons: {in_roi_mean_prob:.3f}")

    print("Building full-grid composite suitability map for visualization...")
    earthvis_idx = dataset.FULL_FEATURE_NAMES.index("earth_visibility")
    earth_vis_map = site.feature_stack[earthvis_idx]
    water_km_map = ext.water_distance_km_grid(site.grid)
    water_goodness_map = 1.0 - np.clip(water_km_map, 0, WATER_DISTANCE_CAP_KM) / WATER_DISTANCE_CAP_KM
    composite_map = (
        WEIGHTS["topology"] * np.nan_to_num(prob_map)
        + WEIGHTS["communications"] * np.nan_to_num(earth_vis_map)
        + WEIGHTS["water"] * np.nan_to_num(water_goodness_map)
    )
    composite_map[~site.valid_mask] = np.nan

    out_png = OUT_DIR / f"{site_name}_prediction.png"
    plot_result(site, prob_map, composite_map, candidates, pois, out_png)

    csv_path = OUT_DIR / f"{site_name}_candidates.csv"
    with open(csv_path, "w") as f:
        f.write(
            "rank,composite_score,topology_prob,area_m2,communications_earth_vis,illumination,"
            "water_km,x_m,y_m,lon_deg,lat_deg,nearest_poi,nearest_poi_km\n"
        )
        for rank, c in enumerate(candidates, 1):
            poi_km = f"{c['nearest_poi_km']:.2f}" if c["nearest_poi_km"] is not None else ""
            f.write(
                f"{rank},{c['composite_score']:.4f},{c['topology_prob']:.4f},{c['area_m2']:.1f},"
                f"{c['communications']:.3f},{c['illumination']:.3f},{c['water_km']:.2f},"
                f"{c['x_m']:.2f},{c['y_m']:.2f},{c['lon_deg']:.5f},{c['lat_deg']:.5f},"
                f"{c['nearest_poi'] or ''},{poi_km}\n"
            )
    print(f"Saved candidate table to {csv_path}")

    poi_csv = OUT_DIR / f"{site_name}_pois.csv"
    with open(poi_csv, "w") as f:
        f.write("name,type,diameter_km,lat,lon,x_m,y_m,inside_site\n")
        for p in pois:
            f.write(f"{p['name']},{p['type']},{p['diameter_km']},{p['lat']:.4f},{p['lon']:.4f},{p['x_m']:.1f},{p['y_m']:.1f},{p['inside_site']}\n")
    print(f"Saved nearby points of interest to {poi_csv}")


if __name__ == "__main__":
    main()
