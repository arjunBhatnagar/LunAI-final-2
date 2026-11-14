# LunAI

LunAI is a 3D viewer for results produced by other programs: the landing-site classifier's output files, and CSV files you upload (spacecraft trajectories or tables of spots on the Moon). It doesn't run any models or plan any routes itself.

## What it shows

- **Landing results:** for Sites 1, 4 and 23 near the lunar south pole, the classifier's own output files (`classifier/LunAI-Classifier2-main/python/outputs`). The **Spot** tab shows:
  - the classifier's top-ranked spots with their scores (and a **Go** button for each)
  - the score split into its parts
  - the classifier's own prediction figure (click to enlarge)
  - nearby named craters
- **Uploaded CSVs:** pick a file in **Upload a CSV**, or drop it anywhere on the page. The file is read in the browser.
  - A **trajectory** (`t`, `rx ry rz`, …) replays at true scale, with Earth fixed at the centre, plus a speed graph, stages and a time scrubber.
  - A **results table** with latitude/longitude columns (for example a `Site*_candidates.csv`) puts every row on the Moon as a blue spot and lists it with all its columns.
- **Go to coordinates:** type a latitude/longitude to move the camera there.
- **Moon missions:** 26 past missions and 7 planned NASA missions, with descriptions, dates, crews and statistics. A mission can also show a flight-path CSV you provide.
- **Context:** an Earth–Moon scene with real directions and lighting for the current date (JPL DE441), LOLA terrain summaries for the three regions, and a Sun & Earth summary for each region (now and over the next 24 h).

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

`app.py` serves the page (`viewer.py`) plus a small API that hands the page the classifier's figures and any stored mission-path CSVs. Three.js and the planet textures load from `cdn.jsdelivr.net` and `threejs.org`, so the page needs internet access. The page is cached for an hour; refresh after adding files to `data/mission_paths`.

## Project structure

- `app.py`: the server (`st.App` with the routes in `lunai/api.py`)
- `viewer.py`: the Streamlit page that embeds the viewer
- `lunai/web/`: the viewer (`index.html`, `styles.css`, `app.js`); `lunai/visuals.py` inlines them into one page
- `lunai/classification/landing_site.py`: reads the classifier's output files (nothing is recalculated)
- `lunai/services.py`: assembles the page data (classifier results, scene snapshot, missions)
- `data/lunar_missions.json`: the mission catalogue; `data/mission_paths/`: optional mission flight-path CSVs
- `classifier/LunAI-Classifier2-main/`: the landing-site classifier project whose results are shown
- `tests/`: pytest suite

## Classifier results

The viewer reads, for each of `Site1`, `Site4` and `Site23`:

| File | Shown as |
| --- | --- |
| `python/outputs/<Site>_candidates.csv` | the ranked spots, their scores and columns, and the region's verdict (best score ≥ 0.75 "good", ≥ 0.5 "possible") |
| `python/outputs/<Site>_pois.csv` | nearby named craters |
| `python/outputs/<Site>_prediction.png` | the classifier's figure, served at `/api/classifier-figure/<Site>` |

To show new results, re-run the classifier so it rewrites those files, then restart Streamlit. Set `LUNAI_LANDING_CLASSIFIER_DIR` if the classifier folder lives elsewhere.

## Uploading CSVs

**Trajectories** need `t` (s, increasing) and `rx ry rz` (m, Earth-centred inertial). Optional columns:

| Column | Units |
| --- | --- |
| `vx vy vz` | m/s (otherwise differentiated from positions) |
| `moon_rx moon_ry moon_rz` | m (otherwise the Moon is held at the scene date's position) |
| `target_rx target_ry target_rz`, `target_distance` | m |
| `mass`, `thrust_N`, `stage_index`, `altitude` | kg, N, integer, m above Earth's surface |
| `mdot`, `ax ay az` | kg/s, m/s² (graphed) |
| `phase`, `utc`, `sun_rx sun_ry sun_rz` | stage name, ISO time (shown in the time bar), sunlight direction |

Everything is drawn at true scale in ECI. Earth's centre stays fixed, the Moon moves to `moon_r` at each sample, and the vehicle is drawn at `r` with no rescaling. Tracks that reach the Moon's surface also get a ground patch with a metric grid. Lunar latitude/longitude and sunlight assume J2000-aligned axes.

**Results tables** need latitude and longitude columns (`lat`/`lon`, `lat_deg`/`lon_deg` or `latitude`/`longitude`). Optional: `composite_score` or `score`, `name`/`label`, and `rank`. Every other column is listed with each spot exactly as written.

## Mission flight paths

Save a trajectory CSV as `data/mission_paths/<mission id>.csv`, using the mission's `id` from `data/lunar_missions.json` (for example `apollo-11.csv`). That mission then gets a **Show the flight path** button. Any past mission can also load a CSV straight from its details. The files are served at `GET /api/mission-path/<mission id>`.

## Updating the mission list

Edit `data/lunar_missions.json`. Each mission needs `id`, `name`, `agency`, `type`, `status` (`success`, `partial`, `failed` or `planned`), `summary` and `description`. Past missions also need a `launch` date (`YYYY-MM-DD`). Planned ones need a `target` and should list `sources`. Planned dates reflect public information as of November 2026.

## Notes

- LOLA measures topography, not water or minerals. "Shelter" is a terrain clue, not measured shadow or ice.
- Region Sun/Earth visibility is computed for the centre of each region from the DE441 ephemeris and a terrain horizon from the region's DEM (about 8–10 km of terrain).
- The overview scene stretches distances and body sizes so everything fits; uploaded trajectories are shown at true scale.
