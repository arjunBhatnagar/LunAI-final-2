# Mission flight-path CSVs

Put a trajectory CSV here named after a mission's `id` in `data/lunar_missions.json`
(for example `apollo-11.csv`, `artemis-2.csv`). Restart Streamlit, then open the mission
in the **Moon missions** panel and press **Watch the flight path**.

The format is the same as the trajectory CSV upload (see the main README): `t`, `rx ry rz`
in metres (Earth-centred inertial), plus any optional columns such as `vx vy vz`,
`moon_rx moon_ry moon_rz`, `target_rx target_ry target_rz`, `mass`, `stage_index`.
Optional text columns `phase` (a stage name) and `utc` (ISO time) are shown in the viewer,
and `sun_rx sun_ry sun_rz` sets the sunlight direction.
