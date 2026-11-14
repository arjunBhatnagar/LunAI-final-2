from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from lunai.frames import body_frames

AU_KM = 149_597_870.7
EARTH_RADIUS_KM = 6_371.0
MOON_RADIUS_KM = 1_737.4
SUN_RADIUS_KM = 695_700.0
SCENE_AU_SCALE = 32.0
SCENE_EARTH_MOON_SEPARATION = 10.5


def _vector3(value: Sequence[float], field: str) -> List[float]:
    if len(value) != 3:
        raise ValueError(f"{field} must contain exactly three coordinates.")
    vector = [float(component) for component in value]
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{field} must contain only finite numbers.")
    return vector


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[index] - b[index] for index in range(3)]


def _add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[index] + b[index] for index in range(3)]


def _scale(vector: Sequence[float], factor: float) -> List[float]:
    return [component * factor for component in vector]


def _validate_graphs(graphs: Any) -> List[Dict[str, Any]]:
    if not isinstance(graphs, list):
        raise ValueError("Input 'graphs' must be an array.")
    for index, graph in enumerate(graphs):
        if not isinstance(graph, Mapping):
            raise ValueError(f"graphs[{index}] must be an object.")
        if not all(isinstance(graph.get(key), str) and graph[key] for key in ("id", "title", "unit")):
            raise ValueError(f"graphs[{index}] must include non-empty id, title, and unit strings.")
        points = graph.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"graphs[{index}].points must be a non-empty array.")
        for point_index, point in enumerate(points):
            if not isinstance(point, Mapping) or not isinstance(point.get("time"), str):
                raise ValueError(f"graphs[{index}].points[{point_index}] must include a time string.")
            if not math.isfinite(float(point.get("value", float("nan")))):
                raise ValueError(f"graphs[{index}].points[{point_index}].value must be finite.")
    return [dict(graph) for graph in graphs]


def _parse_epoch(epoch: Any) -> datetime:
    if not isinstance(epoch, str) or not epoch.strip():
        raise ValueError("Input must include a non-empty 'epoch_utc' timestamp.")
    try:
        parsed = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Input 'epoch_utc' must be an ISO-8601 timestamp.") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def build_simulation_state(
    *,
    epoch_utc: str,
    earth_position_km: Sequence[float],
    moon_position_km: Sequence[float],
    other_bodies: Mapping[str, Mapping[str, Any]] | None = None,
    graphs: List[Dict[str, Any]] | None = None,
    source: str = "External Python calculation",
) -> Dict[str, Any]:
    """Convert Sun-relative body positions into the viewer's input-data contract."""
    epoch = _parse_epoch(epoch_utc)
    earth = _vector3(earth_position_km, "earth_position_km")
    moon = _vector3(moon_position_km, "moon_position_km")
    graph_data = _validate_graphs(graphs or [])
    earth_scene = _scale(earth, SCENE_AU_SCALE / AU_KM)
    earth_to_moon = _subtract(moon, earth)
    earth_moon_distance = _norm(earth_to_moon)
    if earth_moon_distance <= 0:
        raise ValueError("Earth and Moon positions must be distinct.")
    moon_scene = _add(
        earth_scene,
        _scale(earth_to_moon, SCENE_EARTH_MOON_SEPARATION / earth_moon_distance),
    )
    bodies: Dict[str, Dict[str, Any]] = {
        "sun": {"name": "Sun", "position_km": [0.0, 0.0, 0.0], "radius_km": SUN_RADIUS_KM},
        "earth": {"name": "Earth", "position_km": earth, "radius_km": EARTH_RADIUS_KM},
        "moon": {"name": "Moon", "position_km": moon, "radius_km": MOON_RADIUS_KM},
    }
    scene_positions = {"sun": [0.0, 0.0, 0.0], "earth": earth_scene, "moon": moon_scene}
    scene_radii = {"sun": 1.15, "earth": 2.8, "moon": 1.2}
    for body_id, body in (other_bodies or {}).items():
        if body_id in bodies:
            raise ValueError(f"Additional body ID {body_id!r} is reserved.")
        position = _vector3(body.get("position_km", []), f"bodies.{body_id}.position_km")
        radius_km = float(body.get("radius_km", 0.0))
        if not math.isfinite(radius_km) or radius_km < 0:
            raise ValueError(f"bodies.{body_id}.radius_km must be a finite non-negative number.")
        bodies[body_id] = {
            "name": str(body.get("name", body_id)),
            "position_km": position,
            "radius_km": radius_km,
        }
        scene_positions[body_id] = _scale(position, SCENE_AU_SCALE / AU_KM)
        scene_radii[body_id] = max(0.16, min(0.7, math.sqrt(radius_km / EARTH_RADIUS_KM) * 0.12))
    return {
        "schema_version": 1,
        "coordinate_frame": "heliocentric",
        "origin": "Sun",
        "units": "km",
        "epoch_utc": epoch_utc,
        "source": source,
        "bodies": bodies,
        "scene_positions": scene_positions,
        "scene_radii": scene_radii,
        "body_frames": body_frames(epoch),
        "graphs": graph_data,
        "display_note": (
            "Heliocentric directions are preserved, but displayed distances and body sizes are exaggerated "
            "for visibility. Raw input positions remain in kilometers."
        ),
    }


def validate_simulation_state(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate raw Sun-origin positions and build the viewer-ready fields."""
    if data.get("coordinate_frame") != "heliocentric" or data.get("origin") != "Sun":
        raise ValueError("Input must use a heliocentric coordinate frame with the Sun as origin.")
    if data.get("units") != "km":
        raise ValueError("Input position units must be 'km'.")
    bodies = data.get("bodies")
    if not isinstance(bodies, Mapping):
        raise ValueError("Input 'bodies' must be an object containing Earth and Moon.")
    earth = bodies.get("earth")
    moon = bodies.get("moon")
    if not isinstance(earth, Mapping) or not isinstance(moon, Mapping):
        raise ValueError("Input 'bodies' must include 'earth' and 'moon'.")
    epoch = data.get("epoch_utc")
    _parse_epoch(epoch)
    graphs = _validate_graphs(data.get("graphs", []))
    sun = bodies.get("sun")
    if sun is not None:
        if not isinstance(sun, Mapping) or _norm(_vector3(sun.get("position_km", []), "bodies.sun.position_km")) > 1e-6:
            raise ValueError("The Sun position must be [0, 0, 0] in a Sun-origin input.")
    other_bodies = {}
    for body_id, body in bodies.items():
        if body_id in {"sun", "earth", "moon"}:
            continue
        if not isinstance(body, Mapping):
            raise ValueError(f"bodies.{body_id} must be an object.")
        other_bodies[str(body_id)] = body
    return build_simulation_state(
        epoch_utc=epoch,
        earth_position_km=earth.get("position_km", []),
        moon_position_km=moon.get("position_km", []),
        other_bodies=other_bodies,
        graphs=graphs,
        source=str(data.get("source", "External Python calculation")),
    )


def load_simulation_state(path: Path | str) -> Dict[str, Any]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as source_file:
        payload = json.load(source_file)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Simulation input at {source_path} must contain a JSON object.")
    return validate_simulation_state(payload)


@lru_cache(maxsize=1)
def _ephemeris_resources():
    from skyfield.api import load

    kernel_path = Path(__file__).resolve().parents[1] / "de441.bsp"
    if not kernel_path.is_file():
        raise FileNotFoundError(f"JPL DE441 ephemeris kernel not found: {kernel_path}")
    return load(str(kernel_path)), load.timescale()


def generate_de441_state(epoch: datetime | None = None, graph_days: int = 30) -> Dict[str, Any]:
    """Calculate heliocentric Earth/Moon inputs and distance series with JPL DE441."""
    epoch = epoch or datetime.now(timezone.utc)
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    epoch = epoch.astimezone(timezone.utc).replace(microsecond=0)
    kernel, timescale = _ephemeris_resources()
    sun, earth_body, moon_body = kernel["Sun"], kernel["Earth"], kernel["Moon"]

    def positions_at(when: datetime) -> tuple[List[float], List[float]]:
        t = timescale.utc(when.year, when.month, when.day, when.hour, when.minute, when.second)
        sun_position = sun.at(t).position.km
        earth_position = [float(value) for value in earth_body.at(t).position.km - sun_position]
        moon_position = [float(value) for value in moon_body.at(t).position.km - sun_position]
        return earth_position, moon_position

    earth_position, moon_position = positions_at(epoch)
    earth_sun_points: List[Dict[str, Any]] = []
    moon_earth_points: List[Dict[str, Any]] = []
    moon_sun_points: List[Dict[str, Any]] = []
    for day in range(max(1, graph_days) + 1):
        when = epoch + timedelta(days=day)
        sample_earth, sample_moon = positions_at(when)
        timestamp = when.isoformat(timespec="seconds").replace("+00:00", "Z")
        earth_sun_points.append({"time": timestamp, "value": _norm(sample_earth) / AU_KM})
        moon_earth_points.append({"time": timestamp, "value": _norm(_subtract(sample_moon, sample_earth))})
        moon_sun_points.append({"time": timestamp, "value": _norm(sample_moon) / AU_KM})

    return build_simulation_state(
        epoch_utc=epoch.isoformat(timespec="seconds").replace("+00:00", "Z"),
        earth_position_km=earth_position,
        moon_position_km=moon_position,
        source="JPL DE441 via Skyfield",
        graphs=[
            {"id": "earth-sun", "title": "Earth–Sun distance", "unit": "AU", "points": earth_sun_points},
            {"id": "moon-earth", "title": "Moon–Earth distance", "unit": "km", "points": moon_earth_points},
            {"id": "moon-sun", "title": "Moon–Sun distance", "unit": "AU", "points": moon_sun_points},
        ],
    )


def write_de441_state(path: Path | str, epoch: datetime | None = None, graph_days: int = 30) -> None:
    write_simulation_input(path, generate_de441_state(epoch=epoch, graph_days=graph_days))


def write_simulation_input(path: Path | str, state: Mapping[str, Any]) -> None:
    validated = validate_simulation_state(state)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coordinate_frame": validated["coordinate_frame"],
        "origin": validated["origin"],
        "units": validated["units"],
        "epoch_utc": validated["epoch_utc"],
        "source": validated["source"],
        "bodies": validated["bodies"],
        "graphs": validated["graphs"],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "simulation_input.json"
    write_de441_state(output)
    print(f"Wrote heliocentric simulation input to {output}")
