import pytest

from lunai.simulation_data import (
    AU_KM,
    build_simulation_state,
    load_simulation_state,
    validate_simulation_state,
    write_simulation_input,
)


def test_sun_origin_positions_are_preserved_and_display_scale_is_separate():
    state = build_simulation_state(
        epoch_utc="2026-11-13T00:00:00Z",
        earth_position_km=[AU_KM, 0, 0],
        moon_position_km=[AU_KM + 384_400, 0, 0],
    )

    assert state["bodies"]["sun"]["position_km"] == [0.0, 0.0, 0.0]
    assert state["bodies"]["earth"]["position_km"] == [AU_KM, 0.0, 0.0]
    assert state["scene_positions"]["earth"] == [32.0, 0.0, 0.0]
    assert state["scene_positions"]["moon"] == pytest.approx([42.5, 0.0, 0.0])


def test_external_input_round_trips_graphs_and_additional_bodies(tmp_path):
    state = build_simulation_state(
        epoch_utc="2026-11-13T00:00:00Z",
        earth_position_km=[AU_KM, 0, 0],
        moon_position_km=[AU_KM + 384_400, 0, 0],
        other_bodies={"mars": {"name": "Mars", "position_km": [1.5 * AU_KM, 0, 0], "radius_km": 3389}},
        graphs=[{
            "id": "distance",
            "title": "Earth-Sun distance",
            "unit": "AU",
            "points": [{"time": "2026-11-13", "value": 1.0}],
        }],
        source="custom Python module",
    )
    path = tmp_path / "simulation_input.json"
    write_simulation_input(path, state)
    loaded = load_simulation_state(path)

    assert loaded["source"] == "custom Python module"
    assert loaded["scene_positions"]["mars"] == pytest.approx([48.0, 0.0, 0.0])
    assert loaded["graphs"][0]["points"][0]["value"] == 1.0


def test_input_rejects_non_heliocentric_coordinates():
    state = build_simulation_state(
        epoch_utc="2026-11-13T00:00:00Z",
        earth_position_km=[AU_KM, 0, 0],
        moon_position_km=[AU_KM + 384_400, 0, 0],
    )
    state["coordinate_frame"] = "earth-centered"

    with pytest.raises(ValueError, match="heliocentric"):
        validate_simulation_state(state)
