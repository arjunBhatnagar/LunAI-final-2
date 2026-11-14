import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from lunai import services
from lunai.api import routes
from lunai.classification import landing_site
from lunai.missions import load_missions


@pytest.fixture(scope="module")
def client():
    return TestClient(Starlette(routes=routes))


def test_health_reports_classifier(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["classifier"]


def test_only_display_routes_exist(client):
    for path in ("/api/route", "/api/classify", "/api/environment"):
        assert client.post(path, json={}).status_code in (404, 405)


def test_mission_path_endpoint(client, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "MISSION_PATHS_DIR", tmp_path)
    (tmp_path / "apollo-11.csv").write_text("t,rx,ry,rz\n0,7000000,0,0\n60,7000100,1000,0\n", encoding="utf-8")
    (tmp_path / "not-a-mission.csv").write_text("t\n", encoding="utf-8")
    assert services.mission_path_ids() == ["apollo-11"]
    response = client.get("/api/mission-path/apollo-11")
    assert response.status_code == 200 and response.text.startswith("t,rx")
    assert client.get("/api/mission-path/apollo-12").status_code == 404
    assert client.get("/api/mission-path/..%2Fsecrets").status_code == 404


@pytest.mark.skipif(landing_site.figure_path("Site1") is None, reason="classifier outputs not installed")
def test_classifier_figure_is_served(client):
    response = client.get("/api/classifier-figure/Site1")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert client.get("/api/classifier-figure/../../secret").status_code == 404
    assert client.get("/api/classifier-figure/Site2").status_code == 404


def test_mission_catalogue_is_consistent():
    missions = load_missions()["missions"]
    assert {"apollo-11", "artemis-2", "artemis-4"} <= {m["id"] for m in missions}
    walkers = sum(1 for m in missions for c in m.get("crew", []) if "walked on the Moon" in c["role"])
    assert walkers == 12
    planned = [m for m in missions if m["status"] == "planned"]
    assert planned and missions[-len(planned):] == planned  # past first, planned last
    assert all(m.get("sources") for m in planned)
