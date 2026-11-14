"""Small HTTP API next to the Streamlit page (mounted by ``app.py`` with ``st.App``).

It only serves data for display: stored mission flight-path CSVs, the
landing-site classifier's own figures, and terrain heights cut from the local
LOLA elevation models for the 3D surface view.
"""
from __future__ import annotations

import anyio
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

from lunai import services, surface
from lunai.classification.landing_site import figure_path


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "classifier": services.classifier().name, "mission_paths": services.mission_path_ids()})


async def mission_path(request: Request):
    try:
        text = await anyio.to_thread.run_sync(services.mission_path_csv, request.path_params["mission_id"])
        return PlainTextResponse(text, media_type="text/csv")
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


async def classifier_figure(request: Request):
    path = figure_path(request.path_params["site"])
    if path is None:
        return JSONResponse({"error": "No classifier figure for that site."}, status_code=404)
    return FileResponse(path, media_type="image/png")


async def terrain(request: Request) -> JSONResponse:
    try:
        lat, lon = float(request.query_params["lat"]), float(request.query_params["lon"])
        size = float(request.query_params.get("size_km", 8))
        patch = await anyio.to_thread.run_sync(lambda: surface.terrain_patch(lat, lon, size))
    except (KeyError, ValueError) as exc:
        return JSONResponse({"error": f"Give lat and lon as numbers ({exc})."}, status_code=400)
    return JSONResponse(patch or {"available": False})


routes = [
    Route("/api/terrain", terrain, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/mission-path/{mission_id}", mission_path, methods=["GET"]),
    Route("/api/classifier-figure/{site}", classifier_figure, methods=["GET"]),
]
