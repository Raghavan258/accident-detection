"""Simulator and helper-dispatch routes. They do not retrain the production model."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.request import urlopen
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES
from accident_detection.mlops.registry import load_production
from accident_detection.mlops.service import predict_from_features
from accident_detection.synthetic_v2.dispatch import interpret
from accident_detection.synthetic_v2.explain import column_contributions, severity_tier, top_triggers
from accident_detection.synthetic_v2.live import LiveSession
from accident_detection.synthetic_v2.localities import LOCALITIES
from accident_detection.synthetic_v2.query import load_events
from accident_detection.synthetic_v2.ranges import schema_ranges
from accident_detection.synthetic_v2.telemetry import features_from_telemetry

CITY = (12.80, 13.15, 77.50, 77.80)
NAMES = {str(box["name"]).lower() for box in LOCALITIES}
LABELS = {"all", "crash", "near_miss", "normal_driving"}


class SimulateRequest(BaseModel):
    locality: str
    latitude: float
    longitude: float
    peak_g: float = Field(ge=0, le=12)
    speed_before: float = Field(ge=0, le=180)
    speed_after: float = Field(ge=0, le=180)
    vehicle_size: Literal["Light", "Medium", "Heavy"]
    scenario_preset: Literal["crash", "near_miss", "normal"]
    features: dict[str, float] | None = None
    gyro_deg_s: float | None = Field(default=None, ge=0, le=400)
    tier_labels: dict[str, str] | None = None
    force_broadcast: bool = False


class HelperChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    current_filters: dict[str, object] | None = None


class DispatchAckRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=40)
    action: Literal["ambulance", "eta", "secured"]
    eta_min: float | None = Field(default=None, ge=0, le=180)


def build_v2_router(tracking_uri: str) -> APIRouter:
    """Routes for the live synthetic session."""

    session = LiveSession(_baseline())
    router = APIRouter(prefix="/api/v2", tags=["simulator"])
    cache: dict[str, object] = {}
    _stream(router, session)
    _simulate(router, session, tracking_uri, cache)
    _events(router, session)
    _chat(router, session)
    _ranges(router, session)
    _ack(router, session)
    _route(router)
    return router


def _stream(router: APIRouter, session: LiveSession) -> None:
    @router.get("/stream")
    async def stream() -> StreamingResponse:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=32)
        session.subscribers.add(queue)
        return StreamingResponse(_frames(session, queue), media_type="text/event-stream", headers=_sse_headers())


def _simulate(router: APIRouter, session: LiveSession, tracking_uri: str, cache: dict[str, object]) -> None:
    @router.post("/simulate")
    async def simulate(body: SimulateRequest) -> dict[str, object]:
        _inside_city(body)
        features = _features(body)
        event = session.add(body.model_dump(), _score(tracking_uri, cache, features))
        _attach(event, cache.get("bundle"), features, body)
        if body.force_broadcast:
            event["broadcast"] = True
            event["sos_requested"] = True
        if event["broadcast"]:
            await session.broadcast(event)
        return event


def _events(router: APIRouter, session: LiveSession) -> None:
    @router.get("/events")
    def events(label: str = Query(default="all"), locality: str = Query(default="all"), include_live: bool = Query(default=True)) -> dict[str, object]:
        _known_filter(label, locality)
        return session.catalog(label, locality, include_live)


def _chat(router: APIRouter, session: LiveSession) -> None:
    @router.post("/helper-chat")
    def helper_chat(body: HelperChatRequest) -> dict[str, object]:
        catalog = session.catalog("all", "all", True)
        return interpret(list(catalog["events"]), list(catalog["localities"]), body.message, body.current_filters)


async def _frames(session: LiveSession, queue: asyncio.Queue[str]):
    try:
        yield 'event: ready\ndata: {"status":"connected"}\n\n'
        while True:
            item = await _take(queue)
            yield ": keep-alive\n\n" if item is None else f"event: alert\ndata: {item}\n\n"
    finally:
        session.subscribers.discard(queue)


async def _take(queue: asyncio.Queue[str]) -> str | None:
    try:
        return await asyncio.wait_for(queue.get(), timeout=15)
    except TimeoutError:
        return None


def _score(tracking_uri: str, cache: dict[str, object], features: dict[str, float]) -> dict[str, object]:
    bundle = cache.get("bundle")
    if bundle is None:
        bundle = load_production(tracking_uri)
        cache["bundle"] = bundle
    return predict_from_features(bundle, features)


def _features(body: SimulateRequest) -> dict[str, float]:
    if _complete(body.features):
        return {name: float(body.features[name]) for name in DEFENSIBLE_FEATURE_NAMES}
    return features_from_telemetry(body.peak_g, body.speed_before, body.speed_after, body.vehicle_size)


def _complete(features: dict[str, float] | None) -> bool:
    return bool(features) and all(name in features for name in DEFENSIBLE_FEATURE_NAMES)


def _attach(event: dict[str, object], bundle: object, features: dict[str, float], body: SimulateRequest) -> None:
    rows = column_contributions(bundle if isinstance(bundle, dict) else {}, features)
    labels = body.tier_labels or {}
    for row in rows:
        row["tier"] = labels.get(str(row["name"]), "")
    event["contributions"] = rows
    event["top_triggers"] = top_triggers(rows)
    event["severity_tier"] = severity_tier(float(event["peak_horizontal_g"]), str(event["prediction"]))
    event["features"] = features


def _ranges(router: APIRouter, session: LiveSession) -> None:
    @router.get("/schema-ranges")
    def schema() -> dict[str, object]:
        return schema_ranges(session.baseline)


def _ack(router: APIRouter, session: LiveSession) -> None:
    @router.post("/dispatch-ack")
    async def dispatch_ack(body: DispatchAckRequest) -> dict[str, object]:
        update = session.acknowledge(body.event_id, body.action, body.eta_min)
        if update is None:
            raise HTTPException(status_code=404, detail="No live simulation with that event id.")
        await session.broadcast(update)
        return update

    @router.get("/dispatch-status")
    def dispatch_status(event_id: str) -> dict[str, object]:
        return session.updates.get(event_id) or {"event_id": event_id, "message": "", "synthetic": True}


def _inside_city(body: SimulateRequest) -> None:
    lat_min, lat_max, lon_min, lon_max = CITY
    known = body.locality.lower() in NAMES
    inside = lat_min <= body.latitude <= lat_max and lon_min <= body.longitude <= lon_max
    if not known or not inside:
        raise HTTPException(status_code=422, detail="Pick a Bengaluru zone and a point inside the city box.")


def _known_filter(label: str, locality: str) -> None:
    if label not in LABELS or (locality != "all" and locality.lower() not in NAMES):
        raise HTTPException(status_code=422, detail="Unknown label or locality filter.")


def _route(router: APIRouter) -> None:
    @router.get("/route")
    async def route(hub_lat: float, hub_lng: float, crash_lat: float, crash_lng: float) -> dict[str, object]:
        url = osrm_url(hub_lat, hub_lng, crash_lat, crash_lng)
        return {"url": url, "route": await asyncio.to_thread(_fetch_json, url), "synthetic": True}


def osrm_url(hub_lat: float, hub_lng: float, crash_lat: float, crash_lng: float) -> str:
    """Driving route URL used by the helper map."""

    return (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{hub_lng},{hub_lat};{crash_lng},{crash_lat}?overview=full&geometries=geojson"
    )


def _fetch_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=8) as response:
        return json.loads(response.read().decode())


def _sse_headers() -> dict[str, str]:
    return {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def _baseline() -> pd.DataFrame:
    path = Path(os.environ.get("IVERAS_SYNTHETIC", "datasets/synthetic_v2/events.parquet"))
    if not path.exists():
        return pd.DataFrame()
    return load_events(path)
