"""HTTP API for prediction and the live MLOps dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from accident_detection.mlops.catalog import example_rows, feature_schema, system_health
from accident_detection.mlops.dashboard import render_dashboard
from accident_detection.mlops.registry import load_production, rollback_production
from accident_detection.mlops.service import predict_from_features, predict_from_window
from accident_detection.mlops.status import build_status

DEFAULT_URI = "sqlite:///mlops/mlflow.db"


class FeatureRequest(BaseModel):
    features: dict[str, float]


class WindowRequest(BaseModel):
    gsensor: list[list[float]] = Field(default_factory=list)
    gyro: list[list[float]] = Field(default_factory=list)
    gps_speed: list[float] = Field(default_factory=list)
    gyro_is_hd: bool = False
    vehiclesize: str | None = None


class ProductionRequest(BaseModel):
    version: str


def create_app(tracking_uri: str, report_dir: Path | None = None) -> FastAPI:
    """Build an app bound to one MLflow store."""

    app = FastAPI(title="IVERAS crash detector")
    app.state.tracking_uri = tracking_uri
    app.state.report_dir = report_dir
    app.state.db_path = Path(os.environ.get("IVERAS_DB", "data/processed/accident_detection.db"))
    app.state.features_path = Path(os.environ.get("IVERAS_FEATURES", "datasets/processed/features.parquet"))
    _cors(app)
    _mount(app)
    return app


def _mount(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(body: FeatureRequest) -> dict[str, object]:
        return _predict(app, lambda bundle: predict_from_features(bundle, body.features))

    @app.post("/predict/window")
    def predict_window(body: WindowRequest) -> dict[str, object]:
        return _predict(app, lambda bundle: predict_from_window(bundle, body.model_dump()))

    _mount_reads(app)


def _mount_reads(app: FastAPI) -> None:
    @app.get("/mlops/status")
    def mlops_status() -> dict[str, object]:
        return build_status(app.state.tracking_uri, app.state.report_dir)

    @app.get("/mlops/health")
    def mlops_health() -> dict[str, object]:
        return system_health(app.state.tracking_uri, app.state.report_dir, app.state.db_path)

    @app.get("/mlops/schema")
    def mlops_schema() -> dict[str, object]:
        return feature_schema()

    @app.get("/mlops/examples")
    def mlops_examples() -> dict[str, object]:
        return example_rows(app.state.features_path)

    @app.get("/mlops/reports/drift")
    def drift_report() -> FileResponse:
        return _drift_file(app.state.report_dir)

    @app.post("/mlops/registry/production")
    def set_production(body: ProductionRequest) -> dict[str, object]:
        return _promote(app, body.version)

    @app.get("/mlops")
    def mlops_page() -> HTMLResponse:
        status = build_status(app.state.tracking_uri, app.state.report_dir)
        return HTMLResponse(render_dashboard(status))


def _promote(app: FastAPI, version: str) -> dict[str, object]:
    try:
        return rollback_production(app.state.tracking_uri, version)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _cors(app: FastAPI) -> None:
    origins = os.environ.get("IVERAS_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _drift_file(report_dir: Path | None) -> FileResponse:
    path = None if report_dir is None else report_dir / "drift.html"
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="drift report has not been generated")
    return FileResponse(path, media_type="text/html")


def _predict(app: FastAPI, score) -> dict[str, object]:
    try:
        bundle = load_production(app.state.tracking_uri)
        return score(bundle)
    except LookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"missing features: {error}") from error


def _default_app() -> FastAPI:
    report = os.environ.get("IVERAS_REPORT_DIR")
    return create_app(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI), Path(report) if report else None)


app = _default_app()
