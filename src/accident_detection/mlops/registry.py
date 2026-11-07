"""Register, promote, and roll back the crash detector in MLflow."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from accident_detection.mlops import MODEL_NAME

PRODUCTION = "production"
STAGING = "staging"


def register_and_promote(
    tracking_uri: str,
    model_uri: str,
    run_id: str,
    gate_passed: bool,
    *,
    threshold: float,
    feature_names: list[str],
) -> dict[str, object]:
    """Register a run. Production moves only when the validation gate passed."""

    mlflow.set_tracking_uri(tracking_uri)
    client = _client(tracking_uri)
    _ensure_model(client)
    version = mlflow.register_model(model_uri, MODEL_NAME)
    _tag(client, version.version, gate_passed, threshold, feature_names)
    client.set_registered_model_alias(MODEL_NAME, STAGING, version.version)
    promoted = False
    if gate_passed:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION, version.version)
        promoted = True
    return {"name": MODEL_NAME, "version": version.version, "promoted": promoted, "production": _alias(client, PRODUCTION)}


def rollback_production(tracking_uri: str, version: str) -> dict[str, object]:
    """Point production at a version that already passed the validation gate."""

    client = _client(tracking_uri)
    tag = client.get_model_version(MODEL_NAME, version).tags.get("gate_passed")
    if tag != "true":
        raise ValueError(f"version {version} did not pass the validation gate")
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION, version)
    return {"name": MODEL_NAME, "production": version}


def registry_history(tracking_uri: str) -> dict[str, object]:
    """List versions and the current aliases."""

    client = _client(tracking_uri)
    if not _exists(client):
        return {"name": MODEL_NAME, "versions": [], "production": None, "staging": None}
    versions = []
    for item in client.search_model_versions(f"name='{MODEL_NAME}'"):
        versions.append(
            {
                "version": item.version,
                "run_id": item.run_id,
                "tags": dict(item.tags),
                "creation_timestamp": item.creation_timestamp,
                "last_updated_timestamp": item.last_updated_timestamp,
            }
        )
    return {
        "name": MODEL_NAME,
        "versions": versions,
        "production": _alias(client, PRODUCTION),
        "staging": _alias(client, STAGING),
    }


def load_production(tracking_uri: str):
    """Load the production pipeline and its decision sidecar."""

    client = _client(tracking_uri)
    version = _alias(client, PRODUCTION)
    if version is None:
        raise LookupError("no production model is registered")
    info = client.get_model_version(MODEL_NAME, version)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    decision = {"threshold": float(info.tags["threshold"]), "feature_names": info.tags["feature_names"].split(",")}
    return {"model": model, "version": version, "decision": decision}


def _tag(client: MlflowClient, version: str, gate_passed: bool, threshold: float, names: list[str]) -> None:
    client.set_model_version_tag(MODEL_NAME, version, "gate_passed", str(gate_passed).lower())
    client.set_model_version_tag(MODEL_NAME, version, "threshold", str(threshold))
    client.set_model_version_tag(MODEL_NAME, version, "feature_names", ",".join(names))


def _client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def _ensure_model(client: MlflowClient) -> None:
    if not _exists(client):
        client.create_registered_model(MODEL_NAME)


def _exists(client: MlflowClient) -> bool:
    return any(model.name == MODEL_NAME for model in client.search_registered_models())


def _alias(client: MlflowClient, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except Exception:
        return None
