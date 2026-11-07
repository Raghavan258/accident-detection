"""Report Google Cloud readiness. This module does not call cloud APIs."""

from __future__ import annotations

import os

REQUIRED_ENV = ("GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT", "GCS_BUCKET")
SERVICES = ("Cloud Storage", "Vertex AI", "Cloud Build", "Cloud Monitoring")


def cloud_status() -> dict[str, object]:
    """Return disabled when credentials or project settings are absent."""

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    return {
        "enabled": not missing,
        "missing": missing,
        "services": list(SERVICES),
        "apis_called": False,
        "note": "Local training, registry, serving, and monitoring do not use these services.",
    }
