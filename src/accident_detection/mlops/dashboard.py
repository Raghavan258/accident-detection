"""Render the MLOps dashboard from a status payload."""

from __future__ import annotations

import html


def render_dashboard(status: dict[str, object]) -> str:
    """Build an HTML page. Every displayed value comes from `status`."""

    registry = status.get("registry") or {}
    versions = "".join(_version(item) for item in registry.get("versions") or [])
    metrics = _metrics(status.get("validation") or {})
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>IVERAS MLOps</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1917; background: #faf7f2; }}
h1 {{ font-size: 1.6rem; }} section {{ margin: 1.4rem 0; }}
table {{ border-collapse: collapse; }} td, th {{ border-bottom: 1px solid #d6d3d1; padding: 0.3rem 0.6rem; text-align: left; }}
</style></head><body>
<h1>IVERAS model operations</h1>
<p>Production version: {_text(registry.get("production"))}</p>
<p>Staging version: {_text(registry.get("staging"))}</p>
<section><h2>Registered versions</h2><table><tr><th>Version</th><th>Run</th><th>Gate</th></tr>{versions}</table></section>
<section><h2>Validation metrics of the latest tracked run</h2>{metrics}</section>
<section><h2>Dataset and monitoring</h2>
<p>Dataset rows: {_text((status.get("dataset") or {}).get("rows"))}</p>
<p>Validation passed: {_text((status.get("quality") or {}).get("passed"))}</p>
<p>Drift detected: {_text((status.get("drift") or {}).get("drift_detected"))}</p>
<p>Deployment promoted: {_text((status.get("deployment") or {}).get("promoted"))}</p>
</section>
<section><h2>Cloud</h2><p>{_text(status.get("cloud"))}</p></section>
</body></html>"""


def _version(item: dict[str, object]) -> str:
    tags = item.get("tags") or {}
    return f"<tr><td>{_text(item.get('version'))}</td><td>{_text(item.get('run_id'))}</td><td>{_text(tags.get('gate_passed'))}</td></tr>"


def _metrics(validation: dict[str, object]) -> str:
    keys = ("precision_crash", "recall_crash", "f1_crash", "average_precision_crash", "false_positives", "false_negatives")
    rows = "".join(f"<tr><td>{key}</td><td>{_text(validation.get(key))}</td></tr>" for key in keys)
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _text(value: object) -> str:
    return html.escape("none" if value is None else str(value))
