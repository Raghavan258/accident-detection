"""HTTP route for the synthetic-event chatbot."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from accident_detection.synthetic_v2.chat import answer
from accident_detection.synthetic_v2.query import load_events

_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


def mount_synthetic(app: FastAPI) -> None:
    """Add the chatbot route. The table path comes from the environment."""

    @app.post("/synthetic/chat")
    def synthetic_chat(body: ChatRequest) -> dict[str, object]:
        path = Path(os.environ.get("IVERAS_SYNTHETIC", "datasets/synthetic_v2/events.parquet"))
        frame = _frame(path)
        return answer(frame, body.message)


def _frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="synthetic v2 table is not available")
    stamp = path.stat().st_mtime
    cached = _CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    frame = load_events(path)
    _CACHE[str(path)] = (stamp, frame)
    return frame
