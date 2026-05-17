from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dna_sentinel.service import InferenceService

CHECKPOINT = os.getenv("DNA_SENTINEL_CHECKPOINT", "artifacts/dna_sentinel/best.pt")
DEVICE = os.getenv("DNA_SENTINEL_DEVICE", "cpu")

app = FastAPI(title="DNA Sentinel", version="0.1.0")
service: InferenceService | None = None


class PredictRequest(BaseModel):
    sequence_id: str = Field(default="query")
    dna: str = Field(min_length=1)


@app.on_event("startup")
def _startup() -> None:
    global service
    if os.path.exists(CHECKPOINT):
        service = InferenceService(CHECKPOINT, device=DEVICE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if service is not None else "missing_checkpoint", "checkpoint": CHECKPOINT}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail=f"checkpoint not loaded: {CHECKPOINT}")
    return service.predict(req.sequence_id, req.dna)
