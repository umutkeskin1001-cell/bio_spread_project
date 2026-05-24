from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dna_sentinel.utils import InferenceService

service: InferenceService | None = None


def _checkpoint() -> str:
    return os.getenv("CASSIOPEIA_CHECKPOINT", "artifacts/dna_sentinel/cassiopeia_best.pt")


def _device() -> str:
    return os.getenv("CASSIOPEIA_DEVICE", "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    ckpt = _checkpoint()
    if os.path.exists(ckpt):
        service = InferenceService(ckpt, device=_device())
    yield


app = FastAPI(title="Cassiopeia", version="2.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    sequence_id: str = Field(default="query")
    dna: str


class PredictBatchRequest(BaseModel):
    sequences: list[PredictRequest]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if service is not None else "missing_checkpoint", "checkpoint": _checkpoint()}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail=f"checkpoint not loaded: {_checkpoint()}")
    return service.predict(req.sequence_id, req.dna)


@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> list[dict]:
    if service is None:
        raise HTTPException(status_code=503, detail=f"checkpoint not loaded: {_checkpoint()}")
    return service.predict_batch(req.sequences)
