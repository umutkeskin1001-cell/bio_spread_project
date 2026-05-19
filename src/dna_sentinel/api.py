from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dna_sentinel.utils import InferenceService

CHECKPOINT = os.getenv("DNA_SENTINEL_CHECKPOINT", "artifacts/dna_sentinel/kmer_transformer_best.pt")
DEVICE = os.getenv("DNA_SENTINEL_DEVICE", "cpu")

service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    if os.path.exists(CHECKPOINT):
        service = InferenceService(CHECKPOINT, device=DEVICE)
    yield


app = FastAPI(title="DNA Sentinel", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    sequence_id: str = Field(default="query")
    dna: str = Field(min_length=100, pattern=r"^[ACGTRYSWKMBDHVNacgtryswkmbdhvn\s]+$")


class PredictBatchRequest(BaseModel):
    sequences: list[PredictRequest]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if service is not None else "missing_checkpoint", "checkpoint": CHECKPOINT}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail=f"checkpoint not loaded: {CHECKPOINT}")
    return service.predict(req.sequence_id, req.dna)


@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> list[dict]:
    if service is None:
        raise HTTPException(status_code=503, detail=f"checkpoint not loaded: {CHECKPOINT}")
    return [service.predict(s.sequence_id, s.dna) for s in req.sequences]
