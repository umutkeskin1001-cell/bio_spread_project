from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from dna_sentinel.utils import InferenceService, ValidationError, logger

try:
    from importlib.metadata import version as _v
    _VERSION = _v("dna-sentinel")
except Exception:
    _VERSION = "0.2.0"

service: InferenceService | None = None
_uptime: float = 0.0
_MAX_BATCH = int(os.getenv("CASSIOPEIA_MAX_BATCH", "100"))
_MAX_DNA_LEN = int(os.getenv("CASSIOPEIA_MAX_DNA_LEN", "100000"))


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, _uptime
    _uptime = time.time()
    ckpt = _env("CASSIOPEIA_CHECKPOINT", "artifacts/dna_sentinel/cassiopeia_best.pt")
    if os.path.exists(ckpt):
        service = InferenceService(ckpt, device=_env("CASSIOPEIA_DEVICE", "cpu"))
        logger.info(f"model loaded from {ckpt}")
    yield


app = FastAPI(title="Cassiopeia", version=_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    sequence_id: str = Field(default="query")
    dna: str

    @field_validator("dna")
    @classmethod
    def validate_dna(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("DNA sequence is empty")
        if len(v) > _MAX_DNA_LEN:
            raise ValueError(f"DNA sequence too long: {len(v)} > {_MAX_DNA_LEN}")
        return v


class PredictBatchRequest(BaseModel):
    sequences: list[PredictRequest]


@app.get("/health")
def health() -> dict:
    ckpt = _env("CASSIOPEIA_CHECKPOINT", "artifacts/dna_sentinel/cassiopeia_best.pt")
    upt = round(time.time() - _uptime, 1) if _uptime else 0
    return {"status": "ok" if service is not None else "missing_checkpoint", "checkpoint": ckpt, "uptime_seconds": upt}


def _ckpt_err():
    return f"checkpoint not loaded: {_env('CASSIOPEIA_CHECKPOINT', 'artifacts/dna_sentinel/cassiopeia_best.pt')}"


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail=_ckpt_err())
    try:
        return service.predict(req.sequence_id, req.dna)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> list[dict]:
    if service is None:
        raise HTTPException(status_code=503, detail=_ckpt_err())
    if len(req.sequences) > _MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"batch too large, max {_MAX_BATCH} sequences")
    try:
        return service.predict_batch([(s.sequence_id, s.dna) for s in req.sequences])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
