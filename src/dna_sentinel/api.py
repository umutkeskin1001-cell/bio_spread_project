from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from dna_sentinel.utils import InferenceService, ValidationError, logger

try:
    from importlib.metadata import version as _v
    _VERSION = _v("dna-sentinel")
except Exception:
    _VERSION = "0.3.0"

service: InferenceService | None = None
_uptime: float = 0.0
_MAX_BATCH = int(os.getenv("CASSIOPEIA_MAX_BATCH", "100"))
_MAX_DNA_LEN = int(os.getenv("CASSIOPEIA_MAX_DNA_LEN", "300000"))

_CHAMPION_CKPT = "artifacts/cassiopeia_prime_v15/cassiopeia_best.pt"
_ENSEMBLE_CKPT = "artifacts/cassiopeia_prime_v14/cassiopeia_best.pt"
_ENSEMBLE_WEIGHT = 0.53
_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, _uptime
    _uptime = time.time()
    ckpt = _env("CASSIOPEIA_CHECKPOINT", _CHAMPION_CKPT)
    ensemble_ckpt = _env("CASSIOPEIA_ENSEMBLE_CHECKPOINT", "")
    if not ensemble_ckpt:
        ensemble_ckpt = _ENSEMBLE_CKPT if Path(_ENSEMBLE_CKPT).exists() else None
    ew = float(_env("CASSIOPEIA_ENSEMBLE_WEIGHT", str(_ENSEMBLE_WEIGHT)))
    if os.path.exists(ckpt):
        service = InferenceService(ckpt, device=_env("CASSIOPEIA_DEVICE", "cpu"),
                                   ensemble_checkpoint=ensemble_ckpt, ensemble_weight=ew)
        logger.info(f"model loaded from {ckpt}" + (f" + ensemble {ensemble_ckpt}" if ensemble_ckpt else ""))
    else:
        logger.warning(f"checkpoint not found at {ckpt}; /predict and /predict-batch will return 503")
    yield


app = FastAPI(title="Cassiopeia Prime", version=_VERSION, lifespan=lifespan,
              docs_url=None, redoc_url=None)

_MIME_TYPES = {
    ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
    ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml",
    ".ico": "image/x-icon", ".txt": "text/plain",
}
_WEB_FILES: dict[str, Path] = {}
if _WEB_DIR.is_dir():
    for f in _WEB_DIR.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(_WEB_DIR))
            _WEB_FILES[rel] = f


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
    ckpt = _env("CASSIOPEIA_CHECKPOINT", _CHAMPION_CKPT)
    upt = round(time.time() - _uptime, 1) if _uptime else 0
    return {"status": "ok" if service is not None else "missing_checkpoint", "checkpoint": ckpt, "uptime_seconds": upt}


def _ckpt_err():
    return f"checkpoint not loaded: {_env('CASSIOPEIA_CHECKPOINT', _CHAMPION_CKPT)}"


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail=_ckpt_err())
    try:
        return service.predict(req.sequence_id, req.dna)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
def predict_batch_req(req: PredictBatchRequest) -> list[dict]:
    if service is None:
        raise HTTPException(status_code=503, detail=_ckpt_err())
    if len(req.sequences) > _MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"batch too large, max {_MAX_BATCH} sequences")
    try:
        return service.predict_batch([(s.sequence_id, s.dna) for s in req.sequences])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    if not full_path:
        full_path = "index.html"
    if full_path in _WEB_FILES:
        fpath = _WEB_FILES[full_path]
        suffix = fpath.suffix
        media_type = _MIME_TYPES.get(suffix, "application/octet-stream")
        return FileResponse(str(fpath), media_type=media_type)
    # Fallback: serve index.html for any unmatched path (SPA support)
    index = _WEB_FILES.get("index.html")
    if index:
        return FileResponse(str(index), media_type="text/html")
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
