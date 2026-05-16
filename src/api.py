import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bio_spread.constants import ALL_SNAPSHOT_COLS, STATIC_COLS
from bio_spread.serving.service import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BioSpreadAPI")

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/default.yaml")
FEATURE_DIR = os.getenv("FEATURE_DIR", "data/features")
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pt")
PLATT_PATH = os.getenv("PLATT_PATH", "")
DEVICE = os.getenv("DEVICE", "cpu")

service: Optional[InferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Loading BioSpread model from %s...", MODEL_PATH)
    service = InferenceService(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        feature_dir=FEATURE_DIR,
        platt_path=PLATT_PATH,
        device=DEVICE,
    )
    logger.info("BioSpread model loaded successfully")
    yield
    service = None
    logger.info("BioSpread model unloaded")


app = FastAPI(title="BioSpread", lifespan=lifespan)


class SnapshotFeatures(BaseModel):
    n_countries: float = 0.0
    n_hosts: float = 0.0
    years_since_first: float = 0.0
    new_countries_recent: float = 0.0
    new_countries_2y_ago: float = 0.0
    n_records: float = 0.0
    acceleration: float = 0.0
    niche_breadth: float = 0.0


class StaticFeatures(BaseModel):
    log_size: float = 0.0
    gc: float = 0.5
    n_replicon_types: float = 0.0
    n_relaxase_types: float = 0.0
    mobility_score: float = 0.0
    is_conjugative: float = 0.0
    is_mobilizable: float = 0.0
    topology: float = 0.0
    n_orit_types: float = 0.0
    host_range_rank: float = 0.0


class TaxonomyIndices(BaseModel):
    phylum_idx: int = 0
    class_idx: int = 0
    order_idx: int = 0
    family_idx: int = 0
    genus_idx: int = 0


class PredictRequest(BaseModel):
    snapshots: List[SnapshotFeatures] = Field(..., min_length=1)
    static: StaticFeatures
    taxonomy: Optional[TaxonomyIndices] = None


class PredictResponse(BaseModel):
    hazard_year1: float
    hazard_year2: float
    hazard_year3: float
    n_snapshots: int = 0


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]


MAX_BATCH_SIZE = 1024


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = service.predict(
            snapshots=[s.model_dump() for s in req.snapshots],
            static=req.static.model_dump(),
            taxonomy=req.taxonomy.model_dump() if req.taxonomy else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error")
    return PredictResponse(**result)


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(req: BatchPredictRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.requests)} exceeds maximum {MAX_BATCH_SIZE}",
        )
    return [
        PredictResponse(
            **service.predict(
                snapshots=[s.model_dump() for s in r.snapshots],
                static=r.static.model_dump(),
                taxonomy=r.taxonomy.model_dump() if r.taxonomy else None,
            )
        )
        for r in req.requests
    ]


@app.get("/health")
def health():
    if service is None:
        return {"status": "loading", "model": "bio-spread"}
    return {
        "status": "ok",
        "model": "bio-spread",
        "feature_dims": {
            "n_static": len(STATIC_COLS),
            "n_snapshot": len(ALL_SNAPSHOT_COLS),
        },
    }
