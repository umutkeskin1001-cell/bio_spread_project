from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

class DataConfig(BaseModel):
    backbones_path: str = Field(..., description="Path to plasmid_backbones.tsv")
    amr_path: str = Field(..., description="Path to amr.tsv")
    records_path: str = Field(..., description="Path to raw backbone records")
    split_year: int = Field(2020, description="Year to split train and validation")
    snapshot_cache_dir: str = Field("data/snapshots", description="Directory to cache temporal snapshots")

class ModelConfig(BaseModel):
    emb_dim: int = Field(256, description="Gene embedding dimension")
    hidden_dim: int = Field(512, description="Hidden layer dimension")
    max_genes: int = Field(300, description="Maximum number of genes per backbone")
    num_heads: int = Field(4, description="Number of attention heads for GeneEncoder")
    num_layers: int = Field(2, description="Number of Transformer layers for GeneEncoder")
    time_freqs: int = Field(12, description="Number of Fourier frequencies for TimeGate")

class TrainingConfig(BaseModel):
    batch_size: int = Field(128, description="Batch size for training")
    lr: float = Field(0.003, description="Learning rate")
    epochs: int = Field(50, description="Maximum number of epochs")
    patience: int = Field(5, description="Early stopping patience")
    kl_annealing: float = Field(0.1, description="KL divergence annealing coefficient")

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
