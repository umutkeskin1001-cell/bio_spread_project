"""
Sovereign-X: Clean, immutable config. No runtime mutation.
"""

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    backbones_path: str = "data/project_inputs/silver/plasmid_backbones.tsv"
    amr_hits_path: str = "data/project_inputs/silver/plasmid_amr_hits.tsv"
    amr_consensus_path: str = "data/project_inputs/silver/plasmid_amr_consensus.tsv"
    split_year: int = 2020
    val_backbone_frac: float = 0.15
    test_backbone_frac: float = 0.15
    spread_horizon: int = 3
    min_new_countries: int = 1
    require_country_history: bool = True
    feature_dir: str = "data/sovereign_features"


class ModelConfig(BaseModel):
    static_dim: int = 128  # static expert output
    temporal_dim: int = 128  # temporal expert projection
    gru_hidden: int = 192  # GRU hidden dimension
    gru_layers: int = 2  # GRU layers
    gru_dropout: float = 0.15
    dropout: float = 0.15
    max_seq_len: int = 45  # padded sequence length
    n_hazard_steps: int = 3  # 3-year hazard horizon
    taxonomy_embed_dim: int = 8  # per-level embedding dim (5*8=40 total)


class TrainingConfig(BaseModel):
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 50
    patience: int = 10
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    lambda_count: float = 0.15
    lambda_rank: float = 0.10
    lambda_cold: float = 0.25  # cold-start auxiliary loss weight
    lambda_all: float = 1.0  # per-timestep (all snapshots) loss weight
    lambda_gate: float = 0.05  # gate entropy penalty weight
    temporal_masking_prob: float = 0.3  # fraction of batch to mask temporal features
    gaussian_noise_std: float = 0.05  # std of noise added to temporal features
    gate_entropy_target: float = 0.4  # min entropy before penalty kicks in
    calibrate: bool = True  # post-training Platt scaling
    calibrate_cold: bool = True  # separate Platt scaler for cold-start
    seed: int = 42


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
