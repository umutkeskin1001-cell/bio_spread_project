from pydantic import BaseModel, Field, model_validator

from bio_spread.constants import HAZARD_COLS


class DataConfig(BaseModel):
    backbones_path: str = "data/project_inputs/silver/plasmid_backbones.tsv"
    split_year: int = 2020
    val_backbone_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    test_backbone_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    spread_horizon: int = Field(default=3, gt=0)
    require_country_history: bool = True
    feature_dir: str = "data/features"

    @model_validator(mode="after")
    def validate_fracs(self):
        if self.val_backbone_frac + self.test_backbone_frac > 1.0:
            raise ValueError("val_backbone_frac + test_backbone_frac must be <= 1.0")
        return self


class ModelConfig(BaseModel):
    static_dim: int = Field(default=128, gt=0)
    temporal_dim: int = Field(default=128, gt=0)
    gru_hidden: int = Field(default=192, gt=0)
    gru_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.15, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=45, gt=0)
    n_hazard_steps: int = Field(default=3, gt=0)
    taxonomy_embed_dim: int = Field(default=8, gt=0)
    categorical_embed_dim: int = Field(default=16, gt=0)
    use_cross_attention: bool = False

    @model_validator(mode="after")
    def validate_static_dim(self):
        if self.static_dim % 2 != 0:
            raise ValueError(f"static_dim must be even (hazard_proj does static_dim // 2), got {self.static_dim}")
        return self

    @model_validator(mode="after")
    def validate_hazard_steps(self):
        expected = len(HAZARD_COLS)
        if self.n_hazard_steps != expected:
            raise ValueError(f"n_hazard_steps must match HAZARD_COLS ({expected}), got {self.n_hazard_steps}")
        return self


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=64, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=10, gt=0)
    warmup_epochs: int = Field(default=5, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    lambda_count: float = Field(default=0.15, ge=0)
    lambda_rank: float = Field(default=0.10, ge=0)
    lambda_cold: float = Field(default=0.25, ge=0)
    lambda_kd: float = Field(default=1.0, ge=0)
    lambda_all: float = Field(default=1.0, ge=0)
    lambda_gate: float = Field(default=0.05, ge=0)
    temporal_masking_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    gaussian_noise_std: float = Field(default=0.05, ge=0)
    gate_entropy_target: float = Field(default=0.4, ge=0)
    calibrate: bool = True
    calibrate_cold: bool = True
    use_adaptive_loss: bool = False
    use_hard_negative_mining: bool = False
    use_curriculum: bool = False
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_warmup(self):
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be < epochs")
        return self


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
