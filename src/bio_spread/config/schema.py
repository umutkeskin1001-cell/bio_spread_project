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

    # Sovereign-X Ultra feature flags
    use_mamba: bool = False
    use_hyperbolic: bool = False
    use_evidential: bool = False
    use_cagrad: bool = False
    use_retrieval: bool = False

    # Mamba-2 + CausalConv hyperparams
    mamba_d_state: int = Field(default=16, gt=0)
    mamba_n_layers: int = Field(default=4, gt=0)
    conv_kernel: int = Field(default=3, gt=0)

    # Hyperbolic taxonomy hyperparams
    tax_dim_per_level: int = Field(default=16, gt=0)
    hyperbolic_curvature: float = Field(default=-1.0, lt=0.0)

    # CAGrad hyperparams
    cagrad_c: float = Field(default=0.4, gt=0.0, le=1.0)

    # Prototypical retrieval
    prototypes: int = Field(default=512, gt=0)
    prototype_k: int = Field(default=10, gt=0)
    ema_alpha: float = Field(default=0.995, ge=0.0, le=1.0)

    # EDL
    edl_lambda_kl: float = Field(default=0.1, ge=0.0)
    edl_target_smoothing: float = Field(default=0.05, ge=0.0)

    # Phylogenetic smoothness
    phylo_smooth_tau: float = Field(default=0.5, gt=0.0)
    phylo_smooth_k: int = Field(default=24, gt=0)

    # FiT
    fit_heads: int = Field(default=2, gt=0)

    # SSL pretraining
    ssl_pretrain_epochs: int = Field(default=15, ge=0)
    ssl_mask_ratio: float = Field(default=0.3, ge=0.0, le=1.0)

    # ACI
    aci_eta: float = Field(default=0.05, gt=0.0)
    aci_alpha: float = Field(default=0.1, gt=0.0, lt=1.0)

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
    lambda_cold: float = Field(default=0.5, ge=0)
    lambda_kd: float = Field(default=1.0, ge=0)
    lambda_all: float = Field(default=1.0, ge=0)
    lambda_gate: float = Field(default=0.05, ge=0)
    lambda_edl: float = Field(default=1.0, ge=0)
    lambda_phylo: float = Field(default=0.01, ge=0)
    lambda_info_nce: float = Field(default=0.1, ge=0)
    lambda_prior: float = Field(default=0.05, ge=0)
    temporal_masking_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    gaussian_noise_std: float = Field(default=0.05, ge=0)
    gate_entropy_target: float = Field(default=0.4, ge=0)
    calibrate: bool = True
    calibrate_cold: bool = True
    use_adaptive_loss: bool = False
    use_hard_negative_mining: bool = False
    use_curriculum: bool = False
    kd_temperature: float = Field(default=1.0, gt=0.0)
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
