from pathlib import Path
import random

import numpy as np
import torch
import yaml

from bio_spread_reborn.config.schema import Config


def load_config(config_path: str = "config/default.yaml") -> Config:
    """
    Load and validate YAML configuration file using Pydantic.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Pydantic validation
    return Config(**data)


def set_seed(seed: int) -> None:
    """Set all random seeds deterministically.

    Seeds numpy, torch (CPU + CUDA + MPS), and Python random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
