import yaml
from pathlib import Path
from bio_spread_reborn.config.schema import Config

def load_config(config_path: str = "config/default.yaml") -> Config:
    """
    Load and validate YAML configuration file using Pydantic.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Pydantic validation
    return Config(**data)
