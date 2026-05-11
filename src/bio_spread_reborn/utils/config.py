import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
