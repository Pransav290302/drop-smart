"""ML module configuration"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_file: str = "model_config.yaml") -> Dict[str, Any]:
    """Load model configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return load_config("model_config.yaml")


def get_schema_config() -> Dict[str, Any]:
    """Get schema configuration"""
    return load_config("schema_config.yaml")

