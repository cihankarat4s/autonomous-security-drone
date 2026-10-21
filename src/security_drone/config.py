from pathlib import Path

import yaml

# src/security_drone/config.py -> repo root is two levels above the package.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path=DEFAULT_CONFIG_PATH):
    """Read the YAML config into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
