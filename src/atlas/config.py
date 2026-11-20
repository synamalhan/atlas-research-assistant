from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def load(path=None):
    """Read configs/atlas.toml; paths in it are relative to the repo root."""
    with open(path or ROOT / "configs" / "atlas.toml", "rb") as f:
        cfg = tomllib.load(f)
    for k in ("processed_dir", "index_dir"):
        cfg[k] = str(ROOT / cfg[k])
    return cfg
