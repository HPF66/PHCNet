"""Load the locked final PHCNet protocol."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "final_config.json"
MODEL_ORDER = ("esm8m", "esm650m", "protbert")

def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))

FINAL_CONFIG = load_config()
