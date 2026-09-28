"""Load the locked final PHCNet protocol."""

import json
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "final_config.json"
MODEL_ORDER = ("esm8m", "esm650m", "protbert")
RESIDUAL_KEYS = (
    "max_delta", "context_residual_hidden_dim", "context_residual_dropout",
    "physchem_hidden_dim", "physchem_dropout", "physchem_max_delta",
    "physchem_gate_temperature",
)


def resolve_model_config(config, model_name, fold):
    """Apply the published full-model settings for one backbone and fold."""
    fold = str(fold).zfill(2)
    model = copy.deepcopy(config["models"][model_name])
    try:
        locked = model["locked_folds"][fold]
    except KeyError as exc:
        raise ValueError(f"No locked configuration for {model_name}, fold {fold}.") from exc
    residual = dict(locked["residual_config"])
    if set(residual) != set(RESIDUAL_KEYS):
        raise ValueError(f"Incomplete locked residual settings for {model_name}, fold {fold}.")
    model["train_config"]["max_delta"] = residual.pop("max_delta")
    model["architecture"] = residual
    model["source_candidate"] = locked["source_candidate"]
    return model

def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))

FINAL_CONFIG = load_config()
