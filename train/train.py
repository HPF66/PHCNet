#!/usr/bin/env python3
"""Train the locked final PHCNet ensemble without research-only analysis steps."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_loader.data import (
    CATEGORY_FIELDS,
    EXPERIMENT_FIELDS,
    MODIFICATION_FIELDS,
    PeptideV2Dataset,
    build_vocabs,
    collate_batch,
    encode_rows,
    normalize_row,
    read_csv_rows,
    regression_metrics,
    set_seed,
    stable_metrics,
)


MODEL_ORDER = ("esm8m", "esm650m", "protbert")
CACHE_FORMAT = "position_stitched_residue_tokens_v1"
LOCKED_BACKBONE_TOKEN_CAPACITY = 512
LOCKED_CHUNK_RESIDUES = 480
LOCKED_CHUNK_OVERLAP = 64
PRINT_LOCK = threading.Lock()

MODEL_CONSTRUCTOR_KEYS = (
    "input_head",
    "adapter_dim",
    "adapter_dropout",
    "category_dim",
    "hidden_dim",
    "dropout",
    "condition_dropout",
    "condition_scale",
    "max_delta",
    "modification_char_dim",
    "modification_cnn_channels",
    "modification_hidden_dim",
    "modification_kernels",
    "modification_max_length",
    "modification_max_delta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the locked final PHCNet ensemble: ESM2-8M, ESM2-650M, and "
            "ProtBERT frozen features with fixed affine calibration and equal weights."
        )
    )
    parser.add_argument("--config", default="utils/final_config.json")
    parser.add_argument("--result-dir", default="result/final_three_backbone")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-ids", nargs="+", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--skip-cache-build", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--verify-feature-caches",
        action="store_true",
        help="Verify selected frozen feature caches against cache_manifest.json.",
    )
    parser.add_argument("--dry-run", action="store_true")

    # Internal subprocess mode. It is intentionally hidden from normal usage.
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODEL_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--fold", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_line(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def command_text(command: list[str]) -> str:
    return shlex.join(str(value) for value in command)


def normalize_folds(values: list[str] | None, config: dict) -> list[str]:
    source = values if values is not None else config["protocol"]["folds"]
    folds = [str(value).zfill(2) for value in source]
    if not folds or len(folds) != len(set(folds)):
        raise ValueError("Fold identifiers must be present and unique.")
    return folds


def normalize_gpu_ids(values: list[str] | None, device: str) -> list[str]:
    gpu_ids = []
    for value in values or []:
        gpu_ids.extend(str(value).replace(",", " ").split())
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpu-ids must contain unique physical GPU IDs.")
    if gpu_ids and not str(device).startswith("cuda"):
        raise ValueError("--gpu-ids requires a CUDA device.")
    return gpu_ids


def validate_config(config: dict) -> None:
    if set(config.get("models", {})) != set(MODEL_ORDER):
        raise ValueError(f"Config must define exactly {MODEL_ORDER}.")
    protocol = config["protocol"]
    if not protocol.get("same_hyperparameters_across_all_folds"):
        raise ValueError("The final protocol requires shared hyperparameters across folds.")
    if protocol.get("per_fold_hyperparameter_selection"):
        raise ValueError("Per-fold hyperparameter selection is not allowed.")
    if not math.isclose(float(protocol.get("mean_sd_loss_weight", -1)), 0.2):
        raise ValueError("The locked protocol requires lambda_aux=0.2.")
    if not math.isclose(float(protocol.get("mean_sd_noise_floor_log10", -1)), 0.1):
        raise ValueError("The locked protocol requires tau=0.1 log10 units.")
    if protocol.get("mean_sd_weight_mode") != "sd_adaptive":
        raise ValueError("The locked protocol requires sd_adaptive mean-SD weights.")
    weights = config["ensemble"]["official_weights"]
    if set(weights) != set(MODEL_ORDER):
        raise ValueError("Official weights must cover all final backbones.")
    if not math.isclose(sum(float(weights[name]) for name in MODEL_ORDER), 1.0, abs_tol=1e-9):
        raise ValueError(f"Official weights must sum to one: {weights}")

    shared = protocol["unified_adapter"]
    architecture_keys = {
        "physchem_hidden_dim",
        "physchem_dropout",
        "physchem_max_delta",
    }
    for model_name in MODEL_ORDER:
        model = config["models"][model_name]
        for key, expected in shared.items():
            section = model.get("architecture", {}) if key in architecture_keys else model["train_config"]
            if section.get(key) != expected:
                raise ValueError(
                    f"Unified architecture mismatch for {model_name}.{key}: "
                    f"expected {expected!r}, found {section.get(key)!r}."
                )
    long_sequence = protocol["long_sequence"]
    chunk_residues = int(long_sequence["chunk_residues"])
    chunk_overlap = int(long_sequence["chunk_overlap"])
    token_capacity = int(long_sequence["backbone_token_capacity"])
    expected_long_sequence = {
        "mode": "overlap_stitch_before_adapter",
        "chunk_residues": LOCKED_CHUNK_RESIDUES,
        "chunk_overlap": LOCKED_CHUNK_OVERLAP,
        "backbone_token_capacity": LOCKED_BACKBONE_TOKEN_CAPACITY,
        "overlap_merge": "position_aligned_mean",
        "pool_after_stitch": True,
        "truncate_sequences": False,
    }
    for key, expected in expected_long_sequence.items():
        if long_sequence.get(key) != expected:
            raise ValueError(
                f"The final protocol requires long_sequence.{key}={expected!r}; "
                f"found {long_sequence.get(key)!r}."
            )
    if not 0 <= chunk_overlap < chunk_residues <= token_capacity - 2:
        raise ValueError(f"Invalid long-sequence configuration: {long_sequence}")

    expected_backbones = {
        "esm8m": ("facebook/esm2_t6_8M_UR50D", "esm", 320),
        "esm650m": ("facebook/esm2_t33_650M_UR50D", "esm", 1280),
        "protbert": ("Rostlab/prot_bert", "protbert", 1024),
    }
    for model_name, (model_id, backbone_type, hidden_dim) in expected_backbones.items():
        model = config["models"][model_name]
        actual = (
            model.get("backbone_model"),
            model.get("backbone_type"),
            int(model.get("backbone_hidden_dim", -1)),
        )
        if actual != (model_id, backbone_type, hidden_dim):
            raise ValueError(
                f"Locked backbone mismatch for {model_name}: expected "
                f"{(model_id, backbone_type, hidden_dim)!r}, found {actual!r}."
            )


def validate_data_package(config: dict, folds: list[str]) -> dict:
    protocol = config["protocol"]
    dataset_path = resolve_path(protocol["dataset"])
    split_root = resolve_path(protocol["split_root"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")
    dataset_rows = read_csv(dataset_path)
    dataset_ids = []
    exact_ids = set()
    counts: dict[str, int] = {}
    for row in dataset_rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise RuntimeError("Every packaged dataset row must have an ID.")
        label = str(row.get("training_label_type", "exact")).strip().lower() or "exact"
        if label not in {"exact", "mean_sd"}:
            raise RuntimeError(f"Unsupported training label type: {label}")
        dataset_ids.append(row_id)
        counts[label] = counts.get(label, 0) + 1
        if label == "exact":
            exact_ids.add(row_id)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise RuntimeError("The packaged dataset contains duplicate IDs.")
    expected_counts = protocol["expected_dataset_counts"]
    if counts != expected_counts:
        raise RuntimeError(f"Dataset counts differ from the locked protocol: {counts}")

    dataset_id_set = set(dataset_ids)
    pooled_test_ids = []
    for fold in folds:
        split_dir = split_root / f"split_{fold}"
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing packaged split files in {split_dir}")
        train_rows = read_csv(train_path)
        test_rows = read_csv(test_path)
        train_labels = {
            str(row.get("training_label_type", "exact")).strip().lower() or "exact"
            for row in train_rows
        }
        test_labels = {
            str(row.get("training_label_type", "exact")).strip().lower() or "exact"
            for row in test_rows
        }
        if train_labels != {"exact", "mean_sd"} or test_labels != {"exact"}:
            raise RuntimeError(f"Invalid exact/mean-SD split protocol in split_{fold}")
        train_ids = [str(row.get("id", "")).strip() for row in train_rows]
        test_ids = [str(row.get("id", "")).strip() for row in test_rows]
        if not all(train_ids) or not all(test_ids):
            raise RuntimeError(f"Missing IDs in split_{fold}")
        if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
            raise RuntimeError(f"Duplicate IDs in split_{fold}")
        if set(train_ids) & set(test_ids):
            raise RuntimeError(f"Train/test ID overlap in split_{fold}")
        if set(train_ids) | set(test_ids) != dataset_id_set:
            raise RuntimeError(f"split_{fold} does not cover the packaged dataset")
        pooled_test_ids.extend(test_ids)

    locked_folds = [str(value).zfill(2) for value in protocol["folds"]]
    if set(folds) == set(locked_folds):
        if len(pooled_test_ids) != len(set(pooled_test_ids)):
            raise RuntimeError("Outer-test IDs are duplicated across the ten folds.")
        if set(pooled_test_ids) != exact_ids:
            raise RuntimeError("Outer-test folds do not cover exactly the primary labels.")
    return {
        "dataset": str(dataset_path),
        "split_root": str(split_root),
        "dataset_counts": counts,
        "pooled_outer_test_rows": len(pooled_test_ids),
    }


def cache_directory(args: argparse.Namespace) -> Path:
    return resolve_path(args.cache_dir) if args.cache_dir else ROOT / "feature"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_feature_caches(config: dict, models: list[str], cache_dir: Path) -> None:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature cache manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("cache_format") != CACHE_FORMAT:
        raise RuntimeError(f"Unexpected cache format in {manifest_path}")
    expected_manifest_protocol = {
        "backbone_max_length": LOCKED_BACKBONE_TOKEN_CAPACITY,
        "chunk_residues": LOCKED_CHUNK_RESIDUES,
        "chunk_overlap": LOCKED_CHUNK_OVERLAP,
        "truncate_sequences": False,
        "overlap_merge": "position_aligned_mean",
        "pooling_stage": "after_stitch_in_trainable_adapter",
    }
    for key, expected in expected_manifest_protocol.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Feature manifest protocol mismatch for {key}: "
                f"expected {expected!r}, found {manifest.get(key)!r}."
            )
    entries = manifest.get("caches", {})
    for model_name in models:
        cache_name = config["models"][model_name]["cache_name"]
        cache_path = cache_dir / cache_name
        entry = entries.get(model_name)
        if not cache_path.is_file() or not isinstance(entry, dict):
            raise FileNotFoundError(f"Incomplete frozen cache for {model_name}: {cache_path}")
        if entry.get("file") != cache_name:
            raise RuntimeError(f"Cache filename mismatch for {model_name}")
        if entry.get("bytes") != cache_path.stat().st_size:
            raise RuntimeError(f"Cache size mismatch for {model_name}")
        expected_hash = entry.get("sha256")
        if not expected_hash or file_sha256(cache_path) != expected_hash:
            raise RuntimeError(f"Cache SHA256 mismatch for {model_name}")
    print_line(f"Feature cache validation passed: {cache_dir}")


def ensure_feature_caches(
    args: argparse.Namespace,
    config: dict,
    models: list[str],
    split_root: Path,
    cache_dir: Path,
) -> None:
    missing = [
        name
        for name in models
        if not (cache_dir / config["models"][name]["cache_name"]).is_file()
    ]
    if args.skip_cache_build:
        if missing and not args.dry_run:
            raise FileNotFoundError(f"Missing frozen feature caches: {missing}")
        return

    long_sequence = config["protocol"]["long_sequence"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "scratch.get_feature.build_features",
        "--split-root",
        str(split_root),
        "--models",
        *models,
        "--output-dir",
        str(cache_dir),
        "--batch-size",
        str(args.cache_batch_size),
        "--backbone-max-length",
        str(long_sequence["backbone_token_capacity"]),
        "--chunk-residues",
        str(long_sequence["chunk_residues"]),
        "--chunk-overlap",
        str(long_sequence["chunk_overlap"]),
        "--device",
        args.device,
    ]
    if args.force_cache:
        command.append("--force")
    print_line(f"Building or validating frozen features: {command_text(command)}")
    if args.dry_run:
        return
    env = os.environ.copy()
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids[0]
        env["PHCNET_PHYSICAL_GPU"] = args.gpu_ids[0]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    validate_feature_caches(config, models, cache_dir)


def training_label_type(row: dict) -> str:
    label = str(row.get("training_label_type", "")).strip().lower() or "exact"
    if label not in {"exact", "mean_sd"}:
        raise ValueError(f"Unsupported training_label_type={label!r} for {row.get('id', '')!r}")
    return label


def auxiliary_training_only(row: dict) -> bool:
    value = str(row.get("auxiliary_training_only", "")).strip().lower()
    if value in {"", "0", "false", "no", "n"}:
        return False
    if value in {"1", "true", "yes", "y"}:
        return True
    raise ValueError(f"Unsupported auxiliary_training_only={value!r}")


def partition_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    exact, mean_sd = [], []
    for row in rows:
        (mean_sd if training_label_type(row) == "mean_sd" else exact).append(row)
    return exact, mean_sd


def mean_sd_sigma(row: dict) -> float:
    try:
        value = float(row["label_error_log10_delta_method"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric uncertainty for mean-SD row {row.get('id', '')!r}") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid mean-SD uncertainty for {row.get('id', '')!r}")
    return value


def load_outer_split(split_dir: Path, inner_val_fraction: float, inner_val_seed: int):
    outer_train_rows = [
        normalized
        for row in read_csv_rows(split_dir / "train.csv")
        if (normalized := normalize_row(row)) is not None
    ]
    test_rows = [
        normalized
        for row in read_csv_rows(split_dir / "test.csv")
        if (normalized := normalize_row(row)) is not None
    ]
    if not 0.0 < inner_val_fraction < 0.5:
        raise ValueError("inner_val_fraction must be between 0 and 0.5")
    outer_exact, outer_mean_sd = partition_rows(outer_train_rows)
    test_exact, test_mean_sd = partition_rows(test_rows)
    if test_mean_sd or any(auxiliary_training_only(row) for row in test_exact):
        raise RuntimeError("Outer test rows must be exact, non-auxiliary records only.")
    primary_exact = [row for row in outer_exact if not auxiliary_training_only(row)]
    auxiliary_exact = [row for row in outer_exact if auxiliary_training_only(row)]
    if not primary_exact:
        raise RuntimeError("Outer training split has no primary exact rows.")

    n_inner_folds = max(int(round(1.0 / inner_val_fraction)), 2)
    rng = random.Random(inner_val_seed)
    sorted_indices = sorted(range(len(primary_exact)), key=lambda index: float(primary_exact[index]["target"]))
    inner_folds = [[] for _ in range(n_inner_folds)]
    for start in range(0, len(sorted_indices), n_inner_folds):
        block = sorted_indices[start : start + n_inner_folds]
        rng.shuffle(block)
        order = list(range(n_inner_folds))
        rng.shuffle(order)
        for position, row_index in enumerate(block):
            inner_folds[order[position % n_inner_folds]].append(row_index)
    val_indices = set(inner_folds[inner_val_seed % n_inner_folds])
    train_primary = [row for index, row in enumerate(primary_exact) if index not in val_indices]
    val_rows = [row for index, row in enumerate(primary_exact) if index in val_indices]
    return [*train_primary, *auxiliary_exact, *outer_mean_sd], val_rows, test_exact, outer_train_rows


def make_collate_fn():
    from model.modification import build_modification_text, has_modification_evidence
    from model.phcnet import PhyschemModificationSequenceBatch, compute_physicochemical_features

    def collate(rows: list[dict]) -> dict:
        batch = collate_batch(rows)
        sequences = batch["sequences"]
        batch["sequences"] = PhyschemModificationSequenceBatch(
            sequences,
            [build_modification_text(row) for row in rows],
            [has_modification_evidence(row) for row in rows],
            [compute_physicochemical_features(sequence) for sequence in sequences],
        )
        return batch

    return collate


def make_loaders(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], batch_size: int):
    from torch.utils.data import DataLoader

    collate = make_collate_fn()
    return {
        "train": DataLoader(PeptideV2Dataset(train_rows), batch_size=batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(PeptideV2Dataset(val_rows), batch_size=batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(PeptideV2Dataset(test_rows), batch_size=batch_size, shuffle=False, collate_fn=collate),
    }


def target_stats(rows: list[dict]) -> tuple[float, float]:
    values = np.asarray([float(row["target"]) for row in rows], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, std if np.isfinite(std) and std >= 1e-6 else 1.0


def target_weights(target_raw, args: dict):
    import torch

    very_short = math.log10(float(args["very_short_threshold_minutes"]))
    short = math.log10(float(args["short_threshold_minutes"]))
    long = math.log10(float(args["long_threshold_minutes"]))
    if args["weight_mode"] == "bin":
        weights = torch.full_like(target_raw, float(args["mid_weight"]))
        weights = torch.where(target_raw < very_short, torch.full_like(weights, float(args["very_short_weight"])), weights)
        weights = torch.where(
            (target_raw >= very_short) & (target_raw < short),
            torch.full_like(weights, float(args["short_weight"])),
            weights,
        )
        return torch.where(target_raw >= long, torch.full_like(weights, float(args["long_weight"])), weights)

    mid = float(args["mid_weight"])
    weights = torch.full_like(target_raw, mid)
    short_span = max(short - very_short, 1e-6)
    short_factor = ((short - target_raw) / short_span).clamp(0.0, 1.0)
    if float(args["continuous_short_power"]) != 1.0:
        short_factor = short_factor.pow(float(args["continuous_short_power"]))
    short_weights = mid + (float(args["very_short_weight"]) - mid) * short_factor
    weights = torch.maximum(weights, short_weights)
    if float(args["long_weight"]) != mid:
        ramp = max(float(args["continuous_long_ramp_log10"]), 1e-6)
        long_factor = ((target_raw - long) / ramp).clamp(0.0, 1.0)
        if float(args["continuous_long_power"]) != 1.0:
            long_factor = long_factor.pow(float(args["continuous_long_power"]))
        long_weights = mid + (float(args["long_weight"]) - mid) * long_factor
        weights = torch.maximum(weights, long_weights) if float(args["long_weight"]) >= mid else torch.minimum(weights, long_weights)
    return weights.clamp(float(args["min_sample_weight"]), float(args["max_sample_weight"]))


def pairwise_rank_loss(prediction, target, min_gap: float, temperature: float):
    import torch
    import torch.nn.functional as functional

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    prediction_diff = prediction.unsqueeze(1) - prediction.unsqueeze(0)
    mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    mask = mask & (target_diff.abs() >= min_gap)
    if not torch.any(mask):
        return prediction.new_tensor(0.0)
    logits = torch.sign(target_diff[mask]) * prediction_diff[mask] / max(temperature, 1e-6)
    return (target_diff[mask].abs().clamp(max=1.0) * functional.softplus(-logits)).mean()


def weighted_loss(prediction, target, weights, loss_name: str):
    import torch.nn.functional as functional

    per_row = functional.mse_loss(prediction, target, reduction="none") if loss_name == "mse" else functional.smooth_l1_loss(prediction, target, reduction="none")
    return (per_row * weights).sum() / weights.sum().clamp_min(1e-8)


def calibration_loss(prediction, target, weights, bias_weight: float, scale_weight: float):
    if bias_weight <= 0 and scale_weight <= 0:
        return prediction.new_tensor(0.0)
    detached = weights.detach()
    denominator = detached.sum().clamp_min(1e-6)
    pred_mean = (prediction * detached).sum() / denominator
    target_mean = (target.detach() * detached).sum() / denominator
    pred_std = ((((prediction - pred_mean) ** 2) * detached).sum() / denominator).clamp_min(1e-6).sqrt()
    target_std = ((((target.detach() - target_mean) ** 2) * detached).sum() / denominator).clamp_min(1e-6).sqrt()
    loss = prediction.new_tensor(0.0)
    if bias_weight > 0:
        loss = loss + float(bias_weight) * (pred_mean - target_mean).pow(2)
    if scale_weight > 0:
        loss = loss + float(scale_weight) * (pred_std - target_std).pow(2)
    return loss


def train_epoch(model, loader, optimizer, train_config: dict, protocol: dict, device, mean: float, std: float) -> float:
    import torch

    model.train()
    model.esm.eval()
    losses = []
    if protocol.get("mean_sd_weight_mode") != "sd_adaptive":
        raise ValueError("The locked trainer requires sd_adaptive mean-SD weights.")
    noise_floor_sq = float(protocol["mean_sd_noise_floor_log10"]) ** 2
    for batch in loader:
        category_ids = batch["category_ids"].to(device)
        target_raw = batch["target"].to(device)
        target_norm = (target_raw - mean) / std
        weights = target_weights(target_raw, train_config)
        reliability = []
        exact_mask = []
        for row in batch["rows"]:
            is_exact = training_label_type(row) == "exact"
            exact_mask.append(is_exact)
            if is_exact:
                reliability.append(1.0)
            else:
                sigma_sq = mean_sd_sigma(row) ** 2
                reliability.append(float(protocol["mean_sd_loss_weight"]) * noise_floor_sq / (noise_floor_sq + sigma_sq))
        reliability_tensor = torch.tensor(reliability, dtype=weights.dtype, device=device)
        exact_mask_tensor = torch.tensor(exact_mask, dtype=torch.bool, device=device)
        supervised_weights = weights * reliability_tensor
        outputs = model(batch["sequences"], category_ids)
        loss = weighted_loss(outputs["mu"], target_norm, supervised_weights, train_config["loss"])
        if float(train_config["base_loss_weight"]) > 0:
            loss = loss + float(train_config["base_loss_weight"]) * weighted_loss(
                outputs["base_mu"], target_norm, supervised_weights, train_config["loss"]
            )
        if float(train_config["residual_penalty"]) > 0:
            loss = loss + float(train_config["residual_penalty"]) * outputs["residual_delta"].pow(2).mean()
        if float(train_config["pairwise_weight"]) > 0 and int(exact_mask_tensor.sum()) >= 2:
            loss = loss + float(train_config["pairwise_weight"]) * pairwise_rank_loss(
                outputs["mu"][exact_mask_tensor],
                target_raw[exact_mask_tensor],
                float(train_config["pairwise_min_gap"]),
                float(train_config["pairwise_temperature"]),
            )
        if torch.any(exact_mask_tensor):
            loss = loss + calibration_loss(
                outputs["mu"][exact_mask_tensor],
                target_norm[exact_mask_tensor],
                weights[exact_mask_tensor],
                float(train_config["calibration_bias_weight"]),
                float(train_config["calibration_scale_weight"]),
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        if float(train_config["grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def predict_rows(model, loader, device, mean: float, std: float):
    import torch

    model.eval()
    rows, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            category_ids = batch["category_ids"].to(device)
            outputs = model(batch["sequences"], category_ids)
            prediction = outputs["mu"].detach().cpu().numpy() * std + mean
            target = batch["target"].detach().cpu().numpy()
            for index, source in enumerate(batch["rows"]):
                pred_value = float(prediction[index])
                true_value = float(target[index])
                rows.append(
                    {
                        "id": source["id"],
                        "sequence_modified": source.get("sequence_modified", ""),
                        "sequence_esm": source.get("sequence_esm", ""),
                        "true_log10_half_life": true_value,
                        "pred_log10_half_life": pred_value,
                        "true_half_life_minutes": source.get("half_life_minutes", ""),
                        "pred_half_life_minutes": float(10**pred_value),
                    }
                )
                y_true.append(true_value)
                y_pred.append(pred_value)
    return rows, np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    stable = stable_metrics(y_true, y_pred, threshold_minutes=60.0)
    return {
        **regression_metrics(y_true, y_pred),
        "bias_log10": float(np.mean(y_pred - y_true)),
        "accuracy": stable["accuracy"],
        "precision": stable["precision"],
        "recall": stable["recall"],
        "f1": stable["f1"],
        "auc": stable["auc"],
    }


def trainable_state_dict(model) -> dict:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


def train_one_run(config: dict, model_name: str, fold: str, seed: int, result_dir: Path, cache_dir: Path, device_name: str) -> None:
    import torch
    from model import PHCNetUnified

    protocol = config["protocol"]
    model_config = config["models"][model_name]
    train_config = dict(model_config["train_config"])
    architecture = dict(model_config["architecture"])
    split_dir = resolve_path(protocol["split_root"]) / f"split_{fold}"
    outdir = result_dir / "models" / model_name / f"seed_{seed}" / f"split_{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    train_rows, val_rows, test_rows, outer_train_rows = load_outer_split(
        split_dir,
        float(protocol["inner_val_fraction"]),
        int(protocol["inner_val_seed_base"]) + int(fold),
    )
    train_exact, train_mean_sd = partition_rows(train_rows)
    primary_train_exact = [row for row in train_exact if not auxiliary_training_only(row)]
    if not primary_train_exact or not val_rows or not test_rows:
        raise RuntimeError(f"Invalid train/validation/test rows for split_{fold}")
    for row in train_mean_sd:
        mean_sd_sigma(row)
    vocabs = build_vocabs(train_rows)
    for rows in (train_rows, val_rows, test_rows):
        encode_rows(rows, vocabs)
    target_mean, target_std = target_stats(primary_train_exact)
    loaders = make_loaders(train_rows, val_rows, test_rows, int(train_config["batch_size"]))
    cache_path = cache_dir / model_config["cache_name"]
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing feature cache: {cache_path}")
    category_vocab_sizes = [len(vocabs[field]) for field in CATEGORY_FIELDS]
    model_kwargs = {
        key: train_config[key]
        for key in MODEL_CONSTRUCTOR_KEYS
        if key in train_config
    }
    model_kwargs.update(architecture)
    model = PHCNetUnified(
        backbone_model_name=model_config["backbone_model"],
        backbone_type=model_config["backbone_type"],
        category_vocab_sizes=category_vocab_sizes,
        num_experiment_fields=len(EXPERIMENT_FIELDS),
        num_modification_fields=len(MODIFICATION_FIELDS),
        backbone_max_length=int(protocol["long_sequence"]["backbone_token_capacity"]),
        backbone_dtype="float16",
        freeze_backbone=True,
        load_backbone=False,
        backbone_hidden_dim=int(model_config["backbone_hidden_dim"]),
        **model_kwargs,
    ).to(device_name)
    all_sequences = [row["sequence_esm"] for rows in (train_rows, val_rows, test_rows) for row in rows]
    model.precompute_sequences(
        all_sequences,
        batch_size=int(model_config["precompute_batch_size"]),
        device=torch.device(device_name),
        offload_backbone=True,
        cache_path=cache_path,
        expected_cache_format=CACHE_FORMAT,
        expected_chunk_residues=int(protocol["long_sequence"]["chunk_residues"]),
        expected_chunk_overlap=int(protocol["long_sequence"]["chunk_overlap"]),
        expected_synthetic_boundary_tokens=True,
        expected_hidden_dim=int(model_config["backbone_hidden_dim"]),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(train_config["lr"]), weight_decay=float(train_config["weight_decay"]))
    epochs = int(model_config["epochs"])
    patience = int(model_config["patience"])
    best_path = outdir / "best_model.pt"
    history, best_val, best_epoch = [], float("inf"), 0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, train_config, protocol, torch.device(device_name), target_mean, target_std)
        _, val_true, val_pred = predict_rows(model, loaders["val"], torch.device(device_name), target_mean, target_std)
        val_metrics = regression_metrics(val_true, val_pred)
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{key}": value for key, value in val_metrics.items()}})
        print(f"{model_name} seed={seed} fold={fold} epoch={epoch:03d} train_loss={loss:.4f} val_rmse={val_metrics['rmse_log10']:.4f}", flush=True)
        if val_metrics["rmse_log10"] < best_val:
            best_val, best_epoch = val_metrics["rmse_log10"], epoch
            torch.save(
                {
                    "model_state_dict": trainable_state_dict(model),
                    "backbone_model": model_config["backbone_model"],
                    "backbone_type": model_config["backbone_type"],
                    "vocabs": vocabs,
                    "target_mean_log10": target_mean,
                    "target_std_log10": target_std,
                    "best_epoch": best_epoch,
                    "best_val_rmse_log10": best_val,
                },
                best_path,
            )
        if epoch - best_epoch >= patience:
            break

    checkpoint = torch.load(best_path, map_location=torch.device(device_name), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    val_prediction_rows, val_true, val_pred = predict_rows(model, loaders["val"], torch.device(device_name), target_mean, target_std)
    test_prediction_rows, test_true, test_pred = predict_rows(model, loaders["test"], torch.device(device_name), target_mean, target_std)
    write_csv(outdir / "history.csv", history)
    write_csv(outdir / "val_predictions.csv", val_prediction_rows)
    write_csv(outdir / "test_predictions.csv", test_prediction_rows)
    write_json(
        outdir / "metrics.json",
        {
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_rmse_log10": best_val,
            "outer_test_evaluated": True,
            "val": metric_record(val_true, val_pred),
            "test": metric_record(test_true, test_pred),
            "data_protocol": {
                "n_train_primary_exact": len(primary_train_exact),
                "n_train_mean_sd": len(train_mean_sd),
                "n_inner_val_exact": len(val_rows),
                "n_outer_test_exact": len(test_rows),
                "mean_sd_loss_weight": protocol["mean_sd_loss_weight"],
                "mean_sd_noise_floor_log10": protocol["mean_sd_noise_floor_log10"],
                "mean_sd_weight_mode": protocol["mean_sd_weight_mode"],
            },
        },
    )
    write_json(
        outdir / "run_config.json",
        {
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "split_dir": str(split_dir),
            "cache_path": str(cache_path),
            "train_config": train_config,
            "architecture": architecture,
            "target_normalization": {"mean_log10": target_mean, "std_log10": target_std},
            "outer_train_rows": len(outer_train_rows),
        },
    )


def model_output_dir(result_dir: Path, model_name: str, seed: int, fold: str) -> Path:
    return result_dir / "models" / model_name / f"seed_{seed}" / f"split_{fold}"


def model_complete(path: Path) -> bool:
    return all((path / filename).is_file() for filename in ("metrics.json", "val_predictions.csv", "test_predictions.csv"))


def single_run_command(args: argparse.Namespace, config_path: Path, result_dir: Path, cache_dir: Path, model: str, fold: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--single-run",
        "--config",
        str(config_path),
        "--result-dir",
        str(result_dir),
        "--cache-dir",
        str(cache_dir),
        "--model",
        model,
        "--fold",
        fold,
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    return command


def run_training_jobs(args: argparse.Namespace, config_path: Path, config: dict, models: list[str], folds: list[str], seeds: list[int], result_dir: Path, cache_dir: Path) -> None:
    jobs = []
    for model in models:
        for seed in seeds:
            for fold in folds:
                outdir = model_output_dir(result_dir, model, seed, fold)
                if not args.force and model_complete(outdir):
                    continue
                jobs.append((model, fold, seed))
    if not jobs:
        print_line("All requested final-model runs are complete.")
        return

    def execute(job: tuple[str, str, int], gpu_id: str | None) -> None:
        model, fold, seed = job
        command = single_run_command(args, config_path, result_dir, cache_dir, model, fold, seed)
        prefix = f" gpu={gpu_id}" if gpu_id is not None else ""
        print_line(f"{model} seed={seed} split_{fold}: start{prefix}")
        print_line(f"Running: {command_text(command)}")
        if args.dry_run:
            return
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PHCNET_PHYSICAL_GPU"] = gpu_id
        log_path = result_dir / "logs" / model / f"seed_{seed}_split_{fold}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Training failed; see {log_path}")
        print_line(f"{model} seed={seed} split_{fold}: finished{prefix}")

    gpu_ids = args.gpu_ids
    if len(gpu_ids) <= 1:
        for job in jobs:
            execute(job, gpu_ids[0] if gpu_ids else None)
        return
    active_gpus = gpu_ids[: min(len(gpu_ids), len(jobs))]
    assignments = [jobs[index::len(active_gpus)] for index in range(len(active_gpus))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_gpus)) as executor:
        futures = [executor.submit(lambda gpu, assigned: [execute(job, gpu) for job in assigned], gpu, assigned) for gpu, assigned in zip(active_gpus, assignments)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([y_pred, np.ones_like(y_pred)])
    try:
        scale, bias = np.linalg.lstsq(design, y_true, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0, float(np.mean(y_true - y_pred))
    return float(scale), float(bias)


def calibrated_backbone_predictions(result_dir: Path, model_name: str, folds: list[str], seeds: list[int]) -> Path:
    output_dir = result_dir / "backbones" / model_name
    prediction_dir = output_dir / "predictions"
    calibration_rows = []
    for fold in folds:
        per_seed_rows = []
        for seed in seeds:
            run_dir = model_output_dir(result_dir, model_name, seed, fold)
            val_rows = read_csv(run_dir / "val_predictions.csv")
            test_rows = read_csv(run_dir / "test_predictions.csv")
            y_val = np.asarray([float(row["true_log10_half_life"]) for row in val_rows], dtype=float)
            pred_val = np.asarray([float(row["pred_log10_half_life"]) for row in val_rows], dtype=float)
            scale, bias = fit_affine(y_val, pred_val)
            calibrated = np.asarray([float(row["pred_log10_half_life"]) for row in test_rows], dtype=float) * scale + bias
            per_seed_rows.append((test_rows, calibrated))
            calibration_rows.append({"model": model_name, "fold": fold, "seed": seed, "scale": scale, "bias": bias})

        reference_rows, _ = per_seed_rows[0]
        reference_ids = [row["id"] for row in reference_rows]
        reference_true = np.asarray([float(row["true_log10_half_life"]) for row in reference_rows], dtype=float)
        seed_predictions = []
        for rows, prediction in per_seed_rows:
            if [row["id"] for row in rows] != reference_ids:
                raise RuntimeError(f"Seed prediction IDs are misaligned: {model_name} split_{fold}")
            current_true = np.asarray([float(row["true_log10_half_life"]) for row in rows], dtype=float)
            if not np.allclose(current_true, reference_true, atol=1e-8, rtol=0.0):
                raise RuntimeError(f"Seed targets are misaligned: {model_name} split_{fold}")
            seed_predictions.append(prediction)
        mean_prediction = np.mean(seed_predictions, axis=0)
        rows = []
        for index, source in enumerate(reference_rows):
            row = dict(source)
            row["pred_log10_half_life"] = float(mean_prediction[index])
            row["pred_half_life_minutes"] = float(10**mean_prediction[index])
            row["ensemble_method"] = "affine_calibrated_seed_mean"
            for seed, prediction in zip(seeds, seed_predictions):
                row[f"pred_seed_{seed}_log10"] = float(prediction[index])
            rows.append(row)
        write_csv(prediction_dir / f"split_{fold}_test_predictions.csv", rows)
    write_json(output_dir / "affine_calibration.json", {"fit_on": "inner validation predictions only", "parameters": calibration_rows})
    return prediction_dir


def index_predictions(path: Path) -> tuple[dict[str, dict], list[str]]:
    table, order = {}, []
    for row in read_csv(path):
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in table:
            raise RuntimeError(f"Invalid prediction IDs in {path}")
        table[row_id] = row
        order.append(row_id)
    return table, order


def final_ensemble(result_dir: Path, config: dict, folds: list[str]) -> Path:
    weights = {name: float(config["ensemble"]["official_weights"][name]) for name in MODEL_ORDER}
    final_dir = result_dir / "ensemble"
    pooled_rows, pooled_true, pooled_predictions = [], [], {name: [] for name in MODEL_ORDER}
    pooled_final, fold_metrics = [], []
    for fold in folds:
        tables, reference_order, reference_ids = {}, None, None
        for model_name in MODEL_ORDER:
            table, order = index_predictions(result_dir / "backbones" / model_name / "predictions" / f"split_{fold}_test_predictions.csv")
            if reference_ids is None:
                reference_ids, reference_order = set(table), order
            elif set(table) != reference_ids:
                raise RuntimeError(f"Backbone ID mismatch in split_{fold}: {model_name}")
            tables[model_name] = table
        rows, y_true, model_values = [], [], {name: [] for name in MODEL_ORDER}
        for row_id in reference_order or []:
            base = dict(tables[MODEL_ORDER[0]][row_id])
            true_value = float(base["true_log10_half_life"])
            values = {}
            for model_name in MODEL_ORDER:
                row = tables[model_name][row_id]
                if not np.isclose(float(row["true_log10_half_life"]), true_value, atol=1e-8, rtol=0.0):
                    raise RuntimeError(f"Target mismatch for ID={row_id}")
                values[model_name] = float(row["pred_log10_half_life"])
                model_values[model_name].append(values[model_name])
            prediction = sum(weights[name] * values[name] for name in MODEL_ORDER)
            base["pred_log10_half_life"] = prediction
            base["pred_half_life_minutes"] = float(10**prediction)
            base["ensemble_method"] = "fixed_equal_weight_calibrated_log10"
            base["outer_split"] = f"split_{fold}"
            for model_name in MODEL_ORDER:
                base[f"{model_name}_pred_log10_half_life"] = values[model_name]
            rows.append(base)
            y_true.append(true_value)
        true_array = np.asarray(y_true, dtype=float)
        final_array = np.asarray([float(row["pred_log10_half_life"]) for row in rows], dtype=float)
        for model_name in MODEL_ORDER:
            values = np.asarray(model_values[model_name], dtype=float)
            fold_metrics.append({"method": model_name, "split": f"split_{fold}", "n": len(true_array), **metric_record(true_array, values)})
            pooled_predictions[model_name].append(values)
        fold_metrics.append({"method": "final_ensemble", "split": f"split_{fold}", "n": len(true_array), **metric_record(true_array, final_array)})
        write_csv(final_dir / "predictions" / f"split_{fold}_test_predictions.csv", rows)
        pooled_rows.extend(rows)
        pooled_true.append(true_array)
        pooled_final.append(final_array)
    ids = [row["id"] for row in pooled_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Pooled final outer-test IDs are not unique")
    true_all = np.concatenate(pooled_true)
    pooled_metrics = []
    for model_name in MODEL_ORDER:
        pooled_metrics.append({"method": model_name, "n": len(true_all), **metric_record(true_all, np.concatenate(pooled_predictions[model_name]))})
    final_values = np.concatenate(pooled_final)
    pooled_metrics.append({"method": "final_ensemble", "n": len(true_all), **metric_record(true_all, final_values)})
    write_csv(final_dir / "fold_metrics.csv", fold_metrics)
    write_csv(final_dir / "pooled_metrics.csv", pooled_metrics)
    write_csv(final_dir / "all_test_predictions.csv", pooled_rows)
    write_json(
        final_dir / "protocol.json",
        {
            "backbones": list(MODEL_ORDER),
            "weights": weights,
            "weight_space": "affine-calibrated log10 half-life",
            "seed_ensemble": "equal mean of three independently calibrated seeds",
            "outer_test_used_to_select_weights": False,
        },
    )
    return final_dir


def run_single(args: argparse.Namespace, config: dict) -> None:
    if not args.model or not args.fold or args.seed is None:
        raise ValueError("--single-run requires --model, --fold, and --seed")
    result_dir = resolve_path(args.result_dir)
    cache_dir = cache_directory(args)
    train_one_run(config, args.model, str(args.fold).zfill(2), int(args.seed), result_dir, cache_dir, args.device)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = read_json(config_path)
    validate_config(config)
    if args.single_run:
        run_single(args, config)
        return

    folds = normalize_folds(args.folds, config)
    seeds = [int(value) for value in (args.seeds or config["protocol"]["seeds"])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be present and unique.")
    models = list(dict.fromkeys(args.models or list(MODEL_ORDER)))
    args.gpu_ids = normalize_gpu_ids(args.gpu_ids, args.device)
    validation = validate_data_package(config, folds)
    print_line(f"Final PHCNet models={models} folds={folds} seeds={seeds}")
    print_line(f"Data exact={validation['dataset_counts']['exact']} mean_sd={validation['dataset_counts']['mean_sd']} pooled_outer_test={validation['pooled_outer_test_rows']}")
    if args.validate_only:
        if args.verify_feature_caches:
            validate_feature_caches(config, models, cache_directory(args))
        print_line("Final package validation passed.")
        return

    result_dir = resolve_path(args.result_dir)
    cache_dir = cache_directory(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        result_dir / "run_manifest.json",
        {
            "config": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "models": models,
            "folds": folds,
            "seeds": seeds,
            "gpu_ids": args.gpu_ids,
            "cache_dir": str(cache_dir),
            "same_hyperparameters_across_all_folds": True,
            "fixed_equal_weights": config["ensemble"]["official_weights"],
        },
    )
    ensure_feature_caches(args, config, models, Path(validation["split_root"]), cache_dir)
    run_training_jobs(args, config_path, config, models, folds, seeds, result_dir, cache_dir)
    if args.dry_run:
        print_line("Dry run complete.")
        return
    for model_name in models:
        calibrated_backbone_predictions(result_dir, model_name, folds, seeds)
    if set(models) != set(MODEL_ORDER):
        print_line("Final ensemble skipped because all three locked backbones were not requested.")
        return
    final_dir = final_ensemble(result_dir, config, folds)
    final_metrics = read_csv(final_dir / "pooled_metrics.csv")
    best = next(row for row in final_metrics if row["method"] == "final_ensemble")
    print_line(f"Final ensemble complete: RMSE={float(best['rmse_log10']):.4f} MAE={float(best['mae_log10']):.4f} R2={float(best['r2']):.4f}")


if __name__ == "__main__":
    main()
