#!/usr/bin/env python3
"""Build position-stitched frozen token caches for the unified PHCNet ensemble."""

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path

import torch

from model.backbone import (
    PHCNetV2LargeBackboneBoundedResidual,
)
from my_loader.data import (
    CATEGORY_FIELDS,
    EXPERIMENT_FIELDS,
    MODIFICATION_FIELDS,
    normalize_row,
    read_csv_rows,
)


CACHE_FORMAT = "position_stitched_residue_tokens_v1"
LOCKED_BACKBONE_TOKEN_CAPACITY = 512
LOCKED_CHUNK_RESIDUES = 480
LOCKED_CHUNK_OVERLAP = 64
MODEL_SPECS = {
    "esm8m": {
        "backbone_model": "facebook/esm2_t6_8M_UR50D",
        "backbone_type": "esm",
        "cache_name": (
            "esm2_t6_8m_exact_plus_mean_sd_chunk480_overlap64_fp16.pt"
        ),
    },
    "esm650m": {
        "backbone_model": "facebook/esm2_t33_650M_UR50D",
        "backbone_type": "esm",
        "cache_name": (
            "esm2_t33_650m_exact_plus_mean_sd_chunk480_overlap64_fp16.pt"
        ),
    },
    "protbert": {
        "backbone_model": "Rostlab/prot_bert",
        "backbone_type": "protbert",
        "cache_name": "protbert_exact_plus_mean_sd_chunk480_overlap64_fp16.pt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen PLM token caches without discarding residues beyond the "
            "single-window context limit."
        )
    )
    parser.add_argument(
        "--split-root",
        default="data/splits_10fold_exact_plus_mean_sd_train",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--backbone-max-length",
        type=int,
        default=LOCKED_BACKBONE_TOKEN_CAPACITY,
        help="Tokenizer capacity including special tokens; this is not a residue cutoff.",
    )
    parser.add_argument(
        "--chunk-residues",
        type=int,
        default=LOCKED_CHUNK_RESIDUES,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=LOCKED_CHUNK_OVERLAP,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate rows and report chunk statistics without loading a PLM.",
    )
    return parser.parse_args()


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def training_label_type(row):
    value = str(row.get("training_label_type", "exact")).strip().lower()
    return value or "exact"


def collect_rows(split_root):
    rows_by_id = {}
    sequence_order = {}
    for fold in range(1, 11):
        split_dir = split_root / f"split_{fold:02d}"
        for filename in ("train.csv", "test.csv"):
            path = split_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing split file: {path}")
            for raw_row in read_csv_rows(path):
                row = normalize_row(raw_row)
                if row is None:
                    continue
                row_id = str(row.get("id", ""))
                row_key = row_id or f"{row['sequence_esm']}::{row['target']}"
                rows_by_id.setdefault(row_key, row)
                sequence_order.setdefault(str(row["sequence_esm"]), None)

    rows = list(rows_by_id.values())
    invalid_labels = sorted(
        {training_label_type(row) for row in rows} - {"exact", "mean_sd"}
    )
    if invalid_labels:
        raise ValueError(f"Unsupported training label types: {invalid_labels}")
    return rows, list(sequence_order)


def normalize_sequence(sequence, backbone_type):
    normalized = re.sub(r"[^A-Za-z]", "", str(sequence)).upper()
    if backbone_type == "protbert":
        normalized = re.sub(r"[UZOB]", "X", normalized)
    return normalized


def chunk_spans(length, chunk_residues, chunk_overlap):
    if length <= 0:
        raise ValueError("Cannot build a PLM cache for an empty sequence.")
    stride = chunk_residues - chunk_overlap
    spans = []
    for start in range(0, length, stride):
        end = min(start + chunk_residues, length)
        spans.append((start, end))
        if end == length:
            break
    return spans


def cache_matches(
    path,
    spec,
    max_length,
    chunk_residues,
    chunk_overlap,
    sequences,
):
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"Ignoring unreadable cache {path}: {exc}", flush=True)
        return False
    tokens = payload.get("tokens", {})
    return (
        payload.get("cache_format") == CACHE_FORMAT
        and payload.get("backbone_model_name") == spec["backbone_model"]
        and payload.get("backbone_type") == spec["backbone_type"]
        and payload.get("backbone_max_length") == max_length
        and payload.get("chunk_residues") == chunk_residues
        and payload.get("chunk_overlap") == chunk_overlap
        and payload.get("synthetic_boundary_tokens") is True
        and all(sequence in tokens for sequence in sequences)
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_chunk(chunk, backbone_type):
    if backbone_type == "protbert":
        return " ".join(chunk)
    return chunk


@torch.inference_mode()
def build_stitched_cache(
    model,
    sequences,
    *,
    backbone_type,
    batch_size,
    backbone_max_length,
    chunk_residues,
    chunk_overlap,
    device,
    cache_dtype=torch.float16,
):
    tokenizer = model.tokenizer
    backbone = model.esm
    if tokenizer is None:
        raise RuntimeError("Tokenizer was not loaded for cache construction.")

    special_token_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    maximum_residues = backbone_max_length - special_token_count
    if chunk_residues > maximum_residues:
        raise ValueError(
            f"chunk_residues={chunk_residues} exceeds the tokenizer-safe limit "
            f"{maximum_residues} for max_length={backbone_max_length}."
        )

    backbone.to(device)
    backbone.eval()
    cache = {}
    total_chunks = 0
    long_sequences = 0
    max_sequence_length = 0

    for sequence_start in range(0, len(sequences), batch_size):
        raw_batch = sequences[sequence_start : sequence_start + batch_size]
        records = []
        accumulators = {}
        for cache_key in raw_batch:
            sequence = normalize_sequence(cache_key, backbone_type)
            max_sequence_length = max(max_sequence_length, len(sequence))
            spans = chunk_spans(
                len(sequence),
                chunk_residues,
                chunk_overlap,
            )
            if len(spans) > 1:
                long_sequences += 1
            accumulators[cache_key] = {
                "sequence": sequence,
                "sum": None,
                "count": torch.zeros(len(sequence), 1, dtype=torch.float32),
            }
            for start, end in spans:
                records.append(
                    {
                        "cache_key": cache_key,
                        "start": start,
                        "end": end,
                        "text": format_chunk(sequence[start:end], backbone_type),
                    }
                )
            total_chunks += len(spans)

        for chunk_start in range(0, len(records), batch_size):
            record_batch = records[chunk_start : chunk_start + batch_size]
            encoded = tokenizer(
                [record["text"] for record in record_batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
            )
            special_tokens_mask = encoded.pop("special_tokens_mask").bool()
            if encoded["input_ids"].size(1) > backbone_max_length:
                raise RuntimeError(
                    "A chunk exceeded backbone_max_length after tokenization: "
                    f"tokens={encoded['input_ids'].size(1)} "
                    f"limit={backbone_max_length}."
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = backbone(**encoded).last_hidden_state
            attention_mask = encoded["attention_mask"].bool().cpu()
            special_tokens_mask = special_tokens_mask.cpu()

            for row_index, record in enumerate(record_batch):
                residue_mask = attention_mask[row_index] & ~special_tokens_mask[row_index]
                residue_count = int(residue_mask.sum().item())
                expected = int(record["end"] - record["start"])
                if residue_count != expected:
                    raise RuntimeError(
                        "Tokenizer-to-residue alignment failed for "
                        f"{record['cache_key']}: expected={expected}, "
                        f"found={residue_count}."
                    )
                residue_embeddings = (
                    outputs[row_index, residue_mask.to(outputs.device)]
                    .detach()
                    .float()
                    .cpu()
                )
                accumulator = accumulators[record["cache_key"]]
                if accumulator["sum"] is None:
                    accumulator["sum"] = torch.zeros(
                        len(accumulator["sequence"]),
                        residue_embeddings.size(1),
                        dtype=torch.float32,
                    )
                start = int(record["start"])
                end = int(record["end"])
                accumulator["sum"][start:end] += residue_embeddings
                accumulator["count"][start:end] += 1.0

        for cache_key, accumulator in accumulators.items():
            if accumulator["sum"] is None:
                raise RuntimeError(f"No token embeddings were produced for {cache_key}.")
            if torch.any(accumulator["count"] == 0):
                raise RuntimeError(f"Chunk stitching left uncovered residues: {cache_key}")
            stitched = accumulator["sum"] / accumulator["count"]
            boundary = torch.zeros(1, stitched.size(1), dtype=torch.float32)
            cache[cache_key] = torch.cat(
                [boundary, stitched, boundary], dim=0
            ).to(dtype=cache_dtype)

        processed = min(sequence_start + batch_size, len(sequences))
        if processed == len(sequences) or processed % 80 == 0:
            print(
                f"Cached {processed}/{len(sequences)} sequences; "
                f"chunks_so_far={total_chunks}",
                flush=True,
            )

    return cache, {
        "special_tokens_per_chunk": special_token_count,
        "total_chunks": total_chunks,
        "long_sequences": long_sequences,
        "max_sequence_length": max_sequence_length,
    }


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.backbone_max_length <= 2:
        raise ValueError("--backbone-max-length must be greater than two.")
    if args.chunk_residues <= 0:
        raise ValueError("--chunk-residues must be positive.")
    if not 0 <= args.chunk_overlap < args.chunk_residues:
        raise ValueError("--chunk-overlap must be in [0, chunk_residues).")
    locked_values = (
        LOCKED_BACKBONE_TOKEN_CAPACITY,
        LOCKED_CHUNK_RESIDUES,
        LOCKED_CHUNK_OVERLAP,
    )
    requested_values = (
        args.backbone_max_length,
        args.chunk_residues,
        args.chunk_overlap,
    )
    if requested_values != locked_values:
        raise ValueError(
            "The released PHCNet protocol is locked to 480-residue chunks, "
            "64-residue overlap, and a 512-token backbone capacity. "
            f"Requested {requested_values}."
        )

    root = Path(__file__).resolve().parents[2]
    split_root = resolve(root, args.split_root)
    if args.output_dir:
        output_dir = resolve(root, args.output_dir)
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / "huggingface_cache"))
        output_dir = hf_home.expanduser() / "phcnet_token_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, sequences = collect_rows(split_root)
    exact_n = sum(training_label_type(row) == "exact" for row in rows)
    mean_sd_n = sum(training_label_type(row) == "mean_sd" for row in rows)
    if not exact_n or not mean_sd_n:
        raise RuntimeError(
            f"Expected both exact and mean_sd rows; found exact={exact_n}, "
            f"mean_sd={mean_sd_n}."
        )

    clean_lengths = [len(normalize_sequence(sequence, "esm")) for sequence in sequences]
    chunked_n = sum(length > args.chunk_residues for length in clean_lengths)
    max_length = max(clean_lengths)
    print(
        f"Dataset rows={len(rows)} exact={exact_n} mean_sd={mean_sd_n} "
        f"unique_sequences={len(sequences)} chunked_sequences={chunked_n} "
        f"max_residues={max_length}",
        flush=True,
    )
    if args.validate_only:
        print("Chunked cache protocol validation passed.", flush=True)
        return

    manifest_path = output_dir / "cache_manifest.json"
    existing_caches = {}
    if manifest_path.is_file():
        try:
            existing_caches = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("caches", {})
        except (OSError, json.JSONDecodeError):
            existing_caches = {}

    manifest = {
        "cache_format": CACHE_FORMAT,
        "records": len(rows),
        "exact_records": exact_n,
        "mean_sd_records": mean_sd_n,
        "unique_sequences": len(sequences),
        "chunked_sequences": chunked_n,
        "max_sequence_length": max_length,
        "backbone_max_length": args.backbone_max_length,
        "chunk_residues": args.chunk_residues,
        "chunk_overlap": args.chunk_overlap,
        "truncate_sequences": False,
        "overlap_merge": "position_aligned_mean",
        "pooling_stage": "after_stitch_in_trainable_adapter",
        "caches": existing_caches,
    }

    device = torch.device(args.device)
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        cache_path = output_dir / spec["cache_name"]
        complete = cache_matches(
            cache_path,
            spec,
            args.backbone_max_length,
            args.chunk_residues,
            args.chunk_overlap,
            sequences,
        )
        diagnostics = None
        if complete and not args.force:
            print(f"Complete chunked cache already exists: {cache_path}", flush=True)
        else:
            print(f"Building {model_key} chunked cache at {cache_path}", flush=True)
            model = PHCNetV2LargeBackboneBoundedResidual(
                backbone_model_name=spec["backbone_model"],
                backbone_type=spec["backbone_type"],
                category_vocab_sizes=[2] * len(CATEGORY_FIELDS),
                num_experiment_fields=len(EXPERIMENT_FIELDS),
                num_modification_fields=len(MODIFICATION_FIELDS),
                backbone_max_length=args.backbone_max_length,
                backbone_dtype=spec.get("backbone_dtype", "float16"),
                freeze_backbone=True,
                load_backbone=True,
            )
            tokens, diagnostics = build_stitched_cache(
                model,
                sequences,
                backbone_type=spec["backbone_type"],
                batch_size=args.batch_size,
                backbone_max_length=args.backbone_max_length,
                chunk_residues=args.chunk_residues,
                chunk_overlap=args.chunk_overlap,
                device=device,
                cache_dtype=spec.get("cache_dtype", torch.float16),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cache_format": CACHE_FORMAT,
                    "backbone_model_name": spec["backbone_model"],
                    "backbone_type": spec["backbone_type"],
                    "backbone_max_length": args.backbone_max_length,
                    "chunk_residues": args.chunk_residues,
                    "chunk_overlap": args.chunk_overlap,
                    "overlap_merge": "position_aligned_mean",
                    "synthetic_boundary_tokens": True,
                    "cache_dtype": str(spec.get("cache_dtype", torch.float16)),
                    "tokens": tokens,
                    "diagnostics": diagnostics,
                },
                cache_path,
            )
            del tokens
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not cache_matches(
                cache_path,
                spec,
                args.backbone_max_length,
                args.chunk_residues,
                args.chunk_overlap,
                sequences,
            ):
                raise RuntimeError(f"Cache validation failed: {cache_path}")
            print(f"Wrote position-stitched token cache: {cache_path}", flush=True)

        manifest["caches"][model_key] = {
            "file": cache_path.name,
            "backbone_model": spec["backbone_model"],
            "backbone_type": spec["backbone_type"],
            "bytes": cache_path.stat().st_size,
            "sha256": file_sha256(cache_path),
        }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote cache manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
