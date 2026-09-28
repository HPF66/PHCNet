import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass


# Kept only for compatibility with the legacy TrainConfig helper. The locked
# training entry point reads all three backbone IDs from utils/final_config.json.
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

EXPERIMENT_FIELDS = ["species_group", "matrix_group", "vivo_vitro_clean", "assay_group"]
MODIFICATION_FIELDS = [
    "nter_group",
    "cter_group",
    "chem_mod_group",
    "lin_cyc_group",
    "chiral_group",
    "api_source_query",
]
CATEGORY_FIELDS = EXPERIMENT_FIELDS + MODIFICATION_FIELDS

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class TrainConfig:
    esm_model: str = DEFAULT_MODEL_NAME
    chunk_residues: int = 480
    chunk_overlap: int = 64
    backbone_token_capacity: int = 512
    category_dim: int = 16
    hidden_dim: int = 256
    dropout: float = 0.2
    condition_dropout: float = 0.1
    condition_scale: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 80
    patience: int = 15
    loss: str = "smoothl1"
    seed: int = 3407


def default_config_dict():
    return asdict(TrainConfig())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "na", "n/a", "not mentioned", "not reported"}:
        return ""
    return text


def parse_float(value):
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def has_any(text, keywords):
    lowered = clean_text(text).lower()
    return any(keyword in lowered for keyword in keywords)


def normalize_assay(value):
    text = clean_text(value).lower()
    if not text:
        return "unknown"
    if "lc-ms" in text or "lc–ms" in text or "lc/ms" in text or "uplc-ms" in text:
        return "lcms"
    if "hplc" in text:
        return "hplc"
    if "elisa" in text:
        return "elisa"
    if "radioimmunoassay" in text or text == "ria":
        return "ria"
    if "fluores" in text:
        return "fluorescence"
    if "icp-ms" in text:
        return "icpms"
    if "ms" in text:
        return "ms_other"
    return "other"


def normalize_nter(value):
    text = clean_text(value)
    lowered = text.lower()
    if not text:
        return "unknown"
    if lowered == "free":
        return "free"
    if "acetyl" in lowered or lowered in {"ac"}:
        return "acetylation"
    if "pglu" in lowered or "pyroglu" in lowered or "pyroglut" in lowered:
        return "pglu"
    if "peg" in lowered:
        return "peg"
    if has_any(lowered, ["myrist", "palmit", "lipid", "chol", "octanoyl", "fatty"]):
        return "lipid"
    if "glycosyl" in lowered or "sialic" in lowered:
        return "glycosylation"
    if has_any(lowered, ["125i", "radio", "label", "cy5", "fitc", "tetramethylrhodamine", "fluores"]):
        return "label"
    if has_any(lowered, ["fc", "hsa", "albumin", "linker", "fused", "fusion", "abd", "xten", "hgh"]):
        return "fusion_or_linker"
    return "other_modified"


def normalize_cter(value):
    text = clean_text(value)
    lowered = text.lower()
    if not text:
        return "unknown"
    if lowered == "free":
        return "free"
    if "amidation" in lowered or lowered in {"amide", "nh2"}:
        return "amidation"
    if "peg" in lowered:
        return "peg"
    if has_any(lowered, ["myrist", "palmit", "lipid", "chol", "octanoyl", "fatty", "c16", "c18", "c20"]):
        return "lipid"
    if "glycosyl" in lowered:
        return "glycosylation"
    if has_any(lowered, ["125i", "radio", "label", "cy5", "fitc", "tetramethylrhodamine", "fluores"]):
        return "label"
    if has_any(lowered, ["fc", "hsa", "albumin", "linker", "fused", "fusion", "abd", "xten", "hgh", "fab", "scfv"]):
        return "fusion_or_linker"
    if "lysine" in lowered and "added" in lowered:
        return "added_lysine"
    return "other_modified"


def normalize_chem_mod(value):
    text = clean_text(value)
    lowered = text.lower()
    if not text:
        return "none"
    if "aib" in lowered or "aminoisobutyric" in lowered:
        return "aib"
    if "pglu" in lowered or "pyroglu" in lowered or "pyroglut" in lowered:
        return "noncanonical_residue"
    if "all d" in lowered or "d-amino" in lowered or "d " in lowered or "d-" in lowered:
        return "d_amino_acid"
    if "disulfide" in lowered:
        return "disulfide"
    if "reduced amide" in lowered or "reduced carbamate" in lowered or "ψ" in lowered or "Î¨" in lowered:
        return "reduced_bond"
    if "peg" in lowered:
        return "peg"
    if has_any(lowered, ["palmit", "lipid", "fatty", "c16", "c18", "c20", "diacid", "chol"]):
        return "lipid"
    if "glycosyl" in lowered:
        return "glycosylation"
    if has_any(lowered, ["125i", "radio", "label", "cy5", "fitc", "tetramethylrhodamine", "fluores"]):
        return "label"
    if has_any(lowered, ["linker", "fused", "fusion", "fc", "hsa", "albumin", "abd", "xten"]):
        return "fusion_or_linker"
    if has_any(lowered, ["orn", "cit", "harg", "nal", "nle", "oic", "sar", "nme"]):
        return "noncanonical_residue"
    return "other"


def normalize_lin_cyc(value):
    text = clean_text(value)
    lowered = text.lower()
    if not text:
        return "unknown"
    if lowered == "linear":
        return "linear"
    if "disulfide" in lowered or re.search(r"c\s*\d+\s*-\s*c\s*\d+", lowered):
        return "disulfide"
    if "lactam" in lowered:
        return "lactam"
    if "head-to-tail" in lowered or "n-c terminal" in lowered:
        return "head_to_tail"
    if "bicyclic" in lowered:
        return "bicyclic"
    if "cyclic" in lowered or "cyclized" in lowered or "macrocyclic" in lowered:
        return "cyclic_other"
    return "other"


def normalize_chiral(value):
    text = clean_text(value)
    lowered = text.lower()
    if not text:
        return "unknown"
    if lowered == "l":
        return "L"
    if lowered == "d":
        return "D"
    if "mix" in lowered:
        return "Mix"
    return "other"


def clean_sequence_for_esm(sequence):
    text = clean_text(sequence)
    if not text:
        return ""
    replacements = [
        (r"\[ACE\]|\[NHE\]|\[NH2\]", ""),
        (r"\[(AIB)\]", "A"),
        (r"\[(ORN|ORNITHINE)\]", "K"),
        (r"\[(CIT)\]", "R"),
        (r"\[(HARG)\]", "R"),
        (r"\[(?:2-)?NAL\]", "F"),
        (r"alpha-aminoisobutyricacid|aminoisobutyricacid|\bAib\b", "A"),
        (r"\bpGlu\b|\bPGlu\b|\bGlp\b|PyroGlu", "Q"),
        (r"\bSar\b", "G"),
        (r"\bOrn\b", "K"),
        (r"\bCit\b", "R"),
        (r"\bhArg\b|\bHar\b|\bAgp\b", "R"),
        (r"\bNle\b", "L"),
        (r"\bNva\b|\bIva\b", "V"),
        (r"\bCha\b|\bPff\b|\bNal\b|D-Nal", "F"),
        (r"Dallo-Ile|allo-Ile", "I"),
        (r"\bOic\b", "P"),
    ]
    text = text.replace("Î±", "alpha").replace("α", "alpha").replace("Î³", "gamma").replace("γ", "gamma")
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.I)
    text = re.sub(r"\[([^\]]+)\]", lambda m: "".join(c.upper() for c in m.group(1) if c.upper() in CANONICAL_AA), text)
    sequence = "".join(char.upper() for char in text if char.upper() in CANONICAL_AA)
    return sequence


def normalize_row(row):
    target = parse_float(row.get("log10_half_life_minutes"))
    sequence_modified = clean_text(row.get("sequence_modified"))
    sequence_esm = clean_sequence_for_esm(sequence_modified)
    if target is None or not sequence_esm:
        return None

    out = {key: clean_text(value) for key, value in row.items()}
    out["id"] = clean_text(row.get("id"))
    out["sequence_modified"] = sequence_modified
    out["sequence_esm"] = sequence_esm
    out["target"] = target
    out["half_life_minutes"] = parse_float(row.get("half_life_minutes"))
    out["species_group"] = clean_text(row.get("species_group")) or "unknown"
    out["matrix_group"] = clean_text(row.get("matrix_group")) or "unknown"
    out["vivo_vitro_clean"] = clean_text(row.get("vivo_vitro_clean")) or "unknown"
    out["assay_group"] = normalize_assay(row.get("assay"))
    out["nter_group"] = normalize_nter(row.get("nter"))
    out["cter_group"] = normalize_cter(row.get("cter"))
    out["chem_mod_group"] = normalize_chem_mod(row.get("chem_mod"))
    out["lin_cyc_group"] = normalize_lin_cyc(row.get("lin_cyc"))
    out["chiral_group"] = normalize_chiral(row.get("chiral"))
    out["api_source_query"] = clean_text(row.get("api_source_query")) or "unknown"
    return out


def load_split_dir(split_dir):
    split_dir = Path(split_dir)
    split_rows = {}
    for name in ["train", "val", "test"]:
        rows = []
        for row in read_csv_rows(split_dir / f"{name}.csv"):
            normalized = normalize_row(row)
            if normalized is not None:
                rows.append(normalized)
        split_rows[name] = rows
    return split_rows["train"], split_rows["val"], split_rows["test"]


def build_vocabs(train_rows):
    vocabs = {}
    for field in CATEGORY_FIELDS:
        values = sorted({clean_text(row.get(field)) for row in train_rows if clean_text(row.get(field))})
        vocabs[field] = {"<UNK>": 0, **{value: i + 1 for i, value in enumerate(values)}}
    return vocabs


def encode_rows(rows, vocabs):
    for row in rows:
        row["category_ids"] = [
            vocabs[field].get(clean_text(row.get(field)), 0)
            for field in CATEGORY_FIELDS
        ]


class PeptideV2Dataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_batch(batch):
    if torch is None:
        raise RuntimeError("PyTorch is required for collate_batch.")
    return {
        "ids": [row["id"] for row in batch],
        "sequences": [row["sequence_esm"] for row in batch],
        "sequence_modified": [row["sequence_modified"] for row in batch],
        "category_ids": torch.tensor([row["category_ids"] for row in batch], dtype=torch.long),
        "target": torch.tensor([row["target"] for row in batch], dtype=torch.float32),
        "half_life_minutes": [row.get("half_life_minutes") for row in batch],
        "rows": batch,
    }


def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rankdata(values):
    values = np.asarray(values)
    sorter = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[sorter] = np.arange(len(values), dtype=float)
    return ranks


def spearman_corr(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return pearson_corr(rankdata(y_true), rankdata(y_pred))


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae_log10": mae,
        "rmse_log10": rmse,
        "r2": r2,
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
        "median_fold_error": float(np.median(np.power(10.0, np.abs(errors)))),
    }


def binary_auc(y_true_binary, scores):
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true_binary == 1]
    neg = scores[y_true_binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = 0.0
    for value in pos:
        total += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return float(total / (len(pos) * len(neg)))


def stable_metrics(y_true_log, y_pred_log, threshold_minutes=60.0):
    threshold_log = math.log10(threshold_minutes)
    true_binary = np.asarray(y_true_log) >= threshold_log
    pred_binary = np.asarray(y_pred_log) >= threshold_log
    tp = int(np.sum(true_binary & pred_binary))
    tn = int(np.sum(~true_binary & ~pred_binary))
    fp = int(np.sum(~true_binary & pred_binary))
    fn = int(np.sum(true_binary & ~pred_binary))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "stable_threshold_minutes": threshold_minutes,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": binary_auc(true_binary.astype(int), y_pred_log),
    }
