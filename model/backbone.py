import re
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, AutoTokenizer


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 1)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, token_embeddings, attention_mask):
        scores = self.scorer(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)


class PHCNetV2LargeBackboneBoundedResidual(nn.Module):
    """PHCNet head with a frozen FP16 protein-language-model backbone.

    Frozen token embeddings can be precomputed once and held on CPU. The final
    PHCNet protocol constructs those caches with 480-residue chunks and 64
    residues of overlap before this module performs trainable pooling.
    """

    def __init__(
        self,
        backbone_model_name,
        backbone_type,
        category_vocab_sizes,
        num_experiment_fields,
        num_modification_fields,
        category_dim=16,
        hidden_dim=256,
        dropout=0.2,
        condition_dropout=0.1,
        condition_scale=1.0,
        max_delta=0.2,
        backbone_max_length=512,
        backbone_dtype="float16",
        freeze_backbone=True,
        load_backbone=True,
        backbone_hidden_dim=None,
    ):
        super().__init__()
        if backbone_type not in {"esm", "protbert"}:
            raise ValueError("backbone_type must be 'esm' or 'protbert'.")
        if not freeze_backbone:
            raise ValueError("The large-backbone model currently requires a frozen backbone.")

        self.backbone_type = backbone_type
        self.backbone_model_name = backbone_model_name
        self.backbone_max_length = backbone_max_length
        self.condition_dropout = condition_dropout
        self.condition_scale = condition_scale
        self.max_delta = max_delta
        self.num_experiment_fields = num_experiment_fields
        self.num_modification_fields = num_modification_fields
        self._token_cache = {}
        self._backbone_offloaded = False

        dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
        }[backbone_dtype]
        self.tokenizer = None
        self._backbone_loaded = bool(load_backbone)
        if self._backbone_loaded:
            tokenizer_kwargs = {}
            self.tokenizer = AutoTokenizer.from_pretrained(
                backbone_model_name,
                **tokenizer_kwargs,
            )
            self.esm = AutoModel.from_pretrained(
                backbone_model_name,
                torch_dtype=dtype,
                add_pooling_layer=False,
            )
            for param in self.esm.parameters():
                param.requires_grad = False
            self.esm.eval()
            backbone_config = self.esm.config
            backbone_hidden = getattr(
                backbone_config,
                "hidden_size",
                getattr(backbone_config, "d_model", None),
            )
            if (
                backbone_hidden_dim is not None
                and int(backbone_hidden_dim) != int(backbone_hidden)
            ):
                raise ValueError(
                    "Configured backbone_hidden_dim does not match the loaded "
                    f"model: {backbone_hidden_dim} != {backbone_hidden}."
                )
        else:
            self.esm = nn.Identity()
            if backbone_hidden_dim is None:
                backbone_config = AutoConfig.from_pretrained(backbone_model_name)
                backbone_hidden = getattr(
                    backbone_config,
                    "hidden_size",
                    getattr(backbone_config, "d_model", None),
                )
            else:
                backbone_hidden = int(backbone_hidden_dim)
        if backbone_hidden is None:
            raise ValueError(
                f"Could not resolve hidden dimension for {backbone_model_name}."
            )
        self.pool = AttentionPooling(backbone_hidden)

        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size, category_dim) for size in category_vocab_sizes]
        )
        exp_dim = category_dim * num_experiment_fields
        mod_dim = category_dim * num_modification_fields

        self.sequence_net = nn.Sequential(
            nn.Linear(backbone_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.experiment_net = nn.Sequential(
            nn.Linear(exp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.modification_net = nn.Sequential(
            nn.Linear(mod_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.condition_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._zero_init_last_linear(self.residual_head)

    @staticmethod
    def _zero_init_last_linear(module):
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return

    def _format_sequence(self, sequence):
        sequence = re.sub(r"[^A-Za-z]", "", str(sequence)).upper()
        if self.backbone_type == "protbert":
            sequence = re.sub(r"[UZOB]", "X", sequence)
            return " ".join(sequence)
        return sequence

    def _tokenize(self, sequences):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is unavailable in cache-only backbone mode.")
        formatted = [self._format_sequence(sequence) for sequence in sequences]
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        if encoded["input_ids"].size(1) > self.backbone_max_length:
            raise RuntimeError(
                "Direct tokenization would exceed the backbone token capacity. "
                "The final PHCNet protocol does not truncate sequences; build and "
                "load the 480-residue/64-overlap stitched feature cache instead."
            )
        return encoded

    @torch.no_grad()
    def precompute_sequences(
        self,
        sequences,
        batch_size,
        device,
        offload_backbone=True,
        cache_path=None,
        expected_cache_format=None,
        expected_chunk_residues=None,
        expected_chunk_overlap=None,
        expected_synthetic_boundary_tokens=None,
        expected_hidden_dim=None,
    ):
        unique_sequences = list(dict.fromkeys(str(sequence) for sequence in sequences))
        cache_path = Path(cache_path) if cache_path else None
        if cache_path and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            metadata_matches = (
                payload.get("backbone_model_name") == self.backbone_model_name
                and payload.get("backbone_type") == self.backbone_type
                and payload.get("backbone_max_length") == self.backbone_max_length
            )
            if expected_cache_format is not None:
                metadata_matches = metadata_matches and (
                    payload.get("cache_format") == expected_cache_format
                )
            if expected_chunk_residues is not None:
                metadata_matches = metadata_matches and (
                    payload.get("chunk_residues") == expected_chunk_residues
                )
            if expected_chunk_overlap is not None:
                metadata_matches = metadata_matches and (
                    payload.get("chunk_overlap") == expected_chunk_overlap
                )
            if expected_synthetic_boundary_tokens is not None:
                metadata_matches = metadata_matches and (
                    payload.get("synthetic_boundary_tokens")
                    is expected_synthetic_boundary_tokens
                )
            tokens = payload.get("tokens", {})
            if expected_hidden_dim is not None and tokens:
                sample = next(iter(tokens.values()))
                metadata_matches = metadata_matches and (
                    sample.ndim == 2 and sample.size(1) == expected_hidden_dim
                )
            if metadata_matches and all(sequence in tokens for sequence in unique_sequences):
                self._token_cache = {sequence: tokens[sequence] for sequence in unique_sequences}
                if torch.cuda.is_available():
                    self._token_cache = {
                        sequence: tensor.pin_memory()
                        for sequence, tensor in self._token_cache.items()
                    }
                if offload_backbone and self._backbone_loaded:
                    self.esm.to("cpu")
                    self._backbone_offloaded = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(
                    f"Loaded {len(self._token_cache)} cached sequences from {cache_path}; "
                    f"backbone_offloaded={self._backbone_offloaded}."
                )
                return

        if not self._backbone_loaded:
            raise RuntimeError(
                f"A valid token cache is required in cache-only mode: {cache_path}"
            )

        self.esm.to(device)
        self.esm.eval()
        self._token_cache.clear()

        for start in range(0, len(unique_sequences), batch_size):
            raw_batch = unique_sequences[start : start + batch_size]
            encoded = self._tokenize(raw_batch)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = self.esm(**encoded).last_hidden_state
            masks = encoded["attention_mask"]
            for sequence, token_embeddings, mask in zip(raw_batch, outputs, masks):
                length = int(mask.sum().item())
                cached = token_embeddings[:length].detach().to("cpu", dtype=torch.float16)
                if torch.cuda.is_available():
                    cached = cached.pin_memory()
                self._token_cache[sequence] = cached

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backbone_model_name": self.backbone_model_name,
                    "backbone_type": self.backbone_type,
                    "backbone_max_length": self.backbone_max_length,
                    "tokens": self._token_cache,
                },
                cache_path,
            )
            print(f"Wrote frozen token cache: {cache_path}")

        if offload_backbone:
            self.esm.to("cpu")
            self._backbone_offloaded = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"Precomputed {len(self._token_cache)} unique sequences "
            f"with {self.backbone_type}; backbone_offloaded={self._backbone_offloaded}."
        )

    def _encode_cached_sequences(self, sequences, device):
        missing = [str(sequence) for sequence in sequences if str(sequence) not in self._token_cache]
        if missing:
            raise KeyError(f"{len(missing)} sequences are missing from the backbone cache.")
        cached = [self._token_cache[str(sequence)] for sequence in sequences]
        lengths = [tokens.size(0) for tokens in cached]
        padded = pad_sequence(cached, batch_first=True)
        padded = padded.to(device, non_blocking=True).float()
        attention_mask = torch.zeros(
            len(cached), padded.size(1), dtype=torch.long, device=device
        )
        for index, length in enumerate(lengths):
            attention_mask[index, :length] = 1
        return self.pool(padded, attention_mask)

    def encode_sequences(self, sequences, device):
        if self._token_cache:
            return self._encode_cached_sequences(sequences, device)
        if self._backbone_offloaded:
            raise RuntimeError("Backbone was offloaded but the sequence cache is empty.")

        encoded = self._tokenize(sequences)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = torch.no_grad() if not any(p.requires_grad for p in self.esm.parameters()) else nullcontext()
        with context:
            outputs = self.esm(**encoded)
        return self.pool(outputs.last_hidden_state.float(), encoded["attention_mask"])

    def encode_categories(self, category_ids):
        embeddings = [
            embedding(category_ids[:, index])
            for index, embedding in enumerate(self.category_embeddings)
        ]
        exp_embeddings = torch.cat(embeddings[: self.num_experiment_fields], dim=-1)
        mod_embeddings = torch.cat(embeddings[self.num_experiment_fields :], dim=-1)
        return exp_embeddings, mod_embeddings

    def forward(self, sequences, category_ids):
        device = category_ids.device
        seq_embedding = self.encode_sequences(sequences, device)
        h_seq = self.sequence_net(seq_embedding)

        exp_embedding, mod_embedding = self.encode_categories(category_ids)
        h_exp = self.experiment_net(exp_embedding)
        h_mod = self.modification_net(mod_embedding)
        h_condition = self.condition_net(torch.cat([h_exp, h_mod], dim=-1))

        if self.training and self.condition_dropout > 0:
            keep = (
                torch.rand(h_condition.size(0), 1, device=device)
                > self.condition_dropout
            ).float()
            h_condition = h_condition * keep

        gate = self.gate_net(torch.cat([h_seq, h_condition], dim=-1))
        h_base = h_seq + self.condition_scale * gate * h_condition
        base_mu = self.base_head(h_base).squeeze(-1)

        residual_input = torch.cat(
            [h_base, h_condition, h_base * h_condition],
            dim=-1,
        )
        residual_raw = self.residual_head(residual_input).squeeze(-1)
        residual_delta = self.max_delta * torch.tanh(residual_raw)
        mu = base_mu + residual_delta

        return {
            "mu": mu,
            "base_mu": base_mu,
            "residual_delta": residual_delta,
            "condition_gate_mean": gate.mean(dim=-1),
        }
