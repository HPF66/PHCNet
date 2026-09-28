import re
import unicodedata

import torch
from torch import nn

from .adapter import (
    PHCNetV2LargeBackboneAdapter,
)


CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789[](){}<>+-=,.;:/_'*%#@|? "
CHAR_TO_ID = {char: index + 1 for index, char in enumerate(CHAR_VOCAB)}
UNKNOWN_CHAR_ID = len(CHAR_VOCAB) + 1


def parse_kernel_sizes(values):
    """Accept the locked comma-separated config form or an integer sequence."""
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("modification_kernels must be comma-separated integers.") from exc


class ModificationSequenceBatch(list):
    def __init__(self, sequences, modification_texts, modification_gates):
        super().__init__(sequences)
        self.modification_texts = list(modification_texts)
        self.modification_gates = list(modification_gates)


def normalize_modification_text(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", text).strip()


def build_modification_text(row):
    fields = (
        ("nter", row.get("nter")),
        ("cter", row.get("cter")),
        ("chem", row.get("chem_mod")),
        ("linear_cyclic", row.get("lin_cyc")),
        ("chiral", row.get("chiral")),
        ("modified_sequence", row.get("sequence_modified")),
    )
    return " | ".join(
        f"{name}={normalize_modification_text(value)}" for name, value in fields
    )


def has_modification_evidence(row):
    defaults = {
        "nter_group": "free",
        "cter_group": "free",
        "chem_mod_group": "none",
        "lin_cyc_group": "linear",
        "chiral_group": "L",
    }
    if any(str(row.get(field, "")) != value for field, value in defaults.items()):
        return 1.0
    modified = re.sub(r"\s+", "", str(row.get("sequence_modified", "")))
    return float(modified != str(row.get("sequence_esm", "")))


class ModificationTextCNNEncoder(nn.Module):
    def __init__(
        self,
        char_dim=24,
        cnn_channels=32,
        output_dim=96,
        kernels=(3, 5, 9),
        max_length=512,
    ):
        super().__init__()
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("kernels must contain positive odd integers.")
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(
            len(CHAR_VOCAB) + 2,
            char_dim,
            padding_idx=0,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    char_dim,
                    cnn_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
                for kernel in kernels
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(cnn_channels * len(kernels) * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def _batch_to_ids(self, texts, device):
        normalized = [normalize_modification_text(text)[: self.max_length] for text in texts]
        lengths = torch.tensor(
            [len(text) for text in normalized],
            dtype=torch.long,
            device=device,
        )
        width = max(int(lengths.max().item()) if normalized else 0, 1)
        ids = torch.zeros(len(normalized), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(normalized), width, dtype=torch.bool, device=device)
        for row_index, text in enumerate(normalized):
            if not text:
                continue
            ids[row_index, : len(text)] = torch.tensor(
                [CHAR_TO_ID.get(char, UNKNOWN_CHAR_ID) for char in text],
                dtype=torch.long,
                device=device,
            )
            mask[row_index, : len(text)] = True
        return ids, mask, lengths

    def forward(self, texts, device):
        ids, mask, lengths = self._batch_to_ids(texts, device)
        embedded = self.embedding(ids).transpose(1, 2)
        conv_mask = mask.unsqueeze(1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(embedded.dtype)
        pooled = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            mean_features = (features * conv_mask).sum(dim=-1) / denominator
            max_features = features.masked_fill(~conv_mask, -1e4).max(dim=-1).values
            max_features = torch.where(
                lengths.unsqueeze(1) > 0,
                max_features,
                torch.zeros_like(max_features),
            )
            pooled.extend([mean_features, max_features])
        return self.projection(torch.cat(pooled, dim=-1))


class PHCNetV2LargeAdapterModificationAware(PHCNetV2LargeBackboneAdapter):
    """Best large-backbone adapter plus a bounded raw-modification residual."""

    def __init__(
        self,
        *args,
        modification_char_dim=24,
        modification_cnn_channels=32,
        modification_hidden_dim=96,
        modification_kernels=(3, 5, 9),
        modification_max_length=512,
        modification_max_delta=0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if modification_max_delta < 0:
            raise ValueError("modification_max_delta must be nonnegative.")
        hidden_dim = self.base_head[1].in_features
        self.modification_max_delta = float(modification_max_delta)

        # Do not perturb the selected adapter's initialization or training RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.modification_encoder = ModificationTextCNNEncoder(
                char_dim=modification_char_dim,
                cnn_channels=modification_cnn_channels,
                output_dim=modification_hidden_dim,
                kernels=parse_kernel_sizes(modification_kernels),
                max_length=modification_max_length,
            )
            head_dim = max(modification_hidden_dim, 32)
            input_dim = modification_hidden_dim + hidden_dim * 2
            self.modification_residual_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, max(head_dim // 2, 16)),
                nn.ReLU(),
                nn.Linear(max(head_dim // 2, 16), 1),
            )
            self._zero_init_last_linear(self.modification_residual_head)

    @staticmethod
    def _get_modification_inputs(sequences, device):
        if not isinstance(sequences, ModificationSequenceBatch):
            raise TypeError(
                "ModificationSequenceBatch is required for modification-aware inference."
            )
        if len(sequences.modification_texts) != len(sequences):
            raise ValueError("Modification text count does not match sequence count.")
        gates = torch.tensor(
            sequences.modification_gates,
            dtype=torch.float32,
            device=device,
        )
        return sequences.modification_texts, gates

    def forward(self, sequences, category_ids):
        device = category_ids.device
        modification_texts, modification_gate = self._get_modification_inputs(
            sequences, device
        )

        sequence_embedding = self.encode_sequences(sequences, device)
        h_seq = self.sequence_net(sequence_embedding)
        experiment_embedding, modification_embedding = self.encode_categories(category_ids)
        h_experiment = self.experiment_net(experiment_embedding)
        h_modification_category = self.modification_net(modification_embedding)
        h_condition = self.condition_net(
            torch.cat([h_experiment, h_modification_category], dim=-1)
        )
        if self.training and self.condition_dropout > 0:
            keep = (
                torch.rand(h_condition.size(0), 1, device=device)
                > self.condition_dropout
            ).float()
            h_condition = h_condition * keep

        condition_gate = self.gate_net(torch.cat([h_seq, h_condition], dim=-1))
        h_base = h_seq + self.condition_scale * condition_gate * h_condition
        base_mu = self.base_head(h_base).squeeze(-1)

        condition_residual_input = torch.cat(
            [h_base, h_condition, h_base * h_condition],
            dim=-1,
        )
        condition_residual_raw = self.residual_head(condition_residual_input).squeeze(-1)
        condition_residual_delta = self.max_delta * torch.tanh(condition_residual_raw)

        h_modification_text = self.modification_encoder(modification_texts, device)
        modification_residual_input = torch.cat(
            [h_modification_text, h_seq.detach(), h_condition.detach()],
            dim=-1,
        )
        modification_residual_raw = self.modification_residual_head(
            modification_residual_input
        ).squeeze(-1)
        modification_residual_delta = (
            modification_gate
            * self.modification_max_delta
            * torch.tanh(modification_residual_raw)
        )

        residual_delta = condition_residual_delta + modification_residual_delta
        mu = base_mu + residual_delta
        return {
            "mu": mu,
            "base_mu": base_mu,
            "residual_delta": residual_delta,
            "condition_residual_delta": condition_residual_delta,
            "modification_residual_delta": modification_residual_delta,
            "modification_gate": modification_gate,
            "condition_gate_mean": condition_gate.mean(dim=-1),
        }
