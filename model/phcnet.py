"""Large frozen-backbone PHCNet with modification and global physchem residuals."""

import math
import re

import torch
from torch import nn

from .modification import (
    ModificationSequenceBatch,
    PHCNetV2LargeAdapterModificationAware,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
RESIDUE_MASS = {
    "A": 71.0788,
    "C": 103.1388,
    "D": 115.0886,
    "E": 129.1155,
    "F": 147.1766,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "K": 128.1741,
    "L": 113.1594,
    "M": 131.1926,
    "N": 114.1038,
    "P": 97.1167,
    "Q": 128.1307,
    "R": 156.1875,
    "S": 87.0782,
    "T": 101.1051,
    "V": 99.1326,
    "W": 186.2132,
    "Y": 163.1760,
}
GROUPS = (
    ("hydrophobic", frozenset("AVILMFWY")),
    ("polar", frozenset("STNQ")),
    ("acidic", frozenset("DE")),
    ("basic", frozenset("KRH")),
    ("aromatic", frozenset("FWY")),
)
PHYSICOCHEMICAL_FEATURE_NAMES = (
    "length_div_100",
    "log1p_length_div_5",
    "approx_molecular_weight_div_10000",
    *(f"aa_fraction_{amino_acid}" for amino_acid in AMINO_ACIDS),
    *(
        f"dipeptide_fraction_{first}{second}"
        for first in AMINO_ACIDS
        for second in AMINO_ACIDS
    ),
    *(f"group_fraction_{name}" for name, _ in GROUPS),
    "aa_fraction_G_explicit",
    "aa_fraction_P_explicit",
    "aa_fraction_C_explicit",
    "rough_sidechain_charge_per_residue",
)
PHYSICOCHEMICAL_FEATURE_DIM = len(PHYSICOCHEMICAL_FEATURE_NAMES)
if PHYSICOCHEMICAL_FEATURE_DIM != 432:
    raise RuntimeError(
        f"Expected 432 physicochemical features, got {PHYSICOCHEMICAL_FEATURE_DIM}."
    )


class PhyschemModificationSequenceBatch(ModificationSequenceBatch):
    """Modification-aware sequence batch carrying target-independent features."""

    def __init__(
        self,
        sequences,
        modification_texts,
        modification_gates,
        physicochemical_features,
    ):
        super().__init__(sequences, modification_texts, modification_gates)
        if len(physicochemical_features) != len(sequences):
            raise ValueError("Physicochemical feature count does not match batch size.")
        self.physicochemical_features = list(physicochemical_features)


def normalize_amino_acid_sequence(sequence):
    return re.sub(r"[^A-Za-z]", "", str(sequence)).upper()


def compute_physicochemical_features(sequence):
    """Return deterministic composition and coarse chemistry features."""

    sequence = normalize_amino_acid_sequence(sequence)
    length = len(sequence)
    denominator = float(max(length, 1))
    counts = [0] * len(AMINO_ACIDS)
    for amino_acid in sequence:
        index = AA_TO_INDEX.get(amino_acid)
        if index is not None:
            counts[index] += 1

    approximate_mass = 18.0153 + sum(
        RESIDUE_MASS.get(amino_acid, 110.0) for amino_acid in sequence
    )
    if length == 0:
        approximate_mass = 0.0
    amino_acid_fractions = [count / denominator for count in counts]

    dipeptide_counts = [0] * (len(AMINO_ACIDS) ** 2)
    for first, second in zip(sequence, sequence[1:]):
        first_index = AA_TO_INDEX.get(first)
        second_index = AA_TO_INDEX.get(second)
        if first_index is None or second_index is None:
            continue
        dipeptide_counts[first_index * len(AMINO_ACIDS) + second_index] += 1
    dipeptide_denominator = float(max(length - 1, 1))
    dipeptide_fractions = [
        count / dipeptide_denominator for count in dipeptide_counts
    ]

    group_fractions = [
        sum(counts[AA_TO_INDEX[amino_acid]] for amino_acid in members)
        / denominator
        for _, members in GROUPS
    ]
    explicit_fractions = [
        counts[AA_TO_INDEX[amino_acid]] / denominator
        for amino_acid in ("G", "P", "C")
    ]
    rough_charge = (
        counts[AA_TO_INDEX["K"]]
        + counts[AA_TO_INDEX["R"]]
        + 0.1 * counts[AA_TO_INDEX["H"]]
        - counts[AA_TO_INDEX["D"]]
        - counts[AA_TO_INDEX["E"]]
    ) / denominator

    features = [
        length / 100.0,
        math.log1p(length) / 5.0,
        approximate_mass / 10000.0,
        *amino_acid_fractions,
        *dipeptide_fractions,
        *group_fractions,
        *explicit_fractions,
        rough_charge,
    ]
    if len(features) != PHYSICOCHEMICAL_FEATURE_DIM:
        raise RuntimeError(f"Generated {len(features)} physicochemical features.")
    return features


class PHCNetV2LargeAdapterPhyschemModificationAware(
    PHCNetV2LargeAdapterModificationAware
):
    """Large adapter model plus a zero-initialized bounded physchem residual."""

    def __init__(
        self,
        *args,
        physchem_hidden_dim=128,
        physchem_dropout=0.1,
        physchem_max_delta=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if int(physchem_hidden_dim) <= 0:
            raise ValueError("physchem_hidden_dim must be positive.")
        if not 0.0 <= float(physchem_dropout) < 1.0:
            raise ValueError("physchem_dropout must be in [0, 1).")
        if float(physchem_max_delta) < 0:
            raise ValueError("physchem_max_delta must be non-negative.")

        model_hidden_dim = self.base_head[1].in_features
        branch_hidden_dim = int(physchem_hidden_dim)
        self.physchem_max_delta = float(physchem_max_delta)

        # Keep the validated anchor model's initialization stream unchanged.
        with torch.random.fork_rng(devices=[]):
            self.physchem_encoder = nn.Sequential(
                nn.LayerNorm(PHYSICOCHEMICAL_FEATURE_DIM),
                nn.Linear(PHYSICOCHEMICAL_FEATURE_DIM, branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(physchem_dropout)),
                nn.Linear(branch_hidden_dim, model_hidden_dim),
                nn.LayerNorm(model_hidden_dim),
                nn.GELU(),
            )
            fusion_dim = model_hidden_dim * 5
            self.physchem_fusion_net = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(physchem_dropout)),
            )
            self.physchem_gate_head = nn.Sequential(
                nn.Linear(branch_hidden_dim, 1),
                nn.Sigmoid(),
            )
            residual_hidden_dim = max(branch_hidden_dim // 2, 16)
            self.physchem_residual_head = nn.Sequential(
                nn.Linear(branch_hidden_dim, residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(physchem_dropout)),
                nn.Linear(residual_hidden_dim, 1),
            )
            self._zero_init_last_linear(self.physchem_residual_head)

        self.physchem_metadata = {
            "feature_dim": PHYSICOCHEMICAL_FEATURE_DIM,
            "hidden_dim": branch_hidden_dim,
            "dropout": float(physchem_dropout),
            "max_delta_normalized_log10": self.physchem_max_delta,
        }

    @staticmethod
    def _get_physicochemical_inputs(sequences, device):
        if not isinstance(sequences, PhyschemModificationSequenceBatch):
            raise TypeError(
                "PhyschemModificationSequenceBatch is required for physchem fusion."
            )
        features = torch.as_tensor(
            sequences.physicochemical_features,
            dtype=torch.float32,
            device=device,
        )
        expected_shape = (len(sequences), PHYSICOCHEMICAL_FEATURE_DIM)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"Expected physicochemical feature shape {expected_shape}, "
                f"got {tuple(features.shape)}."
            )
        return features

    def forward(self, sequences, category_ids):
        device = category_ids.device
        modification_texts, modification_gate = self._get_modification_inputs(
            sequences, device
        )
        physicochemical_features = self._get_physicochemical_inputs(
            sequences, device
        )

        sequence_embedding = self.encode_sequences(sequences, device)
        h_seq = self.sequence_net(sequence_embedding)
        experiment_embedding, modification_embedding = self.encode_categories(
            category_ids
        )
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
            [h_base, h_condition, h_base * h_condition], dim=-1
        )
        condition_residual_raw = self.residual_head(
            condition_residual_input
        ).squeeze(-1)
        condition_residual_delta = self.max_delta * torch.tanh(
            condition_residual_raw
        )

        h_modification_text = self.modification_encoder(modification_texts, device)
        modification_residual_input = torch.cat(
            [h_modification_text, h_seq.detach(), h_condition.detach()], dim=-1
        )
        modification_residual_raw = self.modification_residual_head(
            modification_residual_input
        ).squeeze(-1)
        modification_residual_delta = (
            modification_gate
            * self.modification_max_delta
            * torch.tanh(modification_residual_raw)
        )

        h_physchem = self.physchem_encoder(physicochemical_features)
        h_seq_anchor = h_seq.detach()
        h_condition_anchor = h_condition.detach()
        physchem_fusion_input = torch.cat(
            [
                h_physchem,
                h_seq_anchor,
                h_condition_anchor,
                h_physchem * h_seq_anchor,
                h_physchem * h_condition_anchor,
            ],
            dim=-1,
        )
        h_physchem_fusion = self.physchem_fusion_net(physchem_fusion_input)
        physchem_gate = self.physchem_gate_head(h_physchem_fusion).squeeze(-1)
        physchem_residual_raw = self.physchem_residual_head(
            h_physchem_fusion
        ).squeeze(-1)
        physchem_residual_delta = (
            physchem_gate
            * self.physchem_max_delta
            * torch.tanh(physchem_residual_raw)
        )

        residual_delta = (
            condition_residual_delta
            + modification_residual_delta
            + physchem_residual_delta
        )
        mu = base_mu + residual_delta
        return {
            "mu": mu,
            "base_mu": base_mu,
            "residual_delta": residual_delta,
            "condition_residual_delta": condition_residual_delta,
            "modification_residual_delta": modification_residual_delta,
            "physchem_residual_delta": physchem_residual_delta,
            "modification_gate": modification_gate,
            "physchem_gate": physchem_gate,
            "condition_gate_mean": condition_gate.mean(dim=-1),
            "physchem_embedding_norm": h_physchem.norm(dim=-1),
        }


# Public name used by all three frozen feature extractors.
PHCNetUnified = PHCNetV2LargeAdapterPhyschemModificationAware
