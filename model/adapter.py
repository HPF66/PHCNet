import torch
from torch import nn

from .backbone import (
    PHCNetV2LargeBackboneBoundedResidual,
)


class CompactTokenPooling(nn.Module):
    """Normalize and compress backbone tokens before sequence pooling."""

    def __init__(self, backbone_dim, adapter_dim, pooling, dropout):
        super().__init__()
        if pooling not in {
            "masked_mean",
            "compact_attention",
            "mean_max",
            "gated_mean_attention",
        }:
            raise ValueError(
                "pooling must be masked_mean, compact_attention, mean_max, "
                "or gated_mean_attention."
            )
        self.pooling = pooling
        self.token_norm = nn.LayerNorm(backbone_dim)
        self.token_adapter = nn.Sequential(
            nn.Linear(backbone_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if pooling in {"compact_attention", "gated_mean_attention"}:
            attention_hidden = max(adapter_dim // 2, 32)
            self.attention_scorer = nn.Sequential(
                nn.Linear(adapter_dim, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1),
            )
        else:
            self.attention_scorer = None
        if pooling == "mean_max":
            self.mean_max_fusion = nn.Sequential(
                nn.Linear(adapter_dim * 2, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.mean_max_fusion = None
        if pooling == "gated_mean_attention":
            self.mean_attention_gate = nn.Sequential(
                nn.Linear(adapter_dim * 2, adapter_dim),
                nn.Sigmoid(),
            )
        else:
            self.mean_attention_gate = None

    @staticmethod
    def residue_mask(attention_mask):
        """Exclude padding plus the first and last special tokens."""
        valid = attention_mask.bool()
        mask = valid.clone()
        if mask.size(1) > 0:
            mask[:, 0] = False
        lengths = valid.sum(dim=1)
        last_indices = (lengths - 1).clamp_min(0).unsqueeze(1)
        mask.scatter_(1, last_indices, False)
        empty = mask.sum(dim=1) == 0
        if torch.any(empty):
            mask[empty] = valid[empty]
        return mask

    def forward(self, token_embeddings, attention_mask):
        mask = self.residue_mask(attention_mask)
        projected = self.token_adapter(self.token_norm(token_embeddings.float()))

        mask_weights = mask.unsqueeze(-1).to(projected.dtype)
        denominator = mask_weights.sum(dim=1).clamp_min(1.0)
        mean_pooled = (projected * mask_weights).sum(dim=1) / denominator
        if self.pooling == "masked_mean":
            return mean_pooled

        if self.pooling == "mean_max":
            max_input = projected.masked_fill(~mask.unsqueeze(-1), -1e4)
            max_pooled = max_input.max(dim=1).values
            return self.mean_max_fusion(torch.cat([mean_pooled, max_pooled], dim=-1))

        scores = self.attention_scorer(projected).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        attention_pooled = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        if self.pooling == "compact_attention":
            return attention_pooled

        gate = self.mean_attention_gate(
            torch.cat([mean_pooled, attention_pooled], dim=-1)
        )
        return gate * attention_pooled + (1.0 - gate) * mean_pooled


class PHCNetV2LargeBackboneAdapter(PHCNetV2LargeBackboneBoundedResidual):
    """Large frozen backbone with a compact embedding-to-PHCNet input head."""

    def __init__(
        self,
        backbone_model_name,
        backbone_type,
        category_vocab_sizes,
        num_experiment_fields,
        num_modification_fields,
        input_head="compact_attention",
        adapter_dim=128,
        adapter_dropout=0.1,
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
        super().__init__(
            backbone_model_name=backbone_model_name,
            backbone_type=backbone_type,
            category_vocab_sizes=category_vocab_sizes,
            num_experiment_fields=num_experiment_fields,
            num_modification_fields=num_modification_fields,
            category_dim=category_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            condition_dropout=condition_dropout,
            condition_scale=condition_scale,
            max_delta=max_delta,
            backbone_max_length=backbone_max_length,
            backbone_dtype=backbone_dtype,
            freeze_backbone=freeze_backbone,
            load_backbone=load_backbone,
            backbone_hidden_dim=backbone_hidden_dim,
        )
        backbone_dim = self.pool.scorer[0].in_features
        self.input_head = input_head
        self.adapter_dim = adapter_dim
        self.pool = CompactTokenPooling(
            backbone_dim=backbone_dim,
            adapter_dim=adapter_dim,
            pooling=input_head,
            dropout=adapter_dropout,
        )
        self.sequence_net = nn.Sequential(
            nn.Linear(adapter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
