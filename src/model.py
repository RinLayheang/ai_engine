"""Two-axis moderation model: one XLM-RoBERTa backbone, two independent heads.

Severity and target get separate heads (not a single combined label) because
routing needs two independent confidences — see Document/ARCHITECTURE.md
section 3, "Two axes".
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from src.labels import Severity, Target


class ModerationModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.severity_head = nn.Linear(hidden_size, len(Severity))
        self.target_head = nn.Linear(hidden_size, len(Target))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(output.last_hidden_state[:, 0])  # <s> token
        return self.severity_head(pooled), self.target_head(pooled)
