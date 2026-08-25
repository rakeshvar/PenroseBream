"""The single direct set Transformer used by PenroseBream."""

from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig


class DirectTransformer(nn.Module):
    """Predict data-endpoint x, y, and scaled angle from a flow state."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_global_tokens = config.num_global_tokens
        self.input_projection = nn.Linear(3, config.d_model)
        self.color_embedding = nn.Embedding(2, config.d_model)
        self.global_tokens = nn.Parameter(
            torch.randn(1, config.num_global_tokens, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 3),
        )

    def forward(self, corrupted: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        if corrupted.ndim != 3 or corrupted.shape[-1] != 3:
            raise ValueError(
                f"Expected corrupted shape (B,N,3), got {tuple(corrupted.shape)}"
            )
        if colors.shape != corrupted.shape[:2]:
            raise ValueError(
                f"Expected colors shape {tuple(corrupted.shape[:2])}, "
                f"got {tuple(colors.shape)}"
            )
        tiles = self.input_projection(corrupted)
        tiles = tiles + self.color_embedding(colors.long())
        globals_ = self.global_tokens.expand(corrupted.shape[0], -1, -1)
        hidden = self.encoder(torch.cat((globals_, tiles), dim=1))
        tile_hidden = hidden[:, self.num_global_tokens :]
        return self.output_projection(self.output_norm(tile_hidden))
