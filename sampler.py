"""Target-free programmatic iterative sampling."""

from __future__ import annotations

import torch

from denoiser import DirectTransformer
from diffuser import canonicalize_xya


@torch.no_grad()
def sample(
    model: DirectTransformer,
    xya: torch.Tensor,
    colors: torch.Tensor,
    num_iters: int = 1,
) -> list[torch.Tensor]:
    """Apply the endpoint model repeatedly to caller-supplied geometry."""
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    if xya.ndim != 3 or xya.shape[-1] != 3:
        raise ValueError(f"Expected xya shape (B,N,3), got {tuple(xya.shape)}")
    if colors.shape != xya.shape[:2]:
        raise ValueError(
            f"Expected colors shape {tuple(xya.shape[:2])}, got {tuple(colors.shape)}"
        )
    model.eval()
    current = canonicalize_xya(xya)
    produced = []
    for _ in range(num_iters):
        current = canonicalize_xya(model(current, colors))
        produced.append(current)
    return produced
