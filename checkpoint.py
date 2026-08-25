"""Atomic epoch checkpoints with exact random-state restoration."""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


def capture_rng(
    training_generator: torch.Generator,
    sampling_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "training_generator": training_generator.get_state(),
        "sampling_generator": sampling_generator.get_state(),
    }


def restore_rng(
    states: dict[str, Any],
    training_generator: torch.Generator,
    sampling_generator: torch.Generator,
) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if states["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    training_generator.set_state(states["training_generator"])
    sampling_generator.set_state(states["sampling_generator"])


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_epoch(
    payload: dict[str, Any],
    checkpoint_directory: Path,
    identifier: str,
    epoch: int,
) -> Path:
    path = checkpoint_directory / f"{identifier}_e{epoch:03d}.pt"
    atomic_save(payload, path)
    return path


def retain_newest_and_best(
    checkpoint_directory: Path,
    identifier: str,
    newest_epoch: int,
    best_epoch: int,
) -> None:
    retain = {newest_epoch, best_epoch}
    for path in checkpoint_directory.glob(f"{identifier}_e*.pt"):
        try:
            epoch = int(path.stem.rsplit("_e", 1)[1])
        except (IndexError, ValueError):
            continue
        if epoch not in retain:
            path.unlink()
