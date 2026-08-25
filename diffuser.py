"""Matched flow path, periodic geometry, schedules, and endpoint losses."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Callable

import torch

PROJECT_DIR = Path(__file__).resolve().parent
SPUR_DIR = Path(
    os.environ.get("PENROSE_SPUR_PATH", PROJECT_DIR.parent / "PenroseSpur")
).resolve()
if str(SPUR_DIR) not in sys.path:
    sys.path.append(str(SPUR_DIR))

from flow_geometry import (  # noqa: E402
    ANGLE_HALF_PERIOD,
    ANGLE_PERIOD,
    GeometryLoss as LossTerms,
    angle_delta,
    canonicalize_xya,
    flow_state,
    pairwise_xya_distance,
    reconstruction_loss,
    wrap_angle,
)


@dataclass
class FlowBatch:
    noise: torch.Tensor
    data: torch.Tensor
    state: torch.Tensor
    time: torch.Tensor


def schedule_time(
    unit_time: torch.Tensor,
    schedule: str,
    *,
    r: float = 2.0,
    k: float = 8.0,
) -> torch.Tensor:
    if bool(((unit_time < 0) | (unit_time > 1)).any()):
        raise ValueError("unit_time must lie in [0,1]")
    if schedule == "uniform":
        result = unit_time
    elif schedule == "exponential":
        if r <= 1 or k <= 0:
            raise ValueError("r must exceed 1 and k must be positive")
        result = (1.0 - torch.pow(r, -k * unit_time)) / (1.0 - r ** (-k))
    elif schedule == "sine":
        result = torch.sin(math.pi * unit_time / 2.0)
    elif schedule == "quadratic":
        result = 1.0 - (1.0 - unit_time.square())
    else:
        raise ValueError(f"Unknown time schedule: {schedule}")
    return result.clamp(0.0, 1.0)


def draw_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    schedule: str,
    *,
    r: float = 2.0,
    k: float = 8.0,
) -> torch.Tensor:
    unit = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    return schedule_time(unit, schedule, r=r, k=k)


def periodic_pair_cost(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return pairwise_xya_distance(data, noise)


def match_noise(
    data: torch.Tensor,
    noise: torch.Tensor,
    lsa: Callable[..., torch.Tensor],
    *,
    workers: int,
) -> torch.Tensor:
    cost = periodic_pair_cost(data, noise)
    permutation = lsa(cost, workers=workers)
    return torch.gather(noise, 1, permutation[..., None].expand_as(noise))


def prepare_flow_batch(
    data: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    *,
    matched: bool,
    lsa: Callable[..., torch.Tensor],
    workers: int,
) -> FlowBatch:
    ordered_noise = match_noise(data, noise, lsa, workers=workers) if matched else noise
    return FlowBatch(
        noise=ordered_noise,
        data=data,
        state=flow_state(ordered_noise, data, time),
        time=time,
    )


def endpoint_loss(prediction: torch.Tensor, target: torch.Tensor, loss: str) -> LossTerms:
    return reconstruction_loss(prediction, target, loss=loss)
