"""Matched flow path, periodic geometry, schedules, and endpoint losses."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
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
    time_xy: torch.Tensor
    time_angle: torch.Tensor


MATCH_TIME_THRESHOLD = 0.95
MAX_LSA_GROUP_SIZE = 64
PENROSE_NEIGHBOUR_FACTOR = (
    0.24 * math.sin(math.pi / 5.0)
    + 0.59 * math.sin(3.0 * math.pi / 10.0)
    + 0.17 * math.sin(2.0 * math.pi / 5.0)
)


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def schedule_time(
    unit_time: torch.Tensor,
    schedule: str,
    *,
    r: float = 2.0,
    k: float = 8.0,
    tail_lim: float = 0.9375,
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
    elif schedule == "tail":
        if not 0.0 <= tail_lim < 1.0:
            raise ValueError("tail_lim must lie in [0,1)")
        result = tail_lim + (1.0 - tail_lim) * unit_time
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
    tail_lim: float = 0.9375,
) -> torch.Tensor:
    unit = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    return schedule_time(unit, schedule, r=r, k=k, tail_lim=tail_lim)


def jitter_lower_limits(
    symmetry: int,
    side: float,
    jitter_xy: float,
    jitter_angle: float,
) -> tuple[float, float]:
    """Return report-calibrated lower times for spatial and angular jitter."""
    if symmetry == 5:
        neighbour_factor = PENROSE_NEIGHBOUR_FACTOR
        angle_noise = 3.0 * jitter_angle / (10.0 * math.sqrt(2.0))
    elif symmetry == 6:
        neighbour_factor = math.sqrt(3.0)
        angle_noise = jitter_angle / (2.0 * math.sqrt(2.0))
    else:
        raise ValueError("symmetry must be 5 or 6")
    if side <= 0 or jitter_xy <= 0 or jitter_angle <= 0:
        raise ValueError("side and jitter targets must be positive")
    xy_noise = (
        jitter_xy
        * math.sqrt(3.0)
        * neighbour_factor
        * side
        / (4.0 * math.sqrt(2.0))
    )
    limits = (1.0 - xy_noise, 1.0 - angle_noise)
    if any(limit < 0.0 or limit >= 1.0 for limit in limits):
        raise ValueError(
            "jitter targets produce lower times outside [0,1): "
            f"xy={limits[0]:.6g}, angle={limits[1]:.6g}"
        )
    return limits


def draw_jitter_times(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    *,
    symmetry: int,
    side: float,
    jitter_xy: float,
    jitter_angle: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw one shared unit time and map it to coordinate-specific tails."""
    lower_xy, lower_angle = jitter_lower_limits(
        symmetry, side, jitter_xy, jitter_angle
    )
    unit = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    time_xy = lower_xy + (1.0 - lower_xy) * unit
    time_angle = lower_angle + (1.0 - lower_angle) * unit
    aggregate = (2.0 * time_xy + time_angle) / 3.0
    return aggregate, time_xy, time_angle


def coordinate_flow_state(
    source: torch.Tensor,
    target: torch.Tensor,
    time_xy: torch.Tensor,
    time_angle: torch.Tensor,
) -> torch.Tensor:
    """Interpolate XY and periodic angle at independent per-sample times."""
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("source and target must share shape (B,N,3)")
    expected = (source.shape[0],)
    if time_xy.shape != expected or time_angle.shape != expected:
        raise ValueError(f"coordinate times must each have shape {expected}")
    if bool(
        ((time_xy < 0) | (time_xy > 1) | (time_angle < 0) | (time_angle > 1)).any()
    ):
        raise ValueError("coordinate times must lie in [0,1]")
    xy_amount = time_xy.to(device=source.device, dtype=source.dtype)[:, None, None]
    angle_amount = time_angle.to(
        device=source.device, dtype=source.dtype
    )[:, None]
    xy = source[..., :2] + xy_amount * (target[..., :2] - source[..., :2])
    angle = wrap_angle(
        source[..., 2] + angle_amount * angle_delta(target[..., 2], source[..., 2])
    )
    return torch.cat((xy, angle[..., None]), dim=-1)


def periodic_pair_cost(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return pairwise_xya_distance(data, noise)


def _chunk_assignment(
    batch_index: int,
    indices: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    data_group = data[indices]
    noise_group = noise[indices]
    xy_delta = data_group[:, None, :2] - noise_group[None, :, :2]
    angle = data_group[:, None, 2] - noise_group[None, :, 2]
    angle = np.remainder(
        angle + ANGLE_HALF_PERIOD, ANGLE_PERIOD
    ) - ANGLE_HALF_PERIOD
    cost = np.square(xy_delta).sum(axis=-1) + np.square(angle)
    rows, columns = linear_sum_assignment(cost)
    return batch_index, indices[rows], indices[columns]


@dataclass
class MatchHandle:
    identity: np.ndarray
    futures: list[Future[tuple[int, np.ndarray, np.ndarray]]]
    task_sizes: tuple[int, ...]
    _resolved: bool = False

    @property
    def task_count(self) -> int:
        return len(self.futures)

    def permutation(self, device: torch.device) -> torch.Tensor:
        if self._resolved:
            raise RuntimeError("MatchHandle has already been resolved")
        result = self.identity.copy()
        for future in self.futures:
            batch, rows, columns = future.result()
            result[batch, rows] = columns
        self._resolved = True
        return torch.from_numpy(result).to(device=device)


class GroupedLSAMatcher:
    """Persistent CPU pool for color-local, time-gated LSA chunks."""

    def __init__(self, workers: int | None = None):
        available = available_cpu_count()
        if workers is not None and workers <= 0:
            raise ValueError("workers must be positive or None")
        self.max_workers = available if workers is None else min(workers, available)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._closed = False

    def submit(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        colors: torch.Tensor,
        time: torch.Tensor,
        *,
        matched: bool,
    ) -> MatchHandle:
        if self._closed:
            raise RuntimeError("GroupedLSAMatcher is closed")
        if data.shape != noise.shape or data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError("data and noise must share shape (B,N,3)")
        if colors.shape != data.shape[:2]:
            raise ValueError("colors must have shape (B,N)")
        if time.shape != (data.shape[0],):
            raise ValueError("time must have shape (B,)")

        batch_size, num_tiles = data.shape[:2]
        identity = np.broadcast_to(
            np.arange(num_tiles, dtype=np.int64), (batch_size, num_tiles)
        ).copy()
        if not matched:
            return MatchHandle(identity, [], ())

        time_cpu = time.detach().to("cpu").numpy()
        active = np.flatnonzero(time_cpu < MATCH_TIME_THRESHOLD)
        if not len(active):
            return MatchHandle(identity, [], ())
        colors_cpu = colors.detach().to("cpu").numpy()
        data_cpu = data.detach().to("cpu").numpy()
        noise_cpu = noise.detach().to("cpu").numpy()
        futures = []
        task_sizes = []
        for batch_index in active:
            for color in np.unique(colors_cpu[batch_index]):
                color_indices = np.flatnonzero(
                    colors_cpu[batch_index] == color
                )
                for start in range(0, len(color_indices), MAX_LSA_GROUP_SIZE):
                    indices = color_indices[start : start + MAX_LSA_GROUP_SIZE]
                    task_sizes.append(len(indices))
                    futures.append(
                        self._executor.submit(
                            _chunk_assignment,
                            int(batch_index),
                            indices,
                            data_cpu[batch_index],
                            noise_cpu[batch_index],
                        )
                    )
        return MatchHandle(identity, futures, tuple(task_sizes))

    def effective_workers(self, task_count: int) -> int:
        return max(0, min(self.max_workers, task_count))

    def shutdown(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> "GroupedLSAMatcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


@dataclass
class PendingFlowBatch:
    data: torch.Tensor
    noise: torch.Tensor
    time: torch.Tensor
    time_xy: torch.Tensor
    time_angle: torch.Tensor
    handle: MatchHandle

    def resolve(self) -> FlowBatch:
        permutation = self.handle.permutation(self.noise.device)
        ordered_noise = torch.gather(
            self.noise, 1, permutation[..., None].expand_as(self.noise)
        )
        return FlowBatch(
            noise=ordered_noise,
            data=self.data,
            state=coordinate_flow_state(
                ordered_noise, self.data, self.time_xy, self.time_angle
            ),
            time=self.time,
            time_xy=self.time_xy,
            time_angle=self.time_angle,
        )


def submit_flow_batch(
    data: torch.Tensor,
    noise: torch.Tensor,
    colors: torch.Tensor,
    time: torch.Tensor,
    matcher: GroupedLSAMatcher,
    *,
    matched: bool,
    time_xy: torch.Tensor | None = None,
    time_angle: torch.Tensor | None = None,
) -> PendingFlowBatch:
    handle = matcher.submit(data, noise, colors, time, matched=matched)
    return PendingFlowBatch(
        data=data,
        noise=noise,
        time=time,
        time_xy=time if time_xy is None else time_xy,
        time_angle=time if time_angle is None else time_angle,
        handle=handle,
    )


def endpoint_loss(prediction: torch.Tensor, target: torch.Tensor, loss: str) -> LossTerms:
    return reconstruction_loss(prediction, target, loss=loss)
