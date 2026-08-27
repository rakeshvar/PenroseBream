"""Paired PenroseSpur evaluation, flow construction, metrics, and SVGs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch

from config import Config, config_from_dict, effective_translation
from denoiser import DirectTransformer
from diffuser import (
    FlowBatch,
    GroupedLSAMatcher,
    PendingFlowBatch,
    draw_time,
    endpoint_loss,
    submit_flow_batch,
)
from sampler import sample


PROJECT_DIR = Path(__file__).resolve().parent
SPUR_DIR = Path(
    os.environ.get("PENROSE_SPUR_PATH", PROJECT_DIR.parent / "PenroseSpur")
).resolve()
if not (SPUR_DIR / "sampler.py").exists():
    raise ImportError(f"PenroseSpur not found at {SPUR_DIR}; set PENROSE_SPUR_PATH")
sys.path.insert(0, str(SPUR_DIR))
from show import (  # noqa: E402
    render_flow_comparison_svg as _show_comparison_svg,
    save_comparison_svg as _show_save_comparison_svg,
    save_evaluation_set as _show_save_evaluation_set,
    xya_shared_viewbox as _show_shared_viewbox,
)


def _load_spur_file(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"penrose_spur_{name}", SPUR_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PenroseSpur module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_spur_sampler = _load_spur_file("sampler")
_spur_lattice_loss = _load_spur_file("lattice_loss")
SpurSampler = _spur_sampler.SpurSampler
ANGLE_SCALE = _spur_sampler.ANGLE_SCALE
lattice_loss = _spur_lattice_loss.lattice_loss


@dataclass
class EvaluationBatch:
    flow: FlowBatch
    colors: torch.Tensor
    labels: torch.Tensor


@dataclass
class RawEvaluationBatch:
    data: torch.Tensor
    noise: torch.Tensor
    time: torch.Tensor
    schedule: str
    colors: torch.Tensor
    labels: torch.Tensor


@dataclass
class PendingEvaluationBatch:
    flow: PendingFlowBatch
    colors: torch.Tensor
    labels: torch.Tensor

    def resolve(self) -> EvaluationBatch:
        return EvaluationBatch(
            flow=self.flow.resolve(),
            colors=self.colors,
            labels=self.labels,
        )


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def build_spur(config: Config, device: torch.device) -> Any:
    return SpurSampler(
        symmetry=config.spur.symmetry,
        num_tiles=config.spur.num_tiles,
        num_ret_tiles=config.spur.num_ret_tiles,
        translation_canvas=effective_translation(config.spur),
        seed=config.spur.seed,
        device=device,
        rotation_canvas=config.spur.rotation_canvas,
        rotation_mask=config.spur.rotation_mask,
    )


def sample_raw_flow_batch(
    spur: Any,
    config: Config,
    batch_size: int,
    generator: torch.Generator,
    *,
    time: float | None = None,
    schedule: str | None = None,
) -> RawEvaluationBatch:
    batch = spur.sample_batch(batch_size, generator=generator)
    data = batch["xya"].float()
    colors = batch["colors"].long()
    noise = spur.sample_noise(batch_size, generator=generator).float()
    selected_schedule = schedule or config.flow.schedule
    if time is None:
        times = draw_time(
            batch_size,
            data.device,
            data.dtype,
            generator,
            selected_schedule,
            r=config.flow.r,
            k=config.flow.k,
            tail_lim=config.flow.tail_lim,
        )
    else:
        if not 0.0 <= time <= 1.0:
            raise ValueError("time must be in [0,1]")
        times = torch.full(
            (batch_size,), time, device=data.device, dtype=data.dtype
        )
    return RawEvaluationBatch(
        data=data,
        noise=noise,
        time=times,
        schedule=selected_schedule,
        colors=colors,
        labels=batch["labels"].long(),
    )


def submit_evaluation_batch(
    raw: RawEvaluationBatch,
    config: Config,
    matcher: GroupedLSAMatcher,
    *,
    matched: bool | None = None,
) -> PendingEvaluationBatch:
    requested_match = config.flow.matched if matched is None else matched
    pending = submit_flow_batch(
        raw.data,
        raw.noise,
        raw.colors,
        raw.time,
        matcher,
        matched=requested_match and raw.schedule != "tail",
    )
    return PendingEvaluationBatch(
        flow=pending,
        colors=raw.colors,
        labels=raw.labels,
    )


def sample_flow_batch(
    spur: Any,
    config: Config,
    batch_size: int,
    generator: torch.Generator,
    *,
    time: float | None = None,
    schedule: str | None = None,
    matched: bool | None = None,
    matcher: GroupedLSAMatcher | None = None,
) -> EvaluationBatch:
    raw = sample_raw_flow_batch(
        spur,
        config,
        batch_size,
        generator,
        time=time,
        schedule=schedule,
    )
    if matcher is not None:
        return submit_evaluation_batch(
            raw, config, matcher, matched=matched
        ).resolve()
    with GroupedLSAMatcher(config.flow.lsa_workers) as owned_matcher:
        return submit_evaluation_batch(
            raw, config, owned_matcher, matched=matched
        ).resolve()


def shared_viewbox(
    geometries: list[torch.Tensor],
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> tuple[float, float, float, float]:
    return _show_shared_viewbox(
        geometries,
        colors,
        symmetry,
        side,
        angle_scale=ANGLE_SCALE,
    ).as_tuple()


def _comparison_svg(
    initial: torch.Tensor,
    produced: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
    mode: str,
    viewbox: tuple[float, float, float, float] | None = None,
) -> str:
    return _show_comparison_svg(
        initial,
        produced,
        colors,
        symmetry,
        side,
        viewbox,
        angle_scale=ANGLE_SCALE,
        mode=mode,
    )


def save_comparison_svg(
    path: Path,
    initial: torch.Tensor,
    produced: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
    viewbox: tuple[float, float, float, float] | None = None,
) -> Path:
    return _show_save_comparison_svg(
        path,
        initial,
        produced,
        colors,
        symmetry,
        side,
        viewbox,
        angle_scale=ANGLE_SCALE,
    )


def _safe_class_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-")
    return safe or "unknown"


def save_evaluation_svgs(
    output: Path,
    class_name: str,
    seed: int,
    initial: torch.Tensor,
    produced: list[torch.Tensor],
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> tuple[Path, ...]:
    stem = f"breamed_{_safe_class_name(class_name)}_{seed}"
    return _show_save_evaluation_set(
        output,
        stem,
        initial,
        produced,
        colors,
        symmetry,
        side,
        angle_scale=ANGLE_SCALE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired PenroseBream evaluation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-n", "--count", type=int)
    times = parser.add_mutually_exclusive_group()
    times.add_argument("-t", "--time", type=float)
    times.add_argument(
        "-s",
        "--time-schedule",
        choices=("uniform", "exponential", "sine", "quadratic", "tail"),
    )
    parser.add_argument("-u", "--unmatched", action="store_true")
    parser.add_argument("-i", "--num-iters", type=int)
    parser.add_argument("-r", "--seed", type=int)
    parser.add_argument("-y", "--symmetry", type=int, choices=(5, 6))
    parser.add_argument("-N", "--num-tiles", type=int)
    parser.add_argument("-x", "--translation", type=float)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    values = copy.deepcopy(checkpoint["config"])
    if args.symmetry is not None:
        values["spur"]["symmetry"] = args.symmetry
    if args.num_tiles is not None:
        values["spur"]["num_tiles"] = args.num_tiles
        values["spur"]["num_ret_tiles"] = args.num_tiles
    if args.translation is not None:
        values["spur"]["translation_canvas"] = args.translation
    config = config_from_dict(values)
    device = choose_device(args.device)
    spur = build_spur(config, device)
    model = DirectTransformer(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    count = config.reverse.n if args.count is None else args.count
    iterations = config.reverse.num_iters if args.num_iters is None else args.num_iters
    seed = config.reverse.seed if args.seed is None else args.seed
    if count <= 0 or iterations <= 0:
        raise ValueError("count and num_iters must be positive")
    with GroupedLSAMatcher(config.flow.lsa_workers) as matcher:
        for index in range(count):
            sample_seed = seed + index
            generator = make_generator(device, sample_seed)
            evaluation = sample_flow_batch(
                spur,
                config,
                1,
                generator,
                time=(
                    config.reverse.time
                    if args.time is None and args.time_schedule is None
                    else args.time
                ),
                schedule=args.time_schedule,
                matched=False if args.unmatched else None,
                matcher=matcher,
            )
            produced = sample(
                model, evaluation.flow.state, evaluation.colors, iterations
            )
            class_name = spur.class_names[int(evaluation.labels[0].item())]
            save_evaluation_svgs(
                args.output,
                class_name,
                sample_seed,
                evaluation.flow.state[0],
                [item[0] for item in produced],
                evaluation.colors[0],
                config.spur.symmetry,
                spur.side,
            )
            for iteration, result in enumerate(produced, start=1):
                terms = endpoint_loss(result, evaluation.flow.data, config.flow.loss)
                print(
                    f"seed={sample_seed} iteration={iteration} "
                    f"t={evaluation.flow.time.item():.6f} "
                    f"loss={terms.total.item():.6f}"
                )
    print(f"Saved {count} paired evaluation sets to {args.output}")


if __name__ == "__main__":
    main()
