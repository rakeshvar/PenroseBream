"""Paired PenroseSpur evaluation, flow construction, metrics, and SVGs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from html import escape
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

from config import Config, config_from_dict, effective_translation
from denoiser import DirectTransformer
from diffuser import (
    FlowBatch,
    draw_time,
    endpoint_loss,
    prepare_flow_batch,
)
from sampler import sample


PROJECT_DIR = Path(__file__).resolve().parent
SPUR_DIR = Path(
    os.environ.get("PENROSE_SPUR_PATH", PROJECT_DIR.parent / "PenroseSpur")
).resolve()
if not (SPUR_DIR / "sampler.py").exists():
    raise ImportError(f"PenroseSpur not found at {SPUR_DIR}; set PENROSE_SPUR_PATH")
sys.path.insert(0, str(SPUR_DIR))


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
_spur_match = _load_spur_file("match")
_spur_lattice_loss = _load_spur_file("lattice_loss")
_spur_convert = _load_spur_file("convert")
_spur_svg = _load_spur_file("svg")
SpurSampler = _spur_sampler.SpurSampler
ANGLE_SCALE = _spur_sampler.ANGLE_SCALE
lattice_loss = _spur_lattice_loss.lattice_loss
PALETTES = _spur_svg.PALETTES


@dataclass
class EvaluationBatch:
    flow: FlowBatch
    colors: torch.Tensor
    labels: torch.Tensor


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


def sample_flow_batch(
    spur: Any,
    config: Config,
    batch_size: int,
    generator: torch.Generator,
    *,
    time: float | None = None,
    schedule: str | None = None,
    matched: bool | None = None,
) -> EvaluationBatch:
    batch = spur.sample_batch(batch_size, generator=generator)
    data = batch["xya"].float()
    colors = batch["colors"].long()
    noise = spur.sample_noise(batch_size, generator=generator).float()
    if time is None:
        times = draw_time(
            batch_size,
            data.device,
            data.dtype,
            generator,
            schedule or config.flow.schedule,
            r=config.flow.r,
            k=config.flow.k,
        )
    else:
        if not 0.0 <= time <= 1.0:
            raise ValueError("time must be in [0,1]")
        times = torch.full(
            (batch_size,), time, device=data.device, dtype=data.dtype
        )
    flow = prepare_flow_batch(
        data,
        noise,
        times,
        matched=config.flow.matched if matched is None else matched,
        lsa=_spur_match.lsa,
        workers=config.flow.lsa_workers,
    )
    return EvaluationBatch(flow=flow, colors=colors, labels=batch["labels"].long())


def _polygons(
    geometry: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> np.ndarray:
    values = geometry.detach().cpu().numpy()
    color_values = colors.detach().cpu().numpy()
    return _spur_convert.vertices(
        symmetry,
        values[:, :2],
        values[:, 2] / ANGLE_SCALE,
        color_values,
        side,
    )


def _points(polygon: np.ndarray) -> str:
    return " ".join(f"{x:.5f},{-y:.5f}" for x, y in polygon)


def shared_viewbox(
    geometries: list[torch.Tensor],
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> tuple[float, float, float, float]:
    polygons = [_polygons(item, colors, symmetry, side) for item in geometries]
    all_vertices = np.concatenate(polygons, axis=0)
    minimum = all_vertices.min(axis=(0, 1))
    maximum = all_vertices.max(axis=(0, 1))
    padding = max(float((maximum - minimum).max()) * 0.04, side * 0.5)
    xmin = float(minimum[0] - padding)
    xmax = float(maximum[0] + padding)
    ymin = float(-maximum[1] - padding)
    ymax = float(-minimum[1] + padding)
    return xmin, ymin, max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)


def _comparison_svg(
    initial: torch.Tensor,
    produced: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
    mode: str,
    viewbox: tuple[float, float, float, float] | None = None,
) -> str:
    if mode not in {"noised", "overlaid", "produced"}:
        raise ValueError(f"Unknown SVG mode: {mode}")
    initial_np = initial.detach().cpu().numpy()
    produced_np = produced.detach().cpu().numpy()
    color_values = colors.detach().cpu().numpy()
    initial_polygons = _polygons(initial, colors, symmetry, side)
    produced_polygons = _polygons(produced, colors, symmetry, side)
    if viewbox is None:
        viewbox = shared_viewbox([initial, produced], colors, symmetry, side)
    xmin, ymin, width, height = viewbox
    display_width = max(1, round(1080 * width / height))
    initial_stroke = max(side / 18.0, width / 1500.0)
    produced_stroke = max(side / 70.0, width / 6000.0)
    line_width = max(side / 65.0, width / 5000.0)
    palette_styles = []
    for index, color in enumerate(PALETTES[symmetry]):
        if color is not None:
            palette_styles.extend(
                (
                    f".fill{index} {{ fill: {color}; }}",
                    f".outline{index} {{ stroke: {color}; }}",
                )
            )
    lines = [
        (
            f'<line class="correspondence" x1="{start[0]:.5f}" y1="{-start[1]:.5f}" '
            f'x2="{end[0]:.5f}" y2="{-end[1]:.5f}"/>'
        )
        for start, end in zip(initial_np[:, :2], produced_np[:, :2])
    ]
    outlines = [
        f'<polygon class="noised outline{int(color)}" points="{_points(polygon)}"/>'
        for polygon, color in zip(initial_polygons, color_values)
    ]
    filled_initial = [
        f'<polygon class="noised-filled fill{int(color)}" points="{_points(polygon)}"/>'
        for polygon, color in zip(initial_polygons, color_values)
    ]
    filled_produced = [
        f'<polygon class="reconstructed fill{int(color)}" points="{_points(polygon)}"/>'
        for polygon, color in zip(produced_polygons, color_values)
    ]
    if mode == "noised":
        elements = filled_initial
    elif mode == "produced":
        elements = filled_produced
    else:
        elements = [*filled_produced, *lines, *outlines]
    return f"""<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1"
     preserveAspectRatio="xMidYMid meet" width="{display_width}" height="1080"
     viewBox="{xmin:.5f} {ymin:.5f} {width:.5f} {height:.5f}">
  <style>
    {escape(chr(10).join(palette_styles))}
    .noised {{ fill: none; stroke-width: {initial_stroke:.5f}; }}
    .noised-filled {{ stroke: #333333; stroke-width: {produced_stroke:.5f}; }}
    .reconstructed {{ stroke: #333333; stroke-width: {produced_stroke:.5f}; }}
    .correspondence {{ stroke: #777777; stroke-opacity: 0.55;
                       stroke-width: {line_width:.5f}; }}
  </style>
  <rect x="{xmin:.5f}" y="{ymin:.5f}" width="{width:.5f}" height="{height:.5f}"
        fill="white"/>
  <g stroke-linejoin="round" vector-effect="non-scaling-stroke">
    {chr(10).join(elements)}
  </g>
</svg>
"""


def save_comparison_svg(
    path: Path,
    initial: torch.Tensor,
    produced: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
    viewbox: tuple[float, float, float, float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _comparison_svg(
            initial, produced, colors, symmetry, side, "overlaid", viewbox
        ),
        encoding="utf-8",
    )
    return path


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
    if not produced:
        raise ValueError("At least one produced iteration is required")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"breamed_{_safe_class_name(class_name)}_{seed}"
    viewbox = shared_viewbox([initial, *produced], colors, symmetry, side)
    initial_path = output / f"{stem}_noised.svg"
    initial_path.write_text(
        _comparison_svg(
            initial, produced[0], colors, symmetry, side, "noised", viewbox
        ),
        encoding="utf-8",
    )
    paths = [initial_path]
    for iteration, result in enumerate(produced, start=1):
        for mode in ("overlaid", "produced"):
            path = output / f"{stem}_{mode}_i{iteration}.svg"
            path.write_text(
                _comparison_svg(
                    initial, result, colors, symmetry, side, mode, viewbox
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return tuple(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired PenroseBream evaluation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-n", "--count", type=int)
    times = parser.add_mutually_exclusive_group()
    times.add_argument("-t", "--time", type=float)
    times.add_argument(
        "-s",
        "--time-schedule",
        choices=("uniform", "exponential", "sine", "quadratic"),
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
    for index in range(count):
        sample_seed = seed + index
        generator = make_generator(device, sample_seed)
        evaluation = sample_flow_batch(
            spur,
            config,
            1,
            generator,
            time=(config.reverse.time if args.time is None and args.time_schedule is None else args.time),
            schedule=args.time_schedule,
            matched=False if args.unmatched else None,
        )
        produced = sample(model, evaluation.flow.state, evaluation.colors, iterations)
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
                f"t={evaluation.flow.time.item():.6f} loss={terms.total.item():.6f}"
            )
    print(f"Saved {count} paired evaluation sets to {args.output}")


if __name__ == "__main__":
    main()
