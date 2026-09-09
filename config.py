"""Typed YAML configuration with strict section and CLI overrides."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
T = TypeVar("T")


@dataclass
class SpurConfig:
    symmetry: int = 5
    num_tiles: int = 96
    num_ret_tiles: int = 96
    translation_canvas: float | None = None
    seed: int | None = None
    rotation_canvas: float = math.pi
    rotation_mask: float = math.pi / 4


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 8
    num_global_tokens: int = 4
    dropout: float = 0.0


@dataclass
class FlowConfig:
    matched: bool = True
    schedule: str | None = None
    jitter_xy: float = 1.0
    jitter_angle: float = 1.0
    r: float = 2.0
    k: float = 8.0
    tail_lim: float = 0.9375
    loss: str = "l2"
    lsa_workers: int | None = None


@dataclass
class TrainConfig:
    batch_size: int = 64
    samples_per_epoch: int = 140_000
    num_epochs: int = 101
    learning_rate: float = 1e-3
    warmup_epochs: int | None = None
    weight_decay: float = 0.01
    min_lr_factor: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"


@dataclass
class ReverseConfig:
    n: int = 1
    seed: int = 1
    num_iters: int = 1
    time: float = 0.0


@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "penrose-bream"
    run_name: str | None = None
    run_id: str | None = None


@dataclass
class OutputConfig:
    directory: str = "outputs"
    resume: str | None = None


@dataclass
class Config:
    spur: SpurConfig
    model: ModelConfig
    flow: FlowConfig
    train: TrainConfig
    reverse: ReverseConfig
    wandb: WandbConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SECTIONS: dict[str, type[Any]] = {
    "spur": SpurConfig,
    "model": ModelConfig,
    "flow": FlowConfig,
    "train": TrainConfig,
    "reverse": ReverseConfig,
    "wandb": WandbConfig,
    "output": OutputConfig,
}

IMMUTABLE_ON_RESUME = (
    "spur.symmetry",
    "spur.num_tiles",
    "spur.num_ret_tiles",
    "spur.translation_canvas",
    "spur.seed",
    "spur.rotation_canvas",
    "spur.rotation_mask",
    "model.d_model",
    "model.num_heads",
    "model.num_layers",
    "model.num_global_tokens",
    "model.dropout",
    "flow.matched",
    "flow.schedule",
    "flow.jitter_xy",
    "flow.jitter_angle",
    "flow.r",
    "flow.k",
    "flow.tail_lim",
    "flow.loss",
    "flow.lsa_workers",
    "train.batch_size",
    "train.samples_per_epoch",
    "train.learning_rate",
    "train.warmup_epochs",
    "train.weight_decay",
    "train.min_lr_factor",
    "train.grad_clip",
    "train.seed",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle) or {}
    if not isinstance(result, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return result


def _merge(target: dict[str, Any], source: dict[str, Any], prefix: str = "") -> None:
    for key, value in source.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise ValueError(f"Unknown configuration key: {dotted}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping for {dotted}")
            _merge(target[key], value, dotted)
        else:
            target[key] = value


def _override(target: dict[str, Any], section: str | None, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got {expression!r}")
    dotted, raw = expression.split("=", 1)
    if section is not None and "." not in dotted:
        dotted = f"{section}.{dotted}"
    keys = dotted.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise ValueError(f"Unknown configuration key: {dotted}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise ValueError(f"Unknown configuration key: {dotted}")
    cursor[keys[-1]] = yaml.safe_load(raw)


def _make_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(values: dict[str, Any]) -> Config:
    missing = set(SECTIONS) - set(values)
    unknown = set(values) - set(SECTIONS)
    if missing or unknown:
        raise ValueError(
            f"Configuration sections mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    config = Config(
        **{
            name: _make_dataclass(section_type, copy.deepcopy(values[name]))
            for name, section_type in SECTIONS.items()
        }
    )
    validate(config)
    return config


def validate(config: Config) -> None:
    if config.spur.symmetry not in (5, 6):
        raise ValueError("spur.symmetry must be 5 or 6")
    if min(config.spur.num_tiles, config.spur.num_ret_tiles) <= 1:
        raise ValueError("tile counts must exceed one")
    if config.model.d_model % config.model.num_heads:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if min(config.model.num_layers, config.model.num_global_tokens) <= 0:
        raise ValueError("model layer and global-token counts must be positive")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0,1)")
    if config.flow.schedule not in (
        None,
        "uniform",
        "exponential",
        "sine",
        "quadratic",
        "tail",
    ):
        raise ValueError(
            "flow.schedule must be null, uniform, exponential, sine, quadratic, or tail"
        )
    if config.flow.jitter_xy <= 0 or config.flow.jitter_angle <= 0:
        raise ValueError("flow jitter targets must be positive")
    if config.flow.loss not in ("l1", "l2"):
        raise ValueError("flow.loss must be l1 or l2")
    if config.flow.r <= 1 or config.flow.k <= 0:
        raise ValueError("flow.r must exceed 1 and flow.k must be positive")
    if not 0.0 <= config.flow.tail_lim < 1.0:
        raise ValueError("flow.tail_lim must be in [0,1)")
    if config.flow.lsa_workers is not None and config.flow.lsa_workers <= 0:
        raise ValueError("flow.lsa_workers must be positive or null")
    if min(
        config.train.batch_size,
        config.train.samples_per_epoch,
        config.train.num_epochs,
        config.reverse.n,
        config.reverse.num_iters,
    ) <= 0:
        raise ValueError("all count settings must be positive")
    if config.train.learning_rate <= 0 or config.train.grad_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if config.train.warmup_epochs is not None and not (
        0 <= config.train.warmup_epochs < config.train.num_epochs
    ):
        raise ValueError("train.warmup_epochs must be in [0, num_epochs)")
    if config.train.weight_decay < 0:
        raise ValueError("train.weight_decay must be nonnegative")
    if not 0.0 < config.train.min_lr_factor <= 1.0:
        raise ValueError("train.min_lr_factor must be in (0,1]")
    if not 0.0 <= config.reverse.time <= 1.0:
        raise ValueError("reverse.time must be in [0,1]")


def effective_translation(config: SpurConfig) -> float:
    if config.translation_canvas is not None:
        return float(config.translation_canvas)
    return 2.0


def batches_per_epoch(config: Config) -> int:
    return math.ceil(config.train.samples_per_epoch / config.train.batch_size)


def make_identifier(config: Config, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    identifier = (
        f"bream{config.spur.num_ret_tiles}_{now:%m%d}_{now:%H%M}_"
        f"{config.model.d_model}x{config.model.num_layers}"
    )
    if config.flow.schedule != "tail" and not config.flow.matched:
        identifier += "_um"
    if config.flow.loss == "l1":
        identifier += "_l1"
    if config.flow.schedule is None:
        def compact(value: float) -> str:
            return f"{value:g}".replace(".", "p")

        identifier += f"_j{compact(config.flow.jitter_xy)}"
        if config.flow.jitter_angle != config.flow.jitter_xy:
            identifier += f"x{compact(config.flow.jitter_angle)}"
    elif config.flow.schedule != "exponential":
        identifier += f"_{config.flow.schedule}"
    return identifier


def nested_value(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        value = value[key]
    return value


def validate_resume_config(config: Config, saved: dict[str, Any]) -> None:
    current = config.to_dict()
    saved = copy.deepcopy(saved)
    saved.setdefault("train", {}).setdefault("warmup_epochs", None)
    saved.setdefault("flow", {}).setdefault("jitter_xy", 1.0)
    saved["flow"].setdefault("jitter_angle", 1.0)
    changed = [
        key
        for key in IMMUTABLE_ON_RESUME
        if nested_value(current, key) != nested_value(saved, key)
    ]
    if changed:
        details = ", ".join(
            f"{key}: checkpoint={nested_value(saved, key)!r}, "
            f"current={nested_value(current, key)!r}"
            for key in changed
        )
        raise ValueError(f"Immutable resume configuration changed: {details}")


def _append_overrides(
    merged: dict[str, Any], section: str, expressions: list[str] | None
) -> None:
    for expression in expressions or []:
        _override(merged, section, expression)


def load_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train PenroseBream")
    parser.add_argument("--config", type=Path, help="Optional experiment YAML")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Checkpoint to resume")
    initialization.add_argument(
        "--init-weights",
        type=Path,
        help="Load model weights while resetting all optimization state",
    )
    parser.add_argument("--symmetry", type=int)
    parser.add_argument("--num-tiles", type=int)
    parser.add_argument("--translation", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output")
    parser.add_argument("--n", type=int)
    parser.add_argument("--num-iters", type=int)
    parser.add_argument("--time", type=float)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "-s",
        "--time-schedule",
        choices=("uniform", "exponential", "sine", "quadratic", "tail"),
    )
    sampling.add_argument(
        "--jitter",
        nargs="*",
        type=float,
        metavar="J",
        help="Calibrated jitter: no values uses 1, one shares J, two set XY and angle",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("-t", "--train", action="append", metavar="KEY=VALUE")
    parser.add_argument("-m", "--model", action="append", metavar="KEY=VALUE")
    parser.add_argument("-p", "--spur", action="append", metavar="KEY=VALUE")
    parser.add_argument("-f", "--flow", action="append", metavar="KEY=VALUE")
    parser.add_argument("-r", "--reverse", action="append", metavar="KEY=VALUE")
    parser.add_argument("--wandb", action="append", metavar="KEY=VALUE")
    parser.add_argument("overrides", nargs="*", help="Dotted key=value overrides")
    args = parser.parse_args(argv)

    merged = _read_yaml(PROJECT_DIR / "config.yaml")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _merge(merged, checkpoint["config"])
    if args.config:
        _merge(merged, _read_yaml(args.config))
    for section, expressions in (
        ("train", args.train),
        ("model", args.model),
        ("spur", args.spur),
        ("flow", args.flow),
        ("reverse", args.reverse),
        ("wandb", args.wandb),
    ):
        _append_overrides(merged, section, expressions)
    for expression in args.overrides:
        _override(merged, None, expression)

    conveniences = {
        ("spur", "symmetry"): args.symmetry,
        ("spur", "num_tiles"): args.num_tiles,
        ("spur", "translation_canvas"): args.translation,
        ("train", "batch_size"): args.batch_size,
        ("train", "learning_rate"): args.learning_rate,
        ("output", "directory"): args.output,
        ("reverse", "n"): args.n,
        ("reverse", "num_iters"): args.num_iters,
        ("reverse", "time"): args.time,
        ("flow", "schedule"): args.time_schedule,
        ("wandb", "project"): args.wandb_project,
        ("wandb", "run_name"): args.wandb_name,
    }
    for (section, key), value in conveniences.items():
        if value is not None:
            merged[section][key] = value
    if args.jitter is not None:
        if len(args.jitter) > 2:
            parser.error("--jitter accepts at most two values: XY [ANGLE]")
        jitter = args.jitter or [1.0]
        merged["flow"]["schedule"] = None
        merged["flow"]["jitter_xy"] = jitter[0]
        merged["flow"]["jitter_angle"] = jitter[-1]
    if args.num_tiles is not None:
        merged["spur"]["num_ret_tiles"] = args.num_tiles
    if args.resume:
        merged["output"]["resume"] = str(args.resume)

    config = config_from_dict(merged)
    print(yaml.safe_dump(config.to_dict(), sort_keys=False).rstrip())
    return config, args
