"""Train the single PenroseBream direct reconstruction model."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from checkpoint import capture_rng, restore_rng, retain_newest_and_best, save_epoch
from config import (
    Config,
    batches_per_epoch,
    load_config,
    make_identifier,
    validate_resume_config,
)
from denoiser import DirectTransformer
from diffuser import endpoint_loss, flow_state
from evaluator import (
    build_spur,
    choose_device,
    lattice_loss,
    make_generator,
    sample_flow_batch,
    save_comparison_svg,
    shared_viewbox,
)
from sampler import sample


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config) -> LambdaLR:
    epochs = config.train.num_epochs
    warmup = min(10, math.floor(0.05 * epochs))
    floor = config.train.min_lr_factor

    def factor(position: int) -> float:
        if warmup and position <= warmup:
            return 0.01 + 0.99 * position / warmup
        if position <= epochs:
            progress = (position - warmup) / max(1, epochs - warmup)
            return floor + (1.0 - floor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return floor

    return LambdaLR(optimizer, lr_lambda=factor)


class WandbLogger:
    def __init__(
        self,
        config: Config,
        run_name: str,
        run_id: str | None,
        resume_existing: bool,
        parameter_counts: dict[str, int],
    ):
        self.run = None
        if not config.wandb.enable:
            return
        if not os.environ.get("WANDB_API_KEY"):
            print("WandB disabled: WANDB_API_KEY is not set")
            return
        try:
            import wandb
        except ModuleNotFoundError:
            print("WandB disabled: package is not installed")
            return
        kwargs: dict[str, Any] = {
            "project": config.wandb.project,
            "name": run_name,
            "config": {**config.to_dict(), "parameter_counts": parameter_counts},
        }
        if run_id:
            kwargs.update(
                id=run_id,
                resume="must" if resume_existing else "never",
            )
        self.run = wandb.init(**kwargs)

    @property
    def run_id(self) -> str | None:
        return self.run.id if self.run is not None else None

    def log(self, metrics: dict[str, float], epoch: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=epoch)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def train(config: Config) -> Path | None:
    seed_everything(config.train.seed)
    device = choose_device(config.train.device)
    resume_path = Path(config.output.resume) if config.output.resume else None
    resume = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path
        else None
    )
    if resume:
        validate_resume_config(config, resume["config"])

    spur = build_spur(config, device)
    model = DirectTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    training_generator = make_generator(device, config.train.seed)
    sampling_generator = make_generator(device, config.reverse.seed)

    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        restore_rng(resume["rng"], training_generator, sampling_generator)
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume["global_step"])
        identifier = str(resume["identifier"])
        output_directory = Path(resume["output_directory"])
        run_name = str(resume["wandb_run_name"])
        wandb_run_id = resume.get("wandb_run_id")
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_primary_metric"])
    else:
        start_epoch = 0
        global_step = 0
        identifier = make_identifier(config)
        output_directory = Path(config.output.directory) / identifier
        run_name = config.wandb.run_name or identifier
        wandb_run_id = config.wandb.run_id or identifier
        best_epoch = -1
        best_loss = math.inf

    checkpoint_directory = output_directory / "checkpoints"
    svg_directory = output_directory / "svg"
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger = WandbLogger(
        config,
        run_name,
        wandb_run_id,
        resume_existing=resume is not None,
        parameter_counts={
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
    )
    steps = batches_per_epoch(config)
    actual_samples = steps * config.train.batch_size
    print(f"Device: {device}")
    print(f"Identifier: {identifier}")
    print(f"Output: {output_directory}")
    print(f"Samples per epoch: {actual_samples} ({steps} batches)")

    last_checkpoint: Path | None = resume_path
    try:
        for epoch in range(start_epoch, config.train.num_epochs):
            model.train()
            loss_sum = 0.0
            xy_sum = 0.0
            angle_sum = 0.0
            gradient_norm_sum = 0.0
            time_sum = 0.0
            for _ in range(steps):
                prepared = sample_flow_batch(
                    spur,
                    config,
                    config.train.batch_size,
                    training_generator,
                )
                prediction = model(prepared.flow.state, prepared.colors)
                terms = endpoint_loss(
                    prediction, prepared.flow.data, config.flow.loss
                )

                optimizer.zero_grad(set_to_none=True)
                terms.total.backward()
                gradient_norm = clip_grad_norm_(
                    model.parameters(), config.train.grad_clip
                )
                optimizer.step()
                global_step += 1
                loss_sum += terms.total.detach().item()
                xy_sum += terms.xy.detach().item()
                angle_sum += terms.angle.detach().item()
                gradient_norm_sum += gradient_norm.detach().item()
                time_sum += prepared.flow.time.mean().item()

            scheduler.step()
            metrics = {
                "average_training_loss": loss_sum / steps,
                "xy_loss": xy_sum / steps,
                "scaled_angle_loss": angle_sum / steps,
                "gradient_norm": gradient_norm_sum / steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "average_time": time_sum / steps,
                "average_noise_fraction": 1.0 - time_sum / steps,
            }
            if metrics["average_training_loss"] < best_loss:
                best_loss = metrics["average_training_loss"]
                best_epoch = epoch

            epoch_pair = sample_flow_batch(
                spur,
                config,
                1,
                sampling_generator,
                time=0.5,
            )
            state_050 = epoch_pair.flow.state
            state_095 = flow_state(
                epoch_pair.flow.noise,
                epoch_pair.flow.data,
                torch.full_like(epoch_pair.flow.time, 0.95),
            )
            produced_050 = sample(model, state_050, epoch_pair.colors, 1)[0]
            produced_095 = sample(model, state_095, epoch_pair.colors, 1)[0]
            metrics["lattice_loss_t050"] = lattice_loss(
                config.spur.symmetry,
                spur.side,
                produced_050,
                epoch_pair.colors,
            ).item()
            metrics["lattice_loss_t095"] = lattice_loss(
                config.spur.symmetry,
                spur.side,
                produced_095,
                epoch_pair.colors,
            ).item()
            checkpoint_id = f"{identifier}_e{epoch:03d}"
            epoch_viewbox = shared_viewbox(
                [
                    state_050[0],
                    produced_050[0],
                    state_095[0],
                    produced_095[0],
                ],
                epoch_pair.colors[0],
                config.spur.symmetry,
                spur.side,
            )
            save_comparison_svg(
                svg_directory / f"{checkpoint_id}_t050.svg",
                state_050[0],
                produced_050[0],
                epoch_pair.colors[0],
                config.spur.symmetry,
                spur.side,
                epoch_viewbox,
            )
            save_comparison_svg(
                svg_directory / f"{checkpoint_id}_t095.svg",
                state_095[0],
                produced_095[0],
                epoch_pair.colors[0],
                config.spur.symmetry,
                spur.side,
                epoch_viewbox,
            )
            logger.log(metrics, epoch)
            print(
                f"Epoch {epoch:03d} loss={metrics['average_training_loss']:.6f} "
                f"xy={metrics['xy_loss']:.6f} "
                f"angle={metrics['scaled_angle_loss']:.6f} "
                f"t={metrics['average_time']:.6f} "
                f"lattice050={metrics['lattice_loss_t050']:.6f} "
                f"lattice095={metrics['lattice_loss_t095']:.6f} "
                f"grad={metrics['gradient_norm']:.6f} "
                f"lr={metrics['learning_rate']:.6g}"
            )

            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "average_training_loss": metrics["average_training_loss"],
                "metrics": metrics,
                "best_epoch": best_epoch,
                "best_primary_metric": best_loss,
                "model": model.state_dict(),
                "flow": {
                    "matched": config.flow.matched,
                    "schedule": config.flow.schedule,
                    "r": config.flow.r,
                    "k": config.flow.k,
                    "loss": config.flow.loss,
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.to_dict(),
                "symmetry": config.spur.symmetry,
                "side": spur.side,
                "num_tiles": config.spur.num_tiles,
                "num_ret_tiles": config.spur.num_ret_tiles,
                "num_classes": len(spur.class_names),
                "class_lookup": list(spur.class_names),
                "identifier": identifier,
                "output_directory": str(output_directory),
                "wandb_run_name": run_name,
                "wandb_run_id": logger.run_id or wandb_run_id,
                "rng": capture_rng(training_generator, sampling_generator),
            }
            last_checkpoint = save_epoch(
                payload, checkpoint_directory, identifier, epoch
            )
            retain_newest_and_best(
                checkpoint_directory, identifier, epoch, best_epoch
            )
    finally:
        logger.finish()
    return last_checkpoint


def main() -> None:
    config, _ = load_config()
    train(config)


if __name__ == "__main__":
    main()
