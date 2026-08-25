"""End-to-end local smoke checks for PenroseBream flow reconstruction."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from checkpoint import retain_newest_and_best  # noqa: E402
from config import batches_per_epoch, config_from_dict, load_config, make_identifier  # noqa: E402
from denoiser import DirectTransformer  # noqa: E402
from diffuser import (  # noqa: E402
    ANGLE_HALF_PERIOD,
    angle_delta,
    canonicalize_xya,
    draw_time,
    endpoint_loss,
    flow_state,
    periodic_pair_cost,
    schedule_time,
)
from evaluator import (  # noqa: E402
    build_parser,
    build_spur,
    make_generator,
    sample_flow_batch,
    save_evaluation_svgs,
)
from sampler import sample  # noqa: E402


def smoke_values(symmetry: int, output: Path, epochs: int = 1) -> dict:
    with (PROJECT_DIR / "config.yaml").open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values = copy.deepcopy(values)
    values["spur"].update(
        symmetry=symmetry,
        num_tiles=8,
        num_ret_tiles=8,
        translation_canvas=2.0,
        seed=7,
    )
    values["model"].update(
        d_model=16,
        num_heads=4,
        num_layers=1,
        num_global_tokens=2,
        dropout=0.0,
    )
    values["flow"].update(lsa_workers=1)
    values["train"].update(
        batch_size=2,
        samples_per_epoch=2,
        num_epochs=epochs,
        device="cpu",
        seed=11,
    )
    values["reverse"].update(n=1, seed=13, num_iters=2, time=0.0)
    values["wandb"]["enable"] = False
    values["output"]["directory"] = str(output)
    return values


def check_config(root: Path) -> None:
    config = config_from_dict(smoke_values(5, root))
    assert batches_per_epoch(config) == 1
    fixed = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert make_identifier(config, fixed) == "bream8_0102_0304_16x1"
    variants = smoke_values(5, root)
    variants["flow"].update(matched=False, loss="l1", schedule="quadratic")
    assert make_identifier(config_from_dict(variants), fixed).endswith(
        "_16x1_um_l1_quadratic"
    )
    with (PROJECT_DIR / "config.yaml").open(encoding="utf-8") as handle:
        defaults = config_from_dict(yaml.safe_load(handle))
    assert defaults.model.d_model == 128
    assert defaults.model.num_layers == 8
    assert defaults.flow.schedule == "exponential"
    assert defaults.flow.r == 2 and defaults.flow.k == 8
    overridden, _ = load_config(
        ["-t", "batch_size=32", "-m", "d_model=256", "-s", "sine", "-f", "loss=l1"]
    )
    assert overridden.train.batch_size == 32
    assert overridden.model.d_model == 256
    assert overridden.flow.schedule == "sine"
    assert overridden.flow.loss == "l1"
    try:
        load_config(["flow.unknown=1"])
    except ValueError as error:
        assert "Unknown configuration key" in str(error)
    else:
        raise AssertionError("Unknown configuration override was accepted")


def check_schedules_and_geometry() -> None:
    unit = torch.linspace(0, 1, 101)
    for name in ("uniform", "exponential", "sine", "quadratic"):
        time = schedule_time(unit, name, r=2, k=8)
        assert time[0] == 0
        assert time[-1] == 1
        assert bool((time[1:] >= time[:-1]).all())
    generator = torch.Generator().manual_seed(3)
    sampled = draw_time(
        32, torch.device("cpu"), torch.float32, generator, "exponential"
    )
    assert sampled.unique().numel() > 1

    noise = torch.tensor([[[0.0, 0.0, 1.70]]])
    data = torch.tensor([[[2.0, 4.0, -1.70]]])
    t0 = flow_state(noise, data, torch.tensor([0.0]))
    t1 = flow_state(noise, data, torch.tensor([1.0]))
    midpoint = flow_state(noise, data, torch.tensor([0.5]))
    assert torch.allclose(t0, canonicalize_xya(noise))
    assert torch.allclose(t1, canonicalize_xya(data))
    assert abs(angle_delta(data[..., 2], noise[..., 2]).item()) < 0.1
    assert abs(midpoint[..., 2].item()) > 1.69

    source = torch.tensor(
        [[[0.0, 0.0, -1.70], [5.0, 0.0, 1.70]]], dtype=torch.float32
    )
    target = torch.tensor(
        [[[5.0, 0.0, -1.70], [0.0, 0.0, 1.70]]], dtype=torch.float32
    )
    cost = periodic_pair_cost(target, source)
    assert cost.shape == (1, 2, 2)
    assert cost[0, 0, 1] < cost[0, 0, 0]


def check_losses_and_sampler(root: Path) -> None:
    config = config_from_dict(smoke_values(6, root))
    model = DirectTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = torch.randn(2, 8, 3)
    state[..., 2].clamp_(-ANGLE_HALF_PERIOD, ANGLE_HALF_PERIOD)
    colors = torch.randint(0, 2, (2, 8))
    target = torch.randn(2, 8, 3)
    target[..., 2].clamp_(-ANGLE_HALF_PERIOD, ANGLE_HALF_PERIOD)
    for loss_name in ("l1", "l2"):
        prediction = model(state, colors)
        terms = endpoint_loss(prediction, target, loss_name)
        assert torch.isfinite(terms.total)
    optimizer.zero_grad(set_to_none=True)
    endpoint_loss(model(state, colors), target, "l2").total.backward()
    optimizer.step()
    produced = sample(model, state, colors, num_iters=3)
    assert len(produced) == 3
    assert all(item.shape == state.shape for item in produced)
    assert all(
        bool((item[..., 2] >= -ANGLE_HALF_PERIOD).all())
        and bool((item[..., 2] < ANGLE_HALF_PERIOD).all())
        for item in produced
    )


def check_evaluator_and_svgs(root: Path) -> None:
    config = config_from_dict(smoke_values(6, root))
    device = torch.device("cpu")
    spur = build_spur(config, device)
    matched = sample_flow_batch(
        spur, config, 2, make_generator(device, 17), time=0.5, matched=True
    )
    unmatched = sample_flow_batch(
        spur, config, 2, make_generator(device, 17), time=0.5, matched=False
    )
    assert torch.equal(matched.flow.data, unmatched.flow.data)
    assert torch.equal(matched.colors, unmatched.colors)
    matched_cost = periodic_pair_cost(
        matched.flow.data, matched.flow.noise
    ).diagonal(dim1=1, dim2=2).sum()
    unmatched_cost = periodic_pair_cost(
        unmatched.flow.data, unmatched.flow.noise
    ).diagonal(dim1=1, dim2=2).sum()
    assert matched_cost <= unmatched_cost

    model = DirectTransformer(config.model)
    outputs = sample(model, matched.flow.state[:1], matched.colors[:1], 3)
    class_name = spur.class_names[int(matched.labels[0].item())]
    paths = save_evaluation_svgs(
        root / "svg",
        class_name,
        17,
        matched.flow.state[0],
        [item[0] for item in outputs],
        matched.colors[0],
        config.spur.symmetry,
        spur.side,
    )
    assert len(paths) == 7
    roots = [ET.parse(path).getroot() for path in paths]
    assert len({item.attrib["viewBox"] for item in roots}) == 1
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert len(roots[0].findall(".//svg:polygon", namespace)) == 8
    assert len(roots[1].findall(".//svg:line", namespace)) == 8

    parser = build_parser()
    try:
        with redirect_stderr(io.StringIO()):
            parser.parse_args(
                ["checkpoint.pt", "-t", "0.5", "-s", "uniform", "-o", "x"]
            )
    except SystemExit:
        pass
    else:
        raise AssertionError("Evaluator accepted both exact time and a schedule")


def check_retention(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    identifier = "bream8_test"
    for epoch in range(4):
        (root / f"{identifier}_e{epoch:03d}.pt").touch()
    retain_newest_and_best(root, identifier, newest_epoch=3, best_epoch=1)
    assert {path.name for path in root.glob("*.pt")} == {
        f"{identifier}_e001.pt",
        f"{identifier}_e003.pt",
    }


def run_command(arguments: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "-u", "train.py", *arguments],
        cwd=PROJECT_DIR,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(
            f"Fresh-process command failed ({result.returncode}):\n{result.stdout}"
        )


def run_evaluator(
    checkpoint: Path, output: Path, environment: dict[str, str]
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "evaluator.py",
            str(checkpoint),
            "-t",
            "0.5",
            "-i",
            "2",
            "-n",
            "1",
            "-d",
            "cpu",
            "-o",
            str(output),
        ],
        cwd=PROJECT_DIR,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(f"Evaluator command failed:\n{result.stdout}")


def write_config(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def newest_checkpoint(output: Path) -> tuple[Path, dict]:
    paths = list(output.glob("*/checkpoints/*_e*.pt"))
    assert paths
    loaded = [
        (path, torch.load(path, map_location="cpu", weights_only=False))
        for path in paths
    ]
    return max(loaded, key=lambda item: item[1]["epoch"])


def check_fresh_process_resume(root: Path) -> None:
    environment = os.environ.copy()
    environment["PENROSE_SPUR_PATH"] = str(PROJECT_DIR.parent / "PenroseSpur")
    output = root / "outputs"
    config_path = root / "first.yaml"
    write_config(config_path, smoke_values(6, output, epochs=1))
    run_command(["--config", str(config_path)], environment)
    first_path, first = newest_checkpoint(output)
    assert first["epoch"] == 0
    assert first["global_step"] == 1
    assert first["flow"]["schedule"] == "exponential"
    assert first["identifier"].startswith("bream8_")
    svg_directory = Path(first["output_directory"]) / "svg"
    svg_050 = svg_directory / f"{first['identifier']}_e000_t050.svg"
    svg_095 = svg_directory / f"{first['identifier']}_e000_t095.svg"
    assert svg_050.exists()
    assert svg_095.exists()
    assert ET.parse(svg_050).getroot().attrib["viewBox"] == (
        ET.parse(svg_095).getroot().attrib["viewBox"]
    )
    evaluator_output = root / "evaluator"
    run_evaluator(first_path, evaluator_output, environment)
    assert len(list(evaluator_output.glob("*.svg"))) == 5

    run_command(["--resume", str(first_path), "-t", "num_epochs=2"], environment)
    newest_path, resumed = newest_checkpoint(output)
    assert resumed["epoch"] == 1
    assert resumed["global_step"] == 2
    assert resumed["identifier"] == first["identifier"]
    assert resumed["output_directory"] == first["output_directory"]
    assert (svg_directory / f"{first['identifier']}_e001_t050.svg").exists()
    assert (svg_directory / f"{first['identifier']}_e001_t095.svg").exists()
    assert 1 <= len(list(newest_path.parent.glob("*.pt"))) <= 2


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="penrose-bream-flow-smoke-") as temporary:
        root = Path(temporary)
        check_config(root / "config")
        check_schedules_and_geometry()
        check_losses_and_sampler(root / "loss")
        check_evaluator_and_svgs(root / "evaluation")
        check_retention(root / "retention")
        check_fresh_process_resume(root / "resume")
    print("PenroseBream flow smoke test passed")


if __name__ == "__main__":
    main()
