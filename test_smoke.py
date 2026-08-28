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
    MAX_LSA_GROUP_SIZE,
    MATCH_TIME_THRESHOLD,
    GroupedLSAMatcher,
    available_cpu_count,
    angle_delta,
    canonicalize_xya,
    coordinate_flow_state,
    draw_jitter_times,
    draw_time,
    endpoint_loss,
    flow_state,
    jitter_lower_limits,
    periodic_pair_cost,
    schedule_time,
)
from evaluator import (  # noqa: E402
    build_parser,
    build_spur,
    make_generator,
    sample_raw_flow_batch,
    sample_flow_batch,
    save_evaluation_svgs,
    submit_evaluation_batch,
)
from sampler import sample  # noqa: E402
from train import (  # noqa: E402
    build_scheduler,
    evaluate_transfer_baseline,
    one_batch_lookahead,
    validate_initialization_checkpoint,
)
from canvas import target_side_for_unit_var  # noqa: E402


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
        samples_per_epoch=4,
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
    assert batches_per_epoch(config) == 2
    fixed = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert make_identifier(config, fixed) == "bream8_0102_0304_16x1_j1"
    variants = smoke_values(5, root)
    variants["flow"].update(matched=False, loss="l1", schedule="quadratic")
    assert make_identifier(config_from_dict(variants), fixed).endswith(
        "_16x1_um_l1_quadratic"
    )
    with (PROJECT_DIR / "config.yaml").open(encoding="utf-8") as handle:
        defaults = config_from_dict(yaml.safe_load(handle))
    assert defaults.model.d_model == 128
    assert defaults.model.num_layers == 8
    assert defaults.flow.schedule is None
    assert defaults.flow.jitter_xy == 1.0
    assert defaults.flow.jitter_angle == 1.0
    assert defaults.flow.r == 2 and defaults.flow.k == 8
    assert defaults.flow.tail_lim == 1.0 - 1.0 / 16.0
    assert defaults.flow.lsa_workers is None
    assert defaults.train.warmup_epochs is None
    overridden, _ = load_config(
        ["-t", "batch_size=32", "-m", "d_model=256", "-s", "sine", "-f", "loss=l1"]
    )
    assert overridden.train.batch_size == 32
    assert overridden.model.d_model == 256
    assert overridden.flow.schedule == "sine"
    assert overridden.flow.loss == "l1"
    tail_cli, _ = load_config(["-s", "tail"])
    assert tail_cli.flow.schedule == "tail"
    jitter_default, _ = load_config(["--jitter"])
    assert jitter_default.flow.schedule is None
    assert jitter_default.flow.jitter_xy == 1.0
    assert jitter_default.flow.jitter_angle == 1.0
    jitter_shared, _ = load_config(["--jitter", "0.75"])
    assert jitter_shared.flow.jitter_xy == 0.75
    assert jitter_shared.flow.jitter_angle == 0.75
    jitter_split, _ = load_config(["--jitter", "0.75", "0.5"])
    assert jitter_split.flow.jitter_xy == 0.75
    assert jitter_split.flow.jitter_angle == 0.5
    try:
        with redirect_stderr(io.StringIO()):
            load_config(["--jitter", "--time-schedule", "tail"])
    except SystemExit:
        pass
    else:
        raise AssertionError("CLI accepted jitter with a time schedule")
    tail_values = smoke_values(5, root)
    tail_values["flow"]["schedule"] = "tail"
    tail = config_from_dict(tail_values)
    assert make_identifier(tail, fixed).endswith("_16x1_tail")
    tail_values["flow"]["matched"] = False
    assert make_identifier(config_from_dict(tail_values), fixed).endswith(
        "_16x1_tail"
    )
    try:
        invalid_tail = smoke_values(5, root)
        invalid_tail["flow"]["tail_lim"] = 1.0
        config_from_dict(invalid_tail)
    except ValueError as error:
        assert "tail_lim" in str(error)
    else:
        raise AssertionError("Invalid tail limit was accepted")
    try:
        load_config(["flow.unknown=1"])
    except ValueError as error:
        assert "Unknown configuration key" in str(error)
    else:
        raise AssertionError("Unknown configuration override was accepted")
    gentle_values = smoke_values(5, root)
    gentle_values["train"].update(
        num_epochs=201,
        learning_rate=1e-4,
        warmup_epochs=40,
        min_lr_factor=0.1,
    )
    gentle = config_from_dict(gentle_values)
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=gentle.train.learning_rate)
    scheduler = build_scheduler(optimizer, gentle)
    assert abs(optimizer.param_groups[0]["lr"] - 1e-6) < 1e-15
    for _ in range(40):
        optimizer.step()
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 1e-4) < 1e-12
    for _ in range(161):
        optimizer.step()
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 1e-5) < 1e-12


def check_schedules_and_geometry() -> None:
    unit = torch.linspace(0, 1, 101)
    for name in ("uniform", "exponential", "sine", "quadratic"):
        time = schedule_time(unit, name, r=2, k=8)
        assert time[0] == 0
        assert time[-1] == 1
        assert bool((time[1:] >= time[:-1]).all())
    tail = schedule_time(unit, "tail", tail_lim=0.9375)
    assert tail[0] == 0.9375
    assert tail[-1] == 1
    assert bool((tail[1:] >= tail[:-1]).all())
    generator = torch.Generator().manual_seed(3)
    sampled = draw_time(
        32, torch.device("cpu"), torch.float32, generator, "exponential"
    )
    assert sampled.unique().numel() > 1
    tail_sampled = draw_time(
        32,
        torch.device("cpu"),
        torch.float32,
        generator,
        "tail",
        tail_lim=0.9375,
    )
    assert tail_sampled.unique().numel() > 1
    assert bool((tail_sampled >= 0.9375).all())

    report_cases = (
        (5, 96, 0.929, 0.788),
        (5, 384, 0.965, 0.788),
        (6, 96, 0.912, 0.646),
        (6, 384, 0.956, 0.646),
    )
    for symmetry, num_tiles, expected_xy, expected_angle in report_cases:
        side = target_side_for_unit_var(symmetry, num_tiles)
        lower_xy, lower_angle = jitter_lower_limits(symmetry, side, 1.0, 1.0)
        assert abs(lower_xy - expected_xy) < 1e-3
        assert abs(lower_angle - expected_angle) < 1e-3

    jitter_generator = torch.Generator().manual_seed(29)
    aggregate, time_xy, time_angle = draw_jitter_times(
        32,
        torch.device("cpu"),
        torch.float32,
        jitter_generator,
        symmetry=5,
        side=0.296,
        jitter_xy=1.0,
        jitter_angle=0.5,
    )
    lower_xy, lower_angle = jitter_lower_limits(5, 0.296, 1.0, 0.5)
    unit_xy = (time_xy - lower_xy) / (1.0 - lower_xy)
    unit_angle = (time_angle - lower_angle) / (1.0 - lower_angle)
    assert torch.allclose(unit_xy, unit_angle, atol=1e-6)
    assert torch.allclose(aggregate, (2.0 * time_xy + time_angle) / 3.0)

    noise = torch.tensor([[[0.0, 0.0, 1.70]]])
    data = torch.tensor([[[2.0, 4.0, -1.70]]])
    t0 = flow_state(noise, data, torch.tensor([0.0]))
    t1 = flow_state(noise, data, torch.tensor([1.0]))
    midpoint = flow_state(noise, data, torch.tensor([0.5]))
    assert torch.allclose(t0, canonicalize_xya(noise))
    assert torch.allclose(t1, canonicalize_xya(data))
    assert abs(angle_delta(data[..., 2], noise[..., 2]).item()) < 0.1
    assert abs(midpoint[..., 2].item()) > 1.69
    coordinate = coordinate_flow_state(
        noise,
        data,
        torch.tensor([0.25]),
        torch.tensor([0.75]),
    )
    assert torch.allclose(coordinate[..., :2], torch.tensor([[[0.5, 1.0]]]))
    expected_angle = noise[..., 2] + 0.75 * angle_delta(
        data[..., 2], noise[..., 2]
    )
    assert torch.allclose(coordinate[..., 2], canonicalize_xya(
        torch.cat((coordinate[..., :2], expected_angle[..., None]), dim=-1)
    )[..., 2])

    source = torch.tensor(
        [[[0.0, 0.0, -1.70], [5.0, 0.0, 1.70]]], dtype=torch.float32
    )
    target = torch.tensor(
        [[[5.0, 0.0, -1.70], [0.0, 0.0, 1.70]]], dtype=torch.float32
    )
    cost = periodic_pair_cost(target, source)
    assert cost.shape == (1, 2, 2)
    assert cost[0, 0, 1] < cost[0, 0, 0]


def check_grouped_matching() -> None:
    identity_time = torch.tensor([MATCH_TIME_THRESHOLD, 1.0])
    values = torch.randn(2, 12, 3, generator=torch.Generator().manual_seed(21))
    colors = torch.tensor([[0] * 6 + [1] * 6] * 2)
    with GroupedLSAMatcher(2) as matcher:
        threshold = matcher.submit(
            values, values.flip(1), colors, identity_time, matched=True
        )
        assert threshold.task_count == 0
        assert torch.equal(
            threshold.permutation(torch.device("cpu")),
            torch.arange(12).expand(2, -1),
        )
        unmatched = matcher.submit(
            values,
            values.flip(1),
            colors,
            torch.zeros(2),
            matched=False,
        )
        assert unmatched.task_count == 0
        assert torch.equal(
            unmatched.permutation(torch.device("cpu")),
            torch.arange(12).expand(2, -1),
        )

    with GroupedLSAMatcher(2) as matcher:
        for group_size, expected in (
            (64, (64,)),
            (65, (64, 1)),
            (128, (64, 64)),
            (256, (64, 64, 64, 64)),
        ):
            generator = torch.Generator().manual_seed(group_size)
            data = torch.randn(1, group_size, 3, generator=generator)
            noise = torch.randn(1, group_size, 3, generator=generator)
            one_color = torch.zeros(1, group_size, dtype=torch.long)
            handle = matcher.submit(
                data, noise, one_color, torch.zeros(1), matched=True
            )
            assert handle.task_sizes == expected
            assert all(size <= MAX_LSA_GROUP_SIZE for size in handle.task_sizes)
            permutation = handle.permutation(torch.device("cpu"))[0]
            assert torch.equal(
                permutation.sort().values, torch.arange(group_size)
            )
            for start in range(0, group_size, MAX_LSA_GROUP_SIZE):
                indices = torch.arange(
                    start, min(start + MAX_LSA_GROUP_SIZE, group_size)
                )
                assigned = permutation[indices]
                assert torch.equal(assigned.sort().values, indices)
                identity_cost = periodic_pair_cost(
                    data[:, indices], noise[:, indices]
                ).diagonal(dim1=1, dim2=2).sum()
                assigned_cost = periodic_pair_cost(
                    data[:, indices], noise[:, assigned]
                ).diagonal(dim1=1, dim2=2).sum()
                assert assigned_cost <= identity_cost + 1e-5

    data = torch.randn(1, 130, 3, generator=torch.Generator().manual_seed(31))
    noise = data.flip(1).clone()
    colors = torch.tensor([[0] * 65 + [1] * 65])
    with GroupedLSAMatcher(None) as matcher:
        handle = matcher.submit(
            data, noise, colors, torch.zeros(1), matched=True
        )
        assert handle.task_sizes == (64, 1, 64, 1)
        assert matcher.effective_workers(handle.task_count) == min(
            available_cpu_count(), handle.task_count
        )
        permutation = handle.permutation(torch.device("cpu"))[0]
        assert torch.equal(colors[0, permutation], colors[0])
        try:
            handle.permutation(torch.device("cpu"))
        except RuntimeError as error:
            assert "already been resolved" in str(error)
        else:
            raise AssertionError("A matching future was consumed more than once")


def check_one_batch_lookahead() -> None:
    events: list[str] = []
    resolve_counts: dict[int, int] = {}

    class Pending:
        def __init__(self, value: int):
            self.value = value

        def resolve(self) -> int:
            resolve_counts[self.value] = resolve_counts.get(self.value, 0) + 1
            events.append(f"resolve{self.value}")
            return self.value

    def submit(value: int) -> Pending:
        events.append(f"submit{value}")
        return Pending(value)

    for value in one_batch_lookahead(range(3), submit):
        events.append(f"train{value}")

    assert events == [
        "submit0",
        "resolve0",
        "submit1",
        "train0",
        "resolve1",
        "submit2",
        "train1",
        "resolve2",
        "train2",
    ]
    assert resolve_counts == {0: 1, 1: 1, 2: 1}
    assert list(one_batch_lookahead([], submit)) == []


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
        spur,
        config,
        2,
        make_generator(device, 17),
        time=0.5,
        schedule="uniform",
        matched=True,
    )
    unmatched = sample_flow_batch(
        spur,
        config,
        2,
        make_generator(device, 17),
        time=0.5,
        schedule="uniform",
        matched=False,
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

    jitter_raw = sample_raw_flow_batch(
        spur, config, 2, make_generator(device, 18)
    )
    assert jitter_raw.schedule is None
    assert not torch.equal(jitter_raw.time_xy, jitter_raw.time_angle)
    with GroupedLSAMatcher(1) as matcher:
        jitter_pending = submit_evaluation_batch(
            jitter_raw, config, matcher, matched=True
        )
        assert jitter_pending.flow.handle.task_count == 0
        jitter_batch = jitter_pending.resolve()
    assert torch.equal(jitter_batch.flow.noise, jitter_raw.noise)

    tail_values = smoke_values(6, root)
    tail_values["flow"]["schedule"] = "tail"
    tail_config = config_from_dict(tail_values)
    tail_spur = build_spur(tail_config, device)
    tail_raw = sample_raw_flow_batch(
        tail_spur,
        tail_config,
        2,
        make_generator(device, 19),
    )
    assert tail_raw.schedule == "tail"
    with GroupedLSAMatcher(1) as matcher:
        tail_pending = submit_evaluation_batch(
            tail_raw, tail_config, matcher, matched=True
        )
        assert tail_pending.flow.handle.task_count == 0
        tail_batch = tail_pending.resolve()
    assert torch.equal(tail_batch.flow.noise, tail_raw.noise)
    override_raw = sample_raw_flow_batch(
        spur,
        config,
        2,
        make_generator(device, 23),
        schedule="tail",
    )
    with GroupedLSAMatcher(1) as matcher:
        override_pending = submit_evaluation_batch(
            override_raw, config, matcher, matched=True
        )
        assert override_pending.flow.handle.task_count == 0
        override_batch = override_pending.resolve()
    assert torch.equal(override_batch.flow.noise, override_raw.noise)

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
    assert (
        parser.parse_args(["checkpoint.pt", "-s", "tail", "-o", "x"]).time_schedule
        == "tail"
    )
    assert parser.parse_args(["checkpoint.pt", "--jitter", "-o", "x"]).jitter == []
    assert parser.parse_args(
        ["checkpoint.pt", "--jitter", "0.5", "0.25", "-o", "x"]
    ).jitter == [0.5, 0.25]


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


def run_command(arguments: list[str], environment: dict[str, str]) -> str:
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
    return result.stdout


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
    assert first["global_step"] == 2
    assert first["flow"]["schedule"] is None
    assert first["flow"]["jitter_xy"] == 1.0
    assert first["flow"]["jitter_angle"] == 1.0
    assert 0.0 <= first["flow"]["jitter_lower_xy"] < 1.0
    assert 0.0 <= first["flow"]["jitter_lower_angle"] < 1.0
    assert first["flow"]["tail_lim"] == 0.9375
    assert first["flow"]["effective_matched"] is False
    assert first["flow"]["match_time_threshold"] == MATCH_TIME_THRESHOLD
    assert first["flow"]["lsa_group_size"] == MAX_LSA_GROUP_SIZE
    assert first["identifier"].startswith("bream8_")
    assert first["identifier"].endswith("_j1")
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
    assert resumed["global_step"] == 4
    assert resumed["identifier"] == first["identifier"]
    assert resumed["output_directory"] == first["output_directory"]
    assert (svg_directory / f"{first['identifier']}_e001_t050.svg").exists()
    assert (svg_directory / f"{first['identifier']}_e001_t095.svg").exists()
    assert 1 <= len(list(newest_path.parent.glob("*.pt"))) <= 2


def check_weight_transfer(root: Path) -> None:
    environment = os.environ.copy()
    environment["PENROSE_SPUR_PATH"] = str(PROJECT_DIR.parent / "PenroseSpur")
    source_output = root / "source"
    source_config = root / "source.yaml"
    write_config(source_config, smoke_values(6, source_output, epochs=1))
    run_command(["--config", str(source_config)], environment)
    source_path, source = newest_checkpoint(source_output)

    legacy = copy.deepcopy(source)
    legacy["config"].pop("flow")
    legacy["config"]["corruption"] = {"alpha": None}
    legacy["epoch"] = 1000
    legacy["global_step"] = 12345
    legacy_path = root / "legacy-e1000.pt"
    torch.save(legacy, legacy_path)

    target_output = root / "target"
    target_values = smoke_values(6, target_output, epochs=1)
    target_values["spur"].update(num_tiles=10, num_ret_tiles=10)
    target_values["flow"].update(schedule="tail", tail_lim=0.97)
    target_values["train"].update(
        learning_rate=1e-4,
        warmup_epochs=0,
        weight_decay=1e-3,
    )
    target_config = root / "target.yaml"
    write_config(target_config, target_values)
    stdout = run_command(
        [
            "--init-weights",
            str(legacy_path),
            "--config",
            str(target_config),
        ],
        environment,
    )
    assert "Transfer baseline loss=" in stdout
    _, transferred = newest_checkpoint(target_output)
    assert transferred["epoch"] == 0
    assert transferred["global_step"] == 2
    assert transferred["identifier"] != source["identifier"]
    assert transferred["initialization"] == {
        "mode": "weights_only",
        "source_checkpoint": str(legacy_path),
        "source_identifier": source["identifier"],
        "source_epoch": 1000,
        "source_global_step": 12345,
        "source_symmetry": 6,
        "source_num_tiles": 8,
        "source_schedule": "legacy_corruption",
        "target_num_tiles": 10,
        "target_schedule": "tail",
        "target_tail_lim": 0.97,
        "target_jitter_xy": 1.0,
        "target_jitter_angle": 1.0,
    }
    assert transferred["transfer_baseline"]["average_time"] >= 0.97
    assert transferred["config"]["train"]["learning_rate"] == 1e-4
    assert transferred["config"]["train"]["weight_decay"] == 1e-3

    target_config_object = config_from_dict(target_values)
    incompatible = copy.deepcopy(legacy)
    incompatible["config"]["model"]["num_layers"] = 2
    try:
        validate_initialization_checkpoint(target_config_object, incompatible)
    except ValueError as error:
        assert "model.num_layers" in str(error)
    else:
        raise AssertionError("Mismatched transfer architecture was accepted")
    incompatible = copy.deepcopy(legacy)
    incompatible["config"]["spur"]["symmetry"] = 5
    try:
        validate_initialization_checkpoint(target_config_object, incompatible)
    except ValueError as error:
        assert "symmetry" in str(error)
    else:
        raise AssertionError("Mismatched transfer symmetry was accepted")

    try:
        with redirect_stderr(io.StringIO()):
            load_config(
                [
                    "--resume",
                    str(source_path),
                    "--init-weights",
                    str(legacy_path),
                ]
            )
    except SystemExit:
        pass
    else:
        raise AssertionError("CLI accepted resume and init-weights together")

    baseline_config = config_from_dict(smoke_values(6, root / "baseline"))
    spur = build_spur(baseline_config, torch.device("cpu"))
    model = DirectTransformer(baseline_config.model)
    before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    matcher = GroupedLSAMatcher(1)
    try:
        metrics = evaluate_transfer_baseline(
            model,
            spur,
            baseline_config,
            matcher,
            make_generator(torch.device("cpu"), 99),
        )
    finally:
        matcher.shutdown()
    assert metrics["loss"] >= 0
    assert all(
        torch.equal(before[key], value)
        for key, value in model.state_dict().items()
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="penrose-bream-flow-smoke-") as temporary:
        root = Path(temporary)
        check_config(root / "config")
        check_schedules_and_geometry()
        check_grouped_matching()
        check_one_batch_lookahead()
        check_losses_and_sampler(root / "loss")
        check_evaluator_and_svgs(root / "evaluation")
        check_retention(root / "retention")
        check_fresh_process_resume(root / "resume")
        check_weight_transfer(root / "transfer")
    print("PenroseBream flow smoke test passed")


if __name__ == "__main__":
    main()
