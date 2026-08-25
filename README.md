# PenroseBream

PenroseBream is a direct endpoint predictor on a flow-style path between
PenroseSpur unit-variance noise and valid tile geometry. It contains one
unconditioned set Transformer and predicts the data endpoint directly; it does
not predict velocity and does not perform reverse diffusion.

## Flow path

PenroseSpur supplies:

- `x0`: Gaussian x/y and uniform scaled angle in `[-sqrt(3), sqrt(3)]`;
- `x1`: a valid centered tiling and its tile colors.

Noise is LSA-matched to data by default using squared XY distance plus squared
shortest-periodic scaled-angle distance. Set `flow.matched=false` for
independent unmatched noise.

For each training tiling, Bream independently draws `u ~ Uniform(0,1)`, maps it
to `t`, and constructs:

```text
xy_t = (1 - t) * xy_0 + t * xy_1
delta_a = wrap(a_1 - a_0)
a_t = wrap(a_0 + t * delta_a)
```

Scaled angle has period `2*sqrt(3)` and is canonicalized to
`[-sqrt(3), sqrt(3))`. The model is not given `t`.

Time schedules are:

```text
uniform:     t = u
exponential: t = (1 - r^(-k*u)) / (1 - r^(-k))   # default r=2, k=8
sine:        t = sin(pi*u/2)
quadratic:   t = 1 - (1 - u^2) = u^2
```

The loss is direct endpoint L2 by default. `flow.loss=l1` selects L1. XY uses
ordinary error and angle uses shortest-periodic error; all three scalar
channels receive equal weight.

## Dependency

Keep `PenroseSpur` as a sibling checkout or set `PENROSE_SPUR_PATH`:

```text
Diffusion/
  PenroseBream/
  PenroseSpur/
```

Bream uses on-the-fly Spur batches only. It has no dataset files, caches,
PenroseDiffusion imports, XLA/JAX/TPU paths, or alternate denoisers.

## Train

Run through the workspace AIVE environment:

```bash
~/.aivenv/bin/python train.py
```

The defaults are `d_model=128`, eight Transformer layers, matched LSA,
exponential time, L2 endpoint loss, and approximately 140,000 fresh tilings per
epoch. Examples:

```bash
~/.aivenv/bin/python train.py \
  -t batch_size=32 -t num_epochs=20 \
  -m d_model=128 -m num_layers=8 \
  -p symmetry=6 -p num_tiles=120 -p num_ret_tiles=120 \
  -s sine -f loss=l1 --output outputs/hex
```

Unknown sections and keys are rejected. The resolved configuration prints
before training.

Each epoch evaluates one deterministic matched pair at `t=0.5` and `t=0.95`,
one model pass each:

```text
svg/<identifier>_e<epoch>_t050.svg
svg/<identifier>_e<epoch>_t095.svg
```

Both are overlaid input/output visualizations. They are local files and are not
uploaded to WandB.

Identifiers follow:

```text
bream<num_tiles>_<MMDD>_<HHMM>_<d_model>x<num_layers>[_um][_l1][_<schedule>]
```

Matched, L2, and exponential are defaults and therefore omit their suffixes.

## Evaluator

`evaluator.py` is the paired evaluation CLI. It samples hidden Spur data and
noise, optionally matches them, constructs `x_t`, compares predictions to the
known `x1`, and writes iterative SVG sets.

Use either an exact time:

```bash
~/.aivenv/bin/python evaluator.py CHECKPOINT -t 0.5 -i 3 -n 8 -o evaluation
```

or a randomly drawn scheduled time:

```bash
~/.aivenv/bin/python evaluator.py CHECKPOINT -s exponential -i 3 -n 8 -o evaluation
```

`-t` and `-s` are mutually exclusive. With neither, exact `t=0` is used.
`-u/--unmatched` evaluates unmatched noise. Other short flags are `-r` seed,
`-y` symmetry, `-N` tile count, `-x` translation, and `-d` device.

For each class/seed:

```text
breamed_<classname>_<seed>_noised.svg
breamed_<classname>_<seed>_overlaid_i<iteration>.svg
breamed_<classname>_<seed>_produced_i<iteration>.svg
```

All files in one set share a viewbox. The initial flow state is filled in the
first file and outlined over each filled produced iteration.

## Programmatic sampler

`sampler.py` never accesses Spur or a hidden target. Call it with tensors:

```python
from sampler import sample

produced = sample(model, xya, colors, num_iters=3)
```

It returns one canonicalized tensor per iteration. No new noise or interpolation
is applied between iterations.

## Checkpoints and metrics

Epoch checkpoints are atomic and retain only newest and best training-loss
epochs. Resume restores model, optimizer, learning-rate scheduler, identifier,
output path, WandB run, global step, and all random states:

```bash
~/.aivenv/bin/python train.py --resume PATH/TO/CHECKPOINT.pt -t num_epochs=150
```

Architecture, Spur geometry, matching, time schedule, and endpoint loss are
immutable on resume. Legacy pre-flow checkpoints are intentionally incompatible.

Scalar metrics include endpoint loss, XY loss, periodic angle loss, average
time, average noise fraction, learning rate, gradient norm, and lattice losses
at both epoch-evaluation times. WandB records scalars, config, and parameter
counts but no SVG artifacts.
