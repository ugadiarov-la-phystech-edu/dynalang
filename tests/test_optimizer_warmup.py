"""Scoped LR warmup (Optimizer warmup_keys): only the matching parameters
warm up linearly from 0 to the full LR; everything else trains at the full LR
from the first step. Both groups see identical gradients here (same input,
same loss structure), so their Adam-preconditioned updates are identical and
the per-step delta ratio equals the warmup schedule factor exactly."""

import numpy as np

from conftest import nj

from dynalang import jaxutils
from dynalang import slot_nets

import jax.numpy as jnp

WARMUP = 4
LR = 1e-2


def _program(opt_kwargs):
  def step(x):
    sc = slot_nets.PLinear(4, name='sc')
    other = slot_nets.PLinear(4, name='other')
    opt = jaxutils.Optimizer(LR, eps=1e-8, **opt_kwargs, name='opt')
    loss = lambda: (sc(x).sum() + other(x).sum()).astype(jnp.float32)
    return opt([sc, other], loss)
  return nj.pure(step)


def _deltas(opt_kwargs, n_steps):
  """Run n_steps updates; return per-step mean |delta| of both kernels."""
  fn = _program(opt_kwargs)
  rng = jnp.asarray(0, jnp.uint32)
  x = jnp.ones((2, 3), jnp.float32)
  _, state = fn({}, rng, x)  # creation call applies update #1 (count 0)
  deltas = []
  for _ in range(n_steps):
    prev = {k: np.asarray(v) for k, v in state.items()}
    _, state = fn(state, rng, x)
    deltas.append({
        k: np.abs(np.asarray(state[k]) - prev[k]).mean()
        for k in ('sc/kernel', 'other/kernel')})
  return deltas


def test_scoped_warmup_ramps_only_matching_keys():
  deltas = _deltas({'warmup': WARMUP, 'warmup_keys': r'^/sc/'}, WARMUP + 2)
  for i, d in enumerate(deltas):
    count = i + 1  # updates already applied before this one
    want = min(count / WARMUP, 1.0)
    assert d['other/kernel'] > 0
    ratio = d['sc/kernel'] / d['other/kernel']
    assert np.isclose(ratio, want, rtol=1e-4), (i, ratio, want)


def test_global_warmup_unchanged():
  # Default warmup_keys: the existing behavior — everything warms up together.
  deltas = _deltas({'warmup': WARMUP}, 2)
  for d in deltas:
    assert np.isclose(d['sc/kernel'], d['other/kernel'], rtol=1e-5)


def test_scoped_warmup_respects_frozen_keys():
  deltas = _deltas(
      {'warmup': WARMUP, 'warmup_keys': r'^/sc/', 'frozen_keys': r'^/other/'},
      3)
  for d in deltas:
    assert d['other/kernel'] == 0.0
    assert d['sc/kernel'] > 0
