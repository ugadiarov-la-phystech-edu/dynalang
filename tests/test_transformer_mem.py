"""Memory fixes (c)+(d): the f32 sinusoidal positional encoding must not
promote an f16 stream to f32 (it silently doubled every transformer head's
activation memory), and rematerialization — per encoder layer (remat flag)
and per imagination step (jax.checkpoint around a scanned ninjax step) — must
be numerically identical in values and gradients."""

import numpy as np

from conftest import allclose, nj, run_pure

from dynalang import jaxutils
from dynalang import slot_nets
from dynalang.transformer import (
    ObjectCentricDynamicsTransformer, TransformerEncoder)

import jax
import jax.numpy as jnp


def test_positional_encoding_keeps_dtype():
  # Zero layers isolates the `src + pe` line: before the fix the f32 encoding
  # promoted an f16 input to f32.
  enc = TransformerEncoder(num_layers=0, d_model=16, nhead=2, name='enc')
  x = jnp.ones((2, 5, 16), jnp.float16)
  out, _ = run_pure(enc, None, x)
  assert out.dtype == jnp.float16, out.dtype
  odt = ObjectCentricDynamicsTransformer(
      num_layers=0, d_model=16, nhead=2, norm=False, name='odt')
  x = jnp.ones((2, 5, 3, 16), jnp.float16)
  out, _ = run_pure(odt, None, x)
  assert out.dtype == jnp.float16, out.dtype


def test_encoder_remat_matches_values_and_grads():
  rng = np.random.default_rng(9)
  x = rng.normal(0, 1, (2, 5, 16)).astype(np.float32)
  ref = TransformerEncoder(
      num_layers=2, d_model=16, nhead=2, feedforward_units=32,
      norm_first=True, norm=True, name='enc')
  want, params = run_pure(ref, None, jnp.asarray(x))
  rem = TransformerEncoder(
      num_layers=2, d_model=16, nhead=2, feedforward_units=32,
      norm_first=True, norm=True, remat=True, name='enc')
  got, _ = run_pure(rem, params, jnp.asarray(x))
  allclose(got, want, 1e-6, what='remat encoder values')

  def grad(module):
    def pure(inp):
      out, _ = nj.pure(module.__call__)(
          dict(params), jnp.asarray(0, jnp.uint32), inp, create=False)
      return out.sum().astype(jnp.float32)
    return jax.grad(pure)(jnp.asarray(x))
  g_ref, g_rem = grad(ref), grad(rem)
  assert np.isfinite(np.asarray(g_rem)).all()
  allclose(g_rem, g_ref, 1e-6, what='remat encoder grads')


def test_scan_step_checkpoint_equivalence():
  # The imag_remat mechanism: a jax.checkpoint-wrapped ninjax step inside
  # jaxutils.scan must match the plain scan in outputs and gradients.
  rng = np.random.default_rng(10)
  x0 = rng.normal(0, 1, (3, 8)).astype(np.float32)

  def rollout(remat, x0):
    cell = slot_nets.PLinear(8, name='cell')
    def step(prev, _):
      return jnp.tanh(cell(prev))
    if remat and not nj.creating():
      step = jax.checkpoint(step)
    out = jaxutils.scan(step, jnp.arange(4), x0, unroll=False)
    return out.sum().astype(jnp.float32)

  _, params = nj.pure(rollout)({}, jnp.asarray(0, jnp.uint32), False,
                               jnp.asarray(x0))

  def grad(remat):
    def pure(inp):
      out, _ = nj.pure(rollout)(dict(params), jnp.asarray(0, jnp.uint32),
                                remat, inp, create=False)
      return out
    return jax.grad(pure)(jnp.asarray(x0))
  g_plain, g_remat = grad(False), grad(True)
  assert np.isfinite(np.asarray(g_remat)).all()
  assert np.abs(np.asarray(g_plain)).max() > 0
  allclose(g_remat, g_plain, 1e-6, what='checkpointed scan step grads')
