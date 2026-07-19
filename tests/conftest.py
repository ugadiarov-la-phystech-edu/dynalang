import pathlib
import sys

import numpy as np
import pytest

# Mirror train.py:14-19 path setup: repo root (for `dynalang.*` namespace
# imports) and dynalang/ (for `embodied` and the torch SlotContrast package).
ROOT = pathlib.Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / 'dynalang'):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import jax  # noqa: E402

# Parity vs torch-CPU float32: forbid TF32 matmuls (JAX defaults to TF32 on
# Ampere+ GPUs, giving ~1e-4 error on 384-dim layers).
jax.config.update('jax_default_matmul_precision', 'highest')

from dynalang import jaxutils  # noqa: E402
from dynalang import ninjax as nj  # noqa: E402

import jax.numpy as jnp  # noqa: E402

# Parity tests compare against float32 torch; only jaxagent ever flips this.
assert jaxutils.COMPUTE_DTYPE is jnp.float32


def torch_or_skip():
  return pytest.importorskip('torch')


def timm_or_skip():
  return pytest.importorskip('timm')


def sd_numpy(module):
  """torch module -> {key: np.ndarray} state dict."""
  return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def run_pure(bound_method, state, *args, **kwargs):
  """Run a ninjax module method purely. state=None initializes from scratch
  and returns (out, created_state); a dict runs with exactly those params
  (create disabled, so missing/mistyped keys fail loudly). Accepts a module
  instance directly (uses its __call__)."""
  if not hasattr(bound_method, '__name__'):
    bound_method = bound_method.__call__
  fn = nj.pure(bound_method)
  rng = jnp.asarray(0, jnp.uint32)  # pure() PRNGKey's scalar seeds
  if state is None:
    return fn({}, rng, *args, **kwargs)
  out, _ = fn(dict(state), rng, *args, **kwargs, create=False)
  return out, state


def init_keys(bound_method, *args, **kwargs):
  """The param key set a module method creates on first call."""
  return set(run_pure(bound_method, None, *args, **kwargs)[1])


def assert_converted_matches(bound_method, converted, *args):
  """Assert converter key set == module-created key set, then run with the
  converted params and return the output."""
  created = init_keys(bound_method, *args)
  missing = created - set(converted)
  extra = set(converted) - created
  assert not missing and not extra, (
      f'missing from converter: {sorted(missing)}\n'
      f'extra in converter: {sorted(extra)}')
  return run_pure(bound_method, converted, *args)[0]


def allclose(got, want, atol, rtol=1e-5, what=''):
  got = np.asarray(got)
  want = np.asarray(want)
  assert got.shape == want.shape, (what, got.shape, want.shape)
  diff = np.abs(got - want).max()
  assert np.allclose(got, want, atol=atol, rtol=rtol), (
      f'{what}: max abs diff {diff:.3e} exceeds atol {atol}')
