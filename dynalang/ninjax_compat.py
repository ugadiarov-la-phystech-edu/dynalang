import functools
from functools import partial as bind

import jax
import jax.numpy as jnp
import ninjax

# Re-export everything from pip ninjax.
from ninjax import (
    FromFlax, FromHaiku, FromOptax, Module, Tree, Variable,
    checkpoint, cond, context, creating, flatten,
    init, scope, unflatten, while_loop,
)


###############################################################################
# Compat wrappers for pure, scan, rng, jit, pmap
###############################################################################


def pure(fun, nested=False):
  """Wrap an impure function with old-style API:
  ``out, state = purified(state, rng, *args, **kwargs)``."""
  new_pure = ninjax.pure(fun, nested=nested)
  @functools.wraps(fun)
  def purified(state, rng, *args, create=None, modify=None, ignore=None,
               **kwargs):
    if hasattr(rng, 'shape') and rng.shape == ():
      rng = jnp.array([rng, rng], jnp.uint32)
    # Old ninjax defaulted create=True; pip ninjax defaults create=False.
    if create is None:
      create = True
    state_out, out = new_pure(
        state, *args, seed=rng, create=create, modify=modify, ignore=ignore,
        **kwargs)
    return out, state_out
  purified.pure = True
  return purified


def rng(amount=None, reserve=16):
  """Alias for ninjax.seed() (renamed in pip ninjax v3.x)."""
  return ninjax.seed(amount, reserve=reserve)


def scan(fun, carry, xs, reverse=False, unroll=1, modify=False):
  """Compat wrapper that supports old-style ``modify`` kwarg.
  - modify=True:  state changes inside each step persist across steps (default
    behavior of pip ninjax scan).
  - modify=False: each step sees the same frozen state; changes are discarded.
  """
  if modify:
    return ninjax.scan(fun, carry, xs, reverse=reverse, unroll=unroll)
  # modify=False: wrap fun so it passes modify=False to discard state changes.
  fun_pure = ninjax.pure(fun, nested=True)
  from ninjax.ninjax import _prerun, seed as _seed, SCOPE
  accessed, modified = _prerun(
      fun_pure, carry, jax.tree.map(lambda x: x[0], xs))
  unchanging = {k: v for k, v in context().items() if k in accessed}
  length = len(jax.tree.leaves(xs)[0])
  seeds = _seed(length, True)
  def inner(carry, x):
    x, sd = x
    state, (carry, y) = fun_pure(
        unchanging, carry, x, create=False, modify=False, seed=sd)
    return carry, y
  carry, ys = jax.lax.scan(inner, carry, (xs, seeds), length, reverse, unroll)
  return carry, ys


def grad(fun, keys, has_aux=False):
  """Thin wrapper forwarding to pip ninjax.grad()."""
  return ninjax.grad(fun, keys, has_aux=has_aux)


def jit(fun, static=None, **kwargs):
  """JIT-compile a pure function. Only the first call may create state."""
  if not getattr(fun, 'pure', False):
    raise ValueError('Use pure() before applying jit().')
  static = static or ()

  @bind(jax.jit, static_argnums=[0], **kwargs)
  def _init(statics, rng, *args, **kw):
    s = fun({}, rng, *args, ignore=True, **dict(statics), **kw)[1]
    return s

  @bind(jax.jit, static_argnums=[0], **kwargs)
  def _apply(statics, state, rng, *args, **kw):
    return fun(state, rng, *args, create=False, **dict(statics), **kw)

  @functools.wraps(fun)
  def wrapper(state, rng, *args, init_only=False, **kw):
    if any([name not in kw for name in static]):
      raise ValueError('Please pass all static arguments by keyword.')
    state = state.copy()
    statics = tuple(sorted([(k, v) for k, v in kw.items() if k in static]))
    kw = {k: v for k, v in kw.items() if k not in static}
    if not hasattr(wrapper, 'keys'):
      created = _init(statics, rng, *args, **kw)
      wrapper.keys = set(created.keys())
      for key, value in created.items():
        if key not in state:
          state[key] = value
    if init_only:
      return state
    else:
      selected = {k: v for k, v in state.items() if k in wrapper.keys}
      out, updated = _apply(statics, selected, rng, *args, **kw)
      return out, {**state, **updated}
  return wrapper


def pmap(fun, axis_name=None, static=None, **kwargs):
  """Parallel-map a pure function across devices."""
  if not getattr(fun, 'pure', False):
    raise ValueError('Use pure() before applying pmap().')
  static = static or ()

  @bind(
      jax.pmap, axis_name=axis_name, static_broadcasted_argnums=[0], **kwargs)
  def _init(statics, rng, *args, **kw):
    return fun({}, rng, *args, ignore=True, **dict(statics), **kw)[1]

  @bind(
      jax.pmap, axis_name=axis_name, static_broadcasted_argnums=[0], **kwargs)
  def _apply(statics, state, rng, *args, **kw):
    return fun(state, rng, *args, create=False, **dict(statics), **kw)

  @functools.wraps(fun)
  def wrapper(state, rng, *args, init_only=False, **kw):
    if any([name not in kw for name in static]):
      raise ValueError('Please pass all static arguments by keyword.')
    state = state.copy()
    statics = tuple(sorted([(k, v) for k, v in kw.items() if k in static]))
    kw = {k: v for k, v in kw.items() if k not in static}
    if not hasattr(wrapper, 'keys'):
      created = _init(statics, rng, *args, **kw)
      wrapper.keys = set(created.keys())
      for key, value in created.items():
        if key not in state:
          state[key] = value
    if init_only:
      return state
    else:
      selected = {k: v for k, v in state.items() if k in wrapper.keys}
      out, updated = _apply(statics, selected, rng, *args, **kw)
      return out, {**state, **updated}
  return wrapper
