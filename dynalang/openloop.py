"""Replay windows for action-conditioned prediction without future observations."""
import jax
import jax.numpy as jnp


def sample_windows(data, post, horizon, context, starts, rng):
  """Select anchors with C observed frames and H future targets.

  action[t] causes observation[t+1]. Reset observations are never prediction
  targets; a terminal observation is allowed, but no rollout continues from it.
  Invalid/padded selections carry zero weight, keeping shapes static under JIT.
  """
  batch, length = data['is_first'].shape
  count = length - context - horizon + 1
  if count < 1:
    raise ValueError('Open-loop loss needs batch_length >= context + horizon')
  anchors = jnp.arange(context - 1, length - horizon)
  b = jnp.repeat(jnp.arange(batch), count)
  t = jnp.tile(anchors, batch)
  # A reset at the first context frame is fine; later resets break the context.
  past = t[:, None] - jnp.arange(context - 2, -1, -1)[None]
  context_valid = ~jnp.any(data['is_first'][b[:, None], past].astype(bool), axis=1)
  future = t[:, None] + jnp.arange(1, horizon + 1)[None]
  previous = future - 1
  last = data.get('is_last', data['is_terminal']).astype(bool)
  last = last | data['is_terminal'].astype(bool)
  step_valid = (~data['is_first'][b[:, None], future].astype(bool) &
                ~last[b[:, None], previous])
  valid = context_valid & jnp.all(step_valid, axis=1)
  # Uniform sampling without replacement among valid anchors; if fewer than
  # starts are available, the remaining selections are masked out.
  scores = jax.random.uniform(rng, (len(t),)) + valid.astype(jnp.float32) * 2
  n = min(starts, len(t))
  indices = jax.lax.top_k(scores, n)[1]
  b, t, weights = b[indices], t[indices], valid[indices].astype(jnp.float32)
  times = t[:, None] + jnp.arange(horizon)[None]
  targets = times + 1
  state = jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x[b, t]), post)
  return state, b, times, targets, weights


def masked_mean(errors, weights):
  weights = weights[:, None]
  return (errors * weights).sum() / jnp.maximum(weights.sum() * errors.shape[1], 1)
