import collections

import numpy as np

from . import base


class BufferedBatchEnv(base.Env):
  """Double-buffered batch env that hides slow resets behind spare envs.
  The agent only ever sees ``num_active`` envs (``len(self) == num_active``),
  but internally we keep ``num_active + num_spare`` underlying envs.
  """

  def __init__(self, envs, num_active, parallel):
    assert all(len(env) == 0 for env in envs)
    assert 0 < num_active <= len(envs)
    self._parallel = parallel not in ('none', None, False)
    self._active = list(envs[:num_active])
    self._spares = collections.deque()
    self._keys = list(self.obs_space.keys())
    self._reset_act = {
        k: np.zeros(v.shape, v.dtype) for k, v in self.act_space.items()} # precompute the reset action
    self._reset_act['reset'] = True
    # Warm up the spares so they are ready by the time active envs finish.
    for env in envs[num_active:]:
      self._dispatch_reset(env)

  @property
  def obs_space(self):
    return self._active[0].obs_space

  @property
  def act_space(self):
    return self._active[0].act_space

  def __len__(self):
    return len(self._active)

  def step(self, action):
    n = len(self._active)
    assert all(len(v) == n for v in action.values()), (
        n, {k: getattr(v, 'shape', None) for k, v in action.items()})
    obs = [None] * n
    pending = [None] * n

    # Decide which finished envs get swapped with a ready spare, and
    # dispatch (non-blocking for parallel workers) the regular/fallback steps.
    swap_indices = []
    available = len(self._spares)
    for i in range(n):
      act = {k: v[i] for k, v in action.items()}
      if bool(act['reset']) and available > 0:
        swap_indices.append(i)
        available -= 1
      else:
        pending[i] = self._active[i].step(act)  # no wait for result (future)

   
    # Kick off background resets for the finished envs first (so the
    # worker processes start the slow reset concurrently), then claim spares.
    for i in swap_indices:
      self._dispatch_reset(self._active[i])
    for i in swap_indices: 
      env, first_obs = self._pop_ready_spare()
      self._active[i] = env
      obs[i] = first_obs
      
    # Resolve the regular steps.
    for i in range(n):
      if obs[i] is None:
        obs[i] = self._resolve(pending[i])
    return {k: np.array([ob[k] for ob in obs]) for k in self._keys}

  def render(self):
    return np.stack([env.render() for env in self._active])

  def close(self):
    for env in self._active:
      self._try_close(env)
    for env, _ in self._spares:
      self._try_close(env)

  def _dispatch_reset(self, env):
    self._spares.append((env, env.step(self._reset_act))) # no wait for result (future)

  def _pop_ready_spare(self):
    # Pop the oldest pending reset (most likely already finished) and block onit
    env, pending = self._spares.popleft() #take env and future result
    return env, self._resolve(pending) # block on result

  def _resolve(self, result):
    return result() if self._parallel else result

  @staticmethod
  def _try_close(env):
    try:
      env.close()
    except Exception:
      pass
