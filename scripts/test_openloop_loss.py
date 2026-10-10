"""Test action alignment, episode boundaries, no observation leakage and gradients."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'dynalang'), str(ROOT / 'scripts')]

import jax
import jax.numpy as jnp
import numpy as np
from dynalang.openloop import sample_windows, masked_mean
from dynalang import ninjax as nj
from dynalang.agent import WorldModel


class WindowTests(unittest.TestCase):
  def test_actions_targets_and_reset_mask(self):
    data = dict(is_first=jnp.array([[1, 0, 0, 0, 0, 1, 0, 0]], bool),
        is_last=jnp.array([[0, 0, 0, 0, 1, 0, 0, 1]], bool),
        is_terminal=jnp.zeros((1, 8), bool),
        action=jnp.arange(8)[None, :, None])
    post = {'deter': jnp.arange(8)[None, :, None].astype(jnp.float32)}
    state, b, times, targets, weights = jax.jit(
        lambda p: sample_windows(data, p, 2, 3, 4, jax.random.PRNGKey(0)))(post)
    chosen = np.flatnonzero(weights)
    self.assertEqual(len(chosen), 1)
    i = chosen[0]
    np.testing.assert_array_equal(times[i], [2, 3])
    np.testing.assert_array_equal(targets[i], [3, 4])
    self.assertEqual(float(state['deter'][i, 0]), 2)
    # No gradient from auxiliary supervision through the observed anchor.
    grad = jax.grad(lambda p: sample_windows(
        data, {'deter': p}, 2, 3, 4, jax.random.PRNGKey(0))[0]['deter'].sum())(
            post['deter'])
    np.testing.assert_array_equal(grad, 0)
    self.assertEqual(float(masked_mean(jnp.ones((4, 2)), jnp.zeros(4))), 0)

  def test_future_slots_are_targets_not_inputs(self):
    import embodied
    from test_savi_encoder import make_config
    config = make_config().update({'batch_size': 1, 'batch_length': 6,
        'octssm.action_mode': 'slot', 'openloop_loss.slot_scale': 1.0,
        'openloop_loss.context': 3, 'openloop_loss.horizon': 3,
        'openloop_loss.starts': 1})
    spaces = {'slot': embodied.Space(np.float32, (7, 128)),
        'reward': embodied.Space(np.float32), 'is_first': embodied.Space(bool),
        'is_terminal': embodied.Space(bool)}
    acts = {'action': embodied.Space(np.float32, (4,), -1, 1)}
    wm = WorldModel(spaces, acts, config,
                    {k: v.shape for k, v in spaces.items()}, name='wm')
    key = jax.random.PRNGKey(0)
    data = dict(slot=jax.random.normal(key, (1, 6, 7, 128)),
        action=jnp.ones((1, 6, 4)) * 0.2,
        reward=jnp.zeros((1, 6)), cont=jnp.ones((1, 6)),
        is_first=jnp.array([[1, 0, 0, 0, 0, 0]], bool),
        is_terminal=jnp.zeros((1, 6), bool))
    def predict(data):
      prev_actions = {'action': jnp.concatenate(
          [jnp.zeros((1, 1, 4)), data['action'][:, :-1]], 1)}
      post = wm.rssm.observe(wm.encoder(data), prev_actions, data['is_first'])
      return wm.openloop_predictions(data, post)[0]
    predicted, variables = nj.pure(predict)({}, key, data)
    fn = jax.jit(nj.pure(predict))
    reference, _ = fn(variables, key, data)
    changed = {**data, 'slot': data['slot'].at[:, 3:].add(100)}
    actual, _ = fn(variables, key, changed)
    np.testing.assert_array_equal(reference, actual)
    changed_actions = {**data, 'action': data['action'].at[:, 2:].set(-0.8)}
    action_prediction, _ = fn(variables, key, changed_actions)
    self.assertGreater(float(jnp.max(jnp.abs(reference - action_prediction))), 1e-5)
    def objective(variables):
      prediction, _ = fn(variables, key, data)
      return jnp.square(prediction - data['slot'][:, 3:]).mean()
    gradients = jax.grad(objective)(variables)
    for component in ['/rssm/', '/dec/slot_proj/']:
      norm = sum(float(jnp.square(v).sum()) for k, v in gradients.items()
                 if component in k)
      self.assertGreater(norm, 0, component)
    self.assertTrue(np.isfinite(np.asarray(predicted)).all())


if __name__ == '__main__':
  unittest.main()
