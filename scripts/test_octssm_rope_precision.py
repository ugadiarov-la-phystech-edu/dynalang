"""RoPE precision and OCTSSM state/cache regressions.

Run: conda run -n ocdreamer python scripts/test_octssm_rope_precision.py
"""
import contextlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'dynalang'))

import jax
import jax.numpy as jnp
import numpy as np

from dynalang import jaxutils, ninjax as nj
from dynalang.nets import ObjectCentricTSSM
from dynalang.transformer import ObjectCentricDynamicsTransformer, apply_rope


DTYPES = (jnp.float32, jnp.float16, jnp.bfloat16)
KEY = jax.random.PRNGKey(42)


@contextlib.contextmanager
def compute_dtype(dtype):
  previous = jaxutils.COMPUTE_DTYPE
  jaxutils.COMPUTE_DTYPE = dtype
  try:
    yield
  finally:
    jaxutils.COMPUTE_DTYPE = previous


class RopePrecisionTest(unittest.TestCase):

  def test_rotation_matches_independent_reference(self):
    positions = np.array([[0, 257, 2049, 65537], [1, 258, 2050, 65538]])
    raw = np.random.default_rng(7).normal(size=(2, 4, 2, 32))
    for dtype in DTYPES:
      with self.subTest(dtype=dtype):
        x = jnp.asarray(raw, dtype)
        # Complex multiplication provides an independent float64 reference.
        xf = np.asarray(x, dtype=np.float64)
        freq = 10000.0 ** (-np.arange(0, 32, 2, dtype=np.float64) / 32)
        phase = positions[..., None, None] * freq
        z = (xf[..., 0::2] + 1j * xf[..., 1::2]) * np.exp(1j * phase)
        expected = np.stack((z.real, z.imag), -1).reshape(x.shape)
        out = jax.jit(apply_rope)(x, jnp.asarray(positions, jnp.int32))
        self.assertEqual(out.dtype, dtype)
        np.testing.assert_allclose(np.asarray(out, np.float32), expected,
                                   atol=0.025, rtol=0.01)
        grad = jax.grad(lambda a: apply_rope(a, jnp.asarray(positions)).astype(
            jnp.float32).sum())(x)
        self.assertTrue(np.isfinite(np.asarray(grad, np.float32)).all())

  def test_relative_position_and_norm(self):
    for dtype in DTYPES:
      with self.subTest(dtype=dtype):
        q = jax.random.normal(KEY, (1, 1, 2, 32)).astype(dtype)
        k = jax.random.normal(jax.random.fold_in(KEY, 1), q.shape).astype(dtype)
        scores = []
        for offset in (0, 256, 2048, 4096):
          qr = apply_rope(q, jnp.array([[offset]], jnp.int32)).astype(jnp.float32)
          kr = apply_rope(k, jnp.array([[offset + 1]], jnp.int32)).astype(jnp.float32)
          scores.append(np.asarray((qr * kr).sum(-1)))
          np.testing.assert_allclose(np.asarray((qr * qr).sum(-1)),
                                     np.asarray((q.astype(jnp.float32) ** 2).sum(-1)),
                                     atol=0.08, rtol=0.01)
        np.testing.assert_allclose(scores, np.repeat(scores[:1], 4, axis=0),
                                   atol=0.06 if dtype == jnp.bfloat16 else 0.008,
                                   rtol=0.002)
    with self.assertRaises(ValueError):
      apply_rope(jnp.ones((1, 1, 1, 3)), jnp.zeros((1, 1)))

  def test_full_incremental_prefixes(self):
    # Before eviction, padded full windows and incremental decoding must agree.
    # Batch elements deliberately use different, large absolute time offsets.
    for dtype in DTYPES:
      for action_mode in ('none', 'slot', 'cross_attn'):
        with self.subTest(dtype=dtype, action_mode=action_mode), compute_dtype(dtype):
          b, length, d, heads, layers = 2, 4, 16, 4, 2
          slots = 4 if action_mode == 'slot' else 3
          xs = jax.random.normal(KEY, (b, length, slots, d)).astype(dtype)
          actions = jax.random.normal(KEY, (b, length, d)).astype(dtype)
          offsets = jnp.array([254, 2046], jnp.int32)

          def run(xs, actions):
            model = ObjectCentricDynamicsTransformer(
                num_layers=layers, d_model=d, nhead=heads,
                feedforward_units=32, dropout=0, norm_first=True,
                norm=True, action_mode=action_mode, pe='rope', name='tf')
            ck = jnp.zeros((b, layers, length, slots, heads, d // heads), dtype)
            cv = jnp.zeros_like(ck)
            window = jnp.zeros_like(xs)
            aw = jnp.zeros_like(actions)
            valid = jnp.zeros((b, length), jnp.float32)
            pairs = []
            for t in range(length):
              window = jnp.concatenate((window[:, 1:], xs[:, t:t+1]), 1)
              aw = jnp.concatenate((aw[:, 1:], actions[:, t:t+1]), 1)
              valid = jnp.concatenate((valid[:, 1:], jnp.ones((b, 1))), 1)
              pos = offsets + t
              inc, ck, cv = model.incremental(
                  xs[:, t], ck, cv, pos, valid=valid,
                  action_embed=actions[:, t] if action_mode == 'cross_attn' else None)
              full = model(
                  window, causal_mask=jnp.triu(jnp.full((length, length), -jnp.inf), 1),
                  key_padding_mask=valid == 0,
                  positions=pos[:, None] - length + 1 + jnp.arange(length)[None],
                  action_embeds=aw if action_mode == 'cross_attn' else None)
              pairs.append((inc, full[:, -1]))
            return pairs

          pairs, _ = nj.pure(run)({}, KEY, xs, actions)
          tol = {jnp.float32: 2e-5, jnp.float16: 0.008, jnp.bfloat16: 0.06}[dtype]
          for inc, full in pairs:
            np.testing.assert_allclose(np.asarray(inc, np.float32),
                                       np.asarray(full, np.float32), atol=tol, rtol=tol)

  def test_state_positions_observe_imagine_and_reset(self):
    for dtype in DTYPES:
      for cached in (False, True):
        with self.subTest(dtype=dtype, cached=cached), compute_dtype(dtype):
          def run(dirty_reset):
            model = ObjectCentricTSSM(
                num_slots=2, deter=8, units=8, stoch=2, classes=2,
                tf_context_length=4, tf_layers=2, tf_heads=2,
                feedforward_units=16, dropout=0, action_mode='slot',
                kv_cache=cached, pe='rope', name='rssm')
            initial = model.initial(2)
            state = {**initial, 'step_pos': jnp.array([2047, 255], jnp.int32)}
            embed = jnp.ones((2, 4, 2, 8), dtype)
            action = jnp.zeros((2, 4, 3), dtype)
            # First batch element resets independently inside the sequence.
            first = jnp.array([[False, False, True, False], [False] * 4])
            post = model.observe(embed, action, first, state)
            prior = model.imagine(action[:, :2], {k: v[:, -1] for k, v in post.items()})
            # A reset must also remove the previous episode's cache/context.
            dirty = {k: v[:, -1] for k, v in post.items()}
            reset_input = jax.tree_util.tree_map(
                lambda old, new: jnp.where(dirty_reset, old, new), dirty, initial)
            reset = model.obs_step(reset_input, action[:, 0], embed[:, 0], jnp.ones(2, bool), False)
            return initial, post, prior, reset

          pure = jax.jit(nj.pure(run))
          (initial, post, prior, reset), _ = pure({}, KEY, True)
          (_, _, _, fresh), _ = pure({}, KEY, False)
          for state in (initial, post, prior, reset):
            self.assertEqual(state['step_pos'].dtype, jnp.int32)
            self.assertEqual(state['deter'].dtype, dtype)
            self.assertTrue(np.isfinite(np.asarray(state['deter'], np.float32)).all())
          np.testing.assert_array_equal(post['step_pos'], [[2048, 2049, 1, 2], [256, 257, 258, 259]])
          np.testing.assert_array_equal(prior['step_pos'], [[3, 4], [260, 261]])
          np.testing.assert_array_equal(reset['step_pos'], [1, 1])
          for key in ('deter', 'tf_context', 'valid', 'action_context') + (('kv_k', 'kv_v') if cached else ()):
            np.testing.assert_array_equal(reset[key], fresh[key])

  def test_cache_after_eviction_matches_banded_attention(self):
    # After eviction the reference is a full sequence with a sliding mask at
    # EVERY layer, not recomputation of only the last input window.
    for dtype in DTYPES:
      with self.subTest(dtype=dtype), compute_dtype(dtype):
        def run(xs):
          model = ObjectCentricDynamicsTransformer(
              num_layers=2, d_model=16, nhead=4, feedforward_units=32,
              dropout=0, norm_first=True, norm=True,
              action_mode='slot', pe='rope', name='tf')
          length, steps, slots = 3, 7, 3
          ck = jnp.zeros((1, 2, length, slots, 4, 4), dtype)
          cv = jnp.zeros_like(ck)
          valid = jnp.zeros((1, length), jnp.float32)
          outputs = []
          for t in range(steps):
            valid = jnp.concatenate((valid[:, 1:], jnp.ones((1, 1))), 1)
            out, ck, cv = model.incremental(
                xs[:, t], ck, cv, jnp.array([2046 + t], jnp.int32), valid=valid)
            outputs.append(out)
          delta = jnp.arange(steps)[:, None] - jnp.arange(steps)[None, :]
          mask = jnp.where((delta >= 0) & (delta < length), 0.0, -jnp.inf)
          full = model(xs, causal_mask=mask, positions=jnp.arange(steps) + 2046)
          return jnp.stack(outputs, axis=1), full

        xs = jax.random.normal(KEY, (1, 7, 3, 16)).astype(dtype)
        (inc, full), _ = nj.pure(run)({}, KEY, xs)
        tol = {jnp.float32: 2e-5, jnp.float16: 0.008, jnp.bfloat16: 0.06}[dtype]
        np.testing.assert_allclose(np.asarray(inc, np.float32),
                                   np.asarray(full, np.float32), atol=tol, rtol=tol)


if __name__ == '__main__':
  unittest.main(verbosity=2)
