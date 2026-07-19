"""MultiEncoder with the in-graph SlotContrast extractor: shapes, slot-state
carry, is_first reset semantics, and the text slot concat. Uses a tiny
randomly-initialized ViT (no torch needed)."""

import numpy as np

from conftest import run_pure

from dynalang import nets

SHAPES = {'image': (32, 32, 3), 'text_embed': (8,)}
SLOTCONTRAST = dict(
    enabled=True, variant='dino_v1_s8', image_key='image', n_slots=3,
    slot_dim=16, input_size=32, depth=2, dim=32, heads=2, force_f32=True,
    remat=False, chunk=0, jax_checkpoint='')
KW = dict(act='silu', norm='layer', winit='normal', fan='avg')


def _encoder():
  return nets.MultiEncoder(
      SHAPES, cnn_keys='$^', mlp_keys='text_embed$', mlp_layers=2,
      mlp_units=16, cnn='resnet', cnn_depth=8, cnn_blocks=0,
      slotcontrast=SLOTCONTRAST, **KW, name='enc')


def _data(batch=2, length=4, seed=0):
  rng = np.random.default_rng(seed)
  data = {
      'image': rng.uniform(0, 1, (batch, length, 32, 32, 3)).astype(np.float32),
      'text_embed': rng.normal(0, 1, (batch, length, 8)).astype(np.float32),
      'is_first': np.zeros((batch, length), bool),
  }
  data['is_first'][:, 0] = True
  return data


def test_shapes_and_synthetic_slot_shapes():
  enc = _encoder()
  assert enc.slotcontrast_enabled
  assert enc.slot_shapes == {'slot': (3, 16)}  # drives octssm num_slots
  data = _data()
  (embed, extras), state = run_pure(enc, None, data, return_slot_state=True)
  assert embed.shape == (2, 4, 4, 16)  # 3 object slots + 1 text slot
  assert extras['slots'].shape == (2, 4, 3, 16)
  assert extras['slot_state'].shape == (2, 3, 16)
  assert np.isfinite(np.asarray(embed)).all()


def test_single_return_without_flag():
  # report/vis call sites keep the plain single-tensor signature.
  enc = _encoder()
  embed, _ = run_pure(enc, None, _data())
  assert embed.shape == (2, 4, 4, 16)


def test_is_first_reset_changes_future_only():
  enc = _encoder()
  data = _data(seed=1)
  _, params = run_pure(enc, None, data, return_slot_state=True)
  (_, ex_a), _ = run_pure(enc, params, data, return_slot_state=True)
  data_b = {**data, 'is_first': data['is_first'].copy()}
  data_b['is_first'][:, 2] = True  # restart mid-sequence
  (_, ex_b), _ = run_pure(enc, params, data_b, return_slot_state=True)
  a, b = np.asarray(ex_a['slots']), np.asarray(ex_b['slots'])
  assert np.allclose(a[:, :2], b[:, :2], atol=1e-6), 'past changed'
  assert np.abs(a[:, 2:] - b[:, 2:]).max() > 1e-4, 'reset had no effect'


def test_policy_single_step_carry():
  enc = _encoder()
  data = _data(seed=2)
  _, params = run_pure(enc, None, data, return_slot_state=True)
  (_, ex_full), _ = run_pure(enc, params, data, return_slot_state=True)
  carry = None
  for t in range(4):
    step = {k: v[:, t] for k, v in data.items()}
    (embed_t, ex_t), _ = run_pure(
        enc, params, step, slot_state=carry, return_slot_state=True)
    assert embed_t.shape == (2, 4, 16)
    carry = ex_t['slot_state']
    got = np.asarray(ex_t['slots'])
    want = np.asarray(ex_full['slots'][:, t])
    assert np.allclose(got, want, atol=1e-5), (t, np.abs(got - want).max())


def test_slot_initial():
  enc = _encoder()
  out, _ = run_pure(enc.slot_initial, None, 5)
  assert out.shape == (5, 3, 16)
  assert np.asarray(out).dtype == np.float32
