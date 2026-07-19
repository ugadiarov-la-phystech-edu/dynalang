"""Featrec wiring after the mode split: with the featrec head present
(fine-tuning mode) the MLPDecoder reconstructs the DINO features from the
world model's per-slot latents via a featproj Linear and there is NO slot
distillation head; without it (frozen mode) the decoder emits only the slot
distillation dist. Tiny randomly-initialized nets, no torch."""

import jax
import jax.numpy as jnp
import numpy as np

from conftest import run_pure

from dynalang import nets
from dynalang import ninjax as nj

SHAPES = {'image': (32, 32, 3), 'text_embed': (8,)}
SLOTCONTRAST = dict(
    enabled=True, variant='dino_v1_s8', image_key='image', n_slots=3,
    slot_dim=16, input_size=32, depth=2, dim=32, heads=2, force_f32=True,
    remat=False, chunk=0, sg_backbone=False, emit_features=True,
    jax_checkpoint='')
KW = dict(act='silu', norm='layer', winit='normal', fan='avg')
N_PATCHES = (32 // 8) ** 2  # input_size 32, dino_v1 patch 8
FEAT_DIM = 32

DEC_KW = dict(
    cnn_keys='$^', mlp_keys='text_embed$', mlp_layers=2, mlp_units=16,
    cnn='resnet', cnn_depth=8, cnn_blocks=0, image_dist='mse',
    vector_dist='mse', inputs=['deter'], featdec_hidden=[24, 24], **KW)


def _encoder():
  return nets.MultiEncoder(
      SHAPES, cnn_keys='$^', mlp_keys='text_embed$', mlp_layers=2,
      mlp_units=16, cnn='resnet', cnn_depth=8, cnn_blocks=0,
      slotcontrast=SLOTCONTRAST, **KW, name='enc')


def _decoder(featrec=True):
  shapes = {'text_embed': (8,), 'slot': (3, 16)}
  if featrec:
    shapes['vit_feature'] = (N_PATCHES, FEAT_DIM)
  return nets.MultiDecoder(shapes, **DEC_KW, name='dec')


def _data(batch=2, length=3, seed=0):
  rng = np.random.default_rng(seed)
  data = {
      'image': rng.uniform(0, 1, (batch, length, 32, 32, 3)).astype(np.float32),
      'text_embed': rng.normal(0, 1, (batch, length, 8)).astype(np.float32),
      'is_first': np.zeros((batch, length), bool),
  }
  data['is_first'][:, 0] = True
  return data


def test_encoder_emits_features():
  enc = _encoder()
  assert enc.slotcontrast_feature_shape == (N_PATCHES, FEAT_DIM)
  (embed, extras), _ = run_pure(enc, None, _data(), return_slot_state=True)
  assert extras['features'].shape == (2, 3, N_PATCHES, FEAT_DIM)
  assert np.isfinite(np.asarray(extras['features'])).all()


def test_policy_path_skips_features():
  # 4D (per-step) input never materializes the featrec target.
  enc = _encoder()
  data = _data()
  _, params = run_pure(enc, None, data, return_slot_state=True)
  step = {k: v[:, 0] for k, v in data.items()}
  (_, extras), _ = run_pure(enc, params, step, return_slot_state=True)
  assert 'features' not in extras


def test_finetune_mode_featrec_replaces_distillation():
  # Featrec head present: vit_feature decoded from the latents, no slot dist.
  dec = _decoder(featrec=True)
  assert dec.featrec_shapes == {'vit_feature': (N_PATCHES, FEAT_DIM)}
  rng = np.random.default_rng(1)
  latents = {'deter': rng.normal(0, 1, (2, 3, 4, 16)).astype(np.float32)}
  dists, params = run_pure(dec, None, latents)
  assert 'vit_feature' in dists
  assert 'slot' not in dists
  assert dists['vit_feature'].mode().shape == (2, 3, N_PATCHES, FEAT_DIM)
  assert any('featproj' in k for k in params), 'featproj Linear missing'
  assert not any('slot_proj' in k for k in params), 'distillation head built'


def test_frozen_mode_distillation_only():
  # No featrec head: slot distillation dist only, no featdec params.
  dec = _decoder(featrec=False)
  assert not dec.featrec_shapes
  rng = np.random.default_rng(2)
  latents = {'deter': rng.normal(0, 1, (2, 3, 4, 16)).astype(np.float32)}
  dists, params = run_pure(dec, None, latents)
  assert 'slot' in dists
  assert 'vit_feature' not in dists
  assert dists['slot'].mode().shape == (2, 3, 3, 16)
  assert not any('featdec' in k for k in params)
  assert not any('featproj' in k for k in params)


def test_featrec_loss_grad_reaches_wm_path_and_extractor():
  # Featrec-from-latents: gradients must reach the featdec, the featproj,
  # and the extractor (through the latents/embed), but not the decoder's
  # unrelated text head.
  enc = _encoder()
  dec = _decoder(featrec=True)
  data = _data(seed=2)

  def loss_fn(data):
    (embed, extras) = enc(data, return_slot_state=True)
    target = jax.lax.stop_gradient(extras['features'].astype(jnp.float32))
    dists = dec({'deter': embed})
    return -dists['vit_feature'].log_prob(target).mean()

  init = nj.pure(loss_fn)
  _, params = init({}, jnp.asarray(0, jnp.uint32), data)

  def pure_loss(params, data):
    out, _ = nj.pure(loss_fn)(params, jnp.asarray(0, jnp.uint32), data,
                              create=False)
    return out

  grads = jax.grad(pure_loss)(params, data)
  norms = {k: float(jnp.abs(v).max()) for k, v in grads.items()}
  featdec = {k: v for k, v in norms.items() if '/featdec/' in f'/{k}/'}
  featproj = {k: v for k, v in norms.items() if '/featproj/' in f'/{k}/'}
  corrector = {k: v for k, v in norms.items() if '/corrector/' in f'/{k}/'}
  init_slots = {k: v for k, v in norms.items() if k.endswith('init_slots')}
  proj = {k: v for k, v in norms.items() if '/slotcontrast/proj/' in f'/{k}/'}
  text_head = {k: v for k, v in norms.items() if '/dec/mlp/' in f'/{k}/'}
  assert featdec and max(featdec.values()) > 0, 'no grad into MLPDecoder'
  assert featproj and max(featproj.values()) > 0, 'no grad into featproj'
  assert corrector and max(corrector.values()) > 0, 'no grad into corrector'
  assert init_slots and max(init_slots.values()) > 0, 'no grad into init'
  assert proj and max(proj.values()) > 0, 'no grad into projection'
  # The decoder's text head is untouched by the featrec loss.
  assert text_head and max(text_head.values()) == 0, {
      k: v for k, v in text_head.items() if v > 0}
