"""Memory optimizations: chunked+rematerialized MLPDecoder (featdec_chunk)
must be numerically identical to the unchunked path (values and gradients),
and sg_backbone must stop gradients at the frozen ViT while leaving the
projection / slot attention / learned init / featdec trainable."""

import numpy as np

from conftest import allclose, nj, run_pure

from dynalang import nets
from dynalang import slot_nets

import jax
import jax.numpy as jnp

KW = dict(act='silu', norm='layer', winit='normal', fan='avg')
SHAPES = {'image': (32, 32, 3), 'text_embed': (8,)}
SLOTCONTRAST = dict(
    enabled=True, variant='dino_v1_s8', image_key='image', n_slots=3,
    slot_dim=16, input_size=32, depth=2, dim=32, heads=2, force_f32=True,
    remat=False, chunk=0, sg_backbone=True, emit_features=True,
    jax_checkpoint='')
N_PATCHES = 16
FEAT_DIM = 32


def test_featdec_chunk_matches_unchunked():
  rng = np.random.default_rng(7)
  slots = rng.normal(0, 1, (4, 2, 3, 16)).astype(np.float32)  # 8 frames
  ref = slot_nets.MLPDecoder(FEAT_DIM, N_PATCHES, (32, 32), name='m')
  (want_recon, want_masks), params = run_pure(ref, None, slots)
  chunked = slot_nets.MLPDecoder(FEAT_DIM, N_PATCHES, (32, 32), chunk=2,
                                 name='m')
  (got_recon, got_masks), _ = run_pure(chunked, params, slots)
  allclose(got_recon, want_recon, 1e-6, what='chunked featdec recon')
  allclose(got_masks, want_masks, 1e-6, what='chunked featdec masks')

  def loss(module):
    def fn(s):
      recon, masks = module(s)
      return (recon.sum() + (masks ** 2).sum()).astype(jnp.float32)
    def pure(s):
      out, _ = nj.pure(fn)(dict(params), jnp.asarray(0, jnp.uint32), s,
                           create=False)
      return out
    return jax.grad(pure)(jnp.asarray(slots))
  g_ref = loss(ref)
  g_chunk = loss(chunked)
  assert np.isfinite(np.asarray(g_chunk)).all()
  allclose(g_chunk, g_ref, 1e-5, what='chunked featdec grad wrt slots')


def test_sg_backbone_blocks_backbone_grads():
  enc = nets.MultiEncoder(
      SHAPES, cnn_keys='$^', mlp_keys='text_embed$', mlp_layers=2,
      mlp_units=16, cnn='resnet', cnn_depth=8, cnn_blocks=0,
      slotcontrast=dict(SLOTCONTRAST), **KW, name='enc')
  dec = nets.MultiDecoder(
      {'text_embed': (8,), 'slot': (3, 16),
       'vit_feature': (N_PATCHES, FEAT_DIM)},
      cnn_keys='$^', mlp_keys='text_embed$', mlp_layers=2, mlp_units=16,
      cnn='resnet', cnn_depth=8, cnn_blocks=0, image_dist='mse',
      vector_dist='mse', inputs=['deter'], featdec_hidden=[24, 24],
      **KW, name='dec')
  rng = np.random.default_rng(8)
  data = {
      'image': rng.uniform(0, 1, (2, 3, 32, 32, 3)).astype(np.float32),
      'text_embed': rng.normal(0, 1, (2, 3, 8)).astype(np.float32),
      'is_first': np.zeros((2, 3), bool),
  }
  data['is_first'][:, 0] = True

  def loss_fn(data):
    embed, extras = enc(data, return_slot_state=True)
    target = jax.lax.stop_gradient(extras['features'].astype(jnp.float32))
    dists = dec({'deter': embed})
    return -dists['vit_feature'].log_prob(target).mean()

  _, params = nj.pure(loss_fn)({}, jnp.asarray(0, jnp.uint32), data)

  def pure_loss(params, data):
    out, _ = nj.pure(loss_fn)(params, jnp.asarray(0, jnp.uint32), data,
                              create=False)
    return out

  grads = jax.grad(pure_loss)(params, data)
  norms = {k: float(jnp.abs(v).max()) for k, v in grads.items()}
  backbone = {k: v for k, v in norms.items() if '/backbone/' in f'/{k}/'}
  trainable = {
      k: v for k, v in norms.items()
      if '/corrector/' in f'/{k}/' or '/proj/' in f'/{k}/'
      or '/featdec/' in f'/{k}/' or k.endswith('init_slots')}
  assert backbone, 'backbone params missing from grads'
  assert max(backbone.values()) == 0, {
      k: v for k, v in backbone.items() if v > 0}
  assert trainable and max(trainable.values()) > 0, 'sg killed all grads'
