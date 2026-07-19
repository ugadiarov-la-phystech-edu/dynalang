"""SlotContrast 'decoder_masks' video logging: layout semantics, parameter
sharing with the featrec head, and end-to-end parity with the torch pipeline
(Resizer(patch_inputs=True, bilinear) + visualizations.masks_on_video)."""

import os

import numpy as np
import pytest

from conftest import allclose, run_pure, torch_or_skip

from dynalang import nets
from dynalang import slot_convert
from dynalang import slot_nets

N_PATCHES = 16  # 4x4 patch grid
FEAT_DIM = 24


def _images(batch=2, length=3, size=8, seed=0):
  rng = np.random.default_rng(seed)
  return rng.uniform(0, 1, (batch, length, size, size, 3)).astype(np.float32)


def test_video_layout_and_semantics():
  # One-hot patch masks: each panel must show the image where its slot's mask
  # is 1 and pure white where it is 0; panel 0 is the untouched original.
  images = _images(size=4)
  masks = np.zeros((2, 3, 2, 16), np.float32)
  masks[..., 0, :8] = 1.0  # slot 0 owns the top half of the 4x4 grid
  masks[..., 1, 8:] = 1.0  # slot 1 owns the bottom half
  video = np.asarray(slot_nets.decoder_masks_video(images, masks))
  assert video.shape == (2, 3, 4, 4 * 3, 3)
  orig, panel0, panel1 = np.split(video, 3, axis=3)
  assert np.allclose(orig, images, atol=1e-6)
  # 4x4 grid on a 4x4 image: patch mask == pixel mask, no interpolation blur
  # away from the boundary rows.
  assert np.allclose(panel0[:, :, 0], images[:, :, 0], atol=1e-6)
  assert np.allclose(panel0[:, :, 3], 1.0, atol=1e-6)
  assert np.allclose(panel1[:, :, 3], images[:, :, 3], atol=1e-6)
  assert np.allclose(panel1[:, :, 0], 1.0, atol=1e-6)


def test_masks_shared_with_featrec_head():
  # decoder_masks must reuse the featrec head's parameters (featproj +
  # MLPDecoder over the latents) and match a manual replay of the same path.
  dec = nets.MultiDecoder(
      {'text_embed': (8,), 'slot': (2, 16),
       'vit_feature': (N_PATCHES, FEAT_DIM)},
      cnn_keys='$^', mlp_keys='text_embed$', mlp_layers=2, mlp_units=16,
      cnn='resnet', cnn_depth=8, cnn_blocks=0, image_dist='mse',
      vector_dist='mse', inputs=['deter'], featdec_hidden=[16, 16],
      act='silu', norm='layer', winit='normal', fan='avg', name='dec')
  rng = np.random.default_rng(1)
  latents = {'deter': rng.normal(0, 1, (2, 3, 3, 16)).astype(np.float32)}
  _, params = run_pure(dec, None, latents)
  masks, _ = run_pure(dec.decoder_masks, params, latents)
  assert masks.shape == (2, 3, 2, N_PATCHES)
  sums = np.asarray(masks).sum(axis=2)
  assert np.allclose(sums, 1.0, atol=1e-5), 'masks must softmax over slots'
  # Manual replay: featproj Linear on the object slots, then the standalone
  # MLPDecoder with the head's own parameters — identical masks.
  obj = latents['deter'][..., :2, :]
  z = obj @ np.asarray(params['dec/featproj/kernel'])
  z = z + np.asarray(params['dec/featproj/bias'])
  featdec_params = {
      k.replace('dec/featdec', 'm'): v for k, v in params.items()
      if k.startswith('dec/featdec/')}
  ref = slot_nets.MLPDecoder(FEAT_DIM, N_PATCHES, (16, 16), name='m')
  (_, ref_masks), _ = run_pure(ref, featdec_params, z.astype(np.float32))
  allclose(masks, ref_masks, 1e-5, what='shared featdec masks')


def test_video_rows_layout():
  # The report stacks the batch vertically: row b of the grid must be exactly
  # sequence b's panel strip, in batch order.
  rng = np.random.default_rng(4)
  video = rng.uniform(0, 1, (6, 3, 4, 12, 3)).astype(np.float32)
  grid = np.asarray(slot_nets.video_rows(video))
  assert grid.shape == (3, 6 * 4, 12, 3)
  for b in range(6):
    row = grid[:, b * 4:(b + 1) * 4]
    assert np.array_equal(row, video[b]), f'row {b} mismatch'


def test_openl_video_rows_semantics():
  # Three rows stacked along height: truth strip, model strip, error. With
  # identical masks the model row equals the truth row and the error row is
  # uniform mid-gray; with different masks the error deviates from 0.5.
  rng = np.random.default_rng(5)
  images = _images(batch=3, length=4, size=4, seed=5)
  logits = rng.normal(0, 1, (3, 4, 2, 16)).astype(np.float32)
  masks = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
  video = np.asarray(
      slot_nets.openl_decoder_masks_video(images, masks, masks))
  assert video.shape == (3, 4, 3 * 4, 4 * 3, 3)
  truth, model, error = np.split(video, 3, axis=2)
  want = np.asarray(slot_nets.decoder_masks_video(images, masks))
  assert np.allclose(truth, want, atol=1e-6)
  assert np.allclose(model, want, atol=1e-6)
  assert np.allclose(error, 0.5, atol=1e-6)
  other = masks[..., ::-1, :]  # swap the two slots' masks
  video = np.asarray(
      slot_nets.openl_decoder_masks_video(images, masks, other))
  truth, model, error = np.split(video, 3, axis=2)
  model_want = np.asarray(slot_nets.decoder_masks_video(images, other))
  assert np.allclose(model, model_want, atol=1e-6)
  assert np.allclose(error, (model_want - want + 1) / 2, atol=1e-6)
  # The original-frames panel is identical in both rows -> zero error there.
  assert np.allclose(error[..., :4, :], 0.5, atol=1e-6)


def test_mlp_decoder_f16_close_to_f32():
  # f16 hidden activations: same params, output within f16 tolerance of the
  # f32 path; params still created as f32; masks still a valid softmax.
  rng = np.random.default_rng(6)
  slots = rng.normal(0, 1, (2, 3, 4, 16)).astype(np.float32)
  ref = slot_nets.MLPDecoder(FEAT_DIM, N_PATCHES, (32, 32), name='m')
  (want_recon, want_masks), params = run_pure(ref, None, slots)
  assert all(np.asarray(v).dtype == np.float32 for v in params.values())
  half = slot_nets.MLPDecoder(FEAT_DIM, N_PATCHES, (32, 32), f16=True,
                              name='m')
  (got_recon, got_masks), _ = run_pure(half, params, slots)
  assert np.asarray(got_recon).dtype == np.float32
  assert np.allclose(np.asarray(got_masks).sum(axis=-2), 1.0, atol=1e-3)
  allclose(got_recon, want_recon, 5e-2, rtol=1e-2, what='f16 featdec recon')
  allclose(got_masks, want_masks, 5e-2, rtol=1e-2, what='f16 featdec masks')


def _torch_reference_video(images, masks):
  """Torch pipeline: patch masks -> bilinear resize (Resizer semantics) ->
  masks_on_video panels, rearranged to the width-concatenated layout."""
  torch = torch_or_skip()
  import torch.nn.functional as F
  b, t, s, p = masks.shape
  size = images.shape[2]
  grid = int(np.sqrt(p))
  m = torch.as_tensor(masks).reshape(b * t, s, grid, grid)
  m = F.interpolate(m, size=(size, size), mode='bilinear', align_corners=False)
  m = m.reshape(b, t, s, size, size)
  video = torch.as_tensor(images).permute(0, 1, 4, 2, 3)  # (B,T,C,H,W)
  panels = [video]
  for i in range(s):
    mask = m[:, :, i:i + 1]
    panels.append(video * mask + (1 - mask))  # masks_on_video formula
  out = torch.cat(panels, dim=-1)  # concat along width
  return out.permute(0, 1, 3, 4, 2).numpy()


def test_torch_parity_random():
  torch_or_skip()
  rng = np.random.default_rng(2)
  images = _images(size=8, seed=2)
  logits = rng.normal(0, 1, (2, 3, 4, 16)).astype(np.float32)
  masks = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
  got = np.asarray(slot_nets.decoder_masks_video(images, masks))
  want = _torch_reference_video(images, masks)
  allclose(got, want, 1e-5, what='decoder_masks video vs torch')


needs_ckpt = pytest.mark.skipif(
    not os.environ.get('SLOTCONTRAST_CKPT'),
    reason='set SLOTCONTRAST_CKPT to run real-checkpoint parity')


@needs_ckpt
def test_real_checkpoint_masks_video():
  # End to end with the real MLPDecoder weights: slots -> masks -> video,
  # against the torch decoder + torch visualization pipeline.
  torch = torch_or_skip()
  from embodied.torch.ocr.slotcontrast.modules import decoders
  sd = torch.load(os.environ['SLOTCONTRAST_CKPT'], map_location='cpu',
                  weights_only=False)['state_dict']
  dec_sd = {
      k.replace('decoder.module.', ''): v
      for k, v in sd.items() if k.startswith('decoder.module.')}
  n_patches, inp_dim = dec_sd['pos_emb'].shape[-2:]
  out_dim = dec_sd['mlp.layers.6.bias'].shape[0] - 1
  ref = decoders.MLPDecoder(
      inp_dim=inp_dim, outp_dim=out_dim,
      hidden_dims=[1024, 1024, 1024], n_patches=n_patches)
  ref.load_state_dict(dec_sd)
  ref.eval()
  rng = np.random.default_rng(3)
  slots = rng.normal(0, 1, (2, 2, 6, inp_dim)).astype(np.float32)
  images = _images(batch=2, length=2, size=64, seed=3)
  with torch.no_grad():
    want_masks = ref(torch.as_tensor(slots.reshape(4, 6, inp_dim)))['masks']
  want_masks = want_masks.numpy().reshape(2, 2, 6, n_patches)
  want = _torch_reference_video(images, want_masks)

  module = slot_nets.MLPDecoder(out_dim, n_patches, (1024,) * 3, name='m')
  converted = slot_convert.convert_mlp_decoder(
      {k: v.detach().cpu().numpy() for k, v in dec_sd.items()}, '', 'm')
  (_, got_masks), _ = run_pure(module, converted, slots)
  got = np.asarray(slot_nets.decoder_masks_video(images, np.asarray(got_masks)))
  diff = np.abs(got - want).max()
  print(f'real ckpt masks video max abs diff: {diff:.3e}')
  allclose(got, want, 5e-4, rtol=1e-3, what='real ckpt masks video')
