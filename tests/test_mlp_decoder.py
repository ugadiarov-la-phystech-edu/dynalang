"""Parity: slot_nets.MLPDecoder vs the torch SlotContrast MLPDecoder
(featrec head), random-init and from the real checkpoint, plus the
MultiEncoder/MultiDecoder featrec wiring."""

import os

import numpy as np
import pytest

from conftest import allclose, assert_converted_matches, init_keys, \
    run_pure, torch_or_skip

torch = torch_or_skip()

from dynalang import slot_convert
from dynalang import slot_nets

ATOL = 1e-5


def _build(inp_dim=16, out_dim=24, hidden=(32, 32), n_patches=12, seed=0):
  from embodied.torch.ocr.slotcontrast.modules import decoders
  torch.manual_seed(seed)
  ref = decoders.MLPDecoder(
      inp_dim=inp_dim, outp_dim=out_dim,
      hidden_dims=list(hidden), n_patches=n_patches)
  ref.eval()
  module = slot_nets.MLPDecoder(out_dim, n_patches, hidden, name='m')
  sd = {k: v.detach().cpu().numpy() for k, v in ref.state_dict().items()}
  converted = slot_convert.convert_mlp_decoder(sd, '', 'm', n_hidden=len(hidden))
  return ref, module, converted


def _run_ref(ref, slots):
  with torch.no_grad():
    out = ref(torch.as_tensor(slots))
  return out['reconstruction'].numpy(), out['masks'].numpy()


def test_random_parity():
  ref, module, converted = _build()
  rng = np.random.default_rng(0)
  slots = rng.normal(0, 1, (2, 5, 16)).astype(np.float32)
  want_recon, want_masks = _run_ref(ref, slots)
  got_recon, got_masks = assert_converted_matches(module, converted, slots)
  allclose(got_recon, want_recon, ATOL, what='recon')
  allclose(got_masks, want_masks, ATOL, what='masks')


def test_key_coverage():
  _, module, converted = _build()
  rng = np.random.default_rng(1)
  slots = rng.normal(0, 1, (2, 5, 16)).astype(np.float32)
  created = init_keys(module, slots)
  expected = slot_convert.decoder_expected_keys('m', n_hidden=2)
  assert created == expected, (created ^ expected)


def test_batched_time_axis():
  # The head runs on (B, T, S, D) slots in WorldModel.loss; leading axes
  # must broadcast like a batch.
  _, module, converted = _build()
  rng = np.random.default_rng(2)
  slots = rng.normal(0, 1, (2, 3, 5, 16)).astype(np.float32)
  (bt_recon, bt_masks), _ = run_pure(module, converted, slots)
  (flat_recon, flat_masks), _ = run_pure(
      module, converted, slots.reshape(6, 5, 16))
  allclose(bt_recon, np.asarray(flat_recon).reshape(2, 3, 12, 24), ATOL,
           what='recon (B,T) vs flat')
  allclose(bt_masks, np.asarray(flat_masks).reshape(2, 3, 5, 12), ATOL,
           what='masks (B,T) vs flat')


needs_ckpt = pytest.mark.skipif(
    not os.environ.get('SLOTCONTRAST_CKPT'),
    reason='set SLOTCONTRAST_CKPT to run real-checkpoint parity')


@needs_ckpt
def test_real_checkpoint_decoder():
  from embodied.torch.ocr.slotcontrast.modules import decoders
  path = os.environ['SLOTCONTRAST_CKPT']
  sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
  dec_sd = {
      k.replace('decoder.module.', ''): v
      for k, v in sd.items() if k.startswith('decoder.module.')}
  assert dec_sd, 'checkpoint has no decoder.module.* keys'
  n_patches, inp_dim = dec_sd['pos_emb'].shape[-2:]
  out_dim = dec_sd['mlp.layers.6.bias'].shape[0] - 1
  ref = decoders.MLPDecoder(
      inp_dim=inp_dim, outp_dim=out_dim,
      hidden_dims=[1024, 1024, 1024], n_patches=n_patches)
  ref.load_state_dict(dec_sd)
  ref.eval()
  module = slot_nets.MLPDecoder(out_dim, n_patches, (1024,) * 3, name='m')
  converted = slot_convert.convert_mlp_decoder(
      {k: v.detach().cpu().numpy() for k, v in dec_sd.items()}, '', 'm')
  rng = np.random.default_rng(3)
  slots = rng.normal(0, 1, (2, 6, inp_dim)).astype(np.float32)
  want_recon, want_masks = _run_ref(ref, slots)
  got_recon, got_masks = assert_converted_matches(module, converted, slots)
  diff = np.abs(np.asarray(got_recon) - want_recon).max()
  print(f'real ckpt decoder max abs diff: {diff:.3e}')
  allclose(got_recon, want_recon, 5e-4, rtol=1e-3, what='real ckpt recon')
  allclose(got_masks, want_masks, 5e-4, rtol=1e-3, what='real ckpt masks')
