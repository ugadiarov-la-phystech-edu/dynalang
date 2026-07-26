"""Video predictor (use_predictor=True): module-level parity of the
TransformerPredictor vs the torch networks.TransformerEncoder, converter
coverage, and the full video recurrence (LatentProcessor + ScanOverTime
semantics: first step SA(init, n_iters=3), later SA(predictor(prev),
n_iters=2), carry = state_predicted) vs a torch reference rollout."""

import numpy as np
import pytest

from conftest import (
    allclose, init_keys, run_pure, sd_numpy, timm_or_skip, torch_or_skip)

torch = torch_or_skip()
timm_or_skip()
pytest.importorskip('omegaconf')

from dynalang import slot_convert
from dynalang import slot_nets

import torch_ref

ATOL = 1e-4


def _predictor_pair(n_blocks=1, dim=64, heads=4, seed=0):
  from embodied.torch.ocr.slotcontrast.modules import networks
  torch.manual_seed(seed)
  ref = networks.TransformerEncoder(dim=dim, n_blocks=n_blocks, n_heads=heads)
  ref.eval()
  ref.requires_grad_(False)
  module = slot_nets.TransformerPredictor(
      dim, n_blocks=n_blocks, heads=heads, name='m')
  converted = slot_convert.convert_predictor(
      sd_numpy(ref), '', 'm', n_blocks=n_blocks)
  return ref, module, converted


@pytest.mark.parametrize('n_blocks', [1, 2])
def test_transformer_predictor_parity(n_blocks):
  ref, module, converted = _predictor_pair(n_blocks=n_blocks)
  x = np.random.default_rng(3).normal(size=(3, 7, 64)).astype(np.float32)
  with torch.no_grad():
    want = ref(torch.as_tensor(x)).numpy()
  got = run_pure(module, converted, x)[0]
  assert got.shape == want.shape
  # f32 noise only; the sqrt(head_dim) logit scale makes multi-block
  # accumulation slightly noisier than standard attention.
  allclose(got, want, 1e-4, what=f'predictor {n_blocks} block(s)')


def test_convert_predictor_key_coverage():
  _, module, converted = _predictor_pair()
  x = np.zeros((2, 5, 64), np.float32)
  assert set(converted) == init_keys(module, x)


def test_expected_keys_include_predictor():
  base = slot_convert.expected_keys('dino_v1_s8', 'm')
  full = slot_convert.expected_keys('dino_v1_s8', 'm', predictor_blocks=1)
  extra = full - base
  assert base < full
  assert all(k.startswith('m/predictor/block0/') for k in extra)
  # 6 sublayers (norm1, norm2, attn/qkv, attn/out_proj, linear1, linear2)
  # x (kernel|scale, bias) each.
  assert len(extra) == 12


def _build(seed=0):
  ref = torch_ref.TorchSlotContrast('dino_v1_s8', seed=seed,
                                    use_predictor=True)
  module = slot_nets.SlotContrastEncoder(
      variant='dino_v1_s8', input_size=ref.input_size, n_slots=8, slot_dim=64,
      force_f32=True, remat=False, use_predictor=True, name='m')
  converted = slot_convert.convert_all(
      ref.state_dict_numpy(), 'dino_v1_s8', 'm', predictor_blocks=1)
  return ref, module, converted


def _data(seed, batch=3, length=6):
  rng = np.random.default_rng(seed)
  images = rng.integers(0, 256, (batch, length, 64, 64, 3), dtype=np.uint8)
  # Env 0 restarts mid-sequence, env 1 only at t=0, env 2 restarts twice.
  is_first = np.array([
      [1, 0, 0, 1, 0, 0],
      [1, 0, 0, 0, 0, 0],
      [1, 0, 1, 0, 0, 1],
  ], bool)[:batch, :length]
  return images, is_first


def test_video_sequence_matches_torch_rollout():
  ref, module, converted = _build()
  images, is_first = _data(0)
  want, want_carry = torch_ref.video_style_rollout(ref, images, is_first)
  got, carry = run_pure(
      module, converted, images.astype(np.float32) / 255.0, is_first)[0]
  assert got.shape == want.shape, (got.shape, want.shape)
  allclose(got, want, ATOL, what='video sequence vs torch rollout')
  allclose(carry, want_carry, ATOL, what='carry == predictor(last slots)')


def test_video_split_sequence_carry_consistency():
  # Running [t0..t2] then [t3..t5] with the returned carry must equal the
  # full-sequence result — the carry now holds state_predicted, and this is
  # exactly what the policy path relies on.
  _, module, converted = _build()
  images, is_first = _data(1)
  images = images.astype(np.float32) / 255.0
  full, _ = run_pure(module, converted, images, is_first)[0]
  part1, carry = run_pure(
      module, converted, images[:, :3], is_first[:, :3])[0]
  part2, _ = run_pure(
      module, converted, images[:, 3:], is_first[:, 3:], carry)[0]
  allclose(part1, full[:, :3], 1e-5, what='split first half')
  allclose(part2, full[:, 3:], 1e-5, what='split second half')


def test_video_single_step_matches_sequence():
  # The policy path uses single-step calls; step-by-step must equal the scan.
  _, module, converted = _build()
  images, is_first = _data(2, batch=2, length=4)
  images = images.astype(np.float32) / 255.0
  full, _ = run_pure(module, converted, images, is_first)[0]
  carry = None
  for t in range(4):
    step, carry = run_pure(
        module, converted, images[:, t], is_first[:, t], carry)[0]
    allclose(step, full[:, t], 1e-5, what=f'single step t={t}')


def test_legacy_mode_untouched_by_predictor_flag():
  # use_predictor=False must create no predictor params and keep the
  # extractor recurrence (existing pickles stay loadable bit-for-bit).
  module = slot_nets.SlotContrastEncoder(
      variant='dino_v1_s8', input_size=224, n_slots=4, slot_dim=64,
      depth=2, dim=32, heads=2, force_f32=True, remat=False, name='m')
  images = np.zeros((2, 3, 32, 32, 3), np.float32)
  is_first = np.ones((2, 3), bool)
  keys = init_keys(module, images, is_first)
  assert not any('/predictor/' in k for k in keys)
