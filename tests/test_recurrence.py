"""Recurrent slot semantics: the JAX sequence scan must match a python loop
implementing BatchSlotExtractorEnv.step over the torch extractor, and the
carry must compose across split sequences (the online policy path)."""

import numpy as np
import pytest

from conftest import allclose, run_pure, torch_or_skip, timm_or_skip

torch_or_skip()
timm_or_skip()
pytest.importorskip('omegaconf')

from dynalang import slot_convert
from dynalang import slot_nets

import torch_ref

ATOL = 1e-4


def _build(seed=0):
  ref = torch_ref.TorchSlotContrast('dino_v1_s8', seed=seed)
  module = slot_nets.SlotContrastEncoder(
      variant='dino_v1_s8', input_size=ref.input_size, n_slots=8, slot_dim=64,
      force_f32=True, remat=False, name='m')
  converted = slot_convert.convert_all(ref.state_dict_numpy(), 'dino_v1_s8', 'm')
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


def test_sequence_matches_env_rollout():
  ref, module, converted = _build()
  images, is_first = _data(0)
  want = torch_ref.env_style_rollout(ref, images, is_first)
  got, carry = run_pure(
      module, converted, images.astype(np.float32) / 255.0, is_first)[0]
  assert got.shape == want.shape, (got.shape, want.shape)
  allclose(got, want, ATOL, what='sequence vs env rollout')
  allclose(carry, want[:, -1], ATOL, what='carry == last step')


def test_split_sequence_carry_consistency():
  # Running [t0..t2] then [t3..t5] with the returned carry must equal the
  # full-sequence result — this is exactly what the policy path relies on.
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


def test_single_step_matches_sequence():
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
