"""Parity: SlotContrastEncoder vs the torch SlotContrast extractor,
assembled from the shipped config yamls with random weights."""

import numpy as np
import pytest

from conftest import allclose, assert_converted_matches, torch_or_skip, \
    timm_or_skip

torch_or_skip()
timm_or_skip()
pytest.importorskip('omegaconf')

from dynalang import slot_convert
from dynalang import slot_nets

import torch_ref

ATOL = 1e-4


def _build(variant, seed=0):
  ref = torch_ref.TorchSlotContrast(variant, seed=seed)
  module = slot_nets.SlotContrastEncoder(
      variant=variant, input_size=ref.input_size, n_slots=8, slot_dim=64,
      force_f32=True, remat=False, name='m')
  converted = slot_convert.convert_all(ref.state_dict_numpy(), variant, 'm')
  return ref, module, converted


@pytest.mark.parametrize('variant', ['dino_v1_s8', 'dinov2_s14'])
def test_episode_start(variant):
  # is_first=True: SA3(SA3(learned_init)) — the initialize_twice path.
  ref, module, converted = _build(variant)
  rng = np.random.default_rng(0)
  images = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
  first_slots = ref(images, previous_slots=None)
  want = ref(images, previous_slots=first_slots)
  is_first = np.ones((2,), bool)
  got, carry = assert_converted_matches(
      module, converted, images.astype(np.float32) / 255.0, is_first)
  allclose(got, want, ATOL, what=f'{variant} episode start')
  allclose(carry, want, ATOL, what=f'{variant} carry')


@pytest.mark.parametrize('variant', ['dino_v1_s8', 'dinov2_s14'])
def test_continuing(variant):
  # is_first=False: SA3(prev_slots).
  ref, module, converted = _build(variant)
  rng = np.random.default_rng(1)
  images = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
  prev = rng.normal(0, 1, (2, 8, 64)).astype(np.float32)
  want = ref(images, previous_slots=prev)
  is_first = np.zeros((2,), bool)
  got, _ = assert_converted_matches(
      module, converted, images.astype(np.float32) / 255.0, is_first, prev)
  allclose(got, want, ATOL, what=f'{variant} continuing')


def test_mixed_is_first():
  # Batched where(): env 0 restarts, env 1 continues — must equal the two
  # torch subset calls.
  ref, module, converted = _build('dino_v1_s8')
  rng = np.random.default_rng(2)
  images = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
  prev = rng.normal(0, 1, (2, 8, 64)).astype(np.float32)
  first_slots = ref(images[:1], previous_slots=None)
  want0 = ref(images[:1], previous_slots=first_slots)
  want1 = ref(images[1:], previous_slots=prev[1:])
  is_first = np.array([True, False])
  got, _ = assert_converted_matches(
      module, converted, images.astype(np.float32) / 255.0, is_first, prev)
  allclose(got[:1], want0, ATOL, what='mixed is_first[0]')
  allclose(got[1:], want1, ATOL, what='mixed is_first[1]')
