"""End-to-end conversion check against a real trained SlotContrast
checkpoint. Skipped unless both env vars are set:

  SLOTCONTRAST_CKPT=/path/to/checkpoint.ckpt \
  SLOTCONTRAST_CFG=/path/to/config.yaml \
  pytest tests/test_real_checkpoint.py -q

Compares the converted JAX encoder against the real torch
SlotContrastExtractor on random images, both is_first branches."""

import os

import numpy as np
import pytest

from conftest import allclose, run_pure, torch_or_skip, timm_or_skip

torch_or_skip()
timm_or_skip()
pytest.importorskip('omegaconf')

CKPT = os.environ.get('SLOTCONTRAST_CKPT', '')
CFG = os.environ.get('SLOTCONTRAST_CFG', '')

pytestmark = pytest.mark.skipif(
    not (CKPT and CFG),
    reason='set SLOTCONTRAST_CKPT and SLOTCONTRAST_CFG to run')

MODEL_TO_VARIANT = {
    'vit_small_patch8_224_dino': 'dino_v1_s8',
    'vit_small_patch14_dinov2': 'dinov2_s14',
}
INPUT_SIZES = {'dino_v1_s8': 224, 'dinov2_s14': 336}


@pytest.fixture(scope='module')
def setup():
  import torch
  from omegaconf import OmegaConf
  from embodied.torch.ocr.slotcontrast.slotcontrast_extractor import (
      SlotContrastExtractor)
  from dynalang import slot_convert
  from dynalang import slot_nets

  model_name = str(OmegaConf.select(OmegaConf.load(CFG), 'globals.DINO_MODEL'))
  variant = MODEL_TO_VARIANT[model_name]
  input_size = INPUT_SIZES[variant]

  ref = SlotContrastExtractor(
      config_path=CFG, checkpoint_path=CKPT, image_size=64,
      device='cpu', backbone_input_size=input_size)

  state_dict = torch.load(CKPT, weights_only=False, map_location='cpu')
  if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
  state_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
  leftover = slot_convert.unconsumed_keys(state_dict, variant)
  assert not leftover, f'unknown checkpoint keys: {leftover}'
  converted = slot_convert.convert_all(state_dict, variant, 'm')

  module = slot_nets.SlotContrastEncoder(
      variant=variant, input_size=input_size,
      n_slots=ref.n_slots, slot_dim=ref.dim,
      force_f32=True, remat=False, name='m')
  return ref, module, converted


# Tolerance note: with TRAINED weights the slot-attention softmax is sharp,
# so float32 rounding gets amplified across the 3-6 compounding iterations.
# Measured on the vizdoom/nslot-6 checkpoint: torch-f32 disagrees with
# torch-f64 by up to 1.9e-1 on adversarial (random-normal) prev slots and
# ~2e-5 on realistic ones, while jax-f32 stays within ~2e-3 of torch-f32 in
# all cases — i.e. the implementations agree far more closely with each other
# than f32 agrees with the f64 ground truth. atol 5e-3 sits above the
# cross-implementation noise and far below any real defect signal (slot
# values are O(1-3)).
ATOL = 5e-3


def test_episode_start(setup):
  ref, module, converted = setup
  rng = np.random.default_rng(0)
  images = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
  first_slots = ref(images, previous_slots=None)
  want = ref(images, previous_slots=first_slots)  # initialize_twice
  got, _ = run_pure(
      module, converted, images.astype(np.float32) / 255.0,
      np.ones((2,), bool))[0]
  diff = np.abs(np.asarray(got) - want).max()
  print(f'real ckpt episode-start max abs diff: {diff:.3e}')
  allclose(got, want, ATOL, rtol=1e-3, what='real ckpt episode start')


def test_continuing(setup):
  ref, module, converted = setup
  rng = np.random.default_rng(1)
  images = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
  prev = rng.normal(0, 1, (2, ref.n_slots, ref.dim)).astype(np.float32)
  want = ref(images, previous_slots=prev)
  got, _ = run_pure(
      module, converted, images.astype(np.float32) / 255.0,
      np.zeros((2,), bool), prev)[0]
  diff = np.abs(np.asarray(got) - want).max()
  print(f'real ckpt continuing max abs diff: {diff:.3e}')
  allclose(got, want, ATOL, rtol=1e-3, what='real ckpt continuing')
