"""Parity: slot_nets resize helpers vs torch/timm interpolation."""

import numpy as np
import pytest

from conftest import allclose, timm_or_skip, torch_or_skip

from dynalang import slot_nets

torch = torch_or_skip()
timm = timm_or_skip()
F = torch.nn.functional


@pytest.mark.parametrize('size', [224, 336])
def test_bilinear_input_resize(size):
  # The torch extractor resizes with F.interpolate(mode='bilinear',
  # align_corners=False) — antialias defaults to False.
  rng = np.random.default_rng(0)
  x = rng.uniform(0, 1, (2, 64, 64, 3)).astype(np.float32)
  ref = F.interpolate(
      torch.from_numpy(x.transpose(0, 3, 1, 2)), size=(size, size),
      mode='bilinear', align_corners=False)
  want = ref.numpy().transpose(0, 2, 3, 1)
  got = np.asarray(slot_nets.resize_bilinear(x, size))
  allclose(got, want, 5e-6, what=f'bilinear 64->{size}')


def test_bilinear_noop():
  x = np.random.rand(2, 224, 224, 3).astype(np.float32)
  got = np.asarray(slot_nets.resize_bilinear(x, 224))
  allclose(got, x, 0, what='bilinear noop')


@pytest.mark.parametrize('new_grid', [24, 16])
def test_pos_embed_resample(new_grid):
  from timm.layers import resample_abs_pos_embed
  rng = np.random.default_rng(1)
  pos = rng.normal(0, 0.02, (1, 37 * 37 + 1, 384)).astype(np.float32)
  want = resample_abs_pos_embed(
      torch.from_numpy(pos), new_size=(new_grid, new_grid),
      num_prefix_tokens=1).numpy()
  got = np.asarray(slot_nets.resample_pos_embed(pos, 37, new_grid))
  allclose(got, want, 1e-5, what=f'pos-embed 37->{new_grid}')


def test_pos_embed_resample_noop():
  pos = np.random.rand(1, 28 * 28 + 1, 384).astype(np.float32)
  got = np.asarray(slot_nets.resample_pos_embed(pos, 28, 28))
  allclose(got, pos, 0, what='pos-embed noop')
