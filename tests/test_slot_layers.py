"""Parity tests: slot_nets primitives vs their torch references."""

import numpy as np
import pytest

from conftest import (
    allclose, assert_converted_matches, run_pure, sd_numpy, torch_or_skip)

from dynalang import slot_convert
from dynalang import slot_nets

torch = torch_or_skip()
nn = torch.nn

ATOL = 1e-5


def _np(x):
  return x.detach().cpu().numpy()


@pytest.mark.parametrize('eps', [1e-5, 1e-6])
def test_layernorm(eps):
  torch.manual_seed(0)
  ref = nn.LayerNorm(32, eps=eps)
  with torch.no_grad():
    ref.weight.uniform_(-1, 1)
    ref.bias.uniform_(-1, 1)
  x = torch.randn(4, 7, 32)
  want = _np(ref(x))
  module = slot_nets.LayerNorm(eps, name='m')
  converted = slot_convert.convert_layernorm(sd_numpy(ref), '', 'm')
  got = assert_converted_matches(module, converted, _np(x))
  allclose(got, want, ATOL, what=f'layernorm eps={eps}')


@pytest.mark.parametrize('bias', [True, False])
def test_plinear(bias):
  torch.manual_seed(1)
  ref = nn.Linear(24, 48, bias=bias)
  x = torch.randn(5, 24)
  want = _np(ref(x))
  module = slot_nets.PLinear(48, bias=bias, name='m')
  converted = slot_convert.convert_linear(sd_numpy(ref), '', 'm', bias=bias)
  got = assert_converted_matches(module, converted, _np(x))
  allclose(got, want, ATOL, what=f'plinear bias={bias}')


def test_grucell():
  torch.manual_seed(2)
  ref = nn.GRUCell(16, 24)
  x = torch.randn(6, 16)
  h = torch.randn(6, 24)
  want = _np(ref(x, h))
  module = slot_nets.GRUCell(name='m')
  converted = slot_convert.convert_gru(sd_numpy(ref), '', 'm')
  got = assert_converted_matches(module, converted, _np(x), _np(h))
  allclose(got, want, ATOL, what='grucell')


def test_two_layer_mlp():
  # Reference: the exact torch module SlotContrast builds for
  # output_transform (networks.two_layer_mlp with initial layer norm).
  from embodied.torch.ocr.slotcontrast.modules import networks
  torch.manual_seed(3)
  ref = networks.MLP(384, 64, [768], initial_layer_norm=True)
  ref.eval()
  x = torch.randn(2, 49, 384)
  with torch.no_grad():
    want = _np(ref(x))
  module = slot_nets.TwoLayerMLP(768, 64, name='m')
  converted = slot_convert.convert_two_layer_mlp(sd_numpy(ref), '', 'm')
  got = assert_converted_matches(module, converted, _np(x))
  allclose(got, want, ATOL, what='two_layer_mlp')


def test_slot_attention():
  from embodied.torch.ocr.slotcontrast.modules import groupers
  torch.manual_seed(4)
  ref = groupers.SlotAttention(
      inp_dim=64, slot_dim=64, n_iters=2, use_mlp=True)
  ref.eval()
  slots = torch.randn(4, 8, 64)
  features = torch.randn(4, 196, 64)
  with torch.no_grad():
    want = _np(ref(slots, features, n_iters=3)['slots'])
  module = slot_nets.SlotAttention(64, 256, n_iters=3, name='m')
  converted = slot_convert.convert_slot_attention(sd_numpy(ref), '', 'm')
  got = assert_converted_matches(module, converted, _np(slots), _np(features))
  allclose(got, want, ATOL, what='slot_attention')


def test_slot_attention_random_init_runs():
  # From-scratch init must produce finite values with sane shapes.
  module = slot_nets.SlotAttention(64, 256, n_iters=3, name='m')
  slots = np.random.randn(2, 8, 64).astype(np.float32)
  feats = np.random.randn(2, 196, 64).astype(np.float32)
  out, _ = run_pure(module, None, slots, feats)
  assert out.shape == (2, 8, 64)
  assert np.isfinite(np.asarray(out)).all()
