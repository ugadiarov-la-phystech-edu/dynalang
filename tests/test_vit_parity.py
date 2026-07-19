"""Parity: slot_nets ViT vs timm (random weights, no network needed).

The reference feature is exactly what TimmExtractor taps (encoders.py:189-223):
the output of blocks[11] (pre-final-norm) with the CLS token stripped.
Captured with a forward hook instead of torchvision FX extraction to avoid
tracing issues with dynamic_img_size."""

import numpy as np
import pytest

from conftest import allclose, assert_converted_matches, sd_numpy, \
    timm_or_skip, torch_or_skip

from dynalang import slot_convert
from dynalang import slot_nets

torch = torch_or_skip()
timm = timm_or_skip()


@pytest.mark.parametrize('layerscale', [False, True])
def test_vit_block(layerscale):
  from timm.models.vision_transformer import Block
  torch.manual_seed(0)
  ref = Block(
      dim=384, num_heads=6, qkv_bias=True,
      init_values=1e-5 if layerscale else None)
  ref.eval()
  x = torch.randn(2, 65, 384)
  with torch.no_grad():
    want = ref(x).numpy()
  module = slot_nets.ViTBlock(384, 6, layerscale=layerscale, name='m')
  converted = slot_convert.convert_vit_block(sd_numpy(ref), '', 'm', layerscale)
  got = assert_converted_matches(module, converted, x.numpy())
  allclose(got, want, 1e-5, what=f'vit_block layerscale={layerscale}')


def _timm_block11_features(model, x):
  """Run timm model, capture blocks[11] output pre-final-norm, strip CLS."""
  captured = {}
  def hook(mod, inp, out):
    captured['out'] = out
  handle = model.blocks[11].register_forward_hook(hook)
  try:
    with torch.no_grad():
      model(x)
  finally:
    handle.remove()
  return captured['out'][:, 1:].numpy()


CASES = [
    # (timm model, variant, depth-of-native-grid, input size, atol)
    ('vit_small_patch8_224_dino', 'dino_v1_s8', 224, 1e-4),
    ('vit_small_patch14_dinov2', 'dinov2_s14', 336, 1e-4),
    ('vit_small_patch14_dinov2', 'dinov2_s14', 518, 1e-4),
]


@pytest.mark.parametrize('model_name,variant,size,atol', CASES)
def test_full_vit(model_name, variant, size, atol):
  torch.manual_seed(1)
  ref = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
  ref.eval()
  spec = slot_nets.VARIANTS[variant]
  x = torch.randn(2, 3, size, size)
  want = _timm_block11_features(ref, x)
  module = slot_nets.ViT(
      depth=12, dim=384, heads=6, patch=spec['patch'],
      native_grid=spec['native_grid'], layerscale=spec['layerscale'],
      name='m')
  converted = slot_convert.convert_vit(
      sd_numpy(ref), '', 'm', depth=12, layerscale=spec['layerscale'])
  x_jax = x.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
  got = assert_converted_matches(module, converted, x_jax)
  assert got.shape == want.shape, (got.shape, want.shape)
  diff = np.abs(np.asarray(got) - want).max()
  print(f'{model_name}@{size}: max abs diff {diff:.3e}')
  allclose(got, want, atol, what=f'{model_name}@{size}')
