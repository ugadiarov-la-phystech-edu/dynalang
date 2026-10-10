"""Differentiable JAX copy of SOLD's frozen spatial-broadcast SAVi decoder.

Weights are constants, outside Ninjax/optimizer state. Gradients flow to slots.
The trusted SAVi checkpoint must remain available when loading a world model.
"""

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class FrozenSAViDecoder:
  def __init__(self, checkpoint_path):
    from embodied.torch.ocr.savi import SAViExtractor
    extractor = SAViExtractor('', checkpoint_path, 64, 'cpu')
    self._initialize(extractor.decoder)
    self.checkpoint_sha256 = hashlib.sha256(
        Path(checkpoint_path).expanduser().read_bytes()).hexdigest()

  @classmethod
  def from_torch(cls, decoder):
    result = cls.__new__(cls)
    result._initialize(decoder)
    return result

  def _initialize(self, decoder):
    import torch
    self.image_size = tuple(decoder.image_size)
    self.slot_dim = decoder.slot_dim
    def constant(tensor):
      with jax.transfer_guard('allow'):
        return jnp.asarray(tensor.detach().cpu().numpy(), jnp.float32)
    pos = decoder.positional_embedding
    self.grid = constant(pos.grid).transpose(0, 2, 3, 1)
    self.pos_weight = constant(pos.projection.weight).transpose(2, 3, 1, 0)
    self.pos_bias = constant(pos.projection.bias)
    self.layers = []
    for layer in decoder.encoder:
      if isinstance(layer, torch.nn.Conv2d):
        self.layers.append(('conv', constant(layer.weight).transpose(2, 3, 1, 0),
            constant(layer.bias) if layer.bias is not None else None,
            tuple(layer.stride), tuple(layer.padding), tuple(layer.dilation),
            layer.groups, type(layer).__name__ == 'SamePadConv2d'))
      elif isinstance(layer, torch.nn.ReLU):
        self.layers.append(('relu',))
      elif not isinstance(layer, torch.nn.Identity):
        raise ValueError(f'Unsupported SAVi decoder layer: {type(layer)}')

  def __call__(self, slots):
    """(..., K, D) slots -> (..., H, W, 3) RGB, matching SOLD's clamp."""
    if slots.shape[-1] != self.slot_dim:
      raise ValueError('SAVi decoder slot dimension does not match checkpoint')
    leading, nslots = slots.shape[:-2], slots.shape[-2]
    h, w = self.image_size
    slots = slots.astype(jnp.float32).reshape(-1, 1, 1, self.slot_dim)
    position = jax.lax.conv_general_dilated(
        self.grid, self.pos_weight, (1, 1), 'VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + self.pos_bias
    x = jnp.broadcast_to(slots, (slots.shape[0], h, w, self.slot_dim)) + position
    for layer in self.layers:
      if layer[0] == 'relu':
        x = jax.nn.relu(x)
        continue
      _, weight, bias, stride, padding, dilation, groups, same = layer
      pads = []
      for size, kernel, step, dil, pad in zip(
          x.shape[1:3], weight.shape[:2], stride, dilation, padding):
        extra = max((math.ceil(size / step) - 1) * step +
                    (kernel - 1) * dil + 1 - size, 0) if same else 0
        pads.append((extra // 2 + pad, extra - extra // 2 + pad))
      x = jax.lax.conv_general_dilated(
          x, weight, stride, tuple(pads), rhs_dilation=dilation,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
          feature_group_count=groups, precision=jax.lax.Precision.HIGHEST)
      if bias is not None:
        x = x + bias
    x = x.reshape(-1, nslots, h, w, 4)
    rgb, masks = jnp.clip(x[..., :3], 0, 1), jax.nn.softmax(x[..., 3:], axis=1)
    return (rgb * masks).sum(1).reshape(*leading, h, w, 3)

  def mse(self, slots, images):
    """Per-frame mean MSE; map/remat bound decoder activation memory."""
    if images.shape[-3:] != (*self.image_size, 3):
      raise ValueError('Image targets must match SAVi checkpoint resolution (H,W,3)')
    leading = slots.shape[:-2]
    flat_slots = slots.reshape(-1, *slots.shape[-2:])
    flat_images = images.reshape(-1, *images.shape[-3:]).astype(jnp.float32)
    def frame_mse(pair):
      slot, image = pair
      return jnp.square(self(slot) - image).mean()
    return jax.lax.map(jax.checkpoint(frame_mse),
                       (flat_slots, flat_images)).reshape(leading)
