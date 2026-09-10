"""Runs a frozen causal SOLV-SAM encoder over images.

Two entry points matter. `forward_episode` describes a whole episode at once
and is the reference implementation; `forward_step` describes one environment
step at a time, carrying the window of past slots in an explicit state so an
online agent gets the same slots without ever seeing a future frame.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from embodied.torch.ocr.solvsam.backbones import build_backbone, freeze
from embodied.torch.ocr.solvsam.config import SolvSamConfig, read_checkpoint_config
from embodied.torch.ocr.solvsam.nets import SolvSam, frame_window_indices


class SolvSamEncoder(nn.Module):

  def __init__(
      self, checkpoint_path, device='cuda', cosmos_checkpoint_dir=None,
      dino_model_name=None, batch_size=64):
    super().__init__()
    self._checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not self._checkpoint_path.is_file():
      raise FileNotFoundError(
          f'no SOLV-SAM checkpoint at {self._checkpoint_path}')
    self._device = torch.device(device)
    self._batch_size = batch_size

    checkpoint = torch.load(
        self._checkpoint_path, map_location='cpu', weights_only=False)
    saved_config = read_checkpoint_config(checkpoint)
    if saved_config is None:
      raise ValueError(
          f'checkpoint {self._checkpoint_path} carries no config metadata, so '
          'the architecture it was trained with cannot be reconstructed')
    self.config = SolvSamConfig.from_checkpoint(saved_config)

    self.backbone = build_backbone(
        self.config, cosmos_checkpoint_dir, dino_model_name)
    self.solv = SolvSam(self.config)
    self._load_slot_weights(checkpoint)

    # Decoding broadcasts every slot over the whole token grid, so it holds
    # num_slots times as much as encoding the same frames does.
    self._decode_batch = max(1, batch_size // self.config.num_slots)

    self.resize = T.Resize(self.config.resize_to)
    if self.backbone.normalize_mean is None:
      self.normalize = nn.Identity()
    else:
      self.normalize = T.Normalize(
          mean=self.backbone.normalize_mean, std=self.backbone.normalize_std)

    self.to(self._device)
    freeze(self)

  @property
  def n_slots(self):
    return self.config.num_slots

  @property
  def slot_dim(self):
    return self.config.slot_dim

  @property
  def device(self):
    return self._device

  def _load_slot_weights(self, checkpoint):
    if 'model' not in checkpoint:
      raise ValueError(
          f'checkpoint {self._checkpoint_path} has no "model" weights')
    state_dict = {}
    for key, value in checkpoint['model'].items():
      if key.startswith('module.'):
        key = key[len('module.'):]
      # The RGB decoder only exists to supervise training and needs the
      # backbone's own decoder blocks, so it is not part of this package.
      if key.startswith('rgb_dec.'):
        continue
      state_dict[key] = value
    missing, unexpected = self.solv.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
      raise ValueError(
          f'checkpoint {self._checkpoint_path} does not match the causal '
          f'SOLV-SAM modules: missing {sorted(missing)}, '
          f'unexpected {sorted(unexpected)}')

  def preprocess(self, images):
    """Bring images to the layout, range and size the backbone expects."""
    images = torch.as_tensor(images, device=self._device)
    if not images.is_floating_point():
      images = images.float() / 255.0
    else:
      images = images.float()
      if images.numel() and images.max() > 1:
        images = images / 255.0
    if images.shape[-1] == 3:
      images = images.permute(0, 3, 1, 2)
    # Resize first, then normalise, matching the training transform.
    return self.normalize(self.resize(images))

  def encode_features(self, images):
    """Backbone tokens of already batched images, `(B, tokens, token_dim)`."""
    features = self.backbone(self.preprocess(images))
    if features.shape[-2:] != (self.config.token_num, self.config.token_dim):
      raise ValueError(
          f'backbone {self.config.encoder} produced features of shape '
          f'{tuple(features.shape)}, expected '
          f'(..., {self.config.token_num}, {self.config.token_dim})')
    return features.float()

  def initial_state(self, batch_size):
    """Empty causal history for `batch_size` environments."""
    history_size = self.config.num_neighbors * self.config.frame_stride
    return {
        'slots': torch.zeros(
            (batch_size, history_size, self.config.num_slots,
             self.config.slot_dim), device=self._device),
        'valid': torch.zeros(
            (batch_size, history_size), dtype=torch.bool,
            device=self._device),
    }

  @torch.no_grad()
  def forward_step(self, images, state=None):
    """Slots of the current frame for a batch of environments.

    Pass `state=None` on the first frame of an episode and the returned state
    back in on every following frame. The state holds the past spatial slots
    the temporal binder needs, so nothing but history reaches the output.
    """
    images = torch.as_tensor(images)
    if images.ndim != 4:
      raise ValueError(
          f'forward_step expects a batch of images, got shape '
          f'{tuple(images.shape)}')

    spatial_slots = self.solv.get_spatial_slots(self.encode_features(images))
    batch_size = spatial_slots.shape[0]

    if state is None:
      state = self.initial_state(batch_size)
    history = torch.as_tensor(
        state['slots'], device=self._device, dtype=spatial_slots.dtype)
    valid = torch.as_tensor(
        state['valid'], device=self._device, dtype=torch.bool)
    history_size = self.config.num_neighbors * self.config.frame_stride
    expected = {
        'slots': (batch_size, history_size, self.config.num_slots,
                  self.config.slot_dim),
        'valid': (batch_size, history_size),
    }
    for name, tensor in (('slots', history), ('valid', valid)):
      if tuple(tensor.shape) != expected[name]:
        raise ValueError(
            f'causal state entry {name!r} has shape {tuple(tensor.shape)}, '
            f'expected {expected[name]}')

    stride = self.config.frame_stride
    ones = torch.ones((batch_size, 1), dtype=torch.bool, device=self._device)
    if self.solv.t_bind is None:
      slots = spatial_slots
    else:
      window = torch.cat([history[:, ::stride], spatial_slots[:, None]], dim=1)
      window_valid = torch.cat([valid[:, ::stride], ones], dim=1)
      slots = self.solv.bind_temporal(
          window.flatten(start_dim=0, end_dim=1), window_valid)

    if history.shape[1]:
      state = {
          'slots': torch.cat([history[:, 1:], spatial_slots[:, None]], dim=1),
          'valid': torch.cat([valid[:, 1:], ones], dim=1),
      }
    return slots, state

  @torch.no_grad()
  def forward_episode(self, images):
    """Slots for every frame of one episode, `(T, num_slots, slot_dim)`.

    Each frame is still described by its own causal window, so this matches a
    run of `forward_step` calls; it is only cheaper because the backbone runs
    once per frame instead of once per window position.
    """
    images = torch.as_tensor(images)
    steps = images.shape[0]
    batch = self._batch_size

    features = torch.cat([
        self.encode_features(images[start:start + batch])
        for start in range(0, steps, batch)], dim=0)
    spatial_slots = torch.cat([
        self.solv.get_spatial_slots(features[start:start + batch])
        for start in range(0, steps, batch)], dim=0)

    if self.solv.t_bind is None:
      return spatial_slots

    indices, valid = frame_window_indices(
        steps, self.config.num_neighbors, self.config.frame_stride,
        spatial_slots.device)
    window_batch = max(1, batch // indices.shape[1])

    slots = []
    for start in range(0, steps, window_batch):
      window = spatial_slots[indices[start:start + window_batch].flatten()]
      slots.append(
          self.solv.bind_temporal(window, valid[start:start + window_batch]))
    return torch.cat(slots, dim=0)

  @property
  def can_decode_rgb(self):
    return self.backbone.can_decode

  @torch.no_grad()
  def decode_patch_masks(self, slots):
    """Visualisation only: per-slot patch masks implied by a slot vector."""
    slots, squeeze = self._batch_slots(slots)
    masks = torch.cat([
        self.solv.decode_patch_masks(slots[start:start + self._decode_batch])
        for start in range(0, slots.shape[0], self._decode_batch)], dim=0)
    return masks[0] if squeeze else masks

  @torch.no_grad()
  def decode_rgb(self, slots):
    """Visualisation only: render slots as images, `(B, 3, H, W)` in [0, 1].

    The slots are decoded into backbone features and those features handed to
    the backbone's own decoder, so this shows what the slots kept of the frame
    rather than what any separately trained pixel decoder can guess.
    """
    slots, squeeze = self._batch_slots(slots)
    images = []
    for start in range(0, slots.shape[0], self._decode_batch):
      features, _ = self.solv.decode(slots[start:start + self._decode_batch])
      images.append(self.backbone.decode(features))
    images = torch.cat(images, dim=0).float().clamp(0, 1)
    return images[0] if squeeze else images

  def _batch_slots(self, slots):
    slots = torch.as_tensor(slots, device=self._device, dtype=torch.float32)
    squeeze = slots.ndim == 2
    return (slots[None] if squeeze else slots), squeeze
