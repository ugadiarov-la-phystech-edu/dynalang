import dataclasses
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn

from embodied.torch.ocr.solvsam import encoder as encoder_module
from embodied.torch.ocr.solvsam.config import SolvSamConfig
from embodied.torch.ocr.solvsam.encoder import SolvSamEncoder
from embodied.torch.ocr.solvsam.nets import SolvSam, frame_window_indices


class FakeBackbone(nn.Module):
  """Deterministic stand-in for a pretrained visual backbone."""

  normalize_mean = None
  normalize_std = None

  def __init__(self, config, *args, **kwargs):
    super().__init__()
    self.config = config

  def forward(self, images):
    wanted = self.config.token_num * self.config.token_dim
    flat = images.mean(dim=1).flatten(start_dim=1)
    assert flat.shape[1] >= wanted, flat.shape
    return flat[:, :wanted].reshape(
        -1, self.config.token_num, self.config.token_dim)


class FakeCosmosEncoder(nn.Module):
  """Shaped like the real tokenizer: a latent plus distribution parameters."""

  def __init__(self, patch_size, token_dim):
    super().__init__()
    self.conv = nn.Conv2d(3, token_dim, patch_size, stride=patch_size)

  def forward(self, images):
    return self.conv(images), (torch.zeros(1), torch.zeros(1))


class FakeCosmosDecoder(nn.Module):

  def __init__(self, patch_size, token_dim):
    super().__init__()
    self.conv = nn.ConvTranspose2d(token_dim, 3, patch_size, stride=patch_size)

  def forward(self, latent):
    return self.conv(latent)


def write_cosmos_jit(directory, config, with_decoder=True):
  directory.mkdir(parents=True, exist_ok=True)
  torch.jit.save(
      torch.jit.script(FakeCosmosEncoder(config.patch_size, config.token_dim)),
      directory / 'encoder.jit')
  if with_decoder:
    torch.jit.save(
        torch.jit.script(FakeCosmosDecoder(config.patch_size, config.token_dim)),
        directory / 'decoder.jit')
  return directory


def tiny_config(num_neighbors=2, frame_stride=2):
  return SolvSamConfig(
      encoder='Cosmos-0.1-Tokenizer-CI16x16',
      resize_to=(32, 32),
      num_slots=3,
      slot_att_iter=1,
      slot_dim=8,
      num_neighbors=num_neighbors,
      frame_stride=frame_stride,
      temporal_depth=1,
  )


def write_checkpoint(path, config):
  torch.manual_seed(0)
  model = SolvSam(config)
  torch.save({
      'config': dataclasses.asdict(config),
      'model': model.state_dict(),
  }, path)


def build_encoder(path, monkeypatch, **kwargs):
  monkeypatch.setattr(
      encoder_module, 'build_backbone',
      lambda config, *args, **kw: FakeBackbone(config))
  return SolvSamEncoder(checkpoint_path=path, device='cpu', **kwargs)


def frames(steps, seed=0):
  rng = np.random.default_rng(seed)
  return rng.integers(0, 256, size=(steps, 32, 32, 3), dtype=np.uint8)


def test_causal_windows_never_reach_into_the_future():
  indices, valid = frame_window_indices(8, num_neighbors=3, frame_stride=2)
  assert indices.shape == (8, 4)
  for current in range(8):
    assert torch.all(indices[current] <= current)
    expected = torch.tensor([
        max(0, current - 6), max(0, current - 4), max(0, current - 2), current])
    assert torch.equal(indices[current], expected)
    assert torch.equal(valid[current], torch.tensor(
        [current >= 6, current >= 4, current >= 2, True]))


def test_streaming_matches_episode_pass(tmp_path, monkeypatch):
  config = tiny_config()
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  encoder = build_encoder(path, monkeypatch)

  images = frames(9)
  episode = encoder.forward_episode(images)

  state = None
  streamed = []
  for image in images:
    slots, state = encoder.forward_step(image[None], state)
    streamed.append(slots)
  streamed = torch.cat(streamed, dim=0)

  assert episode.shape == (9, config.num_slots, config.slot_dim)
  torch.testing.assert_close(streamed, episode, atol=1e-5, rtol=1e-5)


def test_episode_start_does_not_depend_on_earlier_episodes(
    tmp_path, monkeypatch):
  config = tiny_config()
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  encoder = build_encoder(path, monkeypatch)

  images = frames(4, seed=1)
  first_alone, _ = encoder.forward_step(images[:1], None)

  _, state = encoder.forward_step(images[1:2], None)
  _, state = encoder.forward_step(images[2:3], state)
  first_again, _ = encoder.forward_step(images[:1], None)

  torch.testing.assert_close(first_alone, first_again)


def test_batched_steps_match_single_environment(tmp_path, monkeypatch):
  config = tiny_config(num_neighbors=3, frame_stride=1)
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  encoder = build_encoder(path, monkeypatch)

  left, right = frames(5, seed=2), frames(5, seed=3)
  state = None
  batched = []
  for step in range(5):
    slots, state = encoder.forward_step(
        np.stack([left[step], right[step]]), state)
    batched.append(slots)
  batched = torch.stack(batched, dim=1)

  for index, single in enumerate((left, right)):
    state = None
    for step in range(5):
      slots, state = encoder.forward_step(single[step][None], state)
      torch.testing.assert_close(
          slots[0], batched[index, step], atol=1e-5, rtol=1e-5)


def test_rejects_checkpoint_without_causal_binder(tmp_path, monkeypatch):
  config = tiny_config()
  path = tmp_path / 'centered.pt'
  write_checkpoint(path, config)
  # Checkpoints from before the causal rewrite carry no temporal_mode and
  # were trained on a window centred on the current frame.
  saved = torch.load(path, map_location='cpu', weights_only=False)
  del saved['config']['temporal_mode']
  torch.save(saved, path)

  try:
    build_encoder(path, monkeypatch)
  except ValueError as error:
    assert 'temporal_mode' in str(error)
  else:
    raise AssertionError('a centered checkpoint must be rejected')


def test_rejects_checkpoint_with_mismatched_weights(tmp_path, monkeypatch):
  config = tiny_config()
  path = tmp_path / 'partial.pt'
  write_checkpoint(path, config)
  saved = torch.load(path, map_location='cpu', weights_only=False)
  saved['model'] = {
      key: value for key, value in saved['model'].items()
      if not key.startswith('t_bind.')}
  torch.save(saved, path)

  try:
    build_encoder(path, monkeypatch)
  except ValueError as error:
    assert 't_bind' in str(error)
  else:
    raise AssertionError('missing temporal binder weights must be rejected')


def test_decode_patch_masks_sums_to_one_over_slots(tmp_path, monkeypatch):
  config = tiny_config()
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  encoder = build_encoder(path, monkeypatch)

  slots, _ = encoder.forward_step(frames(1, seed=4), None)
  masks = encoder.decode_patch_masks(slots)
  assert masks.shape == (1, config.num_slots, config.token_num)
  torch.testing.assert_close(
      masks.sum(dim=1), torch.ones(1, config.token_num))


def test_cosmos_backbone_encodes_and_renders(tmp_path):
  config = tiny_config()
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  # The directory name is the backbone name, so a shared checkpoint root works.
  write_cosmos_jit(tmp_path / 'weights' / config.encoder, config)

  encoder = SolvSamEncoder(
      checkpoint_path=path, device='cpu',
      cosmos_checkpoint_dir=tmp_path / 'weights')
  assert encoder.can_decode_rgb

  slots, _ = encoder.forward_step(frames(2, seed=5), None)
  assert slots.shape == (2, config.num_slots, config.slot_dim)

  rendered = encoder.decode_rgb(slots)
  assert rendered.shape == (2, 3) + config.resize_to
  assert rendered.min() >= 0 and rendered.max() <= 1

  single = encoder.decode_rgb(slots[0])
  assert single.shape == (3,) + config.resize_to


def test_decoding_in_chunks_changes_nothing(tmp_path):
  config = tiny_config()
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  directory = write_cosmos_jit(tmp_path / config.encoder, config)

  chunked = SolvSamEncoder(
      checkpoint_path=path, device='cpu', cosmos_checkpoint_dir=directory,
      batch_size=4)
  whole = SolvSamEncoder(
      checkpoint_path=path, device='cpu', cosmos_checkpoint_dir=directory,
      batch_size=1024)
  slots, _ = chunked.forward_step(frames(7, seed=7), None)
  assert chunked._decode_batch < len(slots) < whole._decode_batch

  torch.testing.assert_close(
      chunked.decode_patch_masks(slots), whole.decode_patch_masks(slots))
  torch.testing.assert_close(
      chunked.decode_rgb(slots), whole.decode_rgb(slots))


def test_rendering_needs_the_backbone_decoder(tmp_path):
  config = tiny_config()
  path = tmp_path / 'causal.pt'
  write_checkpoint(path, config)
  directory = write_cosmos_jit(
      tmp_path / config.encoder, config, with_decoder=False)

  encoder = SolvSamEncoder(
      checkpoint_path=path, device='cpu', cosmos_checkpoint_dir=directory)
  assert not encoder.can_decode_rgb

  slots, _ = encoder.forward_step(frames(1, seed=6), None)
  try:
    encoder.decode_rgb(slots)
  except FileNotFoundError as error:
    assert 'decoder.jit' in str(error)
  else:
    raise AssertionError('rendering without decoder.jit must fail loudly')
