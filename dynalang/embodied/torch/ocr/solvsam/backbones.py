"""Frozen visual backbones a SOLV-SAM encoder can be trained on top of.

Backbone weights never live in the encoder checkpoint: dyn-O keeps them frozen
and loads them from their own source, so this module has to reach the same
weights the slot modules were fitted against.
"""

from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def freeze(module):
  """Freeze a module, TorchScript included.

  `requires_grad_` is rejected outright on ScriptModules, so the flag has to be
  set on the parameters themselves.
  """
  for parameter in module.parameters():
    parameter.requires_grad = False
  module.eval()
  return module


class Backbone(nn.Module):
  """Turns frames into tokens, and optionally tokens back into pixels."""

  normalize_mean = None
  normalize_std = None

  @property
  def can_decode(self):
    return False

  def decode(self, features):
    raise NotImplementedError(
        f'{type(self).__name__} cannot turn features back into pixels')


class DinoBackbone(Backbone):
  """DINOv2 patch tokens, read straight off the last hidden state.

  dyn-O always instantiates `facebook/dinov2-base` here regardless of the
  configured backbone name, so this does too; the name only decides the patch
  size and token count.

  Pixels cannot be recovered from these features: dyn-O reconstructs them with
  a decoder trained alongside the slots, which is not part of this package.
  """

  normalize_mean = IMAGENET_DEFAULT_MEAN
  normalize_std = IMAGENET_DEFAULT_STD

  def __init__(self, config, model_name='facebook/dinov2-base'):
    super().__init__()
    from transformers import AutoModel
    self.dino = freeze(AutoModel.from_pretrained(model_name))

  def forward(self, images):
    self.dino.eval()
    return self.dino(images).last_hidden_state[:, 1:, :]


class CosmosBackbone(Backbone):
  """Continuous Cosmos image tokenizer, loaded from its TorchScript modules.

  dyn-O rebuilds the tokenizer in eager mode and then fills it from the same
  `encoder.jit` file, so loading that file directly gives the same weights and
  the same graph without vendoring the tokenizer implementation.

  `decoder.jit` is optional and only used to render slots as pixels. Nothing in
  it was trained here: for a Cosmos backbone dyn-O's RGB reconstruction is this
  same frozen decoder, so it carries no weights of its own.
  """

  # Cosmos latents are consumed as-is; dyn-O feeds the tokenizer images in
  # [0, 1] without further normalisation, and the slot modules were fitted on
  # exactly those features.
  normalize_mean = None
  normalize_std = None

  def __init__(self, config, checkpoint_dir):
    super().__init__()
    if not checkpoint_dir:
      raise ValueError(
          f'backbone {config.encoder} needs cosmos_checkpoint_dir to point at '
          'the directory holding its pretrained encoder.jit')
    directory = Path(checkpoint_dir).expanduser().resolve()
    if directory.name != config.encoder and (directory / config.encoder).is_dir():
      directory = directory / config.encoder
    path = directory / 'encoder.jit'
    if not path.is_file():
      raise FileNotFoundError(f'no Cosmos encoder found at {path}')
    self.tokenizer = self._load(path)

    self.feat_res = [size // config.patch_size for size in config.resize_to]
    decoder_path = directory / 'decoder.jit'
    self.decoder = None
    if decoder_path.is_file():
      self.decoder = self._load(decoder_path)

  @staticmethod
  def _load(path):
    """Load a tokenizer half the way dyn-O does, weights cast to float32.

    dyn-O goes through cosmos' `ImageTokenizer(dtype='float32')`, which casts
    the TorchScript weights on load. The released files are not all stored in
    the same dtype -- decoder.jit ships in bfloat16 -- so without the cast it
    rejects the float32 latents everything else here works in.
    """
    return freeze(torch.jit.load(path, map_location='cpu').float())

  @property
  def can_decode(self):
    return self.decoder is not None

  def forward(self, images):
    latent = self.tokenizer(images)
    if isinstance(latent, (tuple, list)):
      # The continuous tokenizer returns (latent, distribution parameters).
      latent = latent[0]
    return rearrange(latent, 'b c h w -> b (h w) c')

  def decode(self, features):
    """Render `(B, tokens, token_dim)` latents back into `(B, 3, H, W)`."""
    if self.decoder is None:
      raise FileNotFoundError(
          'rendering slots as pixels needs decoder.jit next to the Cosmos '
          'encoder.jit that this backbone was loaded from')
    latent = rearrange(
        features, 'b (h w) c -> b c h w',
        h=self.feat_res[0], w=self.feat_res[1])
    images = self.decoder(latent)
    if isinstance(images, (tuple, list)):
      images = images[0]
    return images


def build_backbone(config, cosmos_checkpoint_dir=None, dino_model_name=None):
  if config.encoder.startswith('Cosmos'):
    return CosmosBackbone(config, cosmos_checkpoint_dir)
  if dino_model_name:
    return DinoBackbone(config, dino_model_name)
  return DinoBackbone(config)
