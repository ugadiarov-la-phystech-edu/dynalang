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


def weight_dtype(module, default=torch.float32):
  """The dtype a module's own weights are stored in."""
  for parameter in module.parameters():
    return parameter.dtype
  return default


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
  """Continuous Cosmos image tokenizer, run straight from its released traces.

  dyn-O uses `encoder.jit` only as a weight file: cosmos' loader rebuilds the
  tokenizer in eager mode, fills it from that state dict and casts it to
  float32. Running the trace itself instead keeps the same weights without
  vendoring the tokenizer implementation, at the cost of the dtype the trace was
  captured in -- these files were traced in bfloat16, and a trace freezes the
  `dtype = x.dtype` its patcher read at capture time, so the tokenizer computes
  in bfloat16 no matter what is handed to it. Tensors are therefore brought to
  the module's dtype on the way in and back to float32 on the way out; casting
  the weights instead only breaks the halves that already agree with themselves.

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
    self.tokenizer = freeze(torch.jit.load(path, map_location='cpu'))
    self.tokenizer_dtype = weight_dtype(self.tokenizer)

    self.feat_res = [size // config.patch_size for size in config.resize_to]
    decoder_path = directory / 'decoder.jit'
    self.decoder = None
    self.decoder_dtype = torch.float32
    if decoder_path.is_file():
      self.decoder = freeze(torch.jit.load(decoder_path, map_location='cpu'))
      self.decoder_dtype = weight_dtype(self.decoder)

  @property
  def can_decode(self):
    return self.decoder is not None

  def forward(self, images):
    latent = self.tokenizer(images.to(self.tokenizer_dtype))
    if isinstance(latent, (tuple, list)):
      # The continuous tokenizer returns (latent, distribution parameters).
      latent = latent[0]
    return rearrange(latent, 'b c h w -> b (h w) c').float()

  def decode(self, features):
    """Render `(B, tokens, token_dim)` latents back into `(B, 3, H, W)`."""
    if self.decoder is None:
      raise FileNotFoundError(
          'rendering slots as pixels needs decoder.jit next to the Cosmos '
          'encoder.jit that this backbone was loaded from')
    latent = rearrange(
        features, 'b (h w) c -> b c h w',
        h=self.feat_res[0], w=self.feat_res[1])
    images = self.decoder(latent.to(self.decoder_dtype))
    if isinstance(images, (tuple, list)):
      images = images[0]
    return images.float()


def build_backbone(config, cosmos_checkpoint_dir=None, dino_model_name=None):
  if config.encoder.startswith('Cosmos'):
    return CosmosBackbone(config, cosmos_checkpoint_dir)
  if dino_model_name:
    return DinoBackbone(config, dino_model_name)
  return DinoBackbone(config)
