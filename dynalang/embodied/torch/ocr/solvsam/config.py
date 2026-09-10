"""Inference-side configuration of a frozen causal SOLV-SAM encoder.

The fields mirror the training config of the dyn-O encoder, so a checkpoint
written there can be replayed here. Only the subset that shapes the modules
this package needs is kept; training-only knobs are ignored.
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple


@dataclasses.dataclass
class SolvSamConfig:

  # Visual backbone. Cosmos names encode their patch size, DINO names their
  # patch size and feature width.
  encoder: str = 'Cosmos-0.1-Tokenizer-CI16x16'
  resize_to: Tuple[int, int] = (224, 224)

  # Slot attention.
  num_slots: int = 31
  slot_att_iter: int = 3
  slot_dim: int = 256

  # Causal temporal binding: num_neighbors past frames plus the current one,
  # spaced frame_stride apart. Zero disables the temporal binder.
  num_neighbors: int = 4
  frame_stride: int = 1
  temporal_depth: int = 3
  temporal_mode: str = 'causal'

  # Feature decoder, used here only to turn slots back into patch masks.
  decode_segmentation: bool = False

  # Derived from the backbone name.
  patch_size: int = dataclasses.field(init=False)
  token_dim: int = dataclasses.field(init=False)
  token_num: int = dataclasses.field(init=False)

  def __post_init__(self):
    if self.temporal_mode != 'causal':
      raise ValueError(
          f'unsupported temporal_mode {self.temporal_mode!r}; this package '
          'only runs encoders trained on windows without future frames')
    if self.num_neighbors < 0 or self.frame_stride < 1:
      raise ValueError('num_neighbors must be >= 0 and frame_stride >= 1')
    self.resize_to = tuple(self.resize_to)
    if self.encoder.startswith('Cosmos'):
      self.patch_size = int(self.encoder.split('x')[-1])
      self.token_dim = 16
    else:
      self.patch_size = int(self.encoder.split('-')[2])
      self.token_dim = 768
    height, width = self.resize_to
    if height % self.patch_size or width % self.patch_size:
      raise ValueError(
          f'resize_to {self.resize_to} is not divisible by the patch size '
          f'{self.patch_size} of backbone {self.encoder}')
    self.token_num = (height * width) // (self.patch_size ** 2)

  @property
  def num_frames(self):
    return self.num_neighbors + 1

  @classmethod
  def from_checkpoint(cls, saved_config: Dict[str, Any]):
    """Rebuild the config a checkpoint was trained with.

    Older checkpoints predate `temporal_mode`. Those with a temporal binder
    used a window centred on the current frame, which cannot be evaluated
    online, so they are rejected rather than silently reinterpreted.
    """
    if not isinstance(saved_config, dict):
      saved_config = dataclasses.asdict(saved_config)
    mode = saved_config.get(
        'temporal_mode',
        'centered' if saved_config.get('num_neighbors', 0) > 0 else 'causal')
    if mode != 'causal':
      raise ValueError(
          f'checkpoint was trained with temporal_mode={mode!r}; retrain the '
          'encoder with the causal temporal binder')
    fields = {
        field.name for field in dataclasses.fields(cls) if field.init}
    kwargs = {
        key: value for key, value in saved_config.items() if key in fields}
    kwargs['temporal_mode'] = 'causal'
    return cls(**kwargs)


def read_checkpoint_config(checkpoint: Dict[str, Any]) -> Optional[Dict]:
  """Pull the training config out of a dyn-O encoder checkpoint."""
  saved = checkpoint.get('config', checkpoint.get('args'))
  if saved is None:
    return None
  if not isinstance(saved, dict):
    saved = vars(saved)
  return saved
