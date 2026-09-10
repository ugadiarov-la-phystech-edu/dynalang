"""
SlotExtractor - base interface for extracting slots from images.

Any slot-based model (SlotContrast, SLATE, SAVi, etc.) should
inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

try:
  import jax
  import jax.numpy as jnp
  HAS_JAX = True
except ImportError:
  HAS_JAX = False
  jax = None
  jnp = None


class SlotExtractor(ABC):
  """
  Base class for extracting slots from images.
  
  - n_slots: number of slots
  - dim: dimension of one slot
  - backbone_input_size: if backbone requires fixed size
  """
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  @property
  @abstractmethod
  def n_slots(self) -> int:
    """Number of slots."""
    pass
  
  @property
  @abstractmethod
  def dim(self) -> int:
    """Dimension of one slot."""
    pass
  
  @property
  def backbone_input_size(self) -> Optional[int]:
    """
    Backbone input size (if resize is required).
    If None, images are used as-is.
    """
    return None
  
  @abstractmethod
  def __call__(self, images, previous_slots=None):
    """
    Extract slots from images.
    
    Args:
      images: Batch of images, shape (B, H, W, C) or (B, C, H, W)
      previous_slots: Opaque recurrent carry returned by the previous call,
        with batch dimension first in every array leaf, or None.
    
    Returns:
      slots: Extracted slots, shape (B, n_slots, dim), suitable as an
        observation.
      predicted: Opaque recurrent carry suitable as `previous_slots` on the
        next call. For models without separate state, this can equal `slots`.
    """
    pass
  
  def get_slots(self, images, previous_slots=None, to_numpy=True):
    """
    Convenient method for slot extraction with automatic processing.
    
    Handles:
    - Single image vs batch
    - Conversion to/from numpy
    - Normalization [0, 255] -> [0, 1]
    - Resize if backbone_input_size is needed
    
    Args:
      images: Images in format (H, W, C) or (B, H, W, C),
              dtype uint8 [0, 255]
      previous_slots: Opaque recurrent carry returned by a previous call.
        Every array leaf is unbatched for one image and batched otherwise.
      to_numpy: Convert result to numpy array
    
    Returns:
      slots: shape (n_slots, dim) or (B, n_slots, dim)
      predicted: Opaque carry to pass back as `previous_slots`.
    """
    # Determine single image or batch
    one_image = len(images.shape) == 3
    if one_image:
      batch_images = images[np.newaxis, ...]  # (H, W, C) -> (1, H, W, C)
    else:
      batch_images = images  # (B, H, W, C)
    
    def _tree_map(fn, value):
      if isinstance(value, dict):
        return {key: _tree_map(fn, item) for key, item in value.items()}
      if isinstance(value, tuple):
        return tuple(_tree_map(fn, item) for item in value)
      if isinstance(value, list):
        return [_tree_map(fn, item) for item in value]
      return fn(value)

    if previous_slots is not None and one_image:
      batch_prev_slots = _tree_map(lambda x: x[np.newaxis, ...], previous_slots)
    else:
      batch_prev_slots = previous_slots
    
    # Call slot extraction
    slots, predicted = self(batch_images, previous_slots=batch_prev_slots)

    def _postprocess_leaf(x):
      if one_image:
        x = x[0]
      if to_numpy:
        if HAS_JAX and isinstance(x, jnp.ndarray):
          x = np.array(x)
        elif hasattr(x, 'detach'):  # PyTorch tensor
          x = x.detach().cpu().numpy()
      return x

    return (
        _tree_map(_postprocess_leaf, slots),
        _tree_map(_postprocess_leaf, predicted),
    )


class DummySlotExtractor(SlotExtractor):
  """
  Simple dummy extractor for testing.
  Returns random slots.
  """
  
  def __init__(self, n_slots=4, dim=64):
    super().__init__()
    self._n_slots = n_slots
    self._dim = dim
  
  @property
  def n_slots(self) -> int:
    return self._n_slots
  
  @property
  def dim(self) -> int:
    return self._dim
  
  def __call__(self, images, previous_slots=None):
    """Returns random slots."""
    batch_size = images.shape[0]
    
    # Generate random slots
    slots = np.random.randn(batch_size, self._n_slots, self._dim).astype(np.float32)
    
    # If previous slots exist, add small perturbation
    if previous_slots is not None:
      slots = 0.8 * previous_slots + 0.2 * slots
    
    # No separate transition model: carry state equals the observation.
    return slots, slots