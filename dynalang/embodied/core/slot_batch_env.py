"""
BatchSlotExtractorEnv - batch environment with slot extraction from images.
"""

import functools
import numpy as np

from . import base
from . import space as spacelib
from .batch_env import BatchEnv


def _tree_take(tree, indices):
  if tree is None:
    return None
  if isinstance(tree, dict):
    return {key: _tree_take(value, indices) for key, value in tree.items()}
  if isinstance(tree, tuple):
    return tuple(_tree_take(value, indices) for value in tree)
  if isinstance(tree, list):
    return [_tree_take(value, indices) for value in tree]
  return tree[indices]


def _tree_scatter(tree, indices, values, batch_size):
  if values is None:
    return tree
  if isinstance(values, dict):
    current = tree or {}
    return {
        key: _tree_scatter(current.get(key), indices, value, batch_size)
        for key, value in values.items()
    }
  if isinstance(values, tuple):
    current = tree or (None,) * len(values)
    return tuple(
        _tree_scatter(old, indices, value, batch_size)
        for old, value in zip(current, values)
    )
  if isinstance(values, list):
    current = tree or [None] * len(values)
    return [
        _tree_scatter(old, indices, value, batch_size)
        for old, value in zip(current, values)
    ]
  if tree is None:
    tree = np.zeros((batch_size,) + values.shape[1:], dtype=values.dtype)
  tree[indices] = values
  return tree


class BatchSlotExtractorEnv(BatchEnv):
  """
  Batch environment that automatically extracts slots from images.
  
  Inherits from BatchEnv and adds automatic slot extraction
  after each step(). Supports recurrent slot initialization
  for tracking objects over time.
  
  Args:
    envs: List of environments (as in BatchEnv)
    parallel: Parallel environment processing (as in BatchEnv)
    slot_extractor: SlotExtractor instance (e.g., SlotContrastExtractor)
    image_key: Observation key with image (default 'image')
    use_previous_slots: Whether to use previous slots for initialization
    initialize_twice: Whether to initialize twice at episode start
    flatten_slots: Whether to return slots in flat form (n_slots * dim,)
  """
  
  def __init__(
      self,
      envs,
      parallel,
      slot_extractor,
      image_key='image',
      use_previous_slots=False,
      initialize_twice=False,
      flatten_slots=False
  ):
    self._slot_extractor = slot_extractor
    self._image_key = image_key
    self._use_previous_slots = use_previous_slots
    self._initialize_twice = initialize_twice
    self._flatten_slots = flatten_slots
    
    super().__init__(envs, parallel)
    
    if image_key not in super().obs_space:
      raise ValueError(
          f"Image key '{image_key}' not found in observation space. "
          f"Available keys: {list(super().obs_space.keys())}"
      )

    self._previous_state = None
  
  @functools.cached_property
  def obs_space(self):
    """Extend observation space with slots."""
    spaces = dict(super().obs_space)
    
    if self._flatten_slots:
      # (n_slots * dim,)
      flat_dim = self._slot_extractor.n_slots * self._slot_extractor.dim
      spaces['flatten_slots'] = spacelib.Space(
          np.float32,
          shape=(flat_dim,),
          low=-np.inf,
          high=np.inf
      )
    else:
      # (n_slots, dim)
      spaces['slot'] = spacelib.Space(
          np.float32,
          shape=(self._slot_extractor.n_slots, self._slot_extractor.dim),
          low=-np.inf,
          high=np.inf
      )
    
    return spaces
  
  def step(self, action):

    obs = super().step(action)
    images = obs[self._image_key]  # shape: (n_envs, H, W, C)
    
    if self._use_previous_slots:
      is_first = obs['is_first']
      n_envs = len(self._envs)
      slots = np.zeros(
          (n_envs, self._slot_extractor.n_slots, self._slot_extractor.dim),
          dtype=np.float32,
      )
      next_state = None
      
      if is_first.any():
        first_images = images[is_first]
        first_slots, first_carry = self._slot_extractor.get_slots(
            first_images,
            previous_slots=None,
            to_numpy=True,
        )
        slots[is_first] = first_slots
        
        if self._initialize_twice:
          second_slots, second_carry = self._slot_extractor.get_slots(
              first_images,
              previous_slots=first_carry,
              to_numpy=True,
          )
          slots[is_first] = second_slots
          first_carry = second_carry
        next_state = _tree_scatter(
            next_state, is_first, first_carry, n_envs
        )
      
      if not is_first.all():
        if self._previous_state is None:
          raise RuntimeError(
              "slot extractor has continuing environments but no recurrent state"
          )
        continuing_images = images[~is_first]
        continuing_prev_carry = _tree_take(self._previous_state, ~is_first)
        continuing_slots, continuing_carry = self._slot_extractor.get_slots(
            continuing_images,
            previous_slots=continuing_prev_carry,
            to_numpy=True,
        )
        slots[~is_first] = continuing_slots
        next_state = _tree_scatter(
            next_state, ~is_first, continuing_carry, n_envs
        )
      
      self._previous_state = next_state
    
    else:
      slots, _ = self._slot_extractor.get_slots(
          images,
          previous_slots=None,
          to_numpy=True
      )
    if self._flatten_slots:
      obs['flatten_slots'] = slots.reshape(slots.shape[0], -1)
    else:
      obs['slot'] = slots
    return obs
  
  def close(self):
    """Close all environments."""
    super().close()
    # Clean up slot extractor if needed
    if hasattr(self._slot_extractor, 'close'):
      self._slot_extractor.close()
