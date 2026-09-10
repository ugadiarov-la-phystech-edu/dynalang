import numpy as np
import torch

from embodied.core.slot_extractor import SlotExtractor
from embodied.torch.ocr.solvsam.encoder import SolvSamEncoder


class SolvSamExtractor(torch.nn.Module, SlotExtractor):
  """Frozen causal SOLV-SAM encoder behind the dynalang slot interface.

  The carry is the encoder's window of past slots rather than the previous
  slots themselves, so it stays opaque to the environment wrapper.
  """

  def __init__(
      self, checkpoint_path, device='cuda', cosmos_checkpoint_dir=None,
      dino_model_name=None, batch_size=64):
    torch.nn.Module.__init__(self)
    self.encoder = SolvSamEncoder(
        checkpoint_path=checkpoint_path,
        device=device,
        cosmos_checkpoint_dir=cosmos_checkpoint_dir,
        dino_model_name=dino_model_name,
        batch_size=batch_size,
    )

  @property
  def n_slots(self):
    return self.encoder.n_slots

  @property
  def dim(self):
    return self.encoder.slot_dim

  @property
  def decoder(self):
    """The frozen decoder visualisation reads slots back through."""
    return self.encoder.solv.dec

  def __call__(self, images, previous_slots=None):
    """Extract slots for one step of a batch of environments.

    Args:
      images: numpy array (B, H, W, C), uint8 in [0, 255].
      previous_slots: carry returned by the previous call, or None at the
        first frame of an episode.

    Returns:
      slots: (B, n_slots, dim) slots of the current frame.
      state: carry to pass back as `previous_slots` on the next call.
    """
    return self.encoder.forward_step(np.asarray(images), previous_slots)

  def decode_masks(self, slots):
    """Visualisation only: per-slot patch masks for arbitrary slot vectors."""
    masks = self.encoder.decode_patch_masks(slots)
    return masks.detach().cpu().numpy()

  @property
  def can_decode_images(self):
    return self.encoder.can_decode_rgb

  def decode_images(self, slots):
    """Visualisation only: slots rendered as uint8 images, (..., H, W, 3)."""
    images = self.encoder.decode_rgb(slots)
    images = images.movedim(-3, -1).mul(255).round().to(torch.uint8)
    return images.detach().cpu().numpy()

  def episode_slots(self, images):
    """Reference causal pass over a whole episode of images."""
    slots = self.encoder.forward_episode(np.asarray(images))
    return slots.detach().cpu().numpy()
