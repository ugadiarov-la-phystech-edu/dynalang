"""Color-based object masks for ALE Pong.

ALE Pong uses a fixed palette (not generic white objects):
  background   (144,  72,  17)  brown playfield
  white        (236, 236, 236)  top/bottom walls + ball
  left paddle  (213, 130,  74)  orange CPU paddle + left score digit
  right paddle ( 92, 186,  92)  green player paddle + right score digit

Fixed semantic slots:
  0 scene         — background + horizontal wall lines
  1 left_paddle   — orange pixels (paddle + left score)
  2 right_paddle  — green pixels (paddle + right score)
  3 ball          — small white blob(s), excluding full-width wall lines
"""

import numpy as np

SLOT_NAMES = ('scene', 'left_paddle', 'right_paddle', 'ball')

# Official-ish ALE colors for the standard Pong ROM.
_PALETTE = np.array([
    [144, 72, 17],    # 0 background
    [236, 236, 236],  # 1 white
    [213, 130, 74],   # 2 left / orange
    [92, 186, 92],    # 3 right / green
], dtype=np.float32)

_CLASS_BG = 0
_CLASS_WHITE = 1
_CLASS_LEFT = 2
_CLASS_RIGHT = 3


def _classify_pixels(rgb):
  """Nearest-palette label map, shape (H, W)."""
  flat = rgb.reshape(-1, 3).astype(np.float32)
  dist = ((flat[:, None, :] - _PALETTE[None, :, :]) ** 2).sum(-1)
  return dist.argmin(-1).reshape(rgb.shape[:2])


def _white_wall_mask(white, wall_row_frac=0.4):
  """Full-width white scanlines used as court boundaries."""
  h, w = white.shape
  if h == 0 or w == 0:
    return white
  row_frac = white.sum(axis=1) / float(w)
  wall_rows = row_frac >= wall_row_frac
  return white & wall_rows[:, None]


def segment_pong_rgb(rgb, wall_row_frac=0.4):
  """Return bool masks for each Pong slot from an RGB frame."""
  labels = _classify_pixels(rgb)

  left_paddle = labels == _CLASS_LEFT
  right_paddle = labels == _CLASS_RIGHT

  white = labels == _CLASS_WHITE
  walls = _white_wall_mask(white, wall_row_frac=wall_row_frac)
  ball = white & ~walls

  scene = (labels == _CLASS_BG) | walls
  return [scene, left_paddle, right_paddle, ball]


def build_slot_images(rgb, num_slots=4):
  """Build masked RGB slot images from a Pong frame."""
  masks = segment_pong_rgb(rgb)
  if num_slots > len(masks):
    masks = masks + [np.zeros_like(masks[0])] * (num_slots - len(masks))
  else:
    masks = masks[:num_slots]

  slots = np.zeros((num_slots, *rgb.shape), dtype=np.uint8)
  for i, mask in enumerate(masks):
    slots[i] = rgb * mask[..., None]
  return slots
