"""Color-based object masks for ALE Pong.

ALE Pong uses a fixed palette (not generic white objects):
  background   (144,  72,  17)  brown playfield
  white        (236, 236, 236)  top/bottom walls + ball
  left paddle  (213, 130,  74)  orange CPU paddle + left score digit
  right paddle ( 92, 186,  92)  green player paddle + right score digit

Fixed semantic slots:
  0 scene         — background + horizontal wall lines + both score digits
  1 left_paddle   — orange paddle pixels (score digit excluded)
  2 right_paddle  — green paddle pixels (score digit excluded)
  3 ball          — small white blob(s), excluding full-width wall lines

The two score digits reuse the paddle colors, so they are split off by
position: colored pixels above the top court wall are treated as scoreboard
and folded into the scene slot instead of the paddle slots.
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


def _wall_rows(white, wall_row_frac=0.4):
  """Boolean per-row flag for full-width white scanlines (court walls)."""
  h, w = white.shape
  if h == 0 or w == 0:
    return np.zeros(h, dtype=bool)
  row_frac = white.sum(axis=1) / float(w)
  return row_frac >= wall_row_frac


def _white_wall_mask(white, wall_row_frac=0.4):
  """Full-width white scanlines used as court boundaries."""
  h, w = white.shape
  if h == 0 or w == 0:
    return white
  return white & _wall_rows(white, wall_row_frac=wall_row_frac)[:, None]


def _score_region_mask(white, wall_row_frac=0.4):
  """Rows above the top court wall = scoreboard/HUD band.

  The two score digits live above the topmost full-width wall line, so any
  colored pixel in this band is a score digit rather than a paddle.
  """
  h, w = white.shape
  region = np.zeros((h, w), dtype=bool)
  if h == 0 or w == 0:
    return region
  wall_idx = np.where(_wall_rows(white, wall_row_frac=wall_row_frac))[0]
  if wall_idx.size == 0:
    return region
  top_wall = int(wall_idx[0])
  region[:top_wall, :] = True
  return region


def segment_pong_rgb(rgb, wall_row_frac=0.4):
  """Return bool masks for each Pong slot from an RGB frame.

  Pixels are first separated purely by color (orange -> left, green -> right).
  The score digits share the paddle colors, so as a second step every colored
  pixel that lies in the scoreboard band (above the top court wall) is moved
  out of the paddle masks and into the background/scene mask.
  """
  labels = _classify_pixels(rgb)

  left_paddle = labels == _CLASS_LEFT
  right_paddle = labels == _CLASS_RIGHT

  white = labels == _CLASS_WHITE
  walls = _white_wall_mask(white, wall_row_frac=wall_row_frac)
  ball = white & ~walls

  # Reassign the left/right score digits to the background slot.
  score_region = _score_region_mask(white, wall_row_frac=wall_row_frac)
  left_score = left_paddle & score_region
  right_score = right_paddle & score_region
  left_paddle = left_paddle & ~score_region
  right_paddle = right_paddle & ~score_region

  scene = (labels == _CLASS_BG) | walls | left_score | right_score
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
