"""Adds SlotContrast-vs-world-model mask comparisons to agent.report()'s
output during training (report/openl_slot_mask_overlay,
report/openl_slots_grid_soft, report/openl_slots_grid_hard), mirroring the
truth/model/error convention of agent.py's own openl_{cnn_key} videos.

Both the real ('slot'/'flatten_slots') and world-model-predicted
('model_slot_raw') slot vectors are decoded into per-patch masks via the
extractor's frozen SlotContrast decoder -- the only way to get a spatial
mask for the world model's own prediction, since it only ever
sees/predicts slot vectors, never patch features. Runs as a plain-Python
post-processing step since it needs torch calls that can't happen inside
the jitted agent.report().
"""
import numpy as np


# Distinct colors for up to 10 slots; cycles if there are more.
_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [128, 128, 128],
], dtype=np.uint8)


def _patch_grid(masks):
  """(n_slots, n_patches) -> (n_slots, side, side) float32."""
  n_slots, n_patches = masks.shape
  side = int(round(n_patches ** 0.5))
  assert side * side == n_patches, f'Non-square patch grid: {n_patches}'
  return masks.reshape(n_slots, side, side).astype(np.float32)


def _resize_mask(mask2d, w, h, resample):
  from PIL import Image as PILImage
  return np.asarray(PILImage.fromarray(mask2d, mode='F').resize((w, h), resample))


def _argmax_assignment(masks, w, h):
  from PIL import Image as PILImage
  grid = _patch_grid(masks)
  upsampled = np.stack([_resize_mask(g, w, h, PILImage.NEAREST) for g in grid])
  return upsampled.argmax(0)  # (H, W)


def _upscale(image, factor):
  from PIL import Image as PILImage
  h, w = image.shape[:2]
  return np.asarray(PILImage.fromarray(image).resize(
      (w * factor, h * factor), PILImage.NEAREST))


def masks_to_overlay(image, masks, alpha=0.55):
  """(H, W, 3) uint8 image tinted by each pixel's argmax slot."""
  h, w = image.shape[:2]
  assign = _argmax_assignment(masks, w, h)
  colors = _PALETTE[assign % len(_PALETTE)]
  overlay = image.astype(np.float32) * (1 - alpha) + colors.astype(np.float32) * alpha
  return np.clip(overlay, 0, 255).astype(np.uint8)


def _label_panel(panel, border_color, label):
  from PIL import Image as PILImage, ImageDraw
  pil = PILImage.fromarray(panel)
  draw = ImageDraw.Draw(pil)
  draw.rectangle(
      [0, 0, pil.width - 1, pil.height - 1],
      outline=tuple(int(c) for c in border_color), width=3)
  draw.text((5, 5), label, fill=(0, 0, 0))
  draw.text((4, 4), label, fill=(255, 255, 255))
  return np.asarray(pil)


def _per_slot_mask(arg, i, hard):
  """arg: (H, W) argmax map if hard, else slot i's own resized soft map.
  Returns a (H, W) float32 mask in [0, 1] for slot `i`."""
  if hard:
    return (arg == i).astype(np.float32)
  return arg / (arg.max() + 1e-8)


def _slot_panel(image, mask_hw, w, h, scale):
  from PIL import Image as PILImage
  mask = mask_hw[..., None]
  panel = image.astype(np.float32) * mask + 255.0 * (1 - mask)
  panel = np.clip(panel, 0, 255).astype(np.uint8)
  return np.asarray(PILImage.fromarray(panel).resize(
      (w * scale, h * scale), PILImage.NEAREST))


def _diff_panel(mask_hw, w, h, scale):
  """mask_hw: (H, W) error in [0, 1], agent.py's (model - truth + 1) / 2
  convention -- rendered as flat grayscale, not composited with the image."""
  from PIL import Image as PILImage
  gray = np.clip(mask_hw * 255.0, 0, 255).astype(np.uint8)
  panel = np.stack([gray] * 3, axis=-1)
  return np.asarray(PILImage.fromarray(panel).resize(
      (w * scale, h * scale), PILImage.NEAREST))


def _arrange_grid(panels, cols=None):
  cols = cols or len(panels)
  rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]
  row_imgs = [np.concatenate(row, axis=1) for row in rows]
  max_w = max(r.shape[1] for r in row_imgs)
  row_imgs = [
      np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)), constant_values=255)
      for r in row_imgs]
  return np.concatenate(row_imgs, axis=0)


def combined_slot_grid(
    image, top_masks, bottom_masks, hard, scale=3, cols=None,
    top_label='truth', bottom_label='model'):
  """Per-slot truth/model/error stack: top/bottom are `top_masks`/
  `bottom_masks` (each (n_slots, n_patches)) decoded through the same
  SlotContrast decoder, bottom row is their (model - truth + 1) / 2 diff."""
  from PIL import Image as PILImage
  h, w = image.shape[:2]
  n_slots = top_masks.shape[0]
  top_grid = _patch_grid(top_masks)
  bottom_grid = _patch_grid(bottom_masks)
  top_assign = _argmax_assignment(top_masks, w, h) if hard else None
  bottom_assign = _argmax_assignment(bottom_masks, w, h) if hard else None
  columns = []
  for i in range(n_slots):
    if hard:
      top_mask_i = _per_slot_mask(top_assign, i, hard=True)
      bottom_mask_i = _per_slot_mask(bottom_assign, i, hard=True)
    else:
      top_mask_i = _per_slot_mask(
          _resize_mask(top_grid[i], w, h, PILImage.BILINEAR), i, hard=False)
      bottom_mask_i = _per_slot_mask(
          _resize_mask(bottom_grid[i], w, h, PILImage.BILINEAR), i, hard=False)
    error_i = (bottom_mask_i - top_mask_i + 1) / 2

    color = _PALETTE[i % len(_PALETTE)]
    top_panel = _label_panel(
        _slot_panel(image, top_mask_i, w, h, scale), color, f'S{i} {top_label}')
    bottom_panel = _label_panel(
        _slot_panel(image, bottom_mask_i, w, h, scale), color, f'S{i} {bottom_label}')
    error_panel = _label_panel(
        _diff_panel(error_i, w, h, scale), color, f'S{i} error')
    columns.append(np.concatenate([top_panel, bottom_panel, error_panel], axis=0))
  return _arrange_grid(columns, cols)


def _rgb_error_panel(truth_img, model_img):
  """agent.py's (model - truth + 1) / 2, applied to two rendered
  (H, W, 3) uint8 overlay images instead of raw camera pixels."""
  t = truth_img.astype(np.float32) / 255.0
  m = model_img.astype(np.float32) / 255.0
  error = (m - t + 1) / 2
  return np.clip(error * 255.0, 0, 255).astype(np.uint8)


def overlay_comparison_frame(image, real_masks, wm_masks, scale=4):
  """image | truth argmax overlay | model argmax overlay | their error."""
  truth_overlay = masks_to_overlay(image, real_masks)
  model_overlay = masks_to_overlay(image, wm_masks)
  error = _rgb_error_panel(truth_overlay, model_overlay)
  gray = (128, 128, 128)
  panels = [
      _label_panel(_upscale(image, scale), gray, 'image'),
      _label_panel(_upscale(truth_overlay, scale), gray, 'truth'),
      _label_panel(_upscale(model_overlay, scale), gray, 'model'),
      _label_panel(_upscale(error, scale), gray, 'error'),
  ]
  return np.concatenate(panels, axis=1)


def add_slot_mask_report(env, report, batch, seq_idx=0, max_frames=60):
  model_slot_raw = report.pop('model_slot_raw', None)
  extractor = getattr(env, '_slot_extractor', None)
  if extractor is None or extractor.decoder is None:
    return report
  if model_slot_raw is None:
    return report

  images = np.asarray(batch['image'])[seq_idx][:max_frames]
  model_slot_raw = np.asarray(model_slot_raw)
  if seq_idx >= model_slot_raw.shape[0]:
    return report
  model_slots = model_slot_raw[seq_idx][:max_frames].astype(np.float32)

  if 'slot' in batch:
    real_slots = np.asarray(batch['slot'])[seq_idx][:max_frames].astype(np.float32)
  elif 'flatten_slots' in batch:
    flat = np.asarray(batch['flatten_slots'])[seq_idx][:max_frames].astype(np.float32)
    real_slots = flat.reshape(-1, extractor.n_slots, extractor.dim)
  else:
    return report

  t_len = min(len(images), len(real_slots), len(model_slots))
  images, real_slots, model_slots = images[:t_len], real_slots[:t_len], model_slots[:t_len]

  real_masks = extractor.decode_masks(real_slots)  # (T, n_slots, n_patches)
  wm_masks = extractor.decode_masks(model_slots)  # (T, n_slots, n_patches)

  report['openl_slot_mask_overlay'] = np.stack([
      overlay_comparison_frame(images[i], real_masks[i], wm_masks[i])
      for i in range(t_len)], 0)
  report['openl_slots_grid_soft'] = np.stack([
      combined_slot_grid(images[i], real_masks[i], wm_masks[i], hard=False)
      for i in range(t_len)], 0)
  report['openl_slots_grid_hard'] = np.stack([
      combined_slot_grid(images[i], real_masks[i], wm_masks[i], hard=True)
      for i in range(t_len)], 0)

  return report
