"""Torch SlotContrast reference for extractor-level parity tests.

Replicates slotcontrast_extractor.py:39-62 module assembly from a shipped
config yaml (with pretrained=False so no network/checkpoint is needed) and
its __call__/_forward_torch preprocessing + forward, exposed as plain
functions over numpy arrays."""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

REPO = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = (
    REPO / 'dynalang/embodied/torch/ocr/slotcontrast/config')
CONFIGS = {
    'dino_v1_s8': CONFIG_DIR / 'episode-dataset_pick-specific_dino-v1.yaml',
    'dinov2_s14': CONFIG_DIR / 'episode-dataset_pick-specific.yaml',
}
INPUT_SIZES = {'dino_v1_s8': 224, 'dinov2_s14': 336}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TorchSlotContrast:
  """initializer + encoder + corrector from a SlotContrast config yaml,
  randomly initialized (pretrained=False), eval mode. The predictor is
  omitted: its output never reaches the extracted slots."""

  def __init__(self, variant, seed=0):
    from embodied.torch.ocr.slotcontrast import configuration, modules
    torch.manual_seed(seed)
    config = configuration.load_config(str(CONFIGS[variant]))
    model_config = config.model
    model_config.encoder.backbone.pretrained = False
    self.input_size = INPUT_SIZES[variant]
    self.initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, 'FrameEncoder')
    grouper = modules.build_grouper(model_config.grouper)
    self.encoder = modules.MapOverTime(encoder)
    self.processor = modules.ScanOverTime(modules.build_video(
        model_config.latent_processor, 'LatentProcessor',
        corrector=grouper, predictor=None))
    self.normalization = torchvision.transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)
    for module in (self.initializer, self.encoder, self.processor):
      module.eval()
      module.requires_grad_(False)

  def state_dict_numpy(self):
    """Combined state dict with the checkpoint's key namespace
    (initializer. / encoder.module. / processor.module.corrector.)."""
    out = {}
    for prefix, module in (
        ('initializer', self.initializer),
        ('encoder', self.encoder),
        ('processor', self.processor)):
      for key, value in module.state_dict().items():
        out[f'{prefix}.{key}'] = value.detach().cpu().numpy()
    return out

  def __call__(self, images, previous_slots=None):
    """slotcontrast_extractor.SlotContrastExtractor.__call__ equivalent.
    images: uint8 (B, H, W, C). previous_slots: (B, S, D) or None.
    Returns (B, S, D) float32."""
    x = torch.as_tensor(
        images.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    if x.shape[-1] != self.input_size:
      x = F.interpolate(
          x, size=(self.input_size, self.input_size),
          mode='bilinear', align_corners=False)
    x = self.normalization(x)
    x = x.unsqueeze(1)  # (B, 1, C, H, W) — video path, T=1
    with torch.no_grad():
      features = self.encoder(x)['features']
      if previous_slots is None:
        slots = self.initializer(batch_size=x.shape[0])
      else:
        slots = torch.as_tensor(previous_slots, dtype=torch.float32)
      out = self.processor(slots, features)
    return out['state'][:, 0].numpy()


def env_style_rollout(model, images, is_first, initialize_twice=True):
  """BatchSlotExtractorEnv.step logic (slot_batch_env.py:91-123) over a
  (B, T, H, W, C) uint8 sequence. Returns (B, T, S, D) slots."""
  batch, length = images.shape[:2]
  probe = model(images[:, 0])
  prev = np.zeros_like(probe)
  outputs = []
  for t in range(length):
    frame = images[:, t]
    first = is_first[:, t].astype(bool)
    slots = np.zeros_like(prev)
    if first.any():
      first_slots = model(frame[first], previous_slots=None)
      slots[first] = first_slots
      if initialize_twice:
        slots[first] = model(frame[first], previous_slots=first_slots)
    if (~first).any():
      slots[~first] = model(frame[~first], previous_slots=prev[~first])
    outputs.append(slots.copy())
    prev = slots
  return np.stack(outputs, axis=1)
