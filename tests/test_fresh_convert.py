"""--fresh conversion (train SlotContrast from scratch inside dynalang):
slot_convert.fresh_state_dict builds a freshly initialized torch model in the
lightning-checkpoint key namespace, so the existing convert_all/expected_keys
pipeline and the strict jaxagent loader work unchanged. Backbone weights come
from the timm hub when the config says pretrained (network-gated test);
everything else is random."""

import pathlib

import numpy as np
import pytest

from conftest import torch_or_skip

pytest.importorskip('omegaconf')

from dynalang import slot_convert

CONFIG = (
    pathlib.Path(__file__).resolve().parent.parent /
    'dynalang/embodied/torch/ocr/slotcontrast/config/'
    'episode-dataset_pick-specific.yaml')
VARIANT = 'dinov2_s14'
DECODER_HIDDEN = 3

BACKBONE = 'encoder.module.backbone.model.'
RANDOM_PARTS = ('initializer.', 'processor.module.corrector.',
                'decoder.module.', 'encoder.module.output_transform.')


def _config(pretrained):
  from omegaconf import OmegaConf
  config = OmegaConf.load(CONFIG)
  OmegaConf.update(config, 'model.encoder.backbone.pretrained', pretrained)
  return config


def _fresh(pretrained, seed):
  torch_or_skip()
  return slot_convert.fresh_state_dict(_config(pretrained), seed=seed)


def test_fresh_converts_to_exact_expected_keys():
  sd = _fresh(pretrained=False, seed=0)
  assert not slot_convert.unconsumed_keys(
      sd, VARIANT, decoder_hidden=DECODER_HIDDEN)
  converted = slot_convert.convert_all(
      sd, VARIANT, 'enc', decoder_prefix='dec', decoder_hidden=DECODER_HIDDEN)
  expected = slot_convert.expected_keys(VARIANT, 'enc')
  expected |= slot_convert.decoder_expected_keys('dec', DECODER_HIDDEN)
  assert set(converted) == expected
  for key, value in converted.items():
    assert value.dtype == np.float32, key
    assert np.isfinite(value).all(), key


def test_fresh_seed_controls_random_parts():
  a = _fresh(pretrained=False, seed=0)
  b = _fresh(pretrained=False, seed=0)
  c = _fresh(pretrained=False, seed=1)
  assert set(a) == set(b) == set(c)
  for key in a:
    assert np.array_equal(a[key], b[key]), f'{key}: same seed must reproduce'
  changed = [k for k in a if not np.array_equal(a[k], c[k])]
  for part in RANDOM_PARTS:
    assert any(k.startswith(part) for k in changed), (
        f'{part}* must be randomly initialized (seed-dependent)')


def test_fresh_backbone_is_pretrained_dino():
  # Needs the timm hub weights (network or local cache); skip when offline.
  torch_or_skip()
  timm = pytest.importorskip('timm')
  try:
    ref = timm.create_model('vit_small_patch14_dinov2', pretrained=True)
  except Exception as e:  # download failure
    pytest.skip(f'timm pretrained weights unavailable: {e}')
  sd = _fresh(pretrained=True, seed=0)
  ref_sd = {k: v.detach().cpu().numpy() for k, v in ref.state_dict().items()}
  for key in ('cls_token', 'pos_embed', 'blocks.0.attn.qkv.weight',
              'blocks.11.mlp.fc2.weight'):
    got = sd[BACKBONE + key]
    assert np.array_equal(got, ref_sd[key]), f'backbone {key} != DINO weights'
  # Same backbone regardless of seed.
  sd2 = _fresh(pretrained=True, seed=1)
  assert np.array_equal(
      sd2[BACKBONE + 'blocks.0.attn.qkv.weight'],
      sd[BACKBONE + 'blocks.0.attn.qkv.weight'])
