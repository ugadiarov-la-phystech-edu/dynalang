"""Convert a SlotContrast (PyTorch Lightning) checkpoint to a pickle of flat
ninjax parameters for the JAX slot encoder (dynalang/slot_nets.py).

Usage:
  python scripts/convert_slotcontrast_to_jax.py \
      --checkpoint /path/to/slotcontrast.ckpt \
      --config dynalang/embodied/torch/ocr/slotcontrast/config/episode-dataset_pick-specific_dino-v1.yaml \
      --output slots_dino_v1.pkl \
      [--prefix agent/wm/enc/slotcontrast]

The output pickle holds {ninjax_path: np.float32 array} and is loaded by the
jaxagent hook via `encoder.slotcontrast.jax_checkpoint`.
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(directory))
sys.path.insert(0, str(directory / 'dynalang'))

from dynalang import slot_convert  # noqa: E402

MODEL_TO_VARIANT = {
    'vit_small_patch8_224_dino': 'dino_v1_s8',
    'vit_small_patch14_dinov2': 'dinov2_s14',
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default=None,
                      help='trained SlotContrast .ckpt (omit with --fresh)')
  parser.add_argument('--config', required=True,
                      help='SlotContrast config yaml the checkpoint was trained with')
  parser.add_argument('--output', required=True)
  parser.add_argument('--fresh', action='store_true',
                      help='no checkpoint: convert a freshly initialized model '
                           '(backbone pretrained from the timm hub per the '
                           'config, the rest random) for from-scratch training')
  parser.add_argument('--seed', type=int, default=0,
                      help='torch seed for the random parts of --fresh')
  parser.add_argument('--prefix', default='agent/wm/enc/slotcontrast')
  parser.add_argument('--decoder-prefix', default='agent/wm/dec/featdec',
                      help='ninjax prefix for the MLPDecoder (featrec head)')
  parser.add_argument('--no-decoder', action='store_true',
                      help='skip the MLPDecoder weights (old behavior)')
  args = parser.parse_args()
  assert bool(args.checkpoint) != args.fresh, (
      'pass exactly one of --checkpoint or --fresh')

  import torch
  from omegaconf import OmegaConf

  config = OmegaConf.load(args.config)
  model_name = str(OmegaConf.select(config, 'globals.DINO_MODEL'))
  assert model_name in MODEL_TO_VARIANT, (
      f'Unknown backbone {model_name!r}; known: {list(MODEL_TO_VARIANT)}')
  variant = MODEL_TO_VARIANT[model_name]

  if args.fresh:
    pretrained = OmegaConf.select(
        config, 'model.encoder.backbone.pretrained', default=True)
    print(f'fresh model (seed {args.seed}), backbone pretrained: {pretrained}')
    assert pretrained, (
        'model.encoder.backbone.pretrained is False in this config; a --fresh '
        'conversion would produce a fully random extractor. Enable it to get '
        'the DINO hub weights.')
    state_dict = slot_convert.fresh_state_dict(config, seed=args.seed)
  else:
    state_dict = torch.load(
        args.checkpoint, weights_only=False, map_location='cpu')
    if 'state_dict' in state_dict:
      state_dict = state_dict['state_dict']
    state_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}

  decoder_cfg = OmegaConf.select(config, 'model.decoder')
  with_decoder = not args.no_decoder and decoder_cfg is not None
  decoder_hidden = 3
  if with_decoder:
    assert str(decoder_cfg.name) == 'MLPDecoder', decoder_cfg.name
    decoder_hidden = len(decoder_cfg.hidden_dims)

  leftover = slot_convert.unconsumed_keys(
      state_dict, variant,
      decoder_hidden=decoder_hidden if with_decoder else None)
  assert not leftover, (
      'Checkpoint keys neither mapped nor on the known skip-list '
      f'(unknown structure, refusing to convert): {leftover}')

  converted = slot_convert.convert_all(
      state_dict, variant, args.prefix,
      decoder_prefix=args.decoder_prefix if with_decoder else None,
      decoder_hidden=decoder_hidden)
  expected = slot_convert.expected_keys(variant, args.prefix)
  if with_decoder:
    expected |= slot_convert.decoder_expected_keys(
        args.decoder_prefix, decoder_hidden)
  missing = expected - set(converted)
  extra = set(converted) - expected
  assert not missing and not extra, (missing, extra)
  for key, value in converted.items():
    assert value.dtype == np.float32, (key, value.dtype)

  with open(args.output, 'wb') as f:
    pickle.dump(converted, f)
  total = sum(int(np.prod(v.shape)) for v in converted.values())
  print(f'variant: {variant}')
  print(f'wrote {len(converted)} arrays ({total / 1e6:.2f}M params) '
        f'under prefix {args.prefix!r} to {args.output}')


if __name__ == '__main__':
  main()
