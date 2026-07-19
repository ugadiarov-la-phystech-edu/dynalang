"""
Torch -> ninjax weight conversion for the SlotContrast extractor
(slot_nets.SlotContrastEncoder and its submodules).

All functions take `sd`: a dict {torch_key: np.ndarray} (a torch state_dict
with tensors already converted to numpy), and return a flat dict
{ninjax_path: np.float32 ndarray} whose keys exactly match the parameters
slot_nets modules create.

Layout transforms:
  nn.Linear.weight (out, in)      -> kernel (in, out)          [transpose]
  nn.Conv2d.weight (out, in, kh, kw) -> kernel (kh, kw, in, out) [permute 2,3,1,0]
  nn.LayerNorm.weight / .bias     -> scale / bias              [copy]
  nn.GRUCell.weight_ih (3H, in)   -> w_ih (in, 3H)             [transpose]
  nn.GRUCell.weight_hh (3H, H)    -> w_hh (H, 3H)              [transpose]

Checkpoint key namespace (SlotContrast lightning ckpt, `input_type: video`):
  initializer.slots
  encoder.module.backbone.model.*        (timm ViT)
  encoder.module.output_transform.layers.{0,1,3}.*
  processor.module.corrector.*
  decoder.module.{pos_emb, mlp.layers.{0,2,...}}   (MLPDecoder, optional)
Skipped on purpose (never affect extracted slots):
  processor.module.predictor.*, encoder.module.backbone.model.norm.*,
  image_decoder.*, dynamics_predictor.*, and the target/teacher
  copies some checkpoints carry. decoder.* is skipped unless a
  decoder_prefix is passed to convert_all (featrec fine-tuning).
For `input_type: image` checkpoints the `.module` segment is absent; keys are
normalized with _strip_module() before mapping.
"""

import re

import numpy as np

SKIP_PATTERN = re.compile(
    r'(^|\.)predictor\.|backbone\.model\.norm\.|^decoder\.|^image_decoder\.'
    r'|^dynamics_predictor\.|^target_')


def _f32(x):
  x = np.asarray(x)
  assert np.isfinite(x).all(), 'non-finite values in checkpoint array'
  return x.astype(np.float32)


def _strip_module(key):
  return key.replace('.module.', '.')


def _join(prefix, name):
  return f'{prefix}.{name}' if prefix else name


def convert_layernorm(sd, tkey, jkey):
  """nn.LayerNorm at torch prefix `tkey` -> slot_nets.LayerNorm at `jkey`."""
  return {
      f'{jkey}/scale': _f32(sd[f'{tkey}.weight' if tkey else 'weight']),
      f'{jkey}/bias': _f32(sd[f'{tkey}.bias' if tkey else 'bias']),
  }


def convert_linear(sd, tkey, jkey, bias=True):
  """nn.Linear -> slot_nets.PLinear (kernel transposed)."""
  prefix = f'{tkey}.' if tkey else ''
  out = {f'{jkey}/kernel': _f32(sd[f'{prefix}weight']).T}
  if bias:
    out[f'{jkey}/bias'] = _f32(sd[f'{prefix}bias'])
  return out


def convert_gru(sd, tkey, jkey):
  """nn.GRUCell -> slot_nets.GRUCell."""
  prefix = f'{tkey}.' if tkey else ''
  return {
      f'{jkey}/w_ih': _f32(sd[f'{prefix}weight_ih']).T,
      f'{jkey}/w_hh': _f32(sd[f'{prefix}weight_hh']).T,
      f'{jkey}/b_ih': _f32(sd[f'{prefix}bias_ih']),
      f'{jkey}/b_hh': _f32(sd[f'{prefix}bias_hh']),
  }


def convert_two_layer_mlp(sd, tkey, jkey):
  """SlotContrast networks.MLP with initial LayerNorm (layers.0=LN,
  layers.1=Linear, layers.3=Linear) -> slot_nets.TwoLayerMLP."""
  out = {}
  out.update(convert_layernorm(sd, _join(tkey, 'layers.0'), f'{jkey}/norm'))
  out.update(convert_linear(sd, _join(tkey, 'layers.1'), f'{jkey}/fc1'))
  out.update(convert_linear(sd, _join(tkey, 'layers.3'), f'{jkey}/fc2'))
  return out


def convert_mlp_decoder(sd, tkey, jkey, n_hidden=3):
  """decoders.MLPDecoder -> slot_nets.MLPDecoder. The torch mlp is a
  Sequential [Linear, ReLU] * n_hidden + [Linear], so hidden layer i sits at
  layers.{2i} and the output layer at layers.{2 * n_hidden}."""
  out = {f'{jkey}/pos_emb': _f32(sd[_join(tkey, 'pos_emb')])}
  for i in range(n_hidden):
    out.update(convert_linear(
        sd, _join(tkey, f'mlp.layers.{2 * i}'), f'{jkey}/fc{i}'))
  out.update(convert_linear(
      sd, _join(tkey, f'mlp.layers.{2 * n_hidden}'), f'{jkey}/out'))
  return out


def convert_slot_attention(sd, tkey, jkey):
  """groupers.SlotAttention -> slot_nets.SlotAttention."""
  out = {}
  out.update(convert_layernorm(
      sd, _join(tkey, 'norm_features'), f'{jkey}/norm_features'))
  out.update(convert_layernorm(
      sd, _join(tkey, 'norm_slots'), f'{jkey}/norm_slots'))
  for name in ('to_q', 'to_k', 'to_v'):
    out.update(convert_linear(
        sd, _join(tkey, name), f'{jkey}/{name}', bias=False))
  out.update(convert_gru(sd, _join(tkey, 'gru'), f'{jkey}/gru'))
  out.update(convert_layernorm(
      sd, _join(tkey, 'mlp.layers.0'), f'{jkey}/mlp_norm'))
  out.update(convert_linear(sd, _join(tkey, 'mlp.layers.1'), f'{jkey}/mlp_fc1'))
  out.update(convert_linear(sd, _join(tkey, 'mlp.layers.3'), f'{jkey}/mlp_fc2'))
  return out


def convert_vit_block(sd, tkey, jkey, layerscale):
  """timm Block -> slot_nets.ViTBlock."""
  out = {}
  out.update(convert_layernorm(sd, _join(tkey, 'norm1'), f'{jkey}/norm1'))
  out.update(convert_linear(sd, _join(tkey, 'attn.qkv'), f'{jkey}/attn/qkv'))
  out.update(convert_linear(sd, _join(tkey, 'attn.proj'), f'{jkey}/attn/proj'))
  out.update(convert_layernorm(sd, _join(tkey, 'norm2'), f'{jkey}/norm2'))
  out.update(convert_linear(sd, _join(tkey, 'mlp.fc1'), f'{jkey}/mlp_fc1'))
  out.update(convert_linear(sd, _join(tkey, 'mlp.fc2'), f'{jkey}/mlp_fc2'))
  if layerscale:
    out[f'{jkey}/ls1/gamma'] = _f32(sd[_join(tkey, 'ls1.gamma')])
    out[f'{jkey}/ls2/gamma'] = _f32(sd[_join(tkey, 'ls2.gamma')])
  return out


def convert_vit(sd, tkey, jkey, depth=12, layerscale=False, feature_block=None):
  """timm ViT (up to feature_block inclusive) -> slot_nets.ViT."""
  feature_block = depth - 1 if feature_block is None else feature_block
  prefix = f'{tkey}.' if tkey else ''
  out = {
      f'{jkey}/cls_token': _f32(sd[f'{prefix}cls_token']),
      f'{jkey}/pos_embed': _f32(sd[f'{prefix}pos_embed']),
  }
  proj_w = _f32(sd[f'{prefix}patch_embed.proj.weight'])  # (out, in, kh, kw)
  out[f'{jkey}/patch_embed/kernel'] = proj_w.transpose(2, 3, 1, 0)
  out[f'{jkey}/patch_embed/bias'] = _f32(sd[f'{prefix}patch_embed.proj.bias'])
  for i in range(feature_block + 1):
    out.update(convert_vit_block(
        sd, f'{prefix}blocks.{i}', f'{jkey}/block{i}', layerscale))
  return out


def convert_all(state_dict, variant, prefix, depth=12,
                decoder_prefix=None, decoder_hidden=3):
  """Full SlotContrast checkpoint -> flat ninjax dict under `prefix`
  (e.g. 'agent/wm/enc/slotcontrast'). `variant` in slot_nets.VARIANTS.
  With decoder_prefix (e.g. 'agent/wm/dec/featdec') the MLPDecoder weights
  are converted too, for the featrec fine-tuning head."""
  from . import slot_nets
  layerscale = slot_nets.VARIANTS[variant]['layerscale']
  sd = {_strip_module(k): np.asarray(v) for k, v in state_dict.items()}
  out = {f'{prefix}/init_slots': _f32(sd['initializer.slots'])}
  out.update(convert_vit(
      sd, 'encoder.backbone.model', f'{prefix}/backbone',
      depth=depth, layerscale=layerscale))
  out.update(convert_two_layer_mlp(
      sd, 'encoder.output_transform', f'{prefix}/proj'))
  out.update(convert_slot_attention(
      sd, 'processor.corrector', f'{prefix}/corrector'))
  if decoder_prefix:
    out.update(convert_mlp_decoder(
        sd, 'decoder', decoder_prefix, n_hidden=decoder_hidden))
  return out


def unconsumed_keys(state_dict, variant, depth=12, decoder_hidden=None):
  """Torch keys neither mapped by convert_all nor on the known skip-list.
  Non-empty result means the checkpoint has structure we don't understand —
  the converter should fail loudly rather than silently drop weights."""
  from . import slot_nets
  layerscale = slot_nets.VARIANTS[variant]['layerscale']
  consumed = set()
  consumed.add('initializer.slots')
  vit = 'encoder.backbone.model'
  consumed.update({f'{vit}.cls_token', f'{vit}.pos_embed',
                   f'{vit}.patch_embed.proj.weight',
                   f'{vit}.patch_embed.proj.bias'})
  for i in range(depth):
    b = f'{vit}.blocks.{i}'
    for name in ('norm1', 'norm2'):
      consumed.update({f'{b}.{name}.weight', f'{b}.{name}.bias'})
    for name in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'):
      consumed.update({f'{b}.{name}.weight', f'{b}.{name}.bias'})
    if layerscale:
      consumed.update({f'{b}.ls1.gamma', f'{b}.ls2.gamma'})
  for layer in ('layers.0', 'layers.1', 'layers.3'):
    consumed.update({f'encoder.output_transform.{layer}.weight',
                     f'encoder.output_transform.{layer}.bias'})
  corr = 'processor.corrector'
  for name in ('norm_features', 'norm_slots'):
    consumed.update({f'{corr}.{name}.weight', f'{corr}.{name}.bias'})
  for name in ('to_q', 'to_k', 'to_v'):
    consumed.add(f'{corr}.{name}.weight')
  for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
    consumed.add(f'{corr}.gru.{name}')
  for layer in ('layers.0', 'layers.1', 'layers.3'):
    consumed.update({f'{corr}.mlp.{layer}.weight', f'{corr}.mlp.{layer}.bias'})
  if decoder_hidden is not None:
    consumed.add('decoder.pos_emb')
    for i in range(decoder_hidden + 1):
      consumed.update({f'decoder.mlp.layers.{2 * i}.weight',
                       f'decoder.mlp.layers.{2 * i}.bias'})
  leftover = []
  for key in state_dict:
    norm = _strip_module(key)
    if norm in consumed:
      continue
    if SKIP_PATTERN.search(norm):
      continue
    leftover.append(key)
  return leftover


def decoder_expected_keys(decoder_prefix, n_hidden=3):
  """The ninjax key set convert_mlp_decoder produces (== the params
  slot_nets.MLPDecoder creates)."""
  keys = {f'{decoder_prefix}/pos_emb'}
  for i in range(n_hidden):
    keys.update({f'{decoder_prefix}/fc{i}/kernel', f'{decoder_prefix}/fc{i}/bias'})
  keys.update({f'{decoder_prefix}/out/kernel', f'{decoder_prefix}/out/bias'})
  return keys


def expected_keys(variant, prefix, depth=12):
  """The exact ninjax key set convert_all produces (== the key set
  slot_nets.SlotContrastEncoder creates at init). Used by the converter CLI
  and the jaxagent load hook to assert full coverage."""
  from . import slot_nets
  layerscale = slot_nets.VARIANTS[variant]['layerscale']
  keys = {f'{prefix}/init_slots'}
  vit = f'{prefix}/backbone'
  keys.update({f'{vit}/cls_token', f'{vit}/pos_embed',
               f'{vit}/patch_embed/kernel', f'{vit}/patch_embed/bias'})
  for i in range(depth):
    b = f'{vit}/block{i}'
    for name in ('norm1', 'norm2'):
      keys.update({f'{b}/{name}/scale', f'{b}/{name}/bias'})
    for name in ('attn/qkv', 'attn/proj', 'mlp_fc1', 'mlp_fc2'):
      keys.update({f'{b}/{name}/kernel', f'{b}/{name}/bias'})
    if layerscale:
      keys.update({f'{b}/ls1/gamma', f'{b}/ls2/gamma'})
  proj = f'{prefix}/proj'
  keys.update({f'{proj}/norm/scale', f'{proj}/norm/bias'})
  for name in ('fc1', 'fc2'):
    keys.update({f'{proj}/{name}/kernel', f'{proj}/{name}/bias'})
  corr = f'{prefix}/corrector'
  for name in ('norm_features', 'norm_slots', 'mlp_norm'):
    keys.update({f'{corr}/{name}/scale', f'{corr}/{name}/bias'})
  for name in ('to_q', 'to_k', 'to_v'):
    keys.add(f'{corr}/{name}/kernel')
  keys.update({f'{corr}/gru/w_ih', f'{corr}/gru/w_hh',
               f'{corr}/gru/b_ih', f'{corr}/gru/b_hh'})
  for name in ('mlp_fc1', 'mlp_fc2'):
    keys.update({f'{corr}/{name}/kernel', f'{corr}/{name}/bias'})
  return keys


def fresh_state_dict(config, seed=0):
  """Numpy state dict of a freshly initialized torch SlotContrast model built
  from an OmegaConf config (`config.model`), in the lightning-checkpoint key
  namespace convert_all expects. The backbone keeps whatever
  `model.encoder.backbone.pretrained` says (True in the shipped configs, so
  timm loads the DINO hub weights); slot attention, projection, learned init
  and the MLPDecoder come out randomly initialized — the starting point for
  training SlotContrast from scratch inside dynalang. Needs torch and the
  vendored `embodied.torch.ocr.slotcontrast` package on sys.path."""
  import torch
  # Registers the OmegaConf resolvers (${mul:...} etc.) the shipped configs
  # use; interpolation happens lazily on field access below.
  from embodied.torch.ocr.slotcontrast import configuration  # noqa: F401
  from embodied.torch.ocr.slotcontrast import modules
  torch.manual_seed(seed)
  mc = config.model
  parts = [
      ('initializer', modules.build_initializer(mc.initializer)),
      ('encoder', modules.MapOverTime(
          modules.build_encoder(mc.encoder, 'FrameEncoder'))),
      ('processor', modules.ScanOverTime(modules.build_video(
          mc.latent_processor, 'LatentProcessor',
          corrector=modules.build_grouper(mc.grouper), predictor=None))),
  ]
  if mc.get('decoder') is not None:
    parts.append(('decoder', modules.MapOverTime(
        modules.build_decoder(mc.decoder, str(mc.decoder.name)))))
  out = {}
  for prefix, module in parts:
    for key, value in module.state_dict().items():
      out[f'{prefix}.{key}'] = value.detach().cpu().numpy()
  return out
