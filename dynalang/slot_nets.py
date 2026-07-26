"""
Ninjax/JAX re-implementation of the SlotContrast inference path
(dynalang/embodied/torch/ocr/slotcontrast), so the slot extractor can live
inside MultiEncoder and be trained or fine-tuned with the world model.

Ported modules (parity source of truth in parentheses):
  - FixedLearnedInit          (modules/initializers.py)
  - timm ViT backbone tap     (modules/encoders.py TimmExtractor: block-11
                               output, pre-final-norm, CLS stripped)
  - two_layer_mlp projection  (modules/networks.py via encoders.FrameEncoder)
  - SlotAttention corrector   (modules/groupers.py)

Also ported (use_predictor=True — video-model semantics):
  - TransformerEncoder predictor  (modules/networks.py TransformerEncoder /
                                   TransformerEncoderLayer / Attention)
    transports slots_t -> the slot-attention init at t+1, exactly the
    LatentProcessor + ScanOverTime recurrence the video checkpoints were
    trained with (state_predicted carried, corrector output emitted).

Deliberately skipped (never affect extracted slots): decoders, dynamics,
and the `vit_block_keys12` feature. With use_predictor=False the predictor
is skipped too and the extractor-style recurrence below applies.

Parity-critical conventions:
  - LayerNorm eps differs per layer group: 1e-6 (timm ViT), 1e-5 (torch
    nn.LayerNorm in SlotContrast heads). Never reuse nets.Norm (eps 1e-3).
  - PLinear stores kernel as (in, out) = transpose of torch nn.Linear.weight.
  - GRUCell matches torch nn.GRUCell: gate order r, z, n and the b_hn bias
    inside the r*(...) term.
  - Every frame runs 3 slot-attention iterations (the torch extractor always
    calls its processor at time_step=0, triggering first_step_corrector_args).
  - Weights are stored float32; the compute dtype follows the input dtype,
    with LayerNorm/softmax always reduced in float32.
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import ninjax as nj

f32 = jnp.float32

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Backbone variants (timm model geometry). native_grid is the pos-embed grid
# the checkpoint was trained with; inputs at other resolutions trigger the
# timm dynamic_img_size pos-embed resample (bicubic, antialias).
VARIANTS = {
    'dino_v1_s8': dict(patch=8, native_grid=28, layerscale=False),
    'dinov2_s14': dict(patch=14, native_grid=37, layerscale=True),
}


def switch(pred, lhs, rhs):
  """jaxutils.switch equivalent (local to avoid the heavy jaxutils import)."""
  assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
  mask = pred
  while len(mask.shape) < len(lhs.shape):
    mask = mask[..., None]
  return jnp.where(mask, lhs, rhs)


def resize_bilinear(x, size):
  """torch F.interpolate(mode='bilinear', align_corners=False, antialias=False)
  equivalent for (..., H, W, C) tensors."""
  shape = x.shape[:-3] + (size, size, x.shape[-1])
  if x.shape[-3] == size and x.shape[-2] == size:
    return x
  return jax.image.resize(x, shape, method='bilinear', antialias=False)


def resample_pos_embed(pos, old_grid, new_grid):
  """timm.layers.resample_abs_pos_embed equivalent: bicubic, antialias=True,
  prefix (CLS) token kept unresampled. pos: (1, 1 + old_grid**2, D)."""
  if new_grid == old_grid:
    return pos
  prefix, grid = pos[:, :1], pos[:, 1:]
  dim = pos.shape[-1]
  grid = grid.reshape(1, old_grid, old_grid, dim).astype(f32)
  grid = jax.image.resize(
      grid, (1, new_grid, new_grid, dim), method='bicubic', antialias=True)
  grid = grid.reshape(1, new_grid * new_grid, dim)
  return jnp.concatenate([prefix.astype(f32), grid], axis=1).astype(pos.dtype)


def _torch_linear_init(shape):
  # torch nn.Linear default: kaiming_uniform(a=sqrt(5)) == U(+-1/sqrt(fan_in)).
  bound = 1.0 / np.sqrt(shape[0])
  return jax.random.uniform(nj.rng(), shape, f32, -bound, bound)


def _trunc_normal_init(scale):
  def init(shape):
    return scale * jax.random.truncated_normal(nj.rng(), -2.0, 2.0, shape, f32)
  return init


class LayerNorm(nj.Module):
  """torch nn.LayerNorm over the last axis with configurable eps.
  Params: scale (torch weight), bias. Reduction always in float32."""

  def __init__(self, eps=1e-5):
    self._eps = eps

  def __call__(self, x):
    dtype = x.dtype
    x = x.astype(f32)
    scale = self.get('scale', jnp.ones, x.shape[-1], f32)
    bias = self.get('bias', jnp.zeros, x.shape[-1], f32)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(var + self._eps) * scale + bias
    return x.astype(dtype)


class PLinear(nj.Module):
  """torch-layout linear: kernel (in, out) = nn.Linear.weight.T, bias (out,)."""

  def __init__(self, units, bias=True):
    self._units = units
    self._bias = bias

  def __call__(self, x):
    fan_in = x.shape[-1]
    kernel = self.get('kernel', _torch_linear_init, (fan_in, self._units))
    y = x @ kernel.astype(x.dtype)
    if self._bias:
      # torch bias init also uses U(+-1/sqrt(fan_in)) of the layer input.
      bound = 1.0 / np.sqrt(fan_in)
      init = lambda shape: jax.random.uniform(
          nj.rng(), shape, f32, -bound, bound)
      bias = self.get('bias', init, (self._units,))
      y = y + bias.astype(x.dtype)
    return y


class GRUCell(nj.Module):
  """torch nn.GRUCell parity. Params (transposed for right-matmul):
  w_ih (in, 3H), w_hh (H, 3H), b_ih (3H,), b_hh (3H,). Gates r, z, n."""

  def __call__(self, x, h):
    hidden = h.shape[-1]
    bound = 1.0 / np.sqrt(hidden)
    uniform = lambda shape: jax.random.uniform(
        nj.rng(), shape, f32, -bound, bound)
    w_ih = self.get('w_ih', uniform, (x.shape[-1], 3 * hidden))
    w_hh = self.get('w_hh', uniform, (hidden, 3 * hidden))
    b_ih = self.get('b_ih', uniform, (3 * hidden,))
    b_hh = self.get('b_hh', uniform, (3 * hidden,))
    gi = x @ w_ih.astype(x.dtype) + b_ih.astype(x.dtype)
    gh = h @ w_hh.astype(x.dtype) + b_hh.astype(x.dtype)
    i_r, i_z, i_n = jnp.split(gi, 3, -1)
    h_r, h_z, h_n = jnp.split(gh, 3, -1)
    r = jax.nn.sigmoid(i_r + h_r)
    z = jax.nn.sigmoid(i_z + h_z)
    n = jnp.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h


class TwoLayerMLP(nj.Module):
  """SlotContrast networks.two_layer_mlp: LN(1e-5) -> Linear -> ReLU -> Linear."""

  def __init__(self, hidden, units):
    self._hidden = hidden
    self._units = units

  def __call__(self, x):
    x = self.get('norm', LayerNorm, 1e-5)(x)
    x = self.get('fc1', PLinear, self._hidden)(x)
    x = jax.nn.relu(x)
    x = self.get('fc2', PLinear, self._units)(x)
    return x


class SlotAttention(nj.Module):
  """SlotContrast groupers.SlotAttention with use_gru=True, use_mlp=True.
  Fixed n_iters (the torch extractor always runs the 3-iteration first-step
  path). Mirrors groupers.py:64-100."""

  def __init__(self, dim, hidden, n_iters=3, eps=1e-8):
    self._dim = dim
    self._hidden = hidden
    self._n_iters = n_iters
    self._eps = eps

  def __call__(self, slots, features, n_iters=None):
    n_iters = self._n_iters if n_iters is None else n_iters
    scale = self._dim ** -0.5
    features = self.get('norm_features', LayerNorm, 1e-5)(features)
    keys = self.get('to_k', PLinear, self._dim, bias=False)(features)
    values = self.get('to_v', PLinear, self._dim, bias=False)(features)
    for _ in range(n_iters):
      # NOTE: torch groupers.SlotAttention.step() reassigns
      # `slots = norm_slots(slots)` before to_q, so the GRU hidden state is
      # the NORMED slots, not the raw ones.
      slots = self.get('norm_slots', LayerNorm, 1e-5)(slots)
      queries = self.get('to_q', PLinear, self._dim, bias=False)(slots)
      dots = jnp.einsum('bsd,bfd->bsf', queries, keys) * scale
      # Softmax over the slot axis: slots compete for patches.
      attn = jax.nn.softmax(dots.astype(f32), axis=-2).astype(dots.dtype)
      attn = attn + self._eps
      attn = attn / attn.sum(-1, keepdims=True)
      updates = jnp.einsum('bsf,bfd->bsd', attn, values)
      batch, n_slots, dim = slots.shape
      slots = self.get('gru', GRUCell)(
          updates.reshape(batch * n_slots, dim),
          slots.reshape(batch * n_slots, dim),
      ).reshape(batch, n_slots, dim)
      x = self.get('mlp_norm', LayerNorm, 1e-5)(slots)
      x = self.get('mlp_fc1', PLinear, self._hidden)(x)
      x = jax.nn.relu(x)
      x = self.get('mlp_fc2', PLinear, self._dim)(x)
      slots = slots + x
    return slots


class PredictorAttention(nj.Module):
  """SlotContrast networks.Attention (fused qkv, rows [q;k;v]) as used by the
  video predictor. Parity-critical quirk: the reference divides q by
  `self.scale` (= head_dim**-0.5) in networks.py:516, so the attention logits
  are MULTIPLIED by sqrt(head_dim) — not divided as in standard attention.
  The checkpoints were trained this way; replicate it exactly."""

  def __init__(self, dim, heads):
    assert dim % heads == 0, (dim, heads)
    self._dim = dim
    self._heads = heads

  def __call__(self, x):
    batch, length, dim = x.shape
    heads = self._heads
    head_dim = dim // heads
    qkv = self.get('qkv', PLinear, 3 * dim)(x)
    qkv = qkv.reshape(batch, length, 3, heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = q * (head_dim ** 0.5)  # sic — see class docstring
    attn = jnp.einsum('bnhd,bmhd->bhnm', q, k)
    attn = jax.nn.softmax(attn.astype(f32), axis=-1).astype(attn.dtype)
    out = jnp.einsum('bhnm,bmhd->bnhd', attn, v)
    out = out.reshape(batch, length, dim)
    return self.get('out_proj', PLinear, dim)(out)


class PredictorBlock(nj.Module):
  """SlotContrast networks.TransformerEncoderLayer with norm_first=True:
  pre-norm self-attention + pre-norm ReLU MLP (hidden = 4*dim by default),
  LayerNorm eps 1e-5, no LayerScale (initial_residual_scale=None), dropout 0
  (inference)."""

  def __init__(self, dim, heads, hidden=None):
    self._dim = dim
    self._heads = heads
    self._hidden = 4 * dim if hidden is None else hidden

  def __call__(self, x):
    y = self.get('norm1', LayerNorm, 1e-5)(x)
    y = self.get('attn', PredictorAttention, self._dim, self._heads)(y)
    x = x + y
    y = self.get('norm2', LayerNorm, 1e-5)(x)
    y = self.get('linear1', PLinear, self._hidden)(y)
    y = jax.nn.relu(y)
    y = self.get('linear2', PLinear, self._dim)(y)
    return x + y


class TransformerPredictor(nj.Module):
  """SlotContrast networks.TransformerEncoder — the video-model predictor
  that transports slots_t into the slot-attention initialization for t+1."""

  def __init__(self, dim, n_blocks=1, heads=4, hidden=None):
    self._dim = dim
    self._n_blocks = n_blocks
    self._heads = heads
    self._hidden = hidden

  def __call__(self, slots):
    for i in range(self._n_blocks):
      slots = self.get(
          f'block{i}', PredictorBlock,
          self._dim, self._heads, self._hidden)(slots)
    return slots


class LayerScale(nj.Module):
  """timm LayerScale: y = x * gamma, gamma init 1e-5 (dinov2)."""

  def __init__(self, init_value=1e-5):
    self._init_value = init_value

  def __call__(self, x):
    gamma = self.get(
        'gamma', lambda shape: jnp.full(shape, self._init_value, f32),
        (x.shape[-1],))
    return x * gamma.astype(x.dtype)


class PatchEmbed(nj.Module):
  """timm PatchEmbed: conv(patch, stride=patch, VALID). Kernel HWIO."""

  def __init__(self, patch, dim):
    self._patch = patch
    self._dim = dim

  def __call__(self, x):
    assert x.shape[-3] % self._patch == 0 and x.shape[-2] % self._patch == 0, (
        f'image size {x.shape} not divisible by patch {self._patch}')
    init = lambda shape: _torch_linear_init(
        (int(np.prod(shape[:-1])), shape[-1])).reshape(shape)
    kernel = self.get(
        'kernel', init, (self._patch, self._patch, x.shape[-1], self._dim))
    bias = self.get('bias', jnp.zeros, (self._dim,), f32)
    x = jax.lax.conv_general_dilated(
        x, kernel.astype(x.dtype),
        window_strides=(self._patch, self._patch), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return x + bias.astype(x.dtype)


class ViTAttention(nj.Module):
  """timm Attention: fused qkv (rows [q;k;v]), q scaled by head_dim**-0.5."""

  def __init__(self, dim, heads):
    self._dim = dim
    self._heads = heads

  def __call__(self, x):
    batch, length, dim = x.shape
    heads = self._heads
    head_dim = dim // heads
    qkv = self.get('qkv', PLinear, 3 * dim)(x)
    qkv = qkv.reshape(batch, length, 3, heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = q * (head_dim ** -0.5)
    attn = jnp.einsum('bnhd,bmhd->bhnm', q, k)
    attn = jax.nn.softmax(attn.astype(f32), axis=-1).astype(attn.dtype)
    out = jnp.einsum('bhnm,bmhd->bnhd', attn, v)
    out = out.reshape(batch, length, dim)
    return self.get('proj', PLinear, dim)(out)


class ViTBlock(nj.Module):
  """timm Block (pre-norm, exact GELU, optional LayerScale)."""

  def __init__(self, dim, heads, layerscale=False, mlp_ratio=4):
    self._dim = dim
    self._heads = heads
    self._layerscale = layerscale
    self._mlp_ratio = mlp_ratio

  def __call__(self, x):
    y = self.get('norm1', LayerNorm, 1e-6)(x)
    y = self.get('attn', ViTAttention, self._dim, self._heads)(y)
    if self._layerscale:
      y = self.get('ls1', LayerScale)(y)
    x = x + y
    y = self.get('norm2', LayerNorm, 1e-6)(x)
    y = self.get('mlp_fc1', PLinear, self._dim * self._mlp_ratio)(y)
    y = jax.nn.gelu(y, approximate=False)
    y = self.get('mlp_fc2', PLinear, self._dim)(y)
    if self._layerscale:
      y = self.get('ls2', LayerScale)(y)
    return x + y


class ViT(nj.Module):
  """timm ViT backbone up to (and including) `feature_block`, tapping the
  block output pre-final-norm with the CLS token stripped — exactly the
  TimmExtractor 'vit_block12' feature (encoders.py:189-223). Blocks after
  the tap and the final norm are never instantiated."""

  def __init__(self, depth=12, dim=384, heads=6, patch=8, native_grid=28,
               layerscale=False, feature_block=None, remat=False):
    self._depth = depth
    self._dim = dim
    self._heads = heads
    self._patch = patch
    self._native_grid = native_grid
    self._layerscale = layerscale
    self._feature_block = depth - 1 if feature_block is None else feature_block
    self._remat = remat

  def __call__(self, x):
    # x: (B, H, W, 3), already ImageNet-normalized.
    y = self.get('patch_embed', PatchEmbed, self._patch, self._dim)(x)
    batch, grid_h, grid_w, dim = y.shape
    assert grid_h == grid_w, (grid_h, grid_w)
    y = y.reshape(batch, grid_h * grid_w, dim)
    pos = self.get(
        'pos_embed', _trunc_normal_init(0.02),
        (1, self._native_grid * self._native_grid + 1, self._dim))
    pos = resample_pos_embed(pos, self._native_grid, grid_h)
    cls = self.get(
        'cls_token', _trunc_normal_init(0.02), (1, 1, self._dim))
    cls = jnp.broadcast_to(cls, (batch, 1, dim)).astype(y.dtype)
    y = jnp.concatenate([cls, y], axis=1)
    y = y + pos.astype(y.dtype)
    for i in range(self._feature_block + 1):
      block = self.get(
          'block' + str(i), ViTBlock, self._dim, self._heads, self._layerscale)
      if self._remat and not nj.creating():
        y = jax.checkpoint(lambda z, _block=block: _block(z))(y)
      else:
        y = block(y)
    return y[:, 1:]  # strip CLS; pre-final-norm


class MLPDecoder(nj.Module):
  """SlotContrast decoders.MLPDecoder: reconstructs backbone features
  independently for every (slot, patch) position. Each slot is broadcast over
  all patches, a learned positional embedding is added, and a shared MLP emits
  a feature reconstruction plus an alpha logit per position; alphas are
  softmaxed over the slot axis and used to mix the per-slot reconstructions.
  Mirrors decoders.py:46-91 (training path; no eval_output_size resampling).

  Params: pos_emb (1, 1, P, D) and PLinear fc0..fcN-1 / out, so the torch
  Sequential layers.{0,2,...} map directly (slot_convert.convert_mlp_decoder).
  """

  def __init__(self, out_dim, n_patches, hidden_dims=(1024, 1024, 1024),
               f16=False, chunk=0):
    self._out_dim = out_dim
    self._n_patches = n_patches
    self._hidden_dims = tuple(hidden_dims)
    self._f16 = f16
    self._chunk = chunk

  def __call__(self, slots):
    """slots (..., S, D) -> (recon (..., P, out_dim), masks (..., S, P))."""
    dim = slots.shape[-1]
    init = lambda shape: dim ** -0.5 * jax.random.normal(nj.rng(), shape, f32)
    pos = self.get('pos_emb', init, (1, 1, self._n_patches, dim))
    pos = pos.reshape(self._n_patches, dim)
    # f16: halve the (..., S, P, hidden) activation memory — the dominant
    # cost of this head. Params stay f32; softmax and loss stay f32 below.
    comp = jnp.float16 if self._f16 else slots.dtype

    def run(flat):  # (N, S, D) -> (recon (N, P, out_dim), masks (N, S, P))
      x = flat.astype(comp)[:, :, None, :] + pos.astype(comp)  # (N, S, P, D)
      for i, hidden in enumerate(self._hidden_dims):
        x = self.get(f'fc{i}', PLinear, hidden)(x)
        x = jax.nn.relu(x)
      x = self.get('out', PLinear, self._out_dim + 1)(x)
      recons, alpha = x[..., :-1], x[..., -1:]
      # Softmax over the slot axis: slots compete for each patch.
      masks = jax.nn.softmax(alpha.astype(f32), axis=-3)
      recon = (recons.astype(f32) * masks).sum(-3)
      return recon.astype(slots.dtype), masks[..., 0]

    # Bound peak memory by mapping frames through the MLP in chunks, with
    # rematerialization (the hidden activations are recomputed chunk by chunk
    # in the backward pass instead of being stored for the whole batch). Same
    # divisibility rule as the ViT chunking; creation runs unchunked.
    lead = slots.shape[:-2]
    flat = slots.reshape((-1,) + slots.shape[-2:])
    n = flat.shape[0]
    if (self._chunk and not nj.creating()
        and n > self._chunk and n % self._chunk == 0):
      xs = flat.reshape((n // self._chunk, self._chunk) + flat.shape[1:])
      recon, masks = jax.lax.map(jax.checkpoint(run), xs)
      recon = recon.reshape((n,) + recon.shape[2:])
      masks = masks.reshape((n,) + masks.shape[2:])
    else:
      recon, masks = run(flat)
    return (recon.reshape(lead + recon.shape[1:]),
            masks.reshape(lead + masks.shape[1:]))


def decoder_masks_video(images, masks):
  """SlotContrast 'decoder_masks' visualization (models.py:_log_masks with
  mix_with_source=True + visualizations.masks_on_video): per sequence, a strip
  of panels [original video | slot 0 | slot 1 | ...] concatenated along width,
  where each slot panel is video * mask + (1 - mask) — frame content where the
  slot's soft mask is high, fading to white elsewhere. Patch masks are
  upsampled to the frame size bilinearly, matching the torch
  Resizer(patch_inputs=True, resize_mode='bilinear').

  images: (B, T, H, W, 3) floats in [0, 1]; masks: (B, T, S, P) with P a
  perfect square. Returns (B, T, H, W * (S + 1), 3) float32.
  """
  batch, length, height, width, _ = images.shape
  assert height == width, (height, width)
  n_slots, n_patches = masks.shape[-2:]
  grid = int(np.sqrt(n_patches))
  assert grid * grid == n_patches, n_patches
  images = images.astype(f32)
  masks = masks.astype(f32).reshape(batch * length, n_slots, grid, grid)
  masks = masks.transpose(0, 2, 3, 1)  # slots to channels for the resize
  masks = resize_bilinear(masks, height)
  masks = masks.reshape(batch, length, height, width, n_slots)
  panels = [images]
  for s in range(n_slots):
    mask = masks[..., s:s + 1]
    panels.append(images * mask + (1.0 - mask))
  return jnp.concatenate(panels, axis=-2)


def openl_decoder_masks_video(images, truth_masks, model_masks):
  """Open-loop 'decoder_masks' video, analogous to the openl_* pixel videos:
  three rows stacked along height — truth (masks from the encoder's slots),
  model (masks from the world model's reconstructed/imagined slots, applied to
  the same ground-truth frames), and error (model - truth + 1) / 2, mid-gray
  where they agree.

  images: (B, T, H, W, 3); truth_masks/model_masks: (B, T, S, P).
  Returns (B, T, 3 * H, W * (S + 1), 3) float32.
  """
  truth = decoder_masks_video(images, truth_masks)
  model = decoder_masks_video(images, model_masks)
  error = (model - truth + 1.0) / 2.0
  return jnp.concatenate([truth, model, error], axis=2)


def video_rows(video):
  """Stack a batch of videos vertically: one row per sequence.

  video: (B, T, H, W, C) -> (T, B * H, W, C). Counterpart of
  jaxutils.video_grid, which tiles the batch along width instead.
  """
  batch, length, height, width, chan = video.shape
  return video.transpose(1, 0, 2, 3, 4).reshape(
      length, batch * height, width, chan)


class SlotContrastEncoder(nj.Module):
  """Full SlotContrast extractor: images -> per-frame object slots, with
  recurrent slot initialization matching BatchSlotExtractorEnv semantics
  (use_previous_slots=True, initialize_twice=True).

  __call__ modes:
    single step: images (B, H, W, C), is_first (B,)   -> slots (B, S, D)
    sequence:    images (B, T, H, W, C), is_first (B, T) -> slots (B, T, S, D)
  Returns (slots, carry) where carry is the recurrent state to feed back on
  the next call (policy path), float32: the last timestep's slots in legacy
  mode, predictor(slots) (state_predicted) with use_predictor=True.
  Images are floats in [0, 1] (after Agent.preprocess's /255)."""

  def __init__(self, variant='dino_v1_s8', input_size=224, n_slots=8,
               slot_dim=64, depth=12, dim=384, heads=6, hidden_mult=2,
               force_f32=True, remat=True, chunk=0, sg_backbone=False,
               use_predictor=False, predictor_blocks=1, predictor_heads=4,
               sa_iters=2, sa_first_iters=3):
    spec = VARIANTS[variant]
    self._variant = variant
    self._sg_backbone = sg_backbone
    self._input_size = input_size
    self._n_slots = n_slots
    self._slot_dim = slot_dim
    self._depth = depth
    self._dim = dim
    self._heads = heads
    self._hidden_mult = hidden_mult
    self._force_f32 = force_f32
    self._remat = remat
    self._chunk = chunk
    # Video-model recurrence (use_predictor=True): the LatentProcessor +
    # ScanOverTime semantics the video checkpoints were trained with — the
    # corrector runs sa_first_iters at episode starts (first_step_corrector_
    # args) and sa_iters elsewhere, and the carry is predictor(slots), i.e.
    # state_predicted. With use_predictor=False the legacy extractor
    # semantics apply (initialize_twice, 3 iterations everywhere,
    # carry=slots), keeping existing configs and pickles bit-identical.
    self._use_predictor = use_predictor
    self._predictor_blocks = predictor_blocks
    self._predictor_heads = predictor_heads
    self._sa_iters = sa_iters
    self._sa_first_iters = sa_first_iters
    self._patch = spec['patch']
    self._native_grid = spec['native_grid']
    self._layerscale = spec['layerscale']

  @property
  def n_slots(self):
    return self._n_slots

  @property
  def slot_dim(self):
    return self._slot_dim

  @property
  def n_patches(self):
    assert self._input_size % self._patch == 0
    return (self._input_size // self._patch) ** 2

  @property
  def feat_dim(self):
    return self._dim

  def initial(self, batch_size):
    # Policy-path carry; contents are irrelevant at episode starts because
    # is_first switches to the learned init before the carry is ever read.
    return jnp.zeros((batch_size, self._n_slots, self._slot_dim), f32)

  def _learned_init(self, batch_size, dtype):
    # FixedLearnedInit: learned (1, S, D) broadcast, init std = dim**-0.5.
    init = lambda shape: (
        self._slot_dim ** -0.5) * jax.random.normal(nj.rng(), shape, f32)
    slots = self.get('init_slots', init, (1, self._n_slots, self._slot_dim))
    return jnp.broadcast_to(
        slots, (batch_size, self._n_slots, self._slot_dim)).astype(dtype)

  def features(self, images, return_raw=False):
    """(N, H, W, C) float in [0, 1] -> (N, patches, slot_dim).
    With return_raw also returns the pre-projection backbone features
    (N, patches, feat_dim) — the TimmExtractor 'vit_block12' output that
    SlotContrast's MLPDecoder reconstructs (encoder.backbone_features)."""
    dtype = f32 if self._force_f32 else images.dtype
    x = images.astype(dtype)
    x = resize_bilinear(x, self._input_size)
    mean = jnp.asarray(IMAGENET_MEAN, dtype)
    std = jnp.asarray(IMAGENET_STD, dtype)
    x = (x - mean) / std
    backbone = self.get(
        'backbone', ViT, self._depth, self._dim, self._heads, self._patch,
        self._native_grid, self._layerscale, remat=self._remat)
    proj = self.get(
        'proj', TwoLayerMLP, self._dim * self._hidden_mult, self._slot_dim)
    # Bound peak memory by mapping the ViT over frame chunks. Only applies
    # when the flat batch divides evenly (the training path); smaller batches
    # (e.g. the per-step policy path) run in one piece.
    # sg_backbone: with a frozen backbone (frozen_keys), its gradients are
    # computed and then zeroed by the optimizer; the stop_gradient skips that
    # backward pass entirely — no ViT residuals stored, large memory/compute
    # win. Only enable when the backbone is actually frozen.
    sg = jax.lax.stop_gradient if self._sg_backbone else (lambda v: v)
    n = x.shape[0]
    if (self._chunk and not nj.creating()
        and n > self._chunk and n % self._chunk == 0):
      x = x.reshape((n // self._chunk, self._chunk) + x.shape[1:])
      fn = lambda xs: ((lambda raw: (proj(raw), raw))(sg(backbone(xs))))
      feats, raw = jax.lax.map(fn, x)
      feats = feats.reshape((n,) + feats.shape[2:])
      raw = raw.reshape((n,) + raw.shape[2:])
    else:
      raw = sg(backbone(x))
      feats = proj(raw)
    return (feats, raw) if return_raw else feats

  def _corrector(self):
    return self.get(
        'corrector', SlotAttention,
        self._slot_dim, 4 * self._slot_dim, n_iters=3)

  def _predictor(self):
    return self.get(
        'predictor', TransformerPredictor,
        self._slot_dim, self._predictor_blocks, self._predictor_heads)

  def step(self, prev_slots, feats, is_first):
    """One recurrent step. feats (B, P, D), prev_slots (B, S, D), is_first (B,).
    Returns (slots, carry) — the emitted slots and the state to feed back at
    the next step. The SA/predictor passes are cheap next to the ViT, so both
    branches are always computed and selected with a where().

    use_predictor=True (video-model semantics, LatentProcessor+ScanOverTime):
      episode start: SA(learned_init, sa_first_iters); else SA(prev, sa_iters)
      where prev is the carried state_predicted; carry = predictor(slots).
    use_predictor=False (legacy extractor semantics):
      episode start: SA3(SA3(learned_init)) (initialize_twice); else SA3(prev);
      carry = slots."""
    corrector = self._corrector()
    init = self._learned_init(feats.shape[0], feats.dtype)
    prev = prev_slots.astype(feats.dtype)
    if self._use_predictor:
      first = corrector(init, feats, n_iters=self._sa_first_iters)
      cont = corrector(prev, feats, n_iters=self._sa_iters)
      slots = switch(is_first, first, cont)
      return slots, self._predictor()(slots)
    first = corrector(corrector(init, feats), feats)
    cont = corrector(prev, feats)
    slots = switch(is_first, first, cont)
    return slots, slots

  def __call__(self, images, is_first, carry=None, return_features=False):
    """With return_features additionally returns the raw backbone features
    ((B, P, F) / (B, T, P, F)) as a third element — the featrec target."""
    if len(images.shape) == 4:  # single step (policy path)
      feats = self.features(images, return_raw=True)
      feats, raw = feats
      if carry is None:
        carry = self.initial(images.shape[0])
      slots, carry = self.step(carry, feats, is_first)
      if return_features:
        return slots, carry.astype(f32), raw
      return slots, carry.astype(f32)
    assert len(images.shape) == 5, images.shape
    batch, length = images.shape[:2]
    flat = images.reshape((batch * length,) + images.shape[2:])
    feats, raw = self.features(flat, return_raw=True)
    feats = feats.reshape((batch, length) + feats.shape[1:])
    if carry is None:
      carry = self.initial(batch)
    carry = carry.astype(feats.dtype)
    # Scan over time like RSSM.observe: (B, T, ...) -> (T, B, ...).
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    def scan_step(prev, inputs):
      slots, next_carry = self.step(prev, *inputs)
      return next_carry, slots
    carry, slots = nj.scan(
        scan_step, carry, (swap(feats), swap(is_first)))
    slots = swap(slots)  # (B, T, S, D)
    if return_features:
      raw = raw.reshape((batch, length) + raw.shape[1:])
      return slots, carry.astype(f32), raw
    return slots, carry.astype(f32)
