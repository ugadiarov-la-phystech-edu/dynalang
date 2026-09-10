"""Slot modules of a causal SOLV-SAM encoder, ported for frozen inference.

Module and parameter names match the dyn-O training code, so a checkpoint
written there loads into these modules unchanged. Everything that only exists
to train the encoder is left out: there is no RGB decoder, no attention loss
head and no SAM mask branch, since a causal temporal binder is always trained
without SAM supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Small constant that keeps divisions finite in reduced precision.
EPS = 1e-7


def frame_window_indices(num_steps, num_neighbors, frame_stride, device=None):
  """Index the causal window of every frame, clamped at the episode start.

  Returns `(num_steps, num_neighbors + 1)` indices and a matching validity
  mask. Clamping repeats the first frame so the windows stay rectangular; the
  mask is what tells the temporal binder to ignore those repeats. The last
  column is always the frame being described, so no window reaches into the
  future.
  """
  offsets = torch.arange(-num_neighbors, 1, device=device) * frame_stride
  indices = torch.arange(num_steps, device=device).unsqueeze(dim=1) + offsets
  valid = indices >= 0
  return indices.clamp(0, num_steps - 1), valid


class MLP(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, residual=False,
               layer_order='none'):
    super().__init__()
    self.residual = residual
    self.layer_order = layer_order
    if residual:
      assert input_dim == output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, output_dim)
    self.activation = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.1)
    if layer_order in ('pre', 'post'):
      self.norm = nn.LayerNorm(input_dim)
    else:
      assert layer_order == 'none'

  def forward(self, x):
    inputs = x
    if self.layer_order == 'pre':
      x = self.norm(x)
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    x = self.dropout(x)
    if self.residual:
      x = x + inputs
    if self.layer_order == 'post':
      x = self.norm(x)
    return x


class SpatialBinder(nn.Module):
  """Binds the image tokens of a single frame into slots."""

  def __init__(self, config, input_dim):
    super().__init__()
    self.num_slots = config.num_slots
    self.scale = config.slot_dim ** -0.5
    self.iters = config.slot_att_iter
    self.slot_dim = config.slot_dim

    self.res_h = config.resize_to[0] // config.patch_size
    self.res_w = config.resize_to[1] // config.patch_size
    self.num_tokens = int(self.res_h * self.res_w)

    self.sigma = 5
    xs = torch.linspace(-1, 1, steps=self.res_w)
    ys = torch.linspace(-1, 1, steps=self.res_h)
    xs, ys = torch.meshgrid(xs, ys, indexing='xy')
    xs = xs.reshape(1, 1, -1, 1)
    ys = ys.reshape(1, 1, -1, 1)
    self.abs_grid = nn.Parameter(
        torch.cat([xs, ys], dim=-1), requires_grad=True)
    assert self.abs_grid.shape[2] == self.num_tokens

    # Only used by the relative grid readout that visualisations ask for, but
    # present in every checkpoint.
    self.h = nn.Linear(2, self.slot_dim)

    self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))
    self.S_s = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))
    self.S_p = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))
    init.xavier_uniform_(self.slots)
    init.normal_(self.S_s, mean=0., std=.02)
    init.normal_(self.S_p, mean=0., std=.02)
    sign = torch.sign(self.S_s.data)
    sign[sign == 0] = 1
    self.S_s.data = sign * (self.S_s.data.abs() + 1e-3)

    self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
    self.norm = nn.LayerNorm(self.slot_dim)
    self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
    self.mlp = MLP(
        self.slot_dim, 4 * self.slot_dim, self.slot_dim,
        residual=True, layer_order='pre')

    self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
    self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
    self.g = nn.Linear(2, self.slot_dim)
    self.f = nn.Sequential(
        nn.Linear(self.slot_dim, self.slot_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.slot_dim, self.slot_dim))

    self.initial_mlp = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, input_dim),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim, self.slot_dim),
        nn.LayerNorm(self.slot_dim))

    self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)

  def forward(self, inputs):
    """Map `(bs, num_tokens, token_dim)` features to slots and attention."""
    bs, num_tokens, _ = inputs.shape
    num_slots, slot_dim = self.num_slots, self.slot_dim

    slots = self.slots.expand(bs, num_slots, slot_dim)

    inputs = self.initial_mlp(inputs).unsqueeze(dim=1)
    inputs = inputs.expand(bs, num_slots, num_tokens, slot_dim)

    abs_grid = self.abs_grid.expand(bs, num_slots, self.num_tokens, 2)
    S_s = self.S_s.expand(bs, num_slots, 1, 2)
    S_p = self.S_p.expand(bs, num_slots, 1, 2)

    sign = torch.sign(S_s)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    S_s = sign * (S_s.abs() + EPS)

    attn = None
    unweighted_attn = None
    for step in range(self.iters + 1):
      slots_prev = slots
      slots = self.norm(slots)

      rel_grid = (abs_grid - S_p) / (S_s * self.sigma)
      k = self.f(self.K(inputs) + self.g(rel_grid))
      v = self.f(self.V(inputs) + self.g(rel_grid))

      q = self.Q(slots).unsqueeze(dim=-1)
      dots = torch.einsum('bsdi,bsjd->bsj', q, k) * self.scale

      # Tokens compete over slots, which is what makes the slots partition
      # the frame instead of all describing the same thing.
      attn = dots.softmax(dim=1)
      unweighted_attn = attn

      attn = attn / (attn.sum(dim=-1, keepdim=True) + EPS)
      attn = attn.unsqueeze(dim=2)

      updates = torch.einsum('bsjd,bsij->bsd', v, attn)

      S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn).unsqueeze(dim=2)
      values_ss = torch.pow(abs_grid - S_p, 2)
      S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)
      sign = torch.sign(S_s)
      sign = torch.where(sign == 0, torch.ones_like(sign), sign)
      S_s = sign * (S_s.abs() + EPS)
      S_s = torch.sqrt(S_s).unsqueeze(dim=2)

      if step != self.iters:
        slots = self.gru(
            updates.reshape(-1, slot_dim), slots_prev.reshape(-1, slot_dim))
        slots = slots.reshape(bs, -1, slot_dim)
        slots = self.mlp(slots)

    slots = self.final_layer(slots)
    attn = attn.reshape(bs, num_slots, num_tokens)
    return slots, attn, unweighted_attn


class TemporalBinder(nn.Module):
  """Mixes every slot with itself across the frames of a causal window.

  The slot axis moves into the batch, so one sequence is one slot index over
  time and slots never meet each other here; relating them within a frame
  stays the spatial binder's job. Sharing an index is all that ties a slot to
  the same object across the window, and nothing enforces that, so this
  smooths tracking rather than guaranteeing it.
  """

  def __init__(self, config):
    super().__init__()
    self.slot_dim = config.slot_dim
    self.num_neighbors = config.num_neighbors
    self.num_frames = config.num_frames

    encoder_layer = nn.TransformerEncoderLayer(
        self.slot_dim, nhead=8, dim_feedforward=4 * self.slot_dim,
        batch_first=True)
    self.slot_transformer = nn.TransformerEncoder(
        encoder_layer, config.temporal_depth)

    self.pos_embed_temporal = nn.Parameter(
        torch.Tensor(1, self.num_frames, 1, self.slot_dim))
    init.normal_(self.pos_embed_temporal, mean=0., std=.02)

  def forward(self, slots, frame_valid=None):
    """Bind `(bs * num_frames, num_slots, slot_dim)` windows.

    Returns the current frame of every window, `(bs, num_slots, slot_dim)`.
    """
    _, num_slots, _ = slots.shape

    slots = slots.unflatten(dim=0, sizes=(-1, self.num_frames))
    slots = slots + self.pos_embed_temporal
    bs = slots.shape[0]

    slots = slots.permute(0, 2, 1, 3)
    slots = slots.flatten(start_dim=0, end_dim=1)

    padding_mask = None
    if frame_valid is not None:
      # The current frame is valid by construction, so no sequence is fully
      # masked out.
      padding_mask = (~frame_valid.bool()).repeat_interleave(num_slots, dim=0)

    slots = self.slot_transformer(slots, src_key_padding_mask=padding_mask)
    slots = slots.unflatten(dim=0, sizes=(bs, num_slots))
    return slots[:, :, -1]


class FeatureDecoder(nn.Module):
  """Broadcast decoder that maps a slot back to per-patch features."""

  def __init__(self, config):
    super().__init__()
    hidden_dim = 1024
    self.layer1 = nn.Linear(config.slot_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, hidden_dim)
    self.layer3 = nn.Linear(hidden_dim, hidden_dim)
    self.layer4 = nn.Linear(hidden_dim, hidden_dim)
    self.relu = nn.ReLU(inplace=True)
    self.dino_layer = nn.Linear(hidden_dim, config.token_dim + 1)

    self.decode_segmentation = config.decode_segmentation
    if self.decode_segmentation:
      self.segment_layer = nn.Linear(
          hidden_dim + config.token_dim + 1, config.patch_size ** 2)

  def forward(self, slot_maps):
    slot_maps = self.relu(self.layer1(slot_maps))
    slot_maps = self.relu(self.layer2(slot_maps))
    slot_maps = self.relu(self.layer3(slot_maps))
    slot_maps = self.relu(self.layer4(slot_maps))
    features = self.dino_layer(slot_maps)
    out = {
        'enc_feat_rec': features[..., :-1],
        'enc_feat_rec_logits': features[..., -1:],
    }
    if self.decode_segmentation:
      out['segmentation_logits'] = self.segment_layer(
          torch.cat([slot_maps, features], dim=-1))
    return out


class SolvSam(nn.Module):
  """Frozen slot model: spatial binder, causal temporal binder, decoder."""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.slot_dim = config.slot_dim
    self.slot_num = config.num_slots
    self.token_num = config.token_num
    self.token_dim = config.token_dim
    self.num_neighbors = config.num_neighbors
    self.num_frames = config.num_frames

    self.s_bind = SpatialBinder(config, input_dim=self.token_dim)
    self.t_bind = TemporalBinder(config) if config.num_neighbors else None

    self.dec = FeatureDecoder(config)
    self.pos_dec = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
    init.normal_(self.pos_dec, mean=0., std=.02)

  def get_spatial_slots(self, features):
    """Bind image tokens independently within each frame."""
    return self.s_bind(features)[0]

  def bind_temporal(self, spatial_slots, frame_valid=None):
    """Bind a flattened batch of causal slot windows."""
    if self.t_bind is None:
      return spatial_slots
    return self.t_bind(spatial_slots, frame_valid)

  def get_slots(self, features, frame_valid=None):
    """Slots of the current frame of every window in the batch."""
    return self.bind_temporal(self.get_spatial_slots(features), frame_valid)

  def decode(self, slots):
    """Reconstruct the backbone features a set of slots stands for.

    Returns the blended features, `(bs, num_tokens, token_dim)`, together with
    the weights they were blended by, `(bs, num_slots, num_tokens)`. Those
    weights are what makes a slot readable as a region of the frame.
    """
    slot_maps = slots.unsqueeze(dim=-2).tile(1, 1, self.token_num, 1)
    slot_maps = slot_maps + self.pos_dec.expand_as(slot_maps)
    out = self.dec(slot_maps)
    patch_mask = F.softmax(out['enc_feat_rec_logits'], dim=1)
    features = torch.sum(out['enc_feat_rec'] * patch_mask, dim=1)
    return features, patch_mask.squeeze(dim=-1)

  def decode_patch_masks(self, slots):
    """Per-slot occupancy over patches, `(bs, num_slots, num_tokens)`."""
    return self.decode(slots)[1]
