"""
Ninjax Transformer modules mirroring the torch.nn.Transformer API.

Re-implements Transformer components as nj.Module subclasses so the
parameters live in ninjax's CONTEXT and can be used inside the Dynalang
agent alongside the existing modules in dynalang/nets.py.

Tensor layout (batch-first):
    src : (batch, src_len, d_model)
    tgt : (batch, tgt_len, d_model)
    out : (batch, tgt_len, d_model)

Masks follow PyTorch conventions:
    attn_mask        — additive float mask, -inf blocks a position.
    key_padding_mask — bool array (batch, src_len), True = ignore that key.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp

from . import ninjax as nj
from .nets import Linear, Norm


def _dropout(x: jnp.ndarray, rate: float, training: bool) -> jnp.ndarray:
  if not training or rate == 0.0:
    return x
  keep_prob = 1.0 - rate
  keep = jax.random.bernoulli(nj.rng(), keep_prob, x.shape)
  return jnp.where(keep, x / keep_prob, 0.0)


def sinusoidal_positional_encoding(
    seq_len: int, d_model: int,
) -> jnp.ndarray:
  """Sinusoidal PE from 'Attention Is All You Need'. Returns (seq_len, d_model)."""
  pos = jnp.arange(seq_len)[:, None]        # (seq_len, 1)
  dim = jnp.arange(0, d_model, 2)[None, :]  # (1, d_model//2)
  angle = pos / jnp.power(10000.0, dim / d_model)
  pe = jnp.zeros((seq_len, d_model))
  pe = pe.at[:, 0::2].set(jnp.sin(angle))
  pe = pe.at[:, 1::2].set(jnp.cos(angle))
  return pe


def scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    key_padding_mask: Optional[jnp.ndarray] = None,
    dropout: float = 0.0,
    training: bool = False,
) -> jnp.ndarray:
  """
  q, k, v : (batch, seq, nhead, head_dim)
  Returns : (batch, tgt_seq, nhead, head_dim)
  """
  head_dim = q.shape[-1]
  scale = 1.0 / math.sqrt(head_dim)
  logits = jnp.einsum('btnh,bsnh->bnts', q, k) * scale  # (B, H, Tq, Tk)

  bool_mask: Optional[jnp.ndarray] = None
  if mask is not None:
    bool_mask = jnp.isfinite(mask)
    if bool_mask.ndim == 3:
      bool_mask = bool_mask[:, None, :, :]
  if key_padding_mask is not None:
    kpm = ~key_padding_mask[:, None, None, :]  # (B, 1, 1, Tk)
    bool_mask = kpm if bool_mask is None else (bool_mask & kpm)

  if bool_mask is not None:
    logits = jnp.where(bool_mask, logits, jnp.finfo(logits.dtype).min)

  weights = jax.nn.softmax(logits, axis=-1)
  weights = _dropout(weights, dropout, training)
  return jnp.einsum('bnts,bsnh->btnh', weights, v)  # (B, Tq, H, D)


class MultiHeadAttention(nj.Module):

  def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=True):
    assert embed_dim % num_heads == 0, (embed_dim, num_heads)
    self._embed_dim = embed_dim
    self._num_heads = num_heads
    self._head_dim = embed_dim // num_heads
    self._dropout = dropout
    self._self_attn = self_attn

  def __call__(
      self, query, key, value,
      attn_mask=None, key_padding_mask=None, training=False):
    D = self._embed_dim
    H = self._num_heads
    hd = self._head_dim

    if self._self_attn:
      qkv = self.get('in_proj', Linear, 3 * D)(query)  # (B, T, 3*D)
      qkv = qkv.reshape(*qkv.shape[:-1], 3, H, hd)    # (B, T, 3, H, hd)
      q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
    else:
      q = self.get('q_proj', Linear, D)(query).reshape(*query.shape[:-1], H, hd)
      k = self.get('k_proj', Linear, D)(key).reshape(*key.shape[:-1], H, hd)
      v = self.get('v_proj', Linear, D)(value).reshape(*value.shape[:-1], H, hd)

    attn_out = scaled_dot_product_attention(
        q, k, v,
        mask=attn_mask,
        key_padding_mask=key_padding_mask,
        dropout=self._dropout,
        training=training)

    attn_out = attn_out.reshape(*attn_out.shape[:-2], D)  # (B, T, D)
    return self.get('out_proj', Linear, D)(attn_out)


class TransformerEncoderLayer(nj.Module):

  def __init__(
      self, d_model, nhead, feedforward_units=1024,
      dropout=0.1, norm_first=False):
    self._d_model = d_model
    self._nhead = nhead
    self._feedforward_units = feedforward_units
    self._dropout = dropout
    self._norm_first = norm_first

  def __call__(
      self, src, src_mask=None, src_key_padding_mask=None, training=False):
    attn = lambda x: self.get(
        'self_attn', MultiHeadAttention,
        self._d_model, self._nhead, self._dropout)(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            training=training)

    if self._norm_first:
      normed = self.get('norm1', Norm, 'layer')(src)
      src = src + _dropout(attn(normed), self._dropout, training)

      normed = self.get('norm2', Norm, 'layer')(src)
      ff = jax.nn.relu(
          self.get('linear1', Linear, self._feedforward_units)(normed))
      ff = _dropout(ff, self._dropout, training)
      ff = self.get('linear2', Linear, self._d_model)(ff)
      src = src + _dropout(ff, self._dropout, training)
    else:
      attn_out = attn(src)
      src = self.get('norm1', Norm, 'layer')(
          src + _dropout(attn_out, self._dropout, training))

      ff = jax.nn.relu(self.get('linear1', Linear, self._feedforward_units)(src))
      ff = _dropout(ff, self._dropout, training)
      ff = self.get('linear2', Linear, self._d_model)(ff)
      src = self.get('norm2', Norm, 'layer')(
          src + _dropout(ff, self._dropout, training))
    return src


class TransformerDecoderLayer(nj.Module):

  def __init__(
      self, d_model, nhead, feedforward_units=2048,
      dropout=0.1, norm_first=False):
    self._d_model = d_model
    self._nhead = nhead
    self._feedforward_units = feedforward_units
    self._dropout = dropout
    self._norm_first = norm_first

  def __call__(
      self, tgt, memory,
      tgt_mask=None, memory_mask=None,
      tgt_key_padding_mask=None, memory_key_padding_mask=None,
      training=False):
    self_attn = lambda x: self.get(
        'self_attn', MultiHeadAttention,
        self._d_model, self._nhead, self._dropout)(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            training=training)
    cross_attn = lambda x: self.get(
        'cross_attn', MultiHeadAttention,
        self._d_model, self._nhead, self._dropout, False)(
            x, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            training=training)

    if self._norm_first:
      normed = self.get('norm1', Norm, 'layer')(tgt)
      tgt = tgt + _dropout(self_attn(normed), self._dropout, training)

      normed = self.get('norm2', Norm, 'layer')(tgt)
      tgt = tgt + _dropout(cross_attn(normed), self._dropout, training)

      normed = self.get('norm3', Norm, 'layer')(tgt)
      ff = jax.nn.relu(
          self.get('linear1', Linear, self._feedforward_units)(normed))
      ff = _dropout(ff, self._dropout, training)
      ff = self.get('linear2', Linear, self._d_model)(ff)
      tgt = tgt + _dropout(ff, self._dropout, training)
    else:
      sa = self_attn(tgt)
      tgt = self.get('norm1', Norm, 'layer')(
          tgt + _dropout(sa, self._dropout, training))

      ca = cross_attn(tgt)
      tgt = self.get('norm2', Norm, 'layer')(
          tgt + _dropout(ca, self._dropout, training))

      ff = jax.nn.relu(self.get('linear1', Linear, self._feedforward_units)(tgt))
      ff = _dropout(ff, self._dropout, training)
      ff = self.get('linear2', Linear, self._d_model)(ff)
      tgt = self.get('norm3', Norm, 'layer')(
          tgt + _dropout(ff, self._dropout, training))
    return tgt


class TransformerEncoder(nj.Module):

  def __init__(
      self, num_layers, d_model, nhead, feedforward_units=1024,
      dropout=0.1, norm_first=False, norm=False):
    self._num_layers = num_layers
    self._d_model = d_model
    self._nhead = nhead
    self._feedforward_units = feedforward_units
    self._dropout = dropout
    self._norm_first = norm_first
    self._norm = norm

  def __call__(
      self, src, mask=None, src_key_padding_mask=None, training=False):
    seq_len = src.shape[1]
    pe = sinusoidal_positional_encoding(seq_len, self._d_model)
    x = src + pe[None, :, :]
    x = _dropout(x, self._dropout, training)
    for i in range(self._num_layers):
      x = self.get(
          f'layer_{i}', TransformerEncoderLayer,
          self._d_model, self._nhead, self._feedforward_units,
          self._dropout, self._norm_first)(
              x, src_mask=mask,
              src_key_padding_mask=src_key_padding_mask,
              training=training)
    if self._norm:
      x = self.get('norm', Norm, 'layer')(x)
    return x


class TransformerDecoder(nj.Module):

  def __init__(
      self, num_layers, d_model, nhead, feedforward_units=2048,
      dropout=0.1, norm_first=False, norm=False):
    self._num_layers = num_layers
    self._d_model = d_model
    self._nhead = nhead
    self._feedforward_units = feedforward_units
    self._dropout = dropout
    self._norm_first = norm_first
    self._norm = norm

  def __call__(
      self, tgt, memory,
      tgt_mask=None, memory_mask=None,
      tgt_key_padding_mask=None, memory_key_padding_mask=None,
      training=False):
    x = tgt
    for i in range(self._num_layers):
      x = self.get(
          f'layer_{i}', TransformerDecoderLayer,
          self._d_model, self._nhead, self._feedforward_units,
          self._dropout, self._norm_first)(
              x, memory,
              tgt_mask=tgt_mask,
              memory_mask=memory_mask,
              tgt_key_padding_mask=tgt_key_padding_mask,
              memory_key_padding_mask=memory_key_padding_mask,
              training=training)
    if self._norm:
      x = self.get('norm', Norm, 'layer')(x)
    return x


class Transformer(nj.Module):
  """Full encoder-decoder Transformer, API-compatible with torch.nn.Transformer."""

  def __init__(
      self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
      feedforward_units=2048, dropout=0.1, norm_first=False):
    self._d_model = d_model
    self._nhead = nhead
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._feedforward_units = feedforward_units
    self._dropout = dropout
    self._norm_first = norm_first

  def __call__(
      self, src, tgt,
      src_mask=None, tgt_mask=None, memory_mask=None,
      src_key_padding_mask=None, tgt_key_padding_mask=None,
      memory_key_padding_mask=None, training=False):
    memory = self.get(
        'encoder', TransformerEncoder,
        self._num_encoder_layers, self._d_model, self._nhead,
        self._feedforward_units, self._dropout, self._norm_first, True)(
            src, mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            training=training)
    output = self.get(
        'decoder', TransformerDecoder,
        self._num_decoder_layers, self._d_model, self._nhead,
        self._feedforward_units, self._dropout, self._norm_first, True)(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            training=training)
    return output

  @staticmethod
  def generate_square_subsequent_mask(sz: int) -> jnp.ndarray:
    """Causal additive mask of shape (sz, sz); upper-triangle = -inf."""
    return jnp.triu(jnp.full((sz, sz), -jnp.inf), k=1)


class ObjectCentricDynamicsTransformer(nj.Module):
  """Single-pass spatio-temporal Transformer over object slots.

  All ``T * num_slots`` slot tokens in the context window interact in one
  joint self-attention pass over the flattened sequence, mirroring the
  SlotFormer rollouter.

  Temporal causality is preserved when a ``causal_mask`` is supplied: it is
  block-expanded so a slot at step t can attend to every slot at steps <= t.
  Pass ``causal_mask=None`` for fully bidirectional attention over the window
  (exactly the SlotFormer encoder behaviour).

  Action modes
  ------------
  'none'       : pure slot self-attention.
  'slot'       : action is appended as an extra slot by the caller; identical
                 to 'none' here (just a longer slot axis).
  'cross_attn' : slot tokens cross-attend to the per-step action embedding
                 (encoder layers become decoder layers with action as memory).

  Args:
    num_layers: Number of stacked layers.
    d_model: Hidden dimensionality (must match slot embedding dim).
    nhead: Number of attention heads.
    feedforward_units: Feedforward hidden width inside each layer.
    dropout: Dropout rate.
    norm_first: Use pre-layer-norm (True) or post-layer-norm (False).
    norm: Apply a final layer norm to the output.
    action_mode: One of 'none', 'slot', 'cross_attn'.
  """

  def __init__(
      self, num_layers, d_model, nhead, feedforward_units=1024,
      dropout=0.0, norm_first=True, norm=True, action_mode='none'):
    assert action_mode in ('none', 'slot', 'cross_attn'), action_mode
    self._num_layers = num_layers
    self._d_model = d_model
    self._nhead = nhead
    self._feedforward_units = feedforward_units
    self._dropout = dropout
    self._norm_first = norm_first
    self._norm = norm
    self._action_mode = action_mode

  def __call__(self, x, causal_mask=None, action_embeds=None,
               key_padding_mask=None, training=False):
    """
    Args:
      x: (B, T, num_slots, d_model)
      causal_mask: (T, T) additive float causal mask over time, or None.
      action_embeds: (B, T, d_model) — required when action_mode='cross_attn'.
      key_padding_mask: (B, T) bool, True marks a padded (invalid) time step
                        that no token may attend to. Shared by all slots of
                        that step; expanded to the flattened token axis.
      training: bool.

    Returns:
      x: (B, T, num_slots, d_model)
    """
    B, T, num_slots, dim = x.shape
    assert dim == self._d_model, (dim, self._d_model)

    # Temporal positional encoding, shared across slots of the same step.
    pe = sinusoidal_positional_encoding(T, self._d_model)
    x = x + pe[None, :, None, :]
    x = _dropout(x, self._dropout, training)

    # Flatten time and slots into one joint token sequence: (B, T*num_slots, D).
    # Token index t * num_slots + i corresponds to slot i at time step t.
    x = x.reshape(B, T * num_slots, dim)

    # Block-expand the (T, T) temporal mask to (T*num_slots, T*num_slots) so a
    # slot at step t attends to every slot at steps allowed by causal_mask.
    src_mask = None
    if causal_mask is not None:
      src_mask = jnp.kron(
          causal_mask, jnp.ones((num_slots, num_slots), causal_mask.dtype))

    # Per-step padding flag shared by all slots of the step: (B, T*num_slots).
    slot_key_padding_mask = None
    if key_padding_mask is not None:
      slot_key_padding_mask = jnp.repeat(key_padding_mask, num_slots, axis=1)

    if self._action_mode == 'cross_attn':
      assert action_embeds is not None, \
          "action_mode='cross_attn' requires action_embeds"

      step_mask = jnp.where(jnp.eye(T, dtype=bool), 0.0, -jnp.inf)
      memory_mask = jnp.repeat(step_mask, num_slots, axis=0)

    for i in range(self._num_layers):
      if self._action_mode == 'cross_attn':
        x = self.get(
            f'layer_{i}', TransformerDecoderLayer,
            self._d_model, self._nhead, self._feedforward_units,
            self._dropout, self._norm_first)(
                x, action_embeds,
                tgt_mask=src_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=slot_key_padding_mask,
                memory_key_padding_mask=key_padding_mask,
                training=training)
      else:
        x = self.get(
            f'layer_{i}', TransformerEncoderLayer,
            self._d_model, self._nhead, self._feedforward_units,
            self._dropout, self._norm_first)(
                x, src_mask=src_mask,
                src_key_padding_mask=slot_key_padding_mask,
                training=training)
    if self._norm:
      x = self.get('norm', Norm, 'layer')(x)

    x = x.reshape(B, T, num_slots, dim)
    return x
