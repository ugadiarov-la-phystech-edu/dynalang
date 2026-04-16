"""
Ninjax Transformer modules mirroring the torch.nn.Transformer API.

Re-implements Transformer components as nj.Module subclasses so the
parameters live in ninjax's CONTEXT and can be used inside the Dynalang
agent alongside the existing modules in dynalang/nets.py.

Tensor layout matches torch.nn.Transformer:
    src : (src_len, batch, d_model)
    tgt : (tgt_len, batch, d_model)
    out : (tgt_len, batch, d_model)

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


def _split_heads(x: jnp.ndarray, nhead: int) -> jnp.ndarray:
  """(seq, batch, d_model) -> (batch, nhead, seq, head_dim)"""
  seq, batch, d_model = x.shape
  head_dim = d_model // nhead
  x = x.reshape(seq, batch, nhead, head_dim)
  return x.transpose(1, 2, 0, 3)


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
  """(batch, nhead, seq, head_dim) -> (seq, batch, d_model)"""
  batch, nhead, seq, head_dim = x.shape
  x = x.transpose(2, 0, 1, 3)
  return x.reshape(seq, batch, nhead * head_dim)


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
  q, k, v : (batch, nhead, seq, head_dim)
  Returns : (batch, nhead, tgt_seq, head_dim)
  """
  bool_mask: Optional[jnp.ndarray] = None
  if mask is not None:
    bool_mask = jnp.isfinite(mask)
  if key_padding_mask is not None:
    kpm = ~key_padding_mask[:, None, None, :]
    bool_mask = kpm if bool_mask is None else (bool_mask & kpm)

  head_dim = q.shape[-1]
  scale = 1.0 / math.sqrt(head_dim)
  logits = jnp.einsum('bhtd,bhsd->bhts', q, k) * scale

  if bool_mask is not None:
    logits = jnp.where(bool_mask, logits, jnp.finfo(logits.dtype).min)

  weights = jax.nn.softmax(logits, axis=-1)
  weights = _dropout(weights, dropout, training)
  return jnp.einsum('bhts,bhsd->bhtd', weights, v)


class MultiHeadAttention(nj.Module):

  def __init__(self, embed_dim, num_heads, dropout=0.0):
    assert embed_dim % num_heads == 0, (embed_dim, num_heads)
    self._embed_dim = embed_dim
    self._num_heads = num_heads
    self._dropout = dropout

  def __call__(
      self, query, key, value,
      attn_mask=None, key_padding_mask=None, training=False):
    q = self.get('q_proj', Linear, self._embed_dim)(query)
    k = self.get('k_proj', Linear, self._embed_dim)(key)
    v = self.get('v_proj', Linear, self._embed_dim)(value)

    q = _split_heads(q, self._num_heads)
    k = _split_heads(k, self._num_heads)
    v = _split_heads(v, self._num_heads)

    attn_out = scaled_dot_product_attention(
        q, k, v,
        mask=attn_mask,
        key_padding_mask=key_padding_mask,
        dropout=self._dropout,
        training=training)

    attn_out = _merge_heads(attn_out)
    return self.get('out_proj', Linear, self._embed_dim)(attn_out)


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
        self._d_model, self._nhead, self._dropout)(
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
    seq_len = src.shape[0]
    pe = sinusoidal_positional_encoding(seq_len, self._d_model)
    x = src + pe[:, None, :]
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
