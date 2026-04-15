import einops
import jax
import jax.ad_checkpoint as adc
import jax.numpy as jnp
import numpy as np

from . import ninjax as nj
from .nets import Linear, get_act

f32 = jnp.float32


class Norm(nj.Module):

  def __init__(self, impl, axis: tuple = (-1,), eps: float = 1e-4, scale: bool = True, shift: bool = True):
    self.axis = axis
    self.eps = eps
    self.scale = scale
    self.shift = shift
    self.impl = impl

  def __call__(self, x):
    # ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)

    return self.get('scale', jnp.ones, shape[-1], f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)

    return self.get('shift', jnp.zeros, shape[-1], f32).astype(dtype)


def rope(x, ts=None, inverse=False, maxlen=4096):
  B, T, _, D = x.shape
  if ts is None:
    ts = jnp.ones(B, jnp.int32)[:, None] * jnp.arange(T)[None, :]  # [B, T]
  assert ts.shape == (B, T), (ts.shape, (B, T))
  if inverse:
    ts = -ts
  freq_exponents = (2.0 / D) * jnp.arange(D // 2)  # [D/2]
  timescale = maxlen ** freq_exponents
  radians = ts[:, :, None] / timescale[None, None, :]  # [B, T, D/2]
  radians = radians[..., None, :].astype(x.dtype)  # [B, T, 1, D/2]
  sin, cos = jnp.sin(radians), jnp.cos(radians)
  x1, x2 = jnp.split(x, 2, axis=-1)  # [B, T, H, D/2]
  res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return res


def dropout(x, prob, training):
  if not prob or not training:
    return x
  keep = jax.random.bernoulli(nj.rng(), 1.0 - prob, x.shape)
  return x * keep / (1.0 - prob)


class Attention(nj.Module):

  def __init__(self, heads: int = 8, kv_heads: int = 0, dropout: float = 0.0, rope: bool = True, qknorm: str = 'none',
               bias: bool = True, winit: str = 'normal', outscale: float = 1.0):
    self.heads = heads
    self.kv_heads = kv_heads
    self.dropout = dropout
    self.rope = rope
    self.qknorm = qknorm
    self.bias = bias
    self.winit = winit
    self.outscale = outscale
    self.kw_linear = dict(bias=self.bias, winit=self.winit)

  def __call__(self, x, mask=None, ts=None, training=True):
    B, T, D = x.shape
    kv_heads = self.kv_heads or self.heads
    assert self.heads % kv_heads == 0
    head_ratio = self.heads // kv_heads
    if head_ratio == 1:
      qkv = self.get('qkv', Linear, 3 * D, **self.kw_linear)(x)
      q, k, v = jnp.split(qkv, 3, -1)
    else:
      q = self.get('q', Linear, D, **self.kw_linear)(x)
      k = self.get('k', Linear, D // head_ratio, **self.kw_linear)(x)
      v = self.get('v', Linear, D // head_ratio, **self.kw_linear)(x)
    q = einops.rearrange(q, 'b t (h d) -> b t h d', h=self.heads)
    k = einops.rearrange(k, 'b t (h d) -> b t h d', h=kv_heads)
    v = einops.rearrange(v, 'b t (h d) -> b t h d', h=kv_heads)

    if self.qknorm != 'none':
      q = self.get('normq', Norm, self.qknorm)(q)
      k = self.get('normk', Norm, self.qknorm)(k)

    if self.rope:
      q = rope(q, ts)
      k = rope(k, ts)

    q = einops.rearrange(q, 'b t (h g) d -> b t h g d', h=kv_heads)
    logits = einops.einsum(q, k, 'b tq h g d, b tk h d -> b h g tq tk')
    logits = logits * (1.0 / np.sqrt(k.shape[-1]))
    logits = f32(logits)
    if mask is not None:
      Tq, Tk = q.shape[1], k.shape[1]
      assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
      mask = einops.rearrange(mask, 'b tq tk -> b 1 1 tq tk')
      logits = jnp.where(mask, logits, -1e30)
    weights = jax.nn.softmax(logits)
    weights = weights.astype(x.dtype)
    weights = dropout(weights, self.dropout, training)
    x = einops.einsum(weights, v, 'b h g tq tk, b tk h d -> b tq h g d')
    x = einops.rearrange(x, 'b t h g d -> b t (h g d)')
    x = self.get('proj', Linear, D, **self.kw_linear, outscale=self.outscale)(x)
    return x


class Transformer(nj.Module):
  def __init__(self, units: int = 1024, layers: int = 12, heads: int = 8, ffup: int = 4, act: str = 'silu',
               norm: str = 'layer', glu: bool = False, rope: bool = True, qknorm: str = 'none', bias: bool = True,
               winit: str = 'normal', outscale: float = 1.0):
    self.units = units
    self.layers = layers
    self.heads = heads
    self.ffup = ffup
    self.act = act
    self.norm = norm
    self.glu = glu
    self.rope = rope
    self.qknorm = qknorm
    self.bias = bias
    self.winit = winit
    self.outscale = outscale

  def __call__(self, x, mask=None, ts=None, training=True):
    kw = dict(bias=self.bias, winit=self.winit,)
    ak = dict(heads=self.heads, rope=self.rope, qknorm=self.qknorm, outscale=self.outscale)
    D = x.shape[-1]
    assert D == self.units, (D, self.units)
    for i in range(self.layers):
      with nj.scope(f'layer{i}'):
        skip = x
        x = self.get('norm1', Norm, self.norm)(x)
        x  = self.get('mha', Attention, **kw, **ak)(x, mask, ts, training)
        x += skip
        skip = x
        x = self.get('norm2', Norm, self.norm)(x)
        if self.glu:
          U = max(D, int((D * self.ffup * 2 / 3) // 32 * 32))
          ff1 = self.get('ff1', Linear, U, **kw)
          ff2 = self.get('ff2', Linear, U, **kw)
          ff3 = self.get('ff3', Linear, D, **kw, outscale=self.outscale)
          x = ff3(get_act(self.act)(ff1(x)) * ff2(x))
        else:
          ff1 = self.get('ff1', Linear, D * self.ffup, **kw)
          ff2 = self.get('ff2', Linear, D, **kw, outscale=self.outscale)
          x = ff2(get_act(self.act)(ff1(x)))
        x += skip
    x = self.get('outnorm', Norm, self.norm)(x)
    return x
