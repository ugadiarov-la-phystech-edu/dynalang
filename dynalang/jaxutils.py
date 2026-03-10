import re
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optree
from tensorflow_probability.substrates import jax as tfp

from . import ninjax_compat as nj

tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
COMPUTE_DTYPE = jnp.float32
f32 = jnp.float32
i32 = jnp.int32


def rms(xs):
  xs = jax.tree.leaves(xs)
  count = sum(x.size for x in xs)
  sumsq = jnp.stack([f32(jnp.square(x).sum()) for x in xs]).sum()
  return jnp.sqrt(sumsq / f32(count))

def get_data_axes():
  axes = ('d', 'f')
  for x in axes:
    try:
      jax.lax.axis_index(x)
    except NameError:
      return ()
  return axes


def load_partial_checkpoint(varibs, state, load_key=''):
  ckpt_paths, ckpt_params, _ = optree.tree_flatten_with_path(state)
  ckpt_dict = {k: v for k, v in zip(ckpt_paths, ckpt_params)}
  ckpt_vars = []
  orig_vars = []
  loaded = optree.tree_map_with_path(
    lambda p, x: ckpt_dict[p] if p in ckpt_dict and load_key in p[0] else x,
    varibs)
  optree.tree_map_with_path(
    lambda p, x: ckpt_vars.append(p) if p in ckpt_dict and load_key in p[0] \
      else orig_vars.append(p),
    varibs) # just for logging which paths are loaded
  print("Loaded from ckpt: ")
  print("\n".join([str(x) for x in ckpt_vars]))
  return loaded


def cast_to_compute(values):
  return tree_map(lambda x: x.astype(COMPUTE_DTYPE), values)


def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False


def tensorstats(tensor, prefix=None):
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).max(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = jax.random.permutation(nj.rng(), values)[:amount]
  return values


def scan(fn, inputs, start, unroll=True, modify=False):
  fn2 = lambda carry, inp: (fn(carry, inp),) * 2
  if not unroll:
    return nj.scan(fn2, start, inputs, modify=modify)[1]
  length = len(jax.tree_util.tree_leaves(inputs)[0])
  carrydef = jax.tree_util.tree_structure(start)
  carry = start
  outs = []
  for index in range(length):
    carry, out = fn2(carry, tree_map(lambda x: x[index], inputs))
    flat, treedef = jax.tree_util.tree_flatten(out)
    assert treedef == carrydef, (treedef, carrydef)
    outs.append(flat)
  outs = [
      jnp.stack([carry[i] for carry in outs], 0)
      for i in range(len(outs[0]))]
  return carrydef.unflatten(outs)


def symlog(x):
  return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def switch(pred, lhs, rhs):
  assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
  while len(pred.shape) < len(lhs.shape):
    pred = pred[..., None]
  return jnp.where(pred, lhs, rhs)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=jnp.float32):
    super().__init__(logits, probs, dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    sample = sg(super().sample(sample_shape, seed))
    probs = self._pad(super().probs_parameter(), sample.shape)
    return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

  def _pad(self, tensor, shape):
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor


class MSEDist:

  def __init__(self, mode, dims, agg='sum'):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._agg = agg
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return self._mode

  def mean(self):
    return self._mode

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = ((self._mode - value) ** 2)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class SymlogDist:

  def __init__(self, mode, dims, dist='mse', agg='sum', tol=1e-8):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._dist = dist
    self._agg = agg
    self._tol = tol
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return symexp(self._mode)

  def mean(self):
    return symexp(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    if self._dist == 'mse':
      distance = (self._mode - symlog(value)) ** 2
      distance = jnp.where(distance < self._tol, 0, distance)
    elif self._dist == 'abs':
      distance = jnp.abs(self._mode - symlog(value))
      distance = jnp.where(distance < self._tol, 0, distance)
    else:
      raise NotImplementedError(self._dist)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class TwoHotDist:

  def __init__(self, logits, bins, dims=0, transfwd=None, transbwd=None):
    assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
    self.logits = logits
    self.probs = jax.nn.softmax(logits)
    self.dims = tuple([-x for x in range(1, dims + 1)])
    self.bins = jnp.array(bins)
    self.transfwd = transfwd or (lambda x: x)
    self.transbwd = transbwd or (lambda x: x)
    self.batch_shape = logits.shape[:len(logits.shape) - dims - 1]
    self.event_shape = logits.shape[len(logits.shape) - dims: -1]

  def mean(self):
    return self.transbwd((self.probs * self.bins).sum(-1))

  def mode(self):
    return self.transbwd((self.probs * self.bins).sum(-1))

  def log_prob(self, x):
    x = self.transfwd(x)
    below = (self.bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1
    above = len(self.bins) - (
        self.bins > x[..., None]).astype(jnp.int32).sum(-1)
    below = jnp.clip(below, 0, len(self.bins) - 1)
    above = jnp.clip(above, 0, len(self.bins) - 1)
    equal = (below == above)
    dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
    dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
        jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None] +
        jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None])
    log_pred = self.logits - jax.scipy.special.logsumexp(
        self.logits, -1, keepdims=True)
    return (target * log_pred).sum(-1).sum(self.dims)


def video_grid(video):
  B, T, H, W, C = video.shape
  return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  tgt = target.astype(jnp.float32)
  pos = (tgt > thres).astype(jnp.float32)
  neg = (tgt <= thres).astype(jnp.float32)
  fine_pos = (tgt >= 0.01).astype(jnp.float32)
  fine_zero = ((tgt > -0.01) & (tgt < 0.01)).astype(jnp.float32)
  fine_neg = (tgt <= -0.01).astype(jnp.float32)
  # 
  pred_mean = dist.mean().astype(jnp.float32)
  pred = (pred_mean > thres).astype(jnp.float32)
  pos_pred = (pred_mean >= 0.01).astype(jnp.float32)
  zero_pred = ((pred_mean > -0.01) & (pred_mean < 0.01)).astype(jnp.float32)
  neg_pred = (pred_mean <= -0.01).astype(jnp.float32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      fine_pos_acc=(pos_pred * fine_pos).sum() / fine_pos.sum(),
      fine_neg_acc=(neg_pred * fine_neg).sum() / fine_neg.sum(),
      fine_zero_acc=(zero_pred * fine_zero).sum() / fine_zero.sum(),
      fine_pos_cnt=fine_pos.sum(),
      fine_neg_cnt=fine_neg.sum(),
      rate=pos.mean(),
      avg=target.astype(jnp.float32).mean(),
      pred=dist.mean().astype(jnp.float32).mean(),
  )


class Moments(nj.Module):

  def __init__(
      self, impl='mean_std', decay=0.99, max=1e8, eps=0.0, perclo=5,
      perchi=95):
    self.impl = impl
    self.decay = decay
    self.max = max
    self.eps = eps
    self.perclo = perclo
    self.perchi = perchi
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.step = nj.Variable(jnp.zeros, (), jnp.int32, name='step')
      self.mean = nj.Variable(jnp.zeros, (), jnp.float32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), jnp.float32, name='sqrs')
    elif self.impl == 'min_max':
      self.low = nj.Variable(jnp.zeros, (), jnp.float32, name='low')
      self.high = nj.Variable(jnp.zeros, (), jnp.float32, name='high')
    elif self.impl == 'perc_ema':
      self.low = nj.Variable(jnp.zeros, (), jnp.float32, name='low')
      self.high = nj.Variable(jnp.zeros, (), jnp.float32, name='high')
    elif self.impl == 'perc_ema_corr':
      self.step = nj.Variable(jnp.zeros, (), jnp.int32, name='step')
      self.low = nj.Variable(jnp.zeros, (), jnp.float32, name='low')
      self.high = nj.Variable(jnp.zeros, (), jnp.float32, name='high')
    elif self.impl == 'mean_mag':
      self.mag = nj.Variable(jnp.zeros, (), jnp.float32, name='mag')
    elif self.impl == 'max_mag':
      self.mag = nj.Variable(jnp.zeros, (), jnp.float32, name='mag')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x):
    self.update(x)
    return self.stats()

  def update(self, x):
    if parallel():
      mean = lambda x: jax.lax.pmean(x.mean(), 'i')
      min_ = lambda x: jax.lax.pmin(x.min(), 'i')
      max_ = lambda x: jax.lax.pmax(x.max(), 'i')
      per = lambda x, q: jnp.percentile(jax.lax.all_gather(x, 'i'), q)
    else:
      mean = jnp.mean
      min_ = jnp.min
      max_ = jnp.max
      per = jnp.percentile
    x = sg(x.astype(jnp.float32))
    m = self.decay
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.step.write(self.step.read() + 1)
      self.mean.write(m * self.mean.read() + (1 - m) * mean(x))
      self.sqrs.write(m * self.sqrs.read() + (1 - m) * mean(x * x))
    elif self.impl == 'min_max':
      low, high = min_(x), max_(x)
      self.low.write(m * jnp.minimum(self.low.read(), low) + (1 - m) * low)
      self.high.write(m * jnp.maximum(self.high.read(), high) + (1 - m) * high)
    elif self.impl == 'perc_ema':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write(m * self.low.read() + (1 - m) * low)
      self.high.write(m * self.high.read() + (1 - m) * high)
    elif self.impl == 'perc_ema_corr':
      self.step.write(self.step.read() + 1)
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write(m * self.low.read() + (1 - m) * low)
      self.high.write(m * self.high.read() + (1 - m) * high)
    elif self.impl == 'mean_mag':
      curr = mean(jnp.abs(x))
      self.mag.write(m * self.mag.read() + (1 - m) * curr)
    elif self.impl == 'max_mag':
      curr = max_(jnp.abs(x))
      self.mag.write(m * jnp.maximum(self.mag.read(), curr) + (1 - m) * curr)
    else:
      raise NotImplementedError(self.impl)

  def stats(self):
    if self.impl == 'off':
      return 0.0, 1.0
    elif self.impl == 'mean_std':
      corr = 1 - self.decay ** self.step.read().astype(jnp.float32)
      mean = self.mean.read() / corr
      var = (self.sqrs.read() / corr) - self.mean.read() ** 2
      std = jnp.sqrt(jnp.maximum(var, 1 / self.max ** 2) + self.eps)
      return sg(mean), sg(std)
    elif self.impl == 'min_max':
      offset = self.low.read()
      invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
      return sg(offset), sg(invscale)
    elif self.impl == 'perc_ema':
      offset = self.low.read()
      invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
      return sg(offset), sg(invscale)
    elif self.impl == 'perc_ema_corr':
      corr = 1 - self.decay ** self.step.read().astype(jnp.float32)
      lo = self.low.read() / corr
      hi = self.high.read() / corr
      invscale = jnp.maximum(1 / self.max, hi - lo)
      return sg(lo), sg(invscale)
    elif self.impl == 'mean_mag':
      offset = jnp.array(0)
      invscale = jnp.maximum(1 / self.max, self.mag.read())
      return sg(offset), sg(invscale)
    elif self.impl == 'max_mag':
      offset = jnp.array(0)
      invscale = jnp.maximum(1 / self.max, self.mag.read())
      return sg(offset), sg(invscale)
    else:
      raise NotImplementedError(self.impl)


class Optimizer(nj.Module):


  def __init__(self, modules, opt, summary_depth=2):
    modules = modules if isinstance(modules, (list, tuple)) else (modules,)
    self.modules = modules
    self.opt = opt
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (COMPUTE_DTYPE == jnp.float16)
    self.summary_depth = summary_depth
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

  def __call__(self, lossfn, *args, has_aux=False, **kwargs):
    metrics = {}

    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux

    loss, params, grads, aux = nj.grad(
        lossfn2, self.modules, has_aux=True)(*args, **kwargs)
    if self.scaling:
      loss *= 1 / self.grad_scale.read()

    counts = {k: math.prod(v.shape) for k, v in params.items()}
    if nj.creating():
      print(self._summarize_params(counts, self.summary_depth))

    axes = get_data_axes()
    if axes:
      grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

    if self.scaling:
      invscale = 1 / self.grad_scale.read()
      grads = jax.tree.map(lambda x: x * invscale, grads)

    state = self.sub('state', nj.Tree, self.opt.init, params)
    updates, new_state = self.opt.update(grads, state.read(), params)
    nj.context().update(optax.apply_updates(params, updates))
    state.write(new_state)
    grad_norm = optax.global_norm(grads)
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
    else:
      self.step.write(self.step.read() + 1)
    metrics['loss'] = loss.mean()
    metrics['updates'] = self.step.read()
    metrics['grad_norm'] = grad_norm
    metrics['grad_rms'] = rms(grads)
    metrics['update_rms'] = rms(updates)
    metrics['param_rms'] = rms([x.values for x in self.modules])
    metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
    metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        f32(keep) * self.grad_scale.read() +
        f32(incr) * self.grad_scale.read() * 2 +
        f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
    return finite

  def _summarize_params(self, counts, depth):
    lines = []
    pfxs = []
    for key in counts:
      parts = key.split('/')
      pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
    subcounts = {
        prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
        for prefix in set(pfxs)}
    lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
    for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
      lines.append(f'{count:>14,} {prefix}')
    return '\n'.join(lines)


def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)



def late_grad_clip(value=1.0):
  def init_fn(params):
    return ()
  def update_fn(updates, state, params):
    updates = tree_map(lambda x: jnp.clip(x, -value, value), updates)
    return updates, ()
  return optax.GradientTransformation(init_fn, update_fn)


def tree_keys(params, prefix=''):
  if hasattr(params, 'items'):
    return type(params)({
        k: tree_keys(v, prefix + '/' + k.lstrip('/'))
        for k, v in params.items()})
  elif isinstance(params, (tuple, list)):
    return [tree_keys(x, prefix) for x in params]
  elif isinstance(params, jnp.ndarray):
    return prefix
  else:
    raise TypeError(type(params))


class SlowUpdater:

  def __init__(self, src, dst, fraction=1.0, period=1):
    self.src = src
    self.dst = dst
    self.fraction = fraction
    self.period = period
    self.updates = nj.Variable(jnp.zeros, (), jnp.int32, name='updates')

  def _getm(self, module):
    prefix = module.path + '/'
    ctx = nj.context()
    return {k: v for k, v in ctx.items() if k.startswith(prefix)}

  def _putm(self, module, mapping):
    nj.context().update(mapping)

  def __call__(self):
    assert self._getm(self.src)
    updates = self.updates.read()
    need_init = (updates == 0).astype(jnp.float32)
    need_update = (updates % self.period == 0).astype(jnp.float32)
    mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
    source = {
        k.replace(f'/{self.src.name}/', f'/{self.dst.name}/'): v
        for k, v in self._getm(self.src).items()}
    self._putm(self.dst, tree_map(
        lambda s, d: mix * s + (1 - mix) * d,
        source, self._getm(self.dst)))
    self.updates.write(updates + 1)
