import sys

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj
import optax


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = {
        k: v for k, v in act_space.items() if k != 'reset'}
    self.step = step
    with jax.transfer_guard("allow"):
      dummy_preproc = self.preprocess(
        {k: jnp.ones(v.shape) for k, v in self.obs_space.items()}) 
      preproc_shapes = {k: tuple(v.shape) for k, v in dummy_preproc.items() \
                        if not k.startswith("log_")}
    self.wm = WorldModel(obs_space, self.act_space, config, preproc_shapes, name='wm')
    self.preprocessors = {k: v() for k, v in
                          self.wm.encoder.preprocessors.items()}
    if self.config.run.pretrain_wm_only:
        print("Agent: Pretraining WM only.")
        return
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    if self.config.rssm_type == "token":
      latent = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed, obs["token"], obs['is_first'])
    else:
      latent = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed, obs['is_first'],
          training=(mode != 'eval'))
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    outs = {'eval': task_outs, 'explore': expl_outs, 'train': task_outs}[mode]
    act = {k: v for k, v in outs.items() if k in self.act_space}
    state = ((latent, act), task_state, expl_state)
    act = {
        k: jnp.argmax(act[k], -1).astype(jnp.int32) if s.discrete else act[k]
        for k, s in self.act_space.items()}
    return act, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    if self.config.run.pretrain_wm_only:
        return wm_outs, state, metrics
    # Flatten (batch, seq) -> (batch * seq)
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def train_wm(self, data, state):
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    return wm_outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    if self.config.run.pretrain_wm_only:
      return report
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def vis(self, data, num_obs, num_imagine):
    data = self.preprocess(data)
    return self.wm.vis(data, num_obs, num_imagine)

  def save(self):
    data = jax.tree_util.tree_flatten(jax.tree_util.tree_map(
        jnp.asarray, self.state))[0]
    data = [np.asarray(x) for x in data]
    return data

  def load(self, state):
    self.state = jax.tree_util.tree_flatten(self.state)[1].unflatten(state)

  def preprocess(self, obs):
    spaces = {**self.obs_space, **self.act_space}
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key', 'reset'):
        continue
      space = spaces.get(key)
      if key in self.act_space and space.discrete:
        value = jax.nn.one_hot(value, int(space.high))
      elif key in ("token", "text"):
        value = jax.nn.one_hot(value, self.obs_space[key].high)
      elif len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config, shapes):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
#    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
#    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    if self.config.rssm_type == 'rssm':
      self.rssm = nets.RSSM(**config.rssm, name='rssm')
    elif self.config.rssm_type == 'early':
      self.rssm = nets.EarlyRSSM(**config.early_rssm, name='rssm')
    elif self.config.rssm_type == 'token':
      self.rssm = nets.TokenRSSM(**config.token_rssm, name='rssm')
    elif self.config.rssm_type == 'tssm':
      self.rssm = nets.TSSM(**config.tssm, name='rssm')
    elif self.config.rssm_type == 'octssm':
      octssm_cfg = dict(config.octssm)
      if getattr(self.encoder, 'slot_based', False):
        octssm_cfg['num_slots'] = self.encoder.n_output_slots
        print(f'WorldModel: auto-set octssm.num_slots = '
              f'{self.encoder.n_object_slots} object slot(s) + '
              f'{self.encoder.n_image_slots} image slot(s) + '
              f'{self.encoder.n_text_slots} text slot(s) = '
              f'{octssm_cfg["num_slots"]}')
      elif self.encoder.slot_shapes:
        base_slots = list(self.encoder.slot_shapes.values())[0][0]
        n_text = 1 if len(self.encoder.mlp_shapes) > 0 else 0
        octssm_cfg['num_slots'] = base_slots + n_text
        print(f'WorldModel: auto-set octssm.num_slots = '
              f'{base_slots} object slots + {n_text} text slot(s) = '
              f'{octssm_cfg["num_slots"]}')
      self.rssm = nets.ObjectCentricTSSM(**octssm_cfg, name='rssm')
    else:
      raise NotImplementedError(self.config.rssm_type)
    
    head_dims = 4 if self.config.rssm_type == 'octssm' else 'deter'
    head_bdims = 2 if self.config.rssm_type == 'octssm' else None
 
    if self.config.reward_head.typ == 'mlp':
      mlp_dims = 3 if self.config.rssm_type == 'octssm' else 'deter'
      reward_head = nets.MLP((), dims=mlp_dims, **self.config.reward_head.mlp, name='rew')
    elif self.config.reward_head.typ == 'transformer':
      from .embodied.core.space import Space
      reward_space = Space(np.float32, ())
      reward_head = nets.AggregationTransformerHead(
          reward_space, inputs=['deter', 'stoch'], dims=head_dims, bdims=head_bdims,
          **self.config.reward_head.transformer, name='rew')
    else:
      raise NotImplementedError(f'reward_head.typ: {self.config.reward_head.typ}')
    
    if self.config.cont_head.typ == 'mlp':
      mlp_dims = 3 if self.config.rssm_type == 'octssm' else 'deter'
      cont_head = nets.MLP((), dims=mlp_dims, **self.config.cont_head.mlp, name='cont')
    elif self.config.cont_head.typ == 'transformer':
      from .embodied.core.space import Space
      cont_space = Space(np.float32, ())
      cont_head = nets.AggregationTransformerHead(
          cont_space, inputs=['deter', 'stoch'], dims=head_dims, bdims=head_bdims,
          **self.config.cont_head.transformer, name='cont')
    else:
      raise NotImplementedError(f'cont_head.typ: {self.config.cont_head.typ}')
    
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': reward_head,
        'cont': cont_head}
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    scales.update({k: vector for k in self.heads['decoder'].slot_shapes})
    scales.update({k: image for k in self.heads['decoder'].slot_image_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = {
        k: jnp.zeros(
            (batch_size, *v.shape, int(v.high)) if v.discrete else (batch_size, *v.shape))
        for k, v in self.act_space.items()}
    return prev_latent, prev_action

  def train(self, data, state):
    for key in [x for x in self.config.zero_data_keys if x]:
      data[key] = jnp.zeros_like(data[key])
    modules = [self.encoder, self.rssm, *self.heads.values()]

    if self.config.skip_mlp_training:
      assert not self.config.skip_cnn_training
      enc = self.encoder._cnn
      dec = self.heads['decoder']._cnn
      others = {k: v for k, v in self.heads.items() if k != 'decoder'}
      modules = [self.rssm, enc, dec, *others.values()]
    if self.config.skip_cnn_training:
      assert not self.config.skip_mlp_training
      enc = self.encoder._mlp
      dec = self.heads['decoder']._mlp
      others = {k: v for k, v in self.heads.items() if k != 'decoder'}
      modules = [self.rssm, enc, dec, *others.values()]

    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(
      data,
      zero_mlp=self.config.zero_mlp,
      zero_cnn=self.config.zero_cnn)

    prev_latent, prev_action = state
    prev_actions = {
        k: jnp.concatenate([prev_action[k][:, None], data[k][:, :-1]], 1)
        for k in self.act_space}
    if self.config.rssm_type == "token":
      post = self.rssm.observe(
          prev_actions, embed, data["token"], data['is_first'], prev_latent)
    else:
      post = self.rssm.observe(
          embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      inp = feats if name in self.config.grad_heads else sg(feats)
      out = head(inp)
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    if self.config.rssm_type == "early":
      rssm_losses, prior = self.rssm.loss(post, prev_latent, prev_actions, **self.config.rssm_loss)
    elif self.config.rssm_type == "token":
      rssm_losses, prior = self.rssm.loss(
        post,
        prev_latent,
        prev_actions,
        data["token"],
        **self.config.rssm_loss
      )
    else:
      rssm_losses, prior = self.rssm.loss(post, **self.config.rssm_loss)

    # LM loss
    if self.scales["lm"] > 0:
      print("Adding LM loss")
      next_ac = {k: data[k][:, :-1].reshape((-1, 1, *data[k].shape[2:]))
                 for k in self.act_space}
      context = {k: v[:, :-1].reshape((-1, *v.shape[2:]))
                 for k, v in post.items()}
      one_step_openl = self.heads["decoder"](
        self.rssm.imagine(next_ac, context),
      )
      truth = data["token"][:, 1:].reshape((-1, 1, *data["token"].shape[2:]))
      nll = -(one_step_openl["token"].log_prob(truth)).mean(-1)
#      nll = nll.reshape((data["token"].shape[0], -1)) # (batch, seq - 1)
      lm_loss = (nll * self.scales["lm"]).mean()
    else:
      lm_loss = 0

    losses.update(rssm_losses)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = {k: data[k][:, -1] for k in self.act_space}
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean() + lm_loss, (state, out, metrics)

  def imagine(self, policy, start, horizon, carry=None):
    if carry is None:
      policy = lambda s, c, f=policy: (f(s), {})
      carry = {}
    state_keys = list(self.rssm.initial(1).keys())
    state = {k: v for k, v in start.items() if k in state_keys}
    action, carry = policy(state, carry)
    keys = list(state.keys()) + list(action.keys()) + list(carry.keys())
    assert len(set(keys)) == len(keys), ('Colliding keys', keys)
    def step(prev, _):
      state, action, carry = prev
      state = self.rssm.img_step(state, action)
      action, carry = policy(state, carry)
      return state, action, carry
    states, actions, carries = jaxutils.scan(
        step, jnp.arange(horizon), (state, action, carry),
        self.config.imag_unroll)
    states, actions, carries = tree_map(
        lambda traj, first: jnp.concatenate([first[None], traj], 0),
        (states, actions, carries), (state, action, carry))
    traj = {**states, **actions, **carries}
    if self.config.imag_cont == 'mode':
      cont = self.heads['cont'](traj).mode()
    elif self.config.imag_cont == 'mean':
      cont = self.heads['cont'](traj).mean()
    else:
      raise NotImplementedError(self.config.imag_cont)
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    # data: dict, each val with shape (batch, length, <obs shape>)
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    
    embed = self.encoder(data)
    act_keys = list(self.act_space.keys())
    if self.config.rssm_type == "token":
      context = self.rssm.observe(
          {k: data[k][:6, :5] for k in act_keys},
          embed[:6, :5],
          data['token'][:6, :5],
          data['is_first'][:6, :5])
    else:
      context = self.rssm.observe(
          embed[:6, :5], 
          {k: data[k][:6, :5] for k in act_keys},
          data['is_first'][:6, :5])
    # context:
    # - deter (batch, prefix_len, rssm.deter)
    # - logit, stoch (batch, prefix_len, rssm.stoch, rssm.classes)
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine({k: data[k][:6, 5:] for k in act_keys}, start),
    )
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)

    for key in self.heads['decoder'].slot_image_shapes.keys():
      n_show = min(2, data[key].shape[0])
      truth = data[key][:n_show].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      model = jnp.clip(model[:n_show].astype(jnp.float32), 0.0, 1.0)
      error = jnp.clip((model - truth + 1) / 2, 0.0, 1.0)

      def _slots_to_row(x):
        n, t, s, h, w, c = x.shape
        return x.transpose((0, 1, 3, 2, 4, 5)).reshape((n, t, h, s * w, c))

      video = jnp.concatenate(
          [_slots_to_row(truth), _slots_to_row(model), _slots_to_row(error)], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)

      if 'image' in data and data['image'].shape[-3:-1] == truth.shape[-3:-1]:
        s = truth.shape[2]
        import colorsys
        palette = jnp.asarray(np.array(
            [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 1.0)
             for i in range(s)], dtype=np.float32))
        bg = data['image'][:n_show].astype(jnp.float32)
        if bg.shape[-1] == 1:
          bg = jnp.repeat(bg, 3, axis=-1)

        def _overlay(slots, alpha=0.5):
          mask = (slots.sum(-1, keepdims=True) > 1e-3).astype(jnp.float32)
          colors = palette.reshape((1, 1, s, 1, 1, 3))
          color_sum = (mask * colors).sum(2)
          count = mask.sum(2)
          avg = color_sum / jnp.maximum(count, 1.0)
          any_mask = (count > 0).astype(jnp.float32)
          return bg * (1 - alpha * any_mask) + alpha * any_mask * avg

        overlay = jnp.concatenate([_overlay(truth), _overlay(model)], 2)
        report[f'openl_{key}_overlay'] = jaxutils.video_grid(overlay)
    # 1 step prediction loss for text
    # Calculate text ppl over entire batch and seq len for more context
    # Above is buggy - observe takes in prev_actions, ac taken into this state (see loss)
    # data["action"] is action we took out of this state
    # prev_actions is action we took into this state
    if self.config.run.pretrain_wm_only and "token" in data: 
      prev_actions = {
          k: jnp.concatenate([
              jnp.zeros_like(data[k][:, 0:1]),
              data[k][:, :-1]], 1)
          for k in act_keys}
      embed = self.encoder(data)
      context = self.rssm.observe(
          embed, prev_actions, data["is_first"])
      # a_t is action out of o_t, cut off last timestep since we don't have truth
      context = {k: v[:, :-1].reshape((-1, *v.shape[2:])) for k, v in context.items()}
      next_ac = {k: data[k][:, :-1].reshape((-1, 1, *data[k].shape[2:]))
                 for k in act_keys}
      one_step_openl = self.heads["decoder"](
        self.rssm.imagine(next_ac, context),
      )
      truth = data["token"][:, 1:].reshape((-1, 1, *data["token"].shape[2:]))
      nll = -(one_step_openl["token"].log_prob(truth)).mean(-1)
      ppl = jnp.exp(nll.mean())
      report["token_lm_nll_min"] = nll.min()
      report["token_lm_nll_max"] = nll.max()
      report["token_lm_nll_mean"] = nll.mean()
      report["token_lm_ppl"] = ppl
      # n-step openl text generation
      report["1step_openl_text"] = one_step_openl["token"].sample(seed=nj.rng()).argmax(-1) 
      report["nstep_openl_text"] = openl["token"].sample(seed=nj.rng()).argmax(-1)
    return report

  def vis(self, data, num_obs, num_imagine):
    act_keys = list(self.act_space.keys())
    state = self.initial(len(data["is_first"]))
    prev_actions = {
        k: jnp.concatenate([
            jnp.zeros_like(data[k][:, 0:1]),
            data[k][:, :-1]], 1)
        for k in act_keys}
    embed = self.encoder(data)

    context = self.rssm.observe(
      embed=embed[:, :num_obs],
      action={k: v[:, :num_obs] for k, v in prev_actions.items()},
      is_first=data["is_first"][:, :num_obs])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    end = num_obs + num_imagine
    openl = self.heads['decoder'](
      self.rssm.imagine(
          {k: data[k][:, num_obs-1:end] for k in act_keys}, start),
    )
    reward = self.heads['reward'](
      self.rssm.imagine(
          {k: data[k][:, num_obs-1:end] for k in act_keys}, start),
    )
    return recon, openl, reward
  
  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    if len(act_space) == 1 and list(act_space.values())[0].discrete:
      self.grad = config.actor_grad_disc
    elif len(act_space) == 1:
      self.grad = config.actor_grad_cont
    else:
      self.grad = 'reinforce'
    dist1, dist2 = config.actor_dist_disc, config.actor_dist_cont
    shapes = {
        k: (*s.shape, int(s.high)) if s.discrete else s.shape
        for k, s in act_space.items()}
    dists = {k: dist1 if v.discrete else dist2 for k, v in act_space.items()}
    actor_dims = 4 if config.rssm_type == 'octssm' else 'deter'
    actor_bdims = None  # Auto-detect based on input shape

    if config.actor.typ == 'mlp':
      mlp_dims = 3 if config.rssm_type == 'octssm' else 'deter'
      self.actor = nets.MLP(
          name='actor', dims=mlp_dims, shape=shapes, **config.actor.mlp,
          dist=dists)
    elif config.actor.typ == 'transformer':
      self.actor = nets.AggregationTransformerHead(
          shapes, dists, inputs=['deter', 'stoch'], dims=actor_dims,
          bdims=actor_bdims, **config.actor.transformer, name='actor')
    else:
      raise NotImplementedError(f'actor.typ: {config.actor.typ}')
    
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry, sample=True):
    dist = self.actor(sg(state))
    if isinstance(dist, dict):
      action = {k: v.sample(seed=nj.rng()) if sample else v.mode() for k, v in dist.items()}
    else:
      action = {'action': dist.sample(seed=nj.rng()) if sample else dist.mode()}
    return action, carry

  def train(self, imagine, start, context):
    carry = self.initial(len(start['deter']))
    def loss(start):
      traj = imagine(self.policy, start, self.config.imag_horizon, carry)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    if isinstance(policy, dict):
      logpi = {k: v.log_prob(sg(traj[k]))[:-1] for k, v in policy.items()}
      loss = {
          'backprop': -adv,
          'reinforce': -sum(logpi.values()) * sg(adv),
      }[self.grad]
      ent = {k: v.entropy()[:-1] for k, v in policy.items()}
      actent = self.config.actent
      if isinstance(actent, (int, float)):
        loss -= actent * sum(ent.values())
      else:
        loss -= sum(actent[k] * v for k, v in ent.items())
    else:
      logpi = policy.log_prob(sg(traj['action']))[:-1]
      loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
      ent = policy.entropy()[:-1]
      loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    if isinstance(policy, dict):
      for key, space in self.act_space.items():
        act = jnp.argmax(traj[key], -1) if space.discrete else traj[key]
        metrics.update(jaxutils.tensorstats(act.astype(jnp.float32), f'{key}_action'))
        rand = (ent[key] - policy[key].minent) / (
            policy[key].maxent - policy[key].minent)
        rand = rand.mean(range(2, len(rand.shape)))
        metrics.update(jaxutils.tensorstats(rand, f'{key}_policy_randomness'))
        metrics.update(jaxutils.tensorstats(ent[key], f'{key}_policy_entropy'))
        metrics.update(jaxutils.tensorstats(logpi[key], f'{key}_policy_logprob'))
    else:
      act = traj['action']
      act = jnp.argmax(act, -1) if list(self.act_space.values())[0].discrete else act
      metrics.update(jaxutils.tensorstats(act.astype(jnp.float32), 'action'))
      rand = (ent - policy.minent) / (policy.maxent - policy.minent)
      rand = rand.mean(range(2, len(rand.shape)))
      metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
      metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
      metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    critic_dims = 4 if config.rssm_type == 'octssm' else 'deter'
    critic_bdims = None  # Auto-detect based on input shape

    if config.critic.typ == 'mlp':
      mlp_dims = 3 if config.rssm_type == 'octssm' else 'deter'
      self.net = nets.MLP((), name='net', dims=mlp_dims, **config.critic.mlp)
      self.slow = nets.MLP((), name='slow', dims=mlp_dims, **config.critic.mlp)
    elif config.critic.typ == 'transformer':
      from .embodied.core.space import Space
      critic_space = Space(np.float32, ())
      self.net = nets.AggregationTransformerHead(
          critic_space, inputs=['deter', 'stoch'], dims=critic_dims, bdims=critic_bdims,
          **config.critic.transformer, name='net')
      self.slow = nets.AggregationTransformerHead(
          critic_space, inputs=['deter', 'stoch'], dims=critic_dims, bdims=critic_bdims,
          **config.critic.transformer, name='slow')
    else:
      raise NotImplementedError(f'critic.typ: {config.critic.typ}')
    
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj, slow=self.config.slow_critic_target)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None, slow=False):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['cont']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    if slow:
      value = self.slow(traj).mean()
    else:
      value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]
