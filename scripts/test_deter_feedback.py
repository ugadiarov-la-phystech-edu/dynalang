"""Smoke test for the deter feedback channel in the TSSM/OCTSSM dynamics.

Covers, on a tiny CPU agent:
  1. construction, world model loss, report, and a real optimizer step;
  2. that the step projection widens by exactly `deter` when the channel is on,
     and is untouched when it is off, so old checkpoints still load;
  3. forward: perturbing the start deter changes the imagined trajectory when
     the channel is on and changes nothing at all when it is off;
  4. gradients: 'detached' cuts the step-to-step chain while 'grad' keeps it.

Run:  python scripts/test_deter_feedback.py
"""
import pathlib
import sys

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))
sys.path.append(str(directory / 'dynalang'))

import embodied  # noqa: E402

BATCH, LENGTH, SLOTS, SLOT_DIM, ACTIONS, IMAGE = 4, 8, 3, 8, 5, 16
HORIZON = 4
VOCAB = 16
SEED = np.asarray(7, np.uint32)


def build_agent(rssm_type, deter_feedback, action_mode=None, text=False,
                obs_key=None, **overrides):
  from dynalang import agent as agt
  # Only the object-centric dynamics keeps the slot axis, so every other one
  # defaults to an image; pass obs_key to feed it flattened slots instead.
  obs_key = obs_key or ('slot' if rssm_type == 'octssm' else 'image')
  raw = agt.Agent.configs
  config = embodied.Config(raw['defaults'])
  config = config.update(raw['debug'])
  if rssm_type == 'octssm':
    config = config.update(raw['octssm'])
  # With text the language embedding becomes an extra slot, so num_slots is
  # derived as objects + 1 instead of being taken from the config.
  mlp_keys = 'token$' if text else '$^'
  # Only the transformer dynamics have the channel; the GRU carries deter by
  # construction and would pass the argument on to a Linear layer.
  dynamics = {} if deter_feedback is None else {
      'deter_feedback': deter_feedback}
  if action_mode is not None:
    dynamics['action_mode'] = action_mode
  config = config.update({
      'rssm_type': rssm_type,
      'batch_size': BATCH,
      'batch_length': LENGTH,
      'imag_horizon': HORIZON,
      'logdir': '/tmp/deter_feedback_test',
      'use_slot_extractor': False,
      'encoder': {'mlp_keys': mlp_keys, 'cnn_keys': 'image$'},
      'decoder': {
          'mlp_keys': mlp_keys, 'cnn_keys': 'image$',
          'vector_dist': 'onehot' if text else 'symlog_mse'},
      'jax': {
          'platform': 'cpu', 'precision': 'float32', 'debug': False,
          'transfer_guard': False, 'prealloc': False,
      },
      'run': {'from_checkpoint': '', 'pretrain_wm_only': False},
      'loggers': ['terminal'],
      rssm_type: dynamics,
      **overrides,
  })
  if obs_key == 'slot':
    obs_space = {'slot': embodied.Space(np.float32, (SLOTS, SLOT_DIM))}
  else:
    obs_space = {'image': embodied.Space(np.uint8, (IMAGE, IMAGE, 3), 0, 255)}
  if text:
    obs_space['token'] = embodied.Space(np.int32, (), 0, VOCAB)
  obs_space |= {
      'reward': embodied.Space(np.float32),
      'is_first': embodied.Space(bool),
      'is_last': embodied.Space(bool),
      'is_terminal': embodied.Space(bool),
  }
  act_space = {
      'action': embodied.Space(np.int32, (), 0, ACTIONS),
      'reset': embodied.Space(bool),
  }
  agent = agt.Agent(obs_space, act_space, embodied.Counter(), config)
  return agent, obs_key, config[rssm_type]['deter']


def make_batch(obs_key, seed=0, text=False):
  rng = np.random.RandomState(seed)
  is_first = np.zeros((BATCH, LENGTH), bool)
  is_first[:, 0] = True
  if obs_key == 'slot':
    obs = rng.randn(BATCH, LENGTH, SLOTS, SLOT_DIM).astype(np.float32)
  else:
    obs = rng.randint(
        0, 256, (BATCH, LENGTH, IMAGE, IMAGE, 3)).astype(np.uint8)
  batch = {
      obs_key: obs,
      'reward': rng.randn(BATCH, LENGTH).astype(np.float32),
      'is_first': is_first,
      'is_last': np.zeros((BATCH, LENGTH), bool),
      'is_terminal': np.zeros((BATCH, LENGTH), bool),
      'action': rng.randint(0, ACTIONS, (BATCH, LENGTH)).astype(np.int32),
      'reset': np.zeros((BATCH, LENGTH), bool),
  }
  if text:
    batch['token'] = rng.randint(0, VOCAB, (BATCH, LENGTH)).astype(np.int32)
  return batch


def projection_width(agent):
  """Input width of the per-step projection that feeds the context window."""
  names = [k for k in agent.varibs if k.endswith('rssm/projection_layer/kernel')]
  assert len(names) == 1, names
  return int(np.asarray(agent.varibs[names[0]]).shape[0])


def check_loss_and_report(agent, obs_key, text=False):
  from dynalang import ninjax as nj
  inner = agent.agent

  def loss_fn(data):
    data = inner.preprocess(data)
    loss, (_, _, metrics) = inner.wm.loss(data, inner.wm.initial(BATCH))
    return loss, metrics

  data = make_batch(obs_key, text=text)
  (loss, metrics), _ = nj.pure(loss_fn)(agent.varibs, SEED, data)
  loss = float(np.asarray(loss))
  assert np.isfinite(loss), f'non-finite world model loss: {loss}'
  for key in ('reward_loss_mean', 'dyn_loss_mean', 'model_loss_mean'):
    value = float(np.asarray(metrics[key]))
    assert np.isfinite(value), (key, value)
  report, _ = nj.pure(inner.report)(agent.varibs, SEED, data)
  assert report, 'empty report'
  print(f'  loss={loss:.4f}, report keys={len(report)}')


def check_train_step(agent, obs_key, text=False):
  """A real optimizer step, exercising imagination and the actor-critic."""
  state = None
  metrics = {}
  for step in range(2):
    data = make_batch(obs_key, seed=step, text=text)
    data['rng'] = np.asarray(SEED, np.uint32)
    _, state, mets = agent.train(data, state)
    metrics = mets or metrics
  keys = [k for k in ('extr_return_raw_mean', 'extr_reward_mean') if k in metrics]
  assert keys, f'no actor-critic metrics, got {sorted(metrics)[:10]}'
  for key in keys:
    value = float(np.asarray(metrics[key]))
    assert np.isfinite(value), (key, value)
  shown = ', '.join(f'{k}={float(np.asarray(metrics[k])):.4f}' for k in keys)
  print(f'  train step ok, {shown}')


def _rollout(inner):
  """Pure rollout returning (scalar, trajectory) as a function of start deter."""
  def rollout(deter, data):
    data = inner.preprocess(data)
    rssm = inner.wm.rssm
    start = {**rssm.initial(BATCH), 'deter': deter}
    actions = {k: data[k][:, :HORIZON] for k in inner.wm.act_space}
    traj = rssm.imagine(actions, start)
    # Squared, because the transformer ends in a layer norm: a plain sum over
    # zero-mean features is identically zero and carries no gradient.
    return (traj['deter'] ** 2).sum(), traj['deter']
  return rollout


def check_deter_channel(agent, obs_key, feedback, text=False):
  """Forward sensitivity and gradient flow through the carried deter."""
  import jax
  from dynalang import ninjax as nj
  inner = agent.agent

  start_deter, _ = nj.pure(
      lambda: inner.wm.rssm.initial(BATCH)['deter'])(agent.varibs, SEED)
  base = np.asarray(start_deter, np.float32)
  bumped = base + 1.0
  data = make_batch(obs_key, text=text)

  fn = nj.pure(_rollout(inner))
  (_, traj_a), _ = fn(agent.varibs, SEED, base, data, create=False)
  (_, traj_b), _ = fn(agent.varibs, SEED, bumped, data, create=False)
  forward = float(np.abs(np.asarray(traj_a) - np.asarray(traj_b)).max())

  scalar = lambda d: fn(agent.varibs, SEED, d, data, create=False)[0][0]
  grads = np.asarray(jax.grad(scalar)(base))
  grad = float(np.abs(grads).max())

  print(f'  start deter: forward delta={forward:.6f}, max |grad|={grad:.6f}')
  if feedback == 'none':
    assert forward == 0.0, (
        f'deter_feedback=none still let the start deter reach the rollout '
        f'(delta {forward})')
    assert grad == 0.0, f'deter_feedback=none has a gradient path ({grad})'
  else:
    assert forward > 1e-6, (
        f'deter_feedback={feedback} never reached the rollout; the channel is '
        'not wired into the dynamics')
  if feedback == 'detached':
    assert grad == 0.0, (
        f'deter_feedback=detached leaks gradients ({grad}); sg() is missing')
  if feedback == 'grad':
    assert grad > 0.0, (
        'deter_feedback=grad has no gradient path; sg() applied by mistake')


VARIANTS = {
    'octssm': ('octssm', {}),
    'tssm': ('tssm', {}),
    # The configuration actually used for the homegrid runs: the action is an
    # extra slot and language becomes one more slot on top of the objects.
    'octssm_slot_text': ('octssm', {'action_mode': 'slot', 'text': True}),
}


def check_num_slots(agent, text):
  slots = agent.agent.wm.rssm._num_slots
  expected = SLOTS + 1 if text else SLOTS
  assert slots == expected, (slots, expected)
  print(f'  num_slots: {slots}' + (' (objects + text)' if text else ''))


def main(names=tuple(VARIANTS)):
  for name in names:
    rssm_type, kwargs = VARIANTS[name]
    text = kwargs.get('text', False)
    widths = {}
    deter = None
    for feedback in ('none', 'detached', 'grad'):
      print(f'== {name} deter_feedback={feedback}')
      agent, obs_key, deter = build_agent(rssm_type, feedback, **kwargs)
      widths[feedback] = projection_width(agent)
      print(f'  projection input width: {widths[feedback]}')
      if rssm_type == 'octssm':
        check_num_slots(agent, text)
      check_loss_and_report(agent, obs_key, text)
      check_deter_channel(agent, obs_key, feedback, text)
      check_train_step(agent, obs_key, text)
    for feedback in ('detached', 'grad'):
      assert widths[feedback] == widths['none'] + deter, (
          f'{name} {feedback}: projection input is {widths[feedback]}, '
          f'expected {widths["none"]} + {deter}')
    print(f'== {name}: projection widens by deter={deter} exactly '
          'when the channel is on')
  print('all ok')


if __name__ == '__main__':
  main(tuple(sys.argv[1:]) or tuple(VARIANTS))
