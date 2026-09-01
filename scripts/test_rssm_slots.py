"""Smoke test for the GRU dynamics reading slot observations.

The encoder returns one vector per slot, which only the object-centric
dynamics can consume. For every other dynamics the slot axis is folded into
features and the decoder reads all the slots back out of the flat latent; this
checks that both ends agree and that nothing about the slot-shaped path moved.

Covers, on a tiny CPU agent:
  1. the embedding the GRU sees is flat and as wide as slots x slot_dim;
  2. the decoder still emits slot-shaped means, so the SlotContrast decoder in
     `embodied/run/slot_viz.py` keeps working, and still emits tokens;
  3. world model loss, report, and a real optimizer step with the actor-critic;
  4. octssm is untouched: it keeps the slot axis and its per-slot projector.

Run:  python scripts/test_rssm_slots.py
"""
import pathlib
import sys

import numpy as np

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))
sys.path.append(str(directory / 'dynalang'))

from test_deter_feedback import (  # noqa: E402
    BATCH, LENGTH, SLOTS, SLOT_DIM, SEED, build_agent, check_loss_and_report,
    check_train_step, make_batch)


def run_pure(agent, fn, *args):
  from dynalang import ninjax as nj
  out, _ = nj.pure(fn)(agent.varibs, SEED, *args)
  return out


def embed_shape(agent, data):
  inner = agent.agent

  def fn(data):
    return inner.wm.encoder(inner.preprocess(data))

  return tuple(np.asarray(run_pure(agent, fn, data)).shape)


def decoded_shapes(agent, data):
  inner = agent.agent

  def fn(data):
    data = inner.preprocess(data)
    embed = inner.wm.encoder(data)
    prev_actions = {
        k: np.concatenate([np.zeros_like(data[k][:, :1]), data[k][:, :-1]], 1)
        for k in inner.wm.act_space}
    post = inner.wm.rssm.observe(embed, prev_actions, data['is_first'])
    return {k: v.mode() for k, v in inner.wm.heads['decoder'](post).items()}

  out = run_pure(agent, fn, data)
  return {k: tuple(np.asarray(v).shape) for k, v in out.items()}


def kernel(agent, suffix):
  names = [k for k in agent.varibs if k.endswith(suffix)]
  assert len(names) == 1, (suffix, names)
  return tuple(np.asarray(agent.varibs[names[0]]).shape)


def check_flat(text):
  label = 'slots + text' if text else 'slots'
  print(f'== rssm on slot observations ({label})')
  agent, obs_key, _ = build_agent(
      'rssm', None, text=text, obs_key='slot')
  data = make_batch(obs_key, text=text)

  # The encoder appends the language embedding as one more slot.
  slots = SLOTS + 1 if text else SLOTS
  shape = embed_shape(agent, data)
  assert shape == (BATCH, LENGTH, slots * SLOT_DIM), (
      f'the GRU needs one flat vector per step, got {shape}')
  print(f'  embed {shape}, {slots} slots x {SLOT_DIM} folded into features')

  shapes = decoded_shapes(agent, data)
  # Only the object slots are reconstructed; language goes to the token head.
  assert shapes['slot'] == (BATCH, LENGTH, SLOTS, SLOT_DIM), shapes
  assert ('token' in shapes) == text, shapes
  print(f'  decoded ' + ', '.join(f'{k} {v}' for k, v in sorted(shapes.items())))

  # One map from the whole latent to all slots at once, unlike the slot-shaped
  # path where the same projector runs per slot.
  width = kernel(agent, 'dec/slot_proj/kernel')
  assert width[-1] == SLOTS * SLOT_DIM, width
  print(f'  slot_proj kernel {width}')

  check_loss_and_report(agent, obs_key, text)
  check_train_step(agent, obs_key, text)


def check_octssm_unchanged():
  """The slot-shaped path: a slot axis end to end and a shared projector."""
  print('== octssm on slot observations (regression guard)')
  agent, obs_key, _ = build_agent('octssm', 'none', action_mode='slot',
                                  text=True)
  data = make_batch(obs_key, text=True)
  shape = embed_shape(agent, data)
  assert shape == (BATCH, LENGTH, SLOTS + 1, SLOT_DIM), shape
  shapes = decoded_shapes(agent, data)
  assert shapes['slot'] == (BATCH, LENGTH, SLOTS, SLOT_DIM), shapes
  width = kernel(agent, 'dec/slot_proj/kernel')
  assert width[-1] == SLOT_DIM, width
  print(f'  embed {shape}, slot_proj kernel {width}')


def main():
  check_flat(text=True)
  check_flat(text=False)
  check_octssm_unchanged()
  print('all ok')


if __name__ == '__main__':
  main()
