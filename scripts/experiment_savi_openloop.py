"""Collect disjoint MOF episodes and compare fixed-data world-model objectives.

This measures held-out dynamics prediction, not online policy success. Run each
variant in a separate process to release JAX GPU allocations between variants.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'dynalang')]
import numpy as np


def collect(args):
  os.environ.setdefault('MUJOCO_GL', 'egl')
  spec = importlib.util.spec_from_file_location('mof_pixels', ROOT / 'dynalang/embodied/envs/mof.py')
  mof = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mof)
  mof._patch_gym_goalenv()
  import gym
  import multi_object_fetch  # noqa: F401
  from embodied.torch.ocr.savi import SAViExtractor
  extractor = SAViExtractor('', str(args.checkpoint), 64, args.encoder_device)
  env = gym.make(args.task).unwrapped
  env = mof._ActionRepeat(env, 2)
  env = mof._TimeLimit(env, 50)
  env = mof._PixelLangObs(env, (64, 64), use_language=False, vis=False)
  args.output.mkdir(parents=True, exist_ok=True)
  try:
    for split, count, seed_base in [('train', args.train_episodes, 0),
                                    ('heldout', args.eval_episodes, 100000)]:
      directory = args.output / split
      directory.mkdir(exist_ok=True)
      for episode in range(count):
        seed = seed_base + episode
        rng = np.random.default_rng(seed)
        env.seed(seed)
        obs = env.reset()
        carry = None
        records = {k: [] for k in ['image', 'slot', 'action', 'reward',
                                    'is_first', 'is_last', 'is_terminal']}
        reward, done = 0.0, False
        for t in range(51):
          slots, carry = extractor.get_slots(obs['image'], carry)
          action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
          for k, v in dict(image=obs['image'], slot=slots, action=action,
              reward=np.float32(reward), is_first=t == 0, is_last=done,
              is_terminal=False).items():
            records[k].append(v)
          if done:
            break
          obs, reward, done, info = env.step(action)
        if done and not info.get('TimeLimit.truncated', False):
          records['is_terminal'][-1] = True
        np.savez_compressed(directory / f'episode-{seed:06d}.npz',
                            **{k: np.stack(v) for k, v in records.items()})
        if (episode + 1) % 10 == 0:
          print(f'{split}: {episode + 1}/{count} episodes', flush=True)
  finally:
    env.close()
  manifest = {'task': args.task, 'train_seeds': [0, args.train_episodes - 1],
      'heldout_seeds': [100000, 100000 + args.eval_episodes - 1],
      'action_repeat': 2, 'episode_steps': 50,
      'savi_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
  (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def train(args):
  import jax
  import jax.numpy as jnp
  import embodied
  import ruamel.yaml as yaml
  from dynalang import ninjax as nj, jaxutils
  from dynalang.agent import WorldModel
  from dynalang.savi_decoder import FrozenSAViDecoder

  jaxutils.COMPUTE_DTYPE = jnp.float32
  args.output.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  def load(split):
    episodes = []
    for path in sorted((args.data / split).glob('*.npz')):
      with np.load(path) as ep:
        episodes.append({k: ep[k] for k in ep})
    if not episodes:
      raise ValueError(f'No {split} episodes in {args.data}')
    return episodes
  training, heldout = load('train'), load('heldout')
  presets = yaml.YAML(typ='safe').load((ROOT / 'dynalang/configs.yaml').read_text())
  cfg = embodied.Config(presets['defaults']).update(presets['mof_savi']).update({
      'batch_size': args.batch, 'batch_length': 32,
      'slot_extractor.checkpoint_path': str(args.checkpoint),
      'encoder.mlp_keys': '$^', 'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': '$^', 'decoder.cnn_keys': '$^',
      # Existing slot reconstruction sums over K*D. Normalize it to a mean
      # here so auxiliary mean-MSE coefficients have comparable units.
      'loss_scales.vector': 1.0 / int(np.prod(training[0]['slot'].shape[1:])),
      'octssm': {'deter': 128, 'units': 128, 'stoch': 8, 'classes': 8,
          'tf_layers': 2, 'tf_heads': 4, 'feedforward_units': 512,
          'tf_context_length': 16, 'action_mode': 'slot',
          'kv_cache': True, 'pe': 'rope', 'dropout': 0.0},
      'reward_head.mlp.layers': 2, 'reward_head.mlp.units': 128,
      'cont_head.mlp.layers': 2, 'cont_head.mlp.units': 128,
      'run.pretrain_wm_only': True,
      'savi_image_loss.starts': 4, 'openloop_loss.starts': 4,
  })
  shapes = {k: v.shape[1:] for k, v in training[0].items()}
  spaces = {k: embodied.Space(v.dtype, shapes[k]) for k, v in training[0].items()
            if k != 'action'}
  acts = {'action': embodied.Space(np.float32, shapes['action'], -1, 1)}
  def preprocess(batch):
    return {**{k: jnp.asarray(v, jnp.float32) for k, v in batch.items()},
            'image': jnp.asarray(batch['image'], jnp.float32) / 255,
            'cont': 1 - jnp.asarray(batch['is_terminal'], jnp.float32)}
  def sample():
    records = []
    eligible = [ep for ep in training if len(ep['reward']) >= 32]
    if not eligible:
      raise ValueError('Training episodes shorter than 32 frames')
    for _ in range(args.batch):
      ep = eligible[rng.integers(len(eligible))]
      offset = rng.integers(len(ep['reward']) - 32 + 1)
      records.append({k: v[offset:offset + 32] for k, v in ep.items()})
    batch = {k: np.stack([r[k] for r in records]) for k in records[0]}
    batch['is_first'][:, 0] = True
    return preprocess(batch)

  # Fixed evaluation windows from entire disjoint episodes, never used in updates.
  evaluation = []
  for ep in heldout:
    if len(ep['reward']) >= 31:
      evaluation.append({k: v[:31] for k, v in ep.items()})
  if not evaluation:
    raise ValueError('Held-out episodes shorter than context 16 + horizon 15')
  evaluator_decoder = FrozenSAViDecoder(str(args.checkpoint))
  history, variables, update_number = [], None, 0
  # Slots first at H=3 then H=5; image variant then enables RGB and extends H.
  horizons = [3, 5, 5, 10, 15]
  for stage, horizon in enumerate(horizons):
    slot_scale = 1.0 if args.variant.startswith('openloop') else 0.0
    image_scale = 1.0 if args.variant == 'openloop-image' and stage >= 2 else 0.0
    stage_cfg = cfg.update({'openloop_loss.horizon': horizon,
        'openloop_loss.slot_scale': slot_scale,
        'openloop_loss.image_scale': image_scale,
        'savi_image_loss.scale': 1.0 if args.variant == 'posterior-image' else 0.0})
    stage_cfg.save(str(args.output / f'config-stage-{stage}.yaml'))
    wm = WorldModel(spaces, acts, stage_cfg, shapes, name='wm')
    def update(batch):
      return wm.train(batch, wm.initial(len(batch['reward'])))
    step = jax.jit(nj.pure(update))
    if variables is None:
      def initialize(batch):
        return wm.loss(batch, wm.initial(len(batch['reward'])))
      _, variables = jax.jit(nj.pure(initialize))(
          {}, jax.random.PRNGKey(args.seed), sample())
      digest = hashlib.sha256()
      for name, value in sorted(jax.device_get(variables).items()):
        digest.update(name.encode())
        digest.update(value.tobytes())
      initial_hash = digest.hexdigest()
    def evaluate(batch):
      context = {k: v[:, :16] for k, v in batch.items()}
      actions = {'action': jnp.concatenate(
          [jnp.zeros_like(context['action'][:, :1]), context['action'][:, :-1]], 1)}
      post = wm.rssm.observe(wm.encoder(context), actions, context['is_first'])
      start = {k: v[:, -1] for k, v in post.items()}
      prior = wm.rssm.imagine({'action': batch['action'][:, 15:30]}, start)
      slots = wm.heads['decoder'](prior)['slot'].mode()
      truth = batch['slot'][:, 16:31]
      errors = jnp.square(slots - truth).mean((-1, -2))
      image_errors = evaluator_decoder.mse(slots, batch['image'][:, 16:31])
      persistence = jnp.square(batch['slot'][:, 15:16] - truth).mean((-1, -2))
      return errors, image_errors, persistence
    score = jax.jit(nj.pure(evaluate))
    def measure(label):
      collected = []
      for i in range(0, len(evaluation), args.eval_batch):
        records = evaluation[i:i + args.eval_batch]
        batch = preprocess({k: np.stack([ep[k] for ep in records]) for k in records[0]})
        result, _ = score(variables, jax.random.PRNGKey(123), batch)
        collected.append(jax.device_get(result))
      errors, image_errors, persistence = [np.concatenate([r[j] for r in collected])
                                           for j in range(3)]
      entry = {'variant': args.variant, 'seed': args.seed, 'stage': stage,
          'initial_parameters_sha256': initial_hash,
          'updates': update_number, 'label': label, 'training_horizon': horizon,
          'slot_scale': slot_scale, 'image_scale': image_scale,
          'heldout_episodes': len(evaluation),
          'slot_mse_by_horizon': errors.mean(0).tolist(),
          'image_mse_by_horizon': image_errors.mean(0).tolist(),
          'persistence_slot_mse_by_horizon': persistence.mean(0).tolist()}
      history.append(entry)
      (args.output / 'metrics.json').write_text(json.dumps(history, indent=2))
      print(json.dumps(entry), flush=True)
    if stage == 0:
      measure('initial')
    start_time = time.monotonic()
    for update_index in range(args.updates_per_stage):
      (_, _, metrics), variables = step(
          variables, jax.random.PRNGKey(args.seed + update_number + 1), sample())
      update_number += 1
      if (update_index + 1) % 50 == 0:
        print('update', update_number,
              {k: float(v) for k, v in metrics.items() if k.endswith('_mse')}, flush=True)
    jax.block_until_ready(variables)
    print('stage seconds', time.monotonic() - start_time, flush=True)
    measure('stage_end')
    with (args.output / f'checkpoint-stage-{stage}.pkl').open('wb') as stream:
      pickle.dump(jax.device_get(variables), stream)
  print('Completed', args.variant, flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode', choices=['collect', 'train'])
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/savi/reach_red.ckpt')
  parser.add_argument('--task', default='ReachRed_2Distractors_Dense-v1')
  parser.add_argument('--train-episodes', type=int, default=80)
  parser.add_argument('--eval-episodes', type=int, default=20)
  parser.add_argument('--encoder-device', default='cuda')
  parser.add_argument('--data', type=Path)
  parser.add_argument('--variant', choices=['baseline', 'posterior-image',
      'openloop-slot', 'openloop-image'], default='baseline')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--updates-per-stage', type=int, default=200)
  parser.add_argument('--batch', type=int, default=8)
  parser.add_argument('--eval-batch', type=int, default=2)
  args = parser.parse_args()
  if args.mode == 'train' and args.data is None:
    parser.error('--data is required for train')
  (collect if args.mode == 'collect' else train)(args)


if __name__ == '__main__':
  main()
