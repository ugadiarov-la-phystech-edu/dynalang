"""FPS / scaling benchmark for data collection with BatchEnv.

For each environment (homegrid, langroom, vln) we collect a fixed number of
total env-steps (default 5000) through ``embodied.BatchEnv`` while sweeping the
number of parallel envs N. We measure the *total wall time to accumulate the
target number of steps* and keep increasing N until the time stops improving
(plateau).

Run (inside the conda env that has the envs installed):

    conda activate ocdreamer
    python bench_collect.py --envs homegrid,langroom

Outputs (in bench_collect_out/):
    - results.csv               one row per (env, num_envs)
    - collect_time.png          collection time, build time, fps vs N

Notes:
    - "total_env" semantics: the target is total env-steps across the whole
      batch, so larger N => fewer driver steps. This is the fair way to compare
      collection speed.
    - We report three times per N: ``build_s`` (constructing the BatchEnv,
      incl. spawning worker processes), ``time_per_target`` (steady-state
      collection, normalized to exactly --target steps), and ``total_per_target``
      (build + collection). The first (slow) reset and worker warmup are
      excluded from the collection timing via a short untimed warmup; only
      steady-state stepping is timed.
"""
import argparse
import importlib.util
import pathlib
import sys
import time
import types
from functools import partial as bind

import numpy as np

# --- make `embodied` importable without pulling in the heavy training stack ---
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / 'dynalang'))
for _n in ('embodied.run', 'embodied.replay'):
  sys.modules.setdefault(_n, types.ModuleType(_n))

import embodied  # noqa: E402

import ruamel.yaml as _yaml  # noqa: E402

CONFIGS_PATH = _ROOT / 'dynalang' / 'configs.yaml'
OUT = _ROOT / 'bench_collect_out'


def _load_train_module():
  """Load dynalang/train.py by path so we can reuse its make_env/wrap_env
  without triggering the heavy agent import (that happens only inside main)."""
  path = _ROOT / 'dynalang' / 'train.py'
  spec = importlib.util.spec_from_file_location('dynalang_train', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_train = _load_train_module()
make_env = _train.make_env
wrap_env = _train.wrap_env

# Default sweep over number of envs; pruned to <= max_envs and <= target.
DEFAULT_SWEEP = [1, 2, 4, 8, 16, 24, 32, 48, 64]


# --------------------------------------------------------------------------- #
# Env construction (reuses make_env / wrap_env imported from dynalang/train.py)
# --------------------------------------------------------------------------- #
# Sections of a named config that actually affect env construction/stepping.
# Everything else (encoder/decoder/rssm/run and the agent-only regex keys such
# as ``.*\.vec_keys``) is agent-side and is intentionally ignored so the
# benchmark builds the exact same env objects as training without crashing on
# patterns that match nothing in the defaults.
_ENV_RELEVANT_SECTIONS = ('task', 'env', 'envs', 'wrapper')


def load_config(env_name):
  """Build the embodied.Config for a named env config (e.g. 'homegrid').

  Only the env-relevant sections of the named config are applied (see
  ``_ENV_RELEVANT_SECTIONS``); agent-only keys are dropped.
  """
  configs = _yaml.YAML(typ='safe').load(CONFIGS_PATH.read_text())
  config = embodied.Config(configs['defaults'])
  if env_name in configs:
    relevant = {k: v for k, v in configs[env_name].items()
                if k.split('.')[0] in _ENV_RELEVANT_SECTIONS}
    if relevant:
      config = config.update(relevant)
  return config


def build_batch_env(config, amount, strategy):
  ctor = bind(make_env, config)
  if strategy != 'none':
    ctor = bind(embodied.Parallel, ctor, strategy)
  envs = [ctor() for _ in range(amount)]
  return embodied.BatchEnv(envs, strategy != 'none')


# --------------------------------------------------------------------------- #
# Random policy + collection loop
# --------------------------------------------------------------------------- #
def make_random_policy(act_space, seed=0):
  rng = np.random.default_rng(seed)
  keys = [(k, sp) for k, sp in act_space.items() if k != 'reset']

  def policy(n):
    acts = {}
    for k, sp in keys:
      shape = (n,) + tuple(sp.shape)
      if sp.discrete:
        low = np.broadcast_to(sp.low, sp.shape)
        high = np.broadcast_to(sp.high, sp.shape)  # exclusive upper bound
        acts[k] = rng.integers(low, high, size=shape).astype(sp.dtype)
      else:
        lo = np.where(np.isfinite(sp.low), sp.low, -1.0)
        hi = np.where(np.isfinite(sp.high), sp.high, 1.0)
        acts[k] = rng.uniform(lo, hi, size=shape).astype(sp.dtype)
    return acts

  return policy


def collect(env, target_steps, warmup=3, seed=0):
  """Step ``env`` until >= target_steps env-steps; return timing dict."""
  n = len(env)
  act_space = env.act_space
  policy = make_random_policy(act_space, seed)
  acts = {k: np.zeros((n,) + tuple(sp.shape), sp.dtype)
          for k, sp in act_space.items() if k != 'reset'}
  acts['reset'] = np.ones(n, bool)

  for _ in range(warmup):
    obs = env.step(acts)
    acts = policy(n)
    acts['reset'] = obs['is_last'].copy()

  t0 = time.time()
  env_steps, driver_steps = 0, 0
  while env_steps < target_steps:
    obs = env.step(acts)
    env_steps += n
    driver_steps += 1
    acts = policy(n)
    acts['reset'] = obs['is_last'].copy()
  wall = time.time() - t0

  return {
      'wall': wall,
      'env_steps': env_steps,
      'driver_steps': driver_steps,
      'time_per_target': wall * target_steps / env_steps,
      'fps': env_steps / wall,
  }


# --------------------------------------------------------------------------- #
# Sweep with plateau detection
# --------------------------------------------------------------------------- #
def run_env_sweep(name, strategy, target, sweep, min_improve, repeats, warmup,
                  early_stop):
  print(f'\n=== {name}  (strategy={strategy}, target={target} env-steps) ===')
  print(f'{"N":>4}  {"build(s)":>8}  {"collect(s)":>10}  {"total(s)":>8}  '
        f'{"fps":>9}  {"vs prev":>8}')
  results = []
  prev = None
  for N in sweep:
    config = load_config(name)
    t_build = time.time()
    try:
      env = build_batch_env(config, N, strategy)
    except Exception as e:  # noqa: BLE001
      print(f'  !! could not build {name} with N={N}: '
            f'{type(e).__name__}: {e}')
      if not results:
        return results  # env unavailable at all -> bail out for this env
      break
    build_s = time.time() - t_build
    try:
      runs = [collect(env, target, warmup, seed=r) for r in range(repeats)]
    finally:
      env.close()
    best = min(runs, key=lambda r: r['time_per_target'])
    best['num_envs'] = N
    best['env'] = name
    best['strategy'] = strategy
    best['build_s'] = build_s
    best['total_per_target'] = build_s + best['time_per_target']
    results.append(best)

    if prev is None:
      improvement = float('nan')
      imp_str = '   --'
    else:
      improvement = (prev['time_per_target'] - best['time_per_target']) \
          / prev['time_per_target']
      imp_str = f'{improvement * 100:+6.1f}%'
    print(f'{N:>4}  {best["build_s"]:>8.3f}  {best["time_per_target"]:>10.3f}  '
          f'{best["total_per_target"]:>8.3f}  {best["fps"]:>9.1f}  '
          f'{imp_str:>8}')

    if early_stop and prev is not None and improvement < min_improve:
      print(f'  -> plateau reached (improvement {improvement * 100:+.1f}% '
            f'< {min_improve * 100:.0f}%); stopping sweep for {name}.')
      break
    prev = best
  return results


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(all_results, path):
  cols = ['env', 'strategy', 'num_envs', 'driver_steps', 'env_steps',
          'wall', 'build_s', 'time_per_target', 'total_per_target',
          'fps']
  lines = [','.join(cols)]
  for r in all_results:
    lines.append(','.join(str(r[c]) for c in cols))
  path.write_text('\n'.join(lines) + '\n')
  print(f'\nwrote {path}')


def plot(all_results, target, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  envs = sorted({r['env'] for r in all_results})
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  for name in envs:
    rs = sorted((r for r in all_results if r['env'] == name),
                key=lambda r: r['num_envs'])
    ns = [r['num_envs'] for r in rs]
    axes[0].plot(ns, [r['time_per_target'] for r in rs], 'o-', label=name)
    axes[1].plot(ns, [r['build_s'] for r in rs], 'o-', label=name)
    axes[2].plot(ns, [r['fps'] for r in rs], 'o-', label=name)
  axes[0].set_xlabel('number of envs (N)')
  axes[0].set_ylabel(f'wall time to collect {target} steps (s)')
  axes[0].set_title('Collection time vs N (lower is better)')
  axes[1].set_xlabel('number of envs (N)')
  axes[1].set_ylabel('build / spawn time (s)')
  axes[1].set_title('Batch build time vs N (lower is better)')
  axes[2].set_xlabel('number of envs (N)')
  axes[2].set_ylabel('fps (env-steps / s)')
  axes[2].set_title('FPS vs N (higher is better)')
  for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f'wrote {path}')


# --------------------------------------------------------------------------- #
def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--envs', default='homegrid,langroom',
                 help='comma-separated env config names (homegrid,langroom,vln)')
  p.add_argument('--strategy', default='process',
                 choices=['process', 'thread', 'none'],
                 help='BatchEnv parallelism strategy')
  p.add_argument('--target', type=int, default=5000,
                 help='total env-steps to accumulate per config')
  p.add_argument('--max-envs', type=int, default=None,
                 help='cap on N (default: 2x CPU count, max 64)')
  p.add_argument('--sweep', default=None,
                 help='explicit comma-separated list of N (overrides default)')
  p.add_argument('--repeats', type=int, default=1,
                 help='runs per N (best/min time is kept)')
  p.add_argument('--warmup', type=int, default=3,
                 help='untimed driver steps before timing')
  p.add_argument('--min-improve', type=float, default=0.05,
                 help='plateau threshold: stop when time improves less than this')
  p.add_argument('--no-early-stop', action='store_true',
                 help='run the full sweep without plateau detection')
  p.add_argument('--gpu', type=int, default=None,
                 help='GPU id for the vln/Habitat env via CUDA_VISIBLE_DEVICES '
                      '(CPU envs homegrid/langroom ignore it)')
  p.add_argument('--outdir', default=str(OUT))
  args = p.parse_args()

  import os
  if args.gpu is not None:
    # Set before any worker process is spawned so children inherit it.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  max_envs = args.max_envs or min(64, (os.cpu_count() or 8) * 2)
  if args.sweep:
    sweep = [int(x) for x in args.sweep.split(',')]
  else:
    sweep = DEFAULT_SWEEP
  sweep = sorted({n for n in sweep if 0 < n <= max_envs and n <= args.target})

  outdir = pathlib.Path(args.outdir)
  outdir.mkdir(exist_ok=True)

  env_names = [e.strip() for e in args.envs.split(',') if e.strip()]
  print(f'envs={env_names}  strategy={args.strategy}  target={args.target}  '
        f'sweep={sweep}  early_stop={not args.no_early_stop}')

  all_results = []
  for name in env_names:
    rs = run_env_sweep(
        name, args.strategy, args.target, sweep, args.min_improve,
        args.repeats, args.warmup, early_stop=not args.no_early_stop)
    all_results.extend(rs)

  if not all_results:
    print('\nNo results collected (no env could be built).')
    return

  write_csv(all_results, outdir / 'results.csv')
  plot(all_results, args.target, outdir / 'collect_time.png')

  print('\nBest config per env (min time to collect target):')
  for name in sorted({r['env'] for r in all_results}):
    rs = [r for r in all_results if r['env'] == name]
    best = min(rs, key=lambda r: r['time_per_target'])
    print(f'  {name:10s}  N={best["num_envs"]:>3}  '
          f'collect={best["time_per_target"]:.2f}s  '
          f'build={best["build_s"]:.2f}s  '
          f'total={best["total_per_target"]:.2f}s for {args.target} steps  '
          f'({best["fps"]:.1f} fps)')


if __name__ == '__main__':
  main()
