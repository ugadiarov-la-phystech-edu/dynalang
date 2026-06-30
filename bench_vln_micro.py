"""Micro-benchmark: Habitat reset vs step for a single VLN env.

Breaks down where time goes in one environment (no BatchEnv):

  habitat_reset   inner ``env._env.reset()`` only (Habitat / VLNCE)
  habitat_step    inner ``env._env.step(action)`` only
  vln_reset_step  full ``VLNEnv.step({reset: True, ...})``  (wrapper + render)
  vln_read_step   full step while streaming instruction (no sim step)
  vln_action_step full step after instruction read (sim MOVE_FORWARD)

Run on the server:

    conda activate dynalang-vln
    cd ~/code/dynalang
    python bench_vln_micro.py --gpu 0 --repeats 20 --outdir bench_vln_micro_out

Outputs (in --outdir):
    - micro_results.csv
    - micro_timings.png     bar chart of mean timings (ms)
"""
import argparse
import os
import pathlib
import time

import numpy as np

import bench_collect as bc
import embodied


def _stats(times):
  arr = np.asarray(times, float)
  return {
      'mean_s': float(arr.mean()),
      'median_s': float(np.median(arr)),
      'min_s': float(arr.min()),
      'max_s': float(arr.max()),
      'std_s': float(arr.std()),
  }


def _fmt(name, s):
  return (f'{name:18s}  mean={s["mean_s"]*1e3:7.1f}ms  '
          f'median={s["median_s"]*1e3:7.1f}ms  '
          f'min={s["min_s"]*1e3:7.1f}ms  max={s["max_s"]*1e3:7.1f}ms')


def _make_single_env(config, parallel):
  env = bc.make_env(config)
  if parallel:
    env = embodied.Parallel(lambda: env, 'process')()
  return env


def _unwrap_vln(env):
  """Return (VLNEnv, inner habitat env), unwrapping embodied.Wrapper chain."""
  from embodied.core.parallel import Parallel

  if isinstance(env, Parallel):
    raise RuntimeError(
        'Micro-benchmark needs in-process env; run without --parallel')
  while isinstance(env, embodied.Wrapper):
    env = env.env
  if not hasattr(env, '_env'):
    raise RuntimeError(
        f'Expected VLNEnv after unwrapping, got {type(env).__name__}')
  return env, env._env


def _start_episode(vln):
  vln.step({'action': 0, 'reset': True})


def _advance_past_read(vln, max_steps=512):
  act = {'action': 0, 'reset': False}
  for _ in range(max_steps):
    ob = vln.step(act)
    if not ob['is_read_step']:
      return ob
  raise RuntimeError('instruction read did not finish')


def bench_habitat_reset(inner, warmup, repeats):
  for _ in range(warmup):
    inner.reset()
  times = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    inner.reset()
    times.append(time.perf_counter() - t0)
  return times


def bench_habitat_step(inner, warmup, repeats, action=1):
  for _ in range(warmup):
    inner.step(action)
  times = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    inner.step(action)
    times.append(time.perf_counter() - t0)
  return times


def bench_vln_reset_step(vln, warmup, repeats):
  act = {'action': 0, 'reset': True}
  for _ in range(warmup):
    vln.step(act)
  times = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    vln.step(act)
    times.append(time.perf_counter() - t0)
  return times


def bench_vln_read_step(vln, warmup, repeats):
  """Step during instruction streaming (no Habitat sim step)."""
  act = {'action': 0, 'reset': False}
  times = []
  for i in range(warmup + repeats):
    vln.step({'action': 0, 'reset': True})
    t0 = time.perf_counter()
    ob = vln.step(act)
    dt = time.perf_counter() - t0
    if i >= warmup:
      times.append(dt)
  return times


def bench_vln_action_step(vln, warmup, repeats, action=1):
  act = {'action': action, 'reset': False}
  for _ in range(warmup):
    _start_episode(vln)
    _advance_past_read(vln)
    vln.step(act)
  times = []
  for _ in range(repeats):
    _start_episode(vln)
    _advance_past_read(vln)
    t0 = time.perf_counter()
    vln.step(act)
    times.append(time.perf_counter() - t0)
  return times


def run(config, warmup, repeats, parallel):
  env = _make_single_env(config, parallel)
  try:
    vln, inner = _unwrap_vln(env)
    results = {}
    results['habitat_reset'] = _stats(
        bench_habitat_reset(inner, warmup, repeats))
    results['vln_reset_step'] = _stats(
        bench_vln_reset_step(vln, warmup, repeats))

    _start_episode(vln)
    _advance_past_read(vln)
    results['habitat_step'] = _stats(
        bench_habitat_step(inner, warmup, repeats, action=1))
    results['vln_read_step'] = _stats(
        bench_vln_read_step(vln, warmup, repeats))
    results['vln_action_step'] = _stats(
        bench_vln_action_step(vln, warmup, repeats, action=1))
    return results
  finally:
    env.close()


METRIC_ORDER = (
    'habitat_reset',
    'habitat_step',
    'vln_reset_step',
    'vln_read_step',
    'vln_action_step',
)
METRIC_LABELS = {
    'habitat_reset': 'habitat\nreset',
    'habitat_step': 'habitat\nstep',
    'vln_reset_step': 'VLN reset\nstep',
    'vln_read_step': 'VLN read\nstep',
    'vln_action_step': 'VLN action\nstep',
}


def write_csv(results, path):
  cols = ['metric', 'mean_s', 'median_s', 'min_s', 'max_s', 'std_s']
  lines = [','.join(cols)]
  for name, s in results.items():
    lines.append(','.join([name] + [str(s[c]) for c in cols[1:]]))
  path.write_text('\n'.join(lines) + '\n')
  print(f'wrote {path}')


def plot(results, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  names = [m for m in METRIC_ORDER if m in results]
  means_ms = [results[m]['mean_s'] * 1e3 for m in names]
  stds_ms = [results[m]['std_s'] * 1e3 for m in names]
  colors = ['#e74c3c' if m.startswith('habitat') else '#2980b9' for m in names]
  labels = [METRIC_LABELS[m] for m in names]

  fig, ax = plt.subplots(figsize=(10, 5))
  x = np.arange(len(names))
  bars = ax.bar(x, means_ms, yerr=stds_ms, capsize=4, color=colors, alpha=0.85)
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_ylabel('mean wall time (ms)')
  ax.set_title('Single VLN env: Habitat vs full VLNEnv step timings')
  ax.grid(axis='y', alpha=0.3)

  hr = results.get('habitat_reset', {}).get('mean_s', 0) * 1e3
  hs = results.get('habitat_step', {}).get('mean_s', 0) * 1e3
  vr = results.get('vln_reset_step', {}).get('mean_s', 0) * 1e3
  va = results.get('vln_action_step', {}).get('mean_s', 0) * 1e3
  note = []
  if hs > 0:
    note.append(f'habitat reset/step = {hr / hs:.1f}x')
  if va > 0:
    note.append(f'VLN reset/action = {vr / va:.1f}x')
  if note:
    ax.text(0.02, 0.98, '  |  '.join(note), transform=ax.transAxes,
            va='top', fontsize=9)

  for bar, val in zip(bars, means_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)

  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f'wrote {path}')


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--gpu', type=int, default=None)
  p.add_argument('--warmup', type=int, default=3)
  p.add_argument('--repeats', type=int, default=20)
  p.add_argument('--parallel', action='store_true',
                 help='wrap env in Parallel(process) with 1 worker (adds IPC)')
  p.add_argument('--outdir', default='bench_vln_micro_out')
  args = p.parse_args()

  if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  outdir = pathlib.Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  config = bc.load_config('vln')
  print(f'warmup={args.warmup}  repeats={args.repeats}  parallel={args.parallel}')
  results = run(config, args.warmup, args.repeats, args.parallel)

  print('\n=== single-env micro timings ===')
  for name in ('habitat_reset', 'habitat_step', 'vln_reset_step',
               'vln_read_step', 'vln_action_step'):
    print(_fmt(name, results[name]))

  hr = results['habitat_reset']['mean_s']
  hs = results['habitat_step']['mean_s']
  vr = results['vln_reset_step']['mean_s']
  va = results['vln_action_step']['mean_s']

  print('\n=== breakdown ===')
  if hs > 0:
    print(f'  habitat_reset / habitat_step     = {hr / hs:.1f}x  '
          f'({hr*1e3:.0f}ms vs {hs*1e3:.0f}ms)')
  if va > 0:
    print(f'  vln_reset_step / vln_action_step = {vr / va:.1f}x  '
          f'({vr*1e3:.0f}ms vs {va*1e3:.0f}ms)')
  print(f'  VLN wrapper on reset (approx)    = {(vr - hr)*1e3:.0f}ms  '
        f'(vln_reset_step - habitat_reset)')
  print(f'  VLN wrapper on action (approx)   = {(va - hs)*1e3:.0f}ms  '
        f'(vln_action_step - habitat_step)')

  csv = outdir / 'micro_results.csv'
  write_csv(results, csv)
  plot(results, outdir / 'micro_timings.png')


if __name__ == '__main__':
  main()
