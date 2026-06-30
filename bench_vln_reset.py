"""Benchmark simultaneous reset wall time: BatchEnv vs BufferedBatchEnv.

Measures how long one driver step takes when *all* active envs receive
``reset=True`` at once (after a short untimed warmup). This isolates the
reset bottleneck that ``bench_collect.py`` only sees indirectly during
collection.

Run on the server (inside the conda env with Habitat / VLN_CE):

    conda activate dynalang-vln
    cd ~/code/dynalang
    git checkout vln_two_envs

    # Baseline: plain BatchEnv
    python bench_vln_reset.py \
        --sweep 1,2,4,8,16,32,64,128,252 \
        --spare 0 --gpu 5 \
        --outdir bench_vln_reset_out/baseline

    # Buffered: double buffering (N active + N spare workers)
    python bench_vln_reset.py \
        --sweep 1,2,4,8,16,32,64,128,252 \
        --spare-eq-n --gpu 5 \
        --outdir bench_vln_reset_out/buffered

Outputs (per run, in --outdir):
    - reset_results.csv   one row per (num_envs, spare)
"""
import argparse
import os
import pathlib
import time

import numpy as np

import bench_collect as bc


def reset_acts(env):
  acts = {k: np.zeros((len(env),) + tuple(sp.shape), sp.dtype)
          for k, sp in env.act_space.items() if k != 'reset'}
  acts['reset'] = np.ones(len(env), bool)
  return acts


def bench_reset(env, warmup_resets=2, measure_resets=5):
  acts = reset_acts(env)
  for _ in range(warmup_resets):
    env.step(acts)
  times = []
  for _ in range(measure_resets):
    t0 = time.perf_counter()
    env.step(acts)
    times.append(time.perf_counter() - t0)
  return times


def run_sweep(config, sweep, strategy, spare, spare_eq_n, warmup_resets,
              measure_resets):
  mode = 'BufferedBatchEnv' if spare or spare_eq_n else 'BatchEnv'
  print(f'\n=== reset benchmark ({mode}, strategy={strategy}) ===')
  print(f'{"N":>4}  {"spare":>5}  {"build(s)":>8}  {"mean(s)":>8}  '
        f'{"median(s)":>9}  {"max(s)":>8}')
  rows = []
  for n in sweep:
    spare_n = n if spare_eq_n else spare
    print(f'\n--- N={n} spare={spare_n} ---')
    t_build = time.time()
    try:
      env = bc.build_batch_env(config, n, strategy, spare=spare_n)
    except Exception as e:  # noqa: BLE001
      print(f'  BUILD FAILED: {type(e).__name__}: {e}')
      break
    build_s = time.time() - t_build
    try:
      times = bench_reset(env, warmup_resets, measure_resets)
    finally:
      env.close()
    arr = np.array(times)
    row = {
        'num_envs': n,
        'spare': spare_n,
        'build_s': build_s,
        'reset_mean_s': float(arr.mean()),
        'reset_median_s': float(np.median(arr)),
        'reset_max_s': float(arr.max()),
        'reset_min_s': float(arr.min()),
    }
    rows.append(row)
    print(f'{n:>4}  {spare_n:>5}  {build_s:>8.1f}  {row["reset_mean_s"]:>8.3f}  '
          f'{row["reset_median_s"]:>9.3f}  {row["reset_max_s"]:>8.3f}')
  return rows


def write_csv(rows, path, strategy):
  cols = ['num_envs', 'spare', 'build_s', 'reset_mean_s', 'reset_median_s',
          'reset_max_s', 'reset_min_s', 'strategy']
  lines = [','.join(cols)]
  for r in rows:
    lines.append(','.join(str(r[c]) if c != 'strategy' else strategy
                           for c in cols))
  path.write_text('\n'.join(lines) + '\n')
  print(f'\nwrote {path}')


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--sweep', default='1,2,4,8,16,32,64,128,252')
  p.add_argument('--spare', type=int, default=0,
                 help='fixed spare count for BufferedBatchEnv (0 = BatchEnv)')
  p.add_argument('--spare-eq-n', action='store_true',
                 help='set spare=N at each sweep point (double buffering)')
  p.add_argument('--strategy', default='process',
                 choices=['process', 'thread', 'none'])
  p.add_argument('--gpu', type=int, default=None)
  p.add_argument('--warmup-resets', type=int, default=2)
  p.add_argument('--measure-resets', type=int, default=5)
  p.add_argument('--max-envs', type=int, default=252)
  p.add_argument('--outdir', default='bench_vln_reset_out')
  args = p.parse_args()

  if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  sweep = sorted({int(x) for x in args.sweep.split(',')
                  if 0 < int(x) <= args.max_envs})
  outdir = pathlib.Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  config = bc.load_config('vln')
  mode = 'buffered' if args.spare or args.spare_eq_n else 'baseline'
  print(f'mode={mode}  sweep={sweep}  spare={args.spare}  '
        f'spare_eq_n={args.spare_eq_n}  strategy={args.strategy}')

  rows = run_sweep(
      config, sweep, args.strategy, args.spare, args.spare_eq_n,
      args.warmup_resets, args.measure_resets)
  if not rows:
    print('\nNo results collected.')
    return

  write_csv(rows, outdir / 'reset_results.csv', args.strategy)


if __name__ == '__main__':
  main()
