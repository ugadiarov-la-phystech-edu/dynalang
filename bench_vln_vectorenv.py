"""Compare parallelization backends for the VLN env on the same metrics as
``bench_collect.py`` (build time, collection time, fps), sweeping the number of
parallel envs and the image-resize mode.

Backends compared:
  - batchenv       : our project's embodied.BatchEnv + embodied.Parallel(process)
                     wrapping VLNEnv (vln.py). This is the path bench_collect uses.
  - vectorenv      : Habitat's native VectorEnv parallelizing the SAME VLNEnv
                     (via a thin gym adapter) -> fair, apples-to-apples test of
                     just the parallelization layer (identical per-worker load).
  - vectorenv_raw  : Habitat's VectorEnv over the raw VLNCEWaypointEnv (no
                     T5/text/resize) -> reference number for the simulator alone.

Resize modes (``--resize``):
  - pil64    : VLNEnv resize=True  -> sensor 256, then PIL downscale to 64x64
               (the project's default). [vectorenv_raw: sensor 256, no PIL]
  - raw256   : VLNEnv resize=False -> keep raw 256x256 sensor frame, no PIL.
  - sensor64 : VLNEnv resize=False, sensor_size=64 -> render directly at 64x64.

Run (on the server, in the env that has habitat / VLN_CE installed):

    conda activate dynalang-vln
    cd ~/code/dynalang
    python bench_vln_vectorenv.py \
        --backends batchenv,vectorenv,vectorenv_raw \
        --resize pil64,raw256 \
        --sweep 1,2,4,8 \
        --target 5000 --warmup 1 --repeats 1 --gpu 5

Outputs (in bench_vln_vectorenv_out/, override with --outdir):
    - results.csv          one row per (backend, resize, num_envs)
    - vln_backends.png     collection time / build time / fps vs N, per backend+mode

NOTE: this script reuses bench_collect.collect() for timing. Several pieces
depend on the server-side Habitat/VLN_CE API (VectorEnv import path, the env
constructor, and whether step returns a 4-tuple); they are written against the
standard habitat-lab API and may need small tweaks for this checkout.
"""
import argparse
import pathlib
from functools import partial as bind

import numpy as np

import bench_collect as bc  # sets up sys.path, stubs run/replay, gives collect()
import embodied  # noqa: E402
import gym  # noqa: E402

# GPU is selected via CUDA_VISIBLE_DEVICES (set in main from --gpu), so inside
# every process the visible device is index 0 -- matching how vln.yaml / bench
# select the device. Always pass 0 to the habitat config.
_GPU_ID = 0

VLN_YAML = bc._ROOT / 'dynalang' / 'embodied' / 'envs' / 'vln.yaml'
OUT = bc._ROOT / 'bench_vln_vectorenv_out'

# resize mode -> (vlnenv_resize, vlnenv_sensor_size)
RESIZE_MODES = {
    'pil64': (True, None),     # sensor 256 + PIL -> 64 (project default)
    'raw256': (False, None),   # raw 256, no PIL
    'sensor64': (False, 64),   # render at 64 directly, no PIL
}


# --------------------------------------------------------------------------- #
# Inner VLN env builder (runs inside each worker process)
# --------------------------------------------------------------------------- #
def build_vlnenv(resize, sensor_size, gpu_id, mode='train', dataset='train',
                 use_depth=True, load_embeddings=True):
  from embodied.envs.vln import VLNEnv
  return VLNEnv(
      task='default', mode=mode, dataset=dataset, use_depth=use_depth,
      load_embeddings=load_embeddings, resize=resize, sensor_size=sensor_size,
      gpu_id=gpu_id, log_image_every=0)


# --------------------------------------------------------------------------- #
# Backend 1: our BatchEnv + Parallel(process) over VLNEnv
# --------------------------------------------------------------------------- #
def make_batchenv(n, resize, sensor_size, gpu_id, strategy='process'):
  ctor = bind(build_vlnenv, resize, sensor_size, gpu_id)
  if strategy != 'none':
    ctor = bind(embodied.Parallel, ctor, strategy)
  envs = [ctor() for _ in range(n)]
  return embodied.BatchEnv(envs, strategy != 'none')


# --------------------------------------------------------------------------- #
# Backend 2/3: Habitat VectorEnv, wrapped to look like an embodied BatchEnv so
# bench_collect.collect() can time it unchanged.
# --------------------------------------------------------------------------- #
def _import_vectorenv():
  for mod in ('habitat_lab.habitat', 'habitat',
              'habitat_lab.habitat.core.vector_env'):
    try:
      m = __import__(mod, fromlist=['VectorEnv'])
      return getattr(m, 'VectorEnv')
    except Exception:  # noqa: BLE001
      continue
  raise ImportError('Could not import habitat VectorEnv')


def _unwrap_action(action):
  """Habitat VectorEnv wraps int actions into {'action': {'action': i}} before
  calling env.step(**data); peel that back to a plain int index."""
  while isinstance(action, dict):
    action = action['action']
  return int(action)


class _VLNGymAdapter(gym.Env):
  """Expose a VLNEnv through the gym.Env interface Habitat VectorEnv expects
  (reset()/step(action)->(obs,reward,done,info), plus observation_space,
  action_space and number_of_episodes that VectorEnv queries on init)."""

  def __init__(self, resize, sensor_size, gpu_id):
    self._env = build_vlnenv(resize, sensor_size, gpu_id)
    self._n_actions = len(self._env._disc_act_space)
    self.action_space = gym.spaces.Discrete(self._n_actions)
    # Minimal obs space (VectorEnv only stores it; we never validate against it).
    img = self._env.obs_space['image']
    self.observation_space = gym.spaces.Dict({
        'image': gym.spaces.Box(0, 255, shape=tuple(img.shape), dtype=np.uint8)})
    self.number_of_episodes = None

  def reset(self):
    return self._env.step({'action': 0, 'reset': True})

  def step(self, action):
    ob = self._env.step({'action': _unwrap_action(action), 'reset': False})
    return ob, float(ob['reward']), bool(ob['is_last']), {}

  def close(self):
    try:
      self._env.close()
    except Exception:  # noqa: BLE001
      pass

  def seed(self, seed=None):
    pass


def _make_vln_gym_adapter(args):
  return _VLNGymAdapter(*args)


def build_vln_config(gpu_id, sensor_size, num_envs, dataset='train', mode='train'):
  from VLN_CE.vlnce_baselines.config.default import get_config
  opts = [
      'TASK_CONFIG.DATASET.SPLIT', dataset,
      'TASK_CONFIG.TASK.NDTW.SPLIT', dataset,
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE', mode == 'train',
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE', mode != 'train',
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE', mode == 'train',
  ]
  config = get_config(str(VLN_YAML), opts=opts)
  config.defrost()
  config.NUM_ENVIRONMENTS = num_envs
  config.SIMULATOR_GPU_IDS = [gpu_id]
  config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
  if sensor_size is not None:
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = sensor_size
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = sensor_size
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = sensor_size
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = sensor_size
  config.freeze()
  return config


class VecBatch:
  """Wrap a Habitat VectorEnv so it presents the embodied BatchEnv interface
  (len, act_space, step(acts)->dict-of-arrays with is_first/is_last)."""

  def __init__(self, vec, n_actions):
    self._vec = vec
    self._n = getattr(vec, 'num_envs', None) or len(vec)
    self._n_actions = n_actions
    self._started = False
    self._prev_done = np.zeros(self._n, bool)

  def __len__(self):
    return self._n

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._n_actions),
        'reset': embodied.Space(bool),
    }

  def _pack(self, first, done, reward):
    return {
        'is_first': np.asarray(first, bool),
        'is_last': np.asarray(done, bool),
        'is_terminal': np.asarray(done, bool),
        'reward': np.asarray(reward, np.float32),
    }

  def step(self, acts):
    if not self._started:
      self._vec.reset()
      self._started = True
      z = np.zeros(self._n)
      return self._pack(np.ones(self._n, bool), z.astype(bool), z)
    actions = [int(a) for a in np.asarray(acts['action']).reshape(-1)]
    results = self._vec.step(actions)
    if isinstance(results[0], (tuple, list)):
      done = [bool(r[2]) for r in results]
      reward = [float(r[1]) for r in results]
    else:  # list of plain obs (habitat.Env, not RLEnv): no done signal exposed
      done = [False] * self._n
      reward = [0.0] * self._n
    first = self._prev_done.copy()  # auto-reset means prev-done are fresh now
    self._prev_done = np.asarray(done, bool)
    return self._pack(first, done, reward)

  def close(self):
    try:
      self._vec.close()
    except Exception:  # noqa: BLE001
      pass


def make_vectorenv(n, resize, sensor_size, gpu_id, raw):
  if raw:
    # Use VLN_CE's construct_envs: it builds a habitat VectorEnv over the raw
    # VLNCEWaypointEnv and correctly splits scenes across the N workers.
    from VLN_CE.vlnce_baselines.common.env_utils import construct_envs
    from habitat_lab.habitat_baselines.common.environments import get_env_class
    config = build_vln_config(gpu_id, sensor_size, num_envs=n)
    vec = construct_envs(config, get_env_class(config.ENV_NAME))
    return VecBatch(vec, n_actions=4)
  # Fair branch: habitat VectorEnv over our VLNEnv (via the gym adapter).
  VectorEnv = _import_vectorenv()
  env_fn_args = tuple(((resize, sensor_size, gpu_id),) for _ in range(n))
  vec = VectorEnv(make_env_fn=_make_vln_gym_adapter, env_fn_args=env_fn_args)
  return VecBatch(vec, n_actions=4)


# --------------------------------------------------------------------------- #
def build_backend(backend, n, resize_mode):
  resize, sensor_size = RESIZE_MODES[resize_mode]
  if backend == 'batchenv':
    return make_batchenv(n, resize, sensor_size, _GPU_ID)
  if backend == 'vectorenv':
    return make_vectorenv(n, resize, sensor_size, _GPU_ID, raw=False)
  if backend == 'vectorenv_raw':
    return make_vectorenv(n, resize, sensor_size, _GPU_ID, raw=True)
  raise ValueError(backend)


def run_one(backend, resize_mode, n, target, warmup, repeats):
  import time
  t_build = time.time()
  env = build_backend(backend, n, resize_mode)
  build_s = time.time() - t_build
  try:
    runs = [bc.collect(env, target, warmup, seed=r) for r in range(repeats)]
  finally:
    env.close()
  best = min(runs, key=lambda r: r['time_per_target'])
  best.update({
      'backend': backend, 'resize': resize_mode, 'num_envs': n,
      'build_s': build_s, 'total_per_target': build_s + best['time_per_target'],
  })
  return best


# --------------------------------------------------------------------------- #
def write_csv(results, path):
  cols = ['backend', 'resize', 'num_envs', 'driver_steps', 'env_steps',
          'build_s', 'time_per_target', 'total_per_target', 'fps']
  lines = [','.join(cols)]
  for r in results:
    lines.append(','.join(str(r[c]) for c in cols))
  path.write_text('\n'.join(lines) + '\n')
  print(f'\nwrote {path}')


def plot(results, target, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  series = sorted({(r['backend'], r['resize']) for r in results})
  fig, axes = plt.subplots(1, 4, figsize=(22, 5))
  for backend, resize in series:
    rs = sorted((r for r in results
                 if r['backend'] == backend and r['resize'] == resize),
                key=lambda r: r['num_envs'])
    if not rs:
      continue
    ns = [r['num_envs'] for r in rs]
    label = f'{backend}/{resize}'
    axes[0].plot(ns, [r['time_per_target'] for r in rs], 'o-', label=label)
    axes[1].plot(ns, [r['build_s'] for r in rs], 'o-', label=label)
    axes[2].plot(ns, [r['total_per_target'] for r in rs], 'o-', label=label)
    axes[3].plot(ns, [r['fps'] for r in rs], 'o-', label=label)
  axes[0].set_ylabel(f'wall time to collect {target} steps (s)')
  axes[0].set_title('Collection time vs N')
  axes[1].set_ylabel('build / spawn time (s)')
  axes[1].set_title('Build time vs N')
  axes[2].set_ylabel(f'build + collect {target} steps (s)')
  axes[2].set_title('Total time vs N')
  axes[3].set_ylabel('fps (env-steps / s)')
  axes[3].set_title('FPS vs N')
  for ax in axes:
    ax.set_xlabel('number of envs (N)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)
  print(f'wrote {path}')


# --------------------------------------------------------------------------- #
def main():
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--backends', default='batchenv,vectorenv,vectorenv_raw',
                 help='comma list: batchenv,vectorenv,vectorenv_raw')
  p.add_argument('--resize', default='pil64,raw256',
                 help=f'comma list of resize modes: {",".join(RESIZE_MODES)}')
  p.add_argument('--sweep', default='1,2,4,8',
                 help='comma list of N (number of parallel envs)')
  p.add_argument('--target', type=int, default=5000)
  p.add_argument('--warmup', type=int, default=1)
  p.add_argument('--repeats', type=int, default=1)
  p.add_argument('--gpu', type=int, default=0)
  p.add_argument('--outdir', default=str(OUT))
  args = p.parse_args()

  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  backends = [b.strip() for b in args.backends.split(',') if b.strip()]
  modes = [m.strip() for m in args.resize.split(',') if m.strip()]
  sweep = [int(x) for x in args.sweep.split(',')]
  outdir = pathlib.Path(args.outdir)
  outdir.mkdir(exist_ok=True)

  print(f'backends={backends}  resize={modes}  sweep={sweep}  '
        f'target={args.target}  gpu={args.gpu}')

  results = []
  for backend in backends:
    for mode in modes:
      # raw habitat has no PIL stage, so pil64 == raw256 for it. Skip pil64
      # only when raw256 is also requested (otherwise it's a real duplicate);
      # if raw256 is absent, still run vectorenv_raw once under pil64.
      if (backend == 'vectorenv_raw' and mode == 'pil64'
          and 'raw256' in modes):
        print(f'(skip {backend}/{mode}: identical to {backend}/raw256)')
        continue
      print(f'\n=== {backend} / resize={mode} ===')
      print(f'{"N":>4}  {"build(s)":>8}  {"collect(s)":>10}  {"total(s)":>8}  '
            f'{"fps":>9}')
      for n in sweep:
        try:
          r = run_one(backend, mode, n, args.target, args.warmup,
                      args.repeats)
        except Exception as e:  # noqa: BLE001
          print(f'{n:>4}  !! failed: {type(e).__name__}: {e}')
          continue
        results.append(r)
        print(f'{n:>4}  {r["build_s"]:>8.3f}  {r["time_per_target"]:>10.3f}  '
              f'{r["total_per_target"]:>8.3f}  {r["fps"]:>9.1f}')

  if not results:
    print('\nNo results collected.')
    return
  write_csv(results, outdir / 'results.csv')
  plot(results, args.target, outdir / 'vln_backends.png')


if __name__ == '__main__':
  main()
