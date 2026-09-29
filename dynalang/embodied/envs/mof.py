"""Multi-object Fetch suite for dynalang (pixels + HomeGrid-style language).

Task ids match the gym registrations in multi-object-fetch-with-text, plus
SOLD's range form:

  ReachMulticolor_2Distractors_Dense-v1
  ReachMulticolor_0to4Distractors_Dense-v1
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Sequence, Tuple

import gym
import numpy as np
from gym import spaces


_ENVIRONMENTS = ("Reach", "Push", "Pick")
_REWARDS = ("Dense", "Sparse")


def parse_mof_name(name: str) -> Dict:
  """Parse ``ReachMulticolor_0to4Distractors_Dense-v1`` into env ids."""
  raw = name
  if name.endswith("-v1"):
    name = name[:-3]
  parts = name.split("_")
  if len(parts) != 3 or not parts[1].endswith("Distractors"):
    raise ValueError(
        "Expected '{Reach|Push|Pick}{Task}_{n|min}to{max}Distractors_"
        f"{{Dense|Sparse}}-v1', got {raw!r}"
    )
  env_task, dist_part, reward = parts
  if reward not in _REWARDS:
    raise ValueError(f"Unknown reward type {reward!r} in {raw!r}")

  spec = dist_part[: -len("Distractors")]
  if "to" in spec:
    lo, hi = spec.split("to", 1)
    min_d, max_d = int(lo), int(hi)
  else:
    min_d = max_d = int(spec)
  if min_d < 0 or max_d < min_d:
    raise ValueError(f"Invalid distractor range {spec!r} in {raw!r}")

  environment, task = _split_env_task(env_task)
  ids = []
  for n in range(min_d, max_d + 1):
    if not _valid_num_distractors(task, n):
      continue
    ids.append(f"{env_task}_{n}Distractors_{reward}-v1")
  if not ids:
    raise ValueError(
        f"No registered distractor counts in {raw!r} "
        f"(Odd needs n>=2, OddGroups n>=4)"
    )
  return {
      "env_task": env_task,
      "environment": environment,
      "task": task,
      "min_distractors": min_d,
      "max_distractors": max_d,
      "reward": reward,
      "ids": ids,
  }


def _split_env_task(env_task: str) -> Tuple[str, str]:
  for environment in _ENVIRONMENTS:
    if env_task.startswith(environment):
      task = env_task[len(environment) :]
      if task:
        return environment, task
  raise ValueError(
      f"Task must start with Reach, Push, or Pick, got {env_task!r}"
  )


def _valid_num_distractors(task: str, n: int) -> bool:
  if task.startswith("Odd") and n < 2:
    return False
  if task == "OddGroups" and n < 4:
    return False
  return True


class _ActionRepeat(gym.Wrapper):

  def __init__(self, env: gym.Env, repeat: int) -> None:
    super().__init__(env)
    self._repeat = int(repeat)
    if self._repeat < 1:
      raise ValueError(f"repeat must be >= 1, got {repeat}")

  def step(self, action):
    total = 0.0
    for _ in range(self._repeat):
      obs, reward, done, info = self.env.step(action)
      total += float(reward)
      if done:
        break
    return obs, total, done, info


class _TimeLimit(gym.Wrapper):

  def __init__(self, env: gym.Env, max_steps: int) -> None:
    super().__init__(env)
    self._max_steps = int(max_steps)
    self._elapsed = 0

  def reset(self, **kwargs):
    self._elapsed = 0
    return self.env.reset(**kwargs)

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._elapsed += 1
    info = dict(info)
    if self._elapsed >= self._max_steps:
      info["TimeLimit.truncated"] = not done
      done = True
    return obs, reward, done, info


class _VariableDistractors(gym.Wrapper):

  def __init__(self, env: gym.Env, factory: Callable[[], gym.Env]) -> None:
    super().__init__(env)
    self._factory = factory

  def reset(self, **kwargs):
    try:
      self.env.close()
    except Exception:
      pass
    self.env = self._factory()
    return self.env.reset(**kwargs)


class _PixelLangObs(gym.Wrapper):
  """Replace privileged state with a uint8 image; keep streamed language."""

  def __init__(
      self,
      env: gym.Env,
      image_size: Sequence[int],
      use_language: bool,
      vis: bool,
  ) -> None:
    super().__init__(env)
    self._size = tuple(int(x) for x in image_size)
    self._use_language = bool(use_language)
    self._vis = bool(vis)
    image_space = spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    obs_spaces = {
        "image": image_space,
        "log_success": spaces.Box(0, 1, (), dtype=np.float32),
    }
    if self._use_language:
      base = env.observation_space.spaces
      obs_spaces["token"] = base["token"]
      obs_spaces["is_read_step"] = base["is_read_step"]
    if self._vis:
      obs_spaces["log_image"] = image_space
    self.observation_space = spaces.Dict(obs_spaces)

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    info = dict(getattr(self.env, "_last_info", {}) or {})
    return self._obs(obs, info)

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    info = dict(info)
    info["is_terminal"] = bool(info.get("unstable", False))
    return self._obs(obs, info), reward, done, info

  def render(self, mode="rgb_array", **kwargs):
    kwargs.setdefault("size", self._size)
    return self.env.render(mode=mode, **kwargs)

  def _obs(self, obs, info):
    image = self.env.render(mode="rgb_array", size=self._size)
    image = np.asarray(image, dtype=np.uint8)
    packed = {
        "image": image,
        "log_success": np.float32(bool(info.get("success", False))),
    }
    if self._use_language:
      packed["token"] = np.asarray(obs["token"], dtype=np.uint32)
      packed["is_read_step"] = np.asarray(obs["is_read_step"], dtype=bool)
    if self._vis:
      from multi_object_fetch.utils.viz import overlay_language

      packed["log_image"] = overlay_language(
          image,
          instruction=str(info.get("instruction", getattr(self.env, "instruction", ""))),
          current_piece=str(info.get("log_token_piece", "")),
          token=int(obs["token"]) if self._use_language else None,
      )
    return packed


class MOF:
  """Gym-compatible env; ``wrap_env`` then applies ``FromGym``."""

  def __init__(
      self,
      task: str,
      size: Sequence[int] = (64, 64),
      repeat: int = 2,
      max_steps: int = 50,
      use_language: bool = True,
      vis: bool = False,
      repeat_task_every: int = 20,
  ) -> None:
    if "MUJOCO_GL" not in os.environ:
      os.environ["MUJOCO_GL"] = "egl"
    import multi_object_fetch  # noqa: F401  # gym registrations

    self._parsed = parse_mof_name(task)
    self._size = tuple(int(x) for x in size)
    self._repeat = int(repeat)
    self._max_steps = int(max_steps)
    self._use_language = bool(use_language)
    self._vis = bool(vis)
    self._repeat_task_every = int(repeat_task_every)

    ids: List[str] = self._parsed["ids"]
    if len(ids) == 1:
      self._env = self._make_gym(ids[0])
    else:
      factory = lambda: self._make_gym(random.choice(ids))
      self._env = _VariableDistractors(factory(), factory)

    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    try:
      from . import from_gym
    except ImportError:
      from embodied.envs import from_gym
    self.wrappers = [from_gym.FromGym]

  def _make_gym(self, env_id: str) -> gym.Env:
    try:
      env = gym.make(env_id, disable_env_checker=True)
    except TypeError:
      env = gym.make(env_id)
    env = env.unwrapped
    if self._repeat > 1:
      env = _ActionRepeat(env, self._repeat)
    if self._max_steps > 0:
      env = _TimeLimit(env, self._max_steps)
    if self._use_language:
      from multi_object_fetch.language import LanguageWrapper
      env = LanguageWrapper(
          env,
          repeat_task_every=self._repeat_task_every,
          visualize=False,
      )
    return _PixelLangObs(env, self._size, self._use_language, self._vis)

  def reset(self):
    return self._env.reset()

  def step(self, action):
    return self._env.step(action)

  def render(self, mode="rgb_array", **kwargs):
    kwargs.setdefault("size", self._size)
    return self._env.render(mode=mode, **kwargs)

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass
