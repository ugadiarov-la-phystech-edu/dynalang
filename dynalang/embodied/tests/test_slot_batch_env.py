import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
from embodied.envs.dummy import Dummy


class _StatefulExtractor(embodied.SlotExtractor):

  @property
  def n_slots(self):
    return 1

  @property
  def dim(self):
    return 1

  def __call__(self, images, previous_slots=None):
    batch = len(images)
    if previous_slots is None:
      count = np.ones((batch, 1), np.int32)
    else:
      count = previous_slots['count'] + 1
    slots = count.astype(np.float32)[..., None]
    state = {
        'count': count,
        'history': np.repeat(count[:, :, None], 3, axis=2),
    }
    return slots, state


def _actions(reset):
  return {
      'action': np.zeros(2, np.int32),
      'reset': np.asarray(reset, bool),
  }


def test_opaque_state_survives_asynchronous_resets():
  envs = [
      Dummy('disc', size=(4, 4), length=10),
      Dummy('disc', size=(4, 4), length=10),
  ]
  env = embodied.BatchSlotExtractorEnv(
      envs, parallel=False, slot_extractor=_StatefulExtractor(),
      use_previous_slots=True, initialize_twice=False)

  first = env.step(_actions([True, True]))
  second = env.step(_actions([False, False]))
  mixed = env.step(_actions([True, False]))

  np.testing.assert_array_equal(first['slot'][:, 0, 0], [1, 1])
  np.testing.assert_array_equal(second['slot'][:, 0, 0], [2, 2])
  np.testing.assert_array_equal(mixed['slot'][:, 0, 0], [1, 3])
  np.testing.assert_array_equal(env._previous_state['count'][:, 0], [1, 3])
  assert env._previous_state['history'].shape == (2, 1, 3)
