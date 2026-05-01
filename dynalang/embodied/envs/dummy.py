import embodied
import numpy as np


class Dummy(embodied.Env):

  def __init__(self, task, mode="train", size=(64, 64), length=100, with_text=False):
    assert task in ('cont', 'disc')
    self._task = task
    self._size = size
    self._length = length
    self._step = 0
    self._done = False
    self._with_text = with_text
    if self._with_text:
      self._text_dim = 512
      self._num_tokens = 10  
      self._reading = False
      self._read_step = 0
      self._token_embeds = None
      self._tokens = None

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'vector': embodied.Space(np.float32, (7,)),
        'step': embodied.Space(np.int32, (), 0, self._length),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
    if self._with_text:
      spaces.update({
          'is_read_step': embodied.Space(bool),
          'token': embodied.Space(np.uint32, ()),
          'token_embed': embodied.Space(np.float32, (self._text_dim,)),
      })
    return spaces

  @property
  def act_space(self):
    if self._task == 'cont':
      space = embodied.Space(np.float32, (6,))
    else:
      space = embodied.Space(np.int32, (), 0, 5)
    return {'action': space, 'reset': embodied.Space(bool)}

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      obs = self._obs(0.0, is_first=True)
      if self._with_text:
        self._reading = True
        self._read_step = 0
        self._token_embeds = [np.random.randn(self._text_dim).astype(np.float32) 
                              for _ in range(self._num_tokens)]
        self._tokens = np.arange(100, 100 + self._num_tokens, dtype=np.uint32)
        obs['is_read_step'] = True
        obs['token'] = self._tokens[self._read_step]
        obs['token_embed'] = self._token_embeds[self._read_step]
        self._read_step += 1
      return obs
    # Handle reading phase (similar to messenger)
    if self._with_text and self._reading:
      obs = self._obs(0.0, is_first=(self._step == 0))
      obs['is_read_step'] = True
      obs['token'] = self._tokens[self._read_step]
      obs['token_embed'] = self._token_embeds[self._read_step]
      self._read_step += 1
      if self._read_step >= self._num_tokens:
        self._reading = False
        self._read_step = 0
      return obs
    
    action = action['action']
    if self._task == 'cont':
      pass
      # assert (-1 <= action).all() and (action <= 1).all(), action
    else:
      assert action in range(5), action
    self._step += 1
    self._done = (self._step >= self._length)
    return self._obs(1.0, is_last=self._done, is_terminal=self._done)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=np.zeros(self._size + (3,), np.uint8),
        vector=np.zeros(7, np.float32),
        step=self._step,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        language_info="test string",
    )
    if self._with_text:
      # Empty token for non-reading steps
      obs['is_read_step'] = False
      obs['token'] = np.uint32(0)  
      obs['token_embed'] = np.zeros(self._text_dim, dtype=np.float32)
    return obs


class DummySlot(embodied.Env):

  def __init__(self, task, mode="train", num_slots=8, slot_dim=64, length=100):
    assert task in ('cont', 'disc')
    self._task = task
    self._num_slots = num_slots
    self._slot_dim = slot_dim
    self._length = length
    self._step = 0
    self._done = False

  @property
  def obs_space(self):
    return {
        'slot': embodied.Space(np.float32, (self._num_slots, self._slot_dim)),
        'step': embodied.Space(np.int32, (), 0, self._length),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    if self._task == 'cont':
      space = embodied.Space(np.float32, (6,))
    else:
      space = embodied.Space(np.int32, (), 0, 5)
    return {'action': space, 'reset': embodied.Space(bool)}

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      return self._obs(0.0, is_first=True)
    action = action['action']
    if self._task == 'cont':
      pass
    else:
      assert action in range(5), action
    self._step += 1
    self._done = (self._step >= self._length)
    return self._obs(1.0, is_last=self._done, is_terminal=self._done)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        slot=np.random.randn(self._num_slots, self._slot_dim).astype(np.float32),
        step=self._step,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
