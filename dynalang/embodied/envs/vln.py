import embodied
import numpy as np
from VLN_CE.vlnce_baselines.config.default import get_config
from habitat_lab.habitat_baselines.utils.env_utils import make_env_fn
from habitat_lab.habitat_baselines.common.environments import get_env_class
import os
import random
from PIL import Image, ImageFont, ImageDraw
import pickle

class VLNEnv(embodied.Env):

  def __init__(
    self,
    task=None,
    mode='train',
    size=(64, 64),
    length=500,
    use_text=True,
    use_depth=False,
    use_semantic=False,
    load_embeddings=True,
    dataset='train',
    # For training with expert demos (unused in final version)
    use_expert=0,
    min_use_expert=0,
    anneal_expert_eps=0,
    # Reward for successful episode
    success_reward=1000,
    # Reward if STOP action executed too early
    early_stop_penalty=0,
    # Whether to include additional language beyond instruction in obs
    # (unused in final version)
    use_descriptions=False,
    desc_length=50,
    seed=None,
    gpu_id=0,
    # Annotate log_image every N env steps; 0=never, -1=always. Default from
    # run.log_every is set in train.make_env when this is omitted from config.
    log_image_every=None,
    # If True, downscale RGB/depth to ``size`` in preprocess (PIL). If False,
    # keep the raw simulator-sensor resolution (no PIL resize).
    resize=True,
    # Optional override for the simulator RGB/DEPTH sensor resolution (square).
    # None keeps the value from vlnce_task.yaml (256). Useful to benchmark
    # rendering at the target resolution directly instead of downscaling later.
    sensor_size=None,
  ):
    assert mode in dataset, "Mismatched env mode and dataset"

    self._task = 'cont'
    self._size = size
    self._resize = resize
    self._sensor_size = sensor_size
    self._length = length
    self._step = 0
    self._done = False
    self._mode = mode
    self._use_text = use_text
    self._use_depth = use_depth
    self._use_semantic = use_semantic
    self._load_embeddings = load_embeddings
    self._use_expert = use_expert
    self._use_descriptions = use_descriptions
    self._desc_length = desc_length
    self._min_use_expert = min_use_expert
    self._anneal_expert_eps = anneal_expert_eps
    self._success_reward = success_reward
    self._early_stop_penalty = early_stop_penalty
    # Reading timestep before start of episode
    self.read_step = 0
    # True if we have finished reading the first text input (the whole instr)
    self.done_first_input = False
    # Type of text we are currently inputting ('instr' or 'desc')
    self.cur_text_type = 'instr'
    # Text string currently being streamed
    self.cur_text = ''
    # Number of episodes (for annealing expert episodes if using demos)
    self._num_eps = 0
    self._disc_act_space = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    self._log_image_every = 0 if log_image_every is None else log_image_every
    self._total_steps = 0

    if seed is None:
      seed = 42
    assert self._desc_length <= self._length
    
    config_opts = [
      'TASK_CONFIG.DATASET.SPLIT', dataset,
      'TASK_CONFIG.TASK.NDTW.SPLIT', dataset,
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE', mode == 'train',
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE', mode != 'train',
      'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE', mode == 'train',
    ]
    if mode == 'test':
      config_opts.extend([
        'TASK_CONFIG.TASK.SENSORS',
        ['INSTRUCTION_SENSOR']
      ])
      config_opts.extend([
        'TASK_CONFIG.TASK.MEASUREMENTS',
        ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'ORACLE_SUCCESS', 'NDTW', 'PATH_LENGTH']
      ])
    self.config = get_config(
      os.path.dirname(os.path.realpath(__file__)) + '/vln.yaml',
      opts=config_opts
    )
    self.config.defrost()
    self.config.SIMULATOR_GPU_IDS = [gpu_id]
    self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    if self._sensor_size is not None:
      self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = self._sensor_size
      self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = self._sensor_size
      self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = self._sensor_size
      self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = self._sensor_size
    if use_semantic:
      sensors = list(self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS)
      if 'SEMANTIC_SENSOR' not in sensors:
        sensors.append('SEMANTIC_SENSOR')
      self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = sensors
      self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = size[0]
      self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = size[1]
    self.config.freeze()
    self._env = make_env_fn(
      self.config,
      get_env_class(self.config.ENV_NAME)
    )

    if load_embeddings:
      with open(f"{os.path.dirname(__file__)}/data/vln_embeds_t5.pkl", "rb") as f:
        self.token_cache, self.embed_cache = pickle.load(f)
      self.empty_token_id = self.token_cache["<pad>"]
      self.empty_token_embed = self.embed_cache["<pad>"]
    else:
      self._init_models()

  def _init_models(self):
    """Initialize tokenizer and encoder for embedding online in the env."""
    self.token_cache = {}
    self.embed_cache = {}
    from transformers import T5Tokenizer, T5EncoderModel
    self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
    self.empty_token_id = self.tokenizer.pad_token_id
    self.encoder = T5EncoderModel.from_pretrained("t5-small")
    self.empty_token_embed = self._embed("<pad>")[0][0]    
 
  @property
  def obs_space(self):
    spaces = {k: embodied.Space(v.dtype, v.shape)
              for k, v in self._env.observation_space.items()}
    new_space = {}
    # Output image size: downscaled ``size`` when resizing, else raw sensor res.
    img_hw = tuple(self._size) if self._resize else tuple(spaces['rgb'].shape[:2])
    new_space['image'] = embodied.Space(
      dtype=spaces['rgb'].dtype,
      shape=img_hw + (3,),
      low=np.zeros(img_hw + (3,), dtype=np.int8),
      high=255 * np.ones(img_hw + (3,), dtype=np.int8)
    )
    if self._use_depth:
      new_space['depth'] = new_space['image']

    if self._use_text:
      # use one field for instructions or description text
      new_space.update({
        "token": embodied.Space(
          low=0, high=32100,
          shape=(),
          dtype=np.uint32),
        "token_embed": embodied.Space(
          low=-np.inf, high=np.inf,
          shape=(512,),
          dtype=np.float32),
        "is_read_step": embodied.Space(
          low=np.array(False),
          high=np.array(True),
          shape=(),
          dtype=bool,
        )
      })

    new_space.update({
      f'log_{self._mode}_success': embodied.Space(np.float32),
      f'log_{self._mode}_pl_success': embodied.Space(np.float32),
      f'log_{self._mode}_oracle_success': embodied.Space(np.float32),
      f'log_image': new_space['image'],
      'reward': embodied.Space(np.float32),
      'is_last': embodied.Space(bool),
      'is_terminal': embodied.Space(bool),
      'is_first': embodied.Space(bool),
      'is_demo': embodied.Space(bool),
      'next_expert_ac': embodied.Space(np.int32, (), -1, len(self._disc_act_space))
    })
    return new_space

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, len(self._disc_act_space)),
        'reset': embodied.Space(bool),
    }

  @staticmethod
  def _action_index(action):
    """Discrete index; policy/driver may pass float (e.g. after is_last masking)."""
    return int(np.asarray(action).item())
  
  def step(self, action):
    if self._done or action['reset']:
      self._num_eps += 1 
      self._step = 0
      self.read_step = 0
      self.cur_text = ''
      self.cur_text_type = 'instr'
      self.tokens = [] # for logging
      self._done = False
      self.done_first_input = False
      ob = self._env.reset()
      self.prev_env_ob = ob

      if self._num_eps < self._anneal_expert_eps:
        self._expert_ep = np.random.rand() < self._use_expert - (self._use_expert - self._min_use_expert) / self._anneal_expert_eps * self._num_eps
      elif self._min_use_expert == self._use_expert: 
        self._expert_ep = np.random.rand() < self._use_expert
      else:
        self._expert_ep = np.random.rand() < self._min_use_expert
 
      log_traj_id = ob['instruction']['trajectory_id']
      ob = self.preprocess_obs(ob)
      ob.update({
        'reward': 0,
        'is_first': True,
        'is_last': self._done,
        'is_terminal': False,
        f'log_{self._mode}_success': 0,
        f'log_{self._mode}_pl_success': 0,
        f'log_{self._mode}_oracle_success': 0,
        f'log_language_info': self.cur_text,
        "is_read_step": not self.done_first_input,
        "is_demo": self._expert_ep,
      })
      ob['log_image'] = self._make_log_image(
        ob, self.cur_text, log_traj_id,
        self._disc_act_space[self._action_index(action['action'])],
        is_first=True, is_last=False,
      )

      if self._expert_ep:
        # need to get infos to get gt_actions
        self.next_expert_ac = self.prev_env_ob['shortest_path_sensor'][0]
        ob["next_expert_ac"] = self.next_expert_ac
      else:
        ob["next_expert_ac"] = -1

      self._total_steps += 1
      return ob

    # STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
    action = self._action_index(action['action'])
    
    if self.done_first_input:
      self._step += 1
      ob, rew, dones, infos = self._env.step(action)
      if action == 0: # STOP
        if infos['success']: 
          rew = self._success_reward
        else:
          rew = self._early_stop_penalty # stop action too early
      self.prev_env_ob = ob
    else:
      # Agent needs to listen to instruction first, frozen
      ob = self.prev_env_ob
      rew = 0
      dones = False
      infos = {'success': 0, 'spl': 0, 'oracle_success': 0}
    
    log_traj_id = ob['instruction']['trajectory_id']
    ob = self.preprocess_obs(ob)
    if self._expert_ep:
      self.next_expert_ac = self.prev_env_ob['shortest_path_sensor'][0]
      ob["next_expert_ac"] = self.next_expert_ac
    else:
      ob["next_expert_ac"] = -1
    self._done = (self._step >= self._length) or dones
    ob.update({
      'reward': rew,
      'is_first': False,
      'is_last': (self._step >= self._length) or self._done,
      'is_terminal': self._done,
      "is_read_step": not self.done_first_input,
      "is_demo": self._expert_ep,
      f'log_{self._mode}_success': infos['success'],
      f'log_{self._mode}_pl_success': infos['spl'],
      f'log_{self._mode}_oracle_success': infos['oracle_success'],
      f'log_language_info': self.cur_text,
    })
    is_last = (self._step >= self._length) or self._done
    ob['log_image'] = self._make_log_image(
      ob, self.cur_text, log_traj_id, self._disc_act_space[action],
      is_first=False, is_last=is_last,
    )
    self._total_steps += 1
    return ob

  def _make_log_image(self, ob, instr_text, traj_id, ac, is_first=False, is_last=False):
    """Cheap log_image by default; annotate only at log cadence or episode bounds."""
    if self._log_image_every < 0:
      return self.render_with_text(ob, instr_text, traj_id, ac)
    if self._log_image_every == 0:
      return ob['image'].copy()
    if is_first or is_last or (self._total_steps % self._log_image_every == 0):
      return self.render_with_text(ob, instr_text, traj_id, ac)
    return ob['image'].copy()

  def _embed(self, string):
    """Embed string with encoder or get from cache."""
    string = string.strip().replace('\n', ' ').replace('\r', '')
    
    if string not in self.embed_cache:
      print('Missing from cache!! String:', string)
      tokens = self.tokenizer(string, return_tensors="pt",
                              add_special_tokens=True)  # add </s> separators
      import torch
      with torch.no_grad():
        # (seq, dim)
        embeds = self.encoder(**tokens).last_hidden_state.squeeze(0)
      self.embed_cache[string] = embeds.cpu().numpy()
      self.token_cache[string] = tokens['input_ids'].squeeze(0).cpu().numpy()
    return (
      self.embed_cache[string],
      self.token_cache[string]
    )
   
  def get_embed_text(self, ob): 
    if len(self.tokens) > 0 and self.read_step >= len(self.tokens):
      self.read_step = 0
      self.done_first_input = True
      if self._use_descriptions  and len(ob['descriptions']) > 1 and self._step > 0:
        self.cur_text_type = 'instr' if self.cur_text_type == 'desc' else 'desc'
      else:
        self.cur_text_type = 'instr'

    if self.read_step == 0:
      # sample new text to feed in
      if self.cur_text_type == 'instr':
        self.cur_text = ob['instruction']['text']
      elif self.cur_text_type == 'desc':
        self.cur_text = random.choice(ob['descriptions'])
      else:
        raise NotImplementedError
      self.token_embeds = []
      self.tokens = [] # for logging

      # Remove padding 
      es, ts = self._embed(self.cur_text) # embed sentence
      self.token_embeds = [tok_e for tok_e in es]
      self.tokens = [tok for tok in ts]
      assert len(self.token_embeds) == len(self.tokens)

    # print(self.cur_text, self.read_step, self.tokens[self.read_step])
    new_ob = {
        "token": self.tokens[self.read_step],
        "token_embed": self.token_embeds[self.read_step],
      }
    self.read_step += 1
    return new_ob

  def preprocess_depth(self, depth):
    """Normalize and clip depth images."""
    depth = (np.clip(depth, 0, 5.0) / 5.0 * 255).astype(np.uint8) # Clip to 5m, convert to uint8
    depth = np.repeat(depth, 3, axis=-1)
    if self._resize:
      depth = Image.fromarray(depth)
      depth = depth.resize(self._size)
    depth = np.asarray(depth, dtype=np.uint8)
    return depth
      
  def preprocess_obs(self, ob):
    new_ob = {}
    rgb = ob['rgb']
    if self._resize:
      img = Image.fromarray(rgb)
      img = img.resize(self._size)
      new_ob['image'] = np.asarray(img, dtype=np.uint8)
    else:
      new_ob['image'] = np.asarray(rgb, dtype=np.uint8)
    if self._use_depth:
      new_ob['depth'] = self.preprocess_depth(ob['depth'])
    if self._use_text:
      new_ob.update(self.get_embed_text(ob))
    return new_ob
  
  def render_with_text(self, ob, instr_text, traj_id, ac):
    """Render policy image with debugging information."""
    img = Image.fromarray(ob['image'])
    draw = ImageDraw.Draw(img)
    # Define the maximum width of the text
    max_width = 256

    # Calculate the height of the text
    instr_text = 'Instruction: ' + instr_text
    instr_text = instr_text.strip().replace('\n', ' ').replace('\r', ' ')
    instr_text = instr_text.encode("ascii", "ignore").decode()
    try:
      text_width = draw.textlength(instr_text)
    except (AttributeError, ValueError):
      text_width, _ = draw.textsize(instr_text)
    if text_width == 0:
      text_width = len(instr_text) * 6
    max_len = int((max_width / text_width) * len(instr_text))
    wrapped_text = "\n".join([instr_text[i:i+max_len] for i in range(0, len(instr_text), max_len)])

    draw.text((0, 0), 'Trajectory ID: {}, Mode: {}'.format(traj_id, self._mode), (0, 0, 0))
    draw.text((0, 15), "Action: {}".format(ac), (0, 0, 0))
    draw.multiline_text((0, 30), wrapped_text, fill=(0, 0, 0))
    img = np.asarray(img).copy()

    # annotate videos
    if ob[f'log_{self._mode}_success']:
      img[:5, :, 1] =  255
    if ob[f'log_{self._mode}_oracle_success']:
      img[:5, :, 2] =  255
    
    img = np.clip(img, 0, 255)
    return img
