import embodied
import numpy as np

from PIL import Image, ImageFont, ImageDraw


class HomeGrid(embodied.Env):

  def __init__(
    self,
    task,
    size=(64, 64),
    max_steps=100,
    num_trashobjs=2,
    num_trashcans=2,
    p_teleport=0.05,
    p_unsafe=0.,
    fixed_state=None,
    vis=False,
    use_object_slots=False,
    num_slots=10,
    slot_tile_size=32,
  ):
    from . import from_gym
    import homegrid
    import gym
    assert task in ("task", "future", "dynamics", "corrections")
    env = gym.make(f"homegrid-{task}", 
                   disable_env_checker=True,
                   max_steps=max_steps,
                   num_trashobjs=num_trashobjs,
                   num_trashcans=num_trashcans,
                   p_teleport=p_teleport,
                   p_unsafe=p_unsafe,
                   fixed_state=fixed_state)
    if use_object_slots:
      env = SlotImageWrapper(
          env, num_slots=num_slots, tile_size=slot_tile_size, out_size=size)
    env = homegrid.wrappers.Gym26Wrapper(env)
    self._env = env
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    self.wrappers = [
      from_gym.FromGym,
      lambda e: embodied.wrappers.ResizeImage(e, size),
    ]
    self.vis = vis

  def reset(self):
    obs = self._env.reset()
    if self.vis:
      obs["log_image"] = self.render_with_text(obs["log_language_info"])
    return obs

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    if self.vis:
      obs["log_image"] = self.render_with_text(obs["log_language_info"])
    return obs, rew, done, info

  def render(self):
    return self._env.render(mode="rgb_array")

  def render_with_text(self, text):
    img = self._env.render(mode="rgb_array")
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (0, 0, 0))
    draw.text((0, 45), "Action: {}".format(self._env.prev_action), (0, 0, 0))
    img = np.asarray(img)
    return img

  def init_from_state(self, state):
    self._env.init_from_state(state)


class SlotImageWrapper:
  """Produces a fixed number of per-object masked RGB images ("slot images").

  Idea (simplified slot extractor, not SlotContrast):
    * take the ground-truth instance segmentation mask from the environment,
    * for every tracked object, mask the agent-POV RGB image so that only that
      object is visible (everything else is black),
    * stack these masked images into a fixed-size tensor `slot_image` of shape
      (num_slots, H, W, 3); slots with no object are zero-padded.

  Each object keeps the SAME slot index for the whole episode (tracked by the
  object's persistent python id via a global object tracker), so a slot image
  can be empty for some frames (object out of view) and reappear later in the
  same slot.

  The masking logic is ported from ocdreamer/collect_homegrid.py.

  Slot groups (semantic, fixed layout — ~8 slots max):
    scene     — wall + all floor types (tile/carpet/wood)
    agent     — agent sprite + direction arrow
    fixtures  — static furniture (sofa, table, fridge, …)
    trashcan  — all trash cans merged (Storage)
    trash     — one slot per trash type (bottle/fruit/papers/plates)
  """

  SCENE_CLASSES = frozenset({"wall", "tile", "carpet", "wood"})
  TRASH_TYPES = frozenset({"bottle", "fruit", "papers", "plates"})
  TRASHCAN_TYPES = frozenset({"recycling_bin", "trash_bin", "compost_bin"})
  # Pixel keys without an instance id (floors, walls).
  STUFF_CLASSES = frozenset({"background", *SCENE_CLASSES})
  SLOT_GROUP_NAMES = {
      "scene": "scene",
      "agent": "agent",
      "fixtures": "fixtures",
      "trashcan": "trashcan",
      "trash": "trash",
  }

  def __init__(self, env, num_slots=10, tile_size=32, out_size=(64, 64)):
    from homegrid.base import (
        FloorWithObject, Inanimate, Pickable, Storage, Wall,
        CENTERED_VIEW, USE_AGENT_TEXTURE, AGENT_TEXTURE,
    )
    from homegrid import rendering
    from homegrid.rendering import point_in_triangle, rotate_fn
    from gym import spaces

    try:
      from homegrid.wrappers import PART_TO_WHOLE, _COMPOSITE_GROUPS
    except ImportError:
      PART_TO_WHOLE = {
          "rugl": "rug", "rugr": "rug",
          "chairl": "chair", "chairr": "chair",
          "sofa_side": "sofa",
      }
      _COMPOSITE_GROUPS = {"rugl": "rug", "rugr": "rug"}

    self.env = env
    self.num_slots = num_slots
    self.tile_size = tile_size
    self.out_size = tuple(out_size)

    # Stash homegrid symbols so the per-frame builder can use them.
    self._hg = dict(
        FloorWithObject=FloorWithObject, Inanimate=Inanimate,
        Pickable=Pickable, Storage=Storage, Wall=Wall,
        CENTERED_VIEW=CENTERED_VIEW, USE_AGENT_TEXTURE=USE_AGENT_TEXTURE,
        AGENT_TEXTURE=AGENT_TEXTURE, rendering=rendering,
        point_in_triangle=point_in_triangle, rotate_fn=rotate_fn,
        PART_TO_WHOLE=PART_TO_WHOLE, _COMPOSITE_GROUPS=_COMPOSITE_GROUPS)

    self.action_space = env.action_space
    self.metadata = getattr(env, "metadata", {})

    slot_space = spaces.Box(
        low=0, high=255,
        shape=(num_slots, self.out_size[0], self.out_size[1], 3),
        dtype="uint8")
    self.observation_space = spaces.Dict(
        {**env.observation_space.spaces, "slot_image": slot_space})

    # Per-episode state.
    self._global_obj_tracker = {}
    self._slot_of_gid = {}
    self._next_slot = 0

  # -- gym plumbing ---------------------------------------------------------
  def __getattr__(self, name):
    return getattr(self.env, name)

  @property
  def unwrapped(self):
    return self.env.unwrapped

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)

  def reset(self, **kwargs):
    self._global_obj_tracker = {}
    self._slot_of_gid = {}
    self._next_slot = 0
    result = self.env.reset(**kwargs)
    if isinstance(result, tuple):
      obs, info = result
      return self._add_slots(obs), info
    return self._add_slots(result)

  def step(self, action):
    result = self.env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
      return self._add_slots(obs), reward, terminated, truncated, info
    obs, reward, done, info = result
    return self._add_slots(obs), reward, done, info

  # -- slot image construction ---------------------------------------------
  def _add_slots(self, obs):
    obs = dict(obs)
    obs["slot_image"] = self._build_slot_images(obs["image"])
    return obs

  def _resize(self, img):
    if img.shape[:2] == self.out_size:
      return img
    pil = Image.fromarray(img)
    pil = pil.resize((self.out_size[1], self.out_size[0]), Image.NEAREST)
    return np.asarray(pil)

  def _build_slot_images(self, rgb_image):
    seg, gid_to_info = self._build_instance_mask()
    h_px, w_px = seg.shape
    # Align the RGB image to the mask resolution if necessary.
    if rgb_image.shape[:2] != (h_px, w_px):
      pil = Image.fromarray(rgb_image).resize((w_px, h_px), Image.NEAREST)
      rgb_image = np.asarray(pil)

    slots = np.zeros(
        (self.num_slots, self.out_size[0], self.out_size[1], 3), dtype=np.uint8)

    for gid, (name, pix_ids) in gid_to_info.items():
      if isinstance(pix_ids, (int, np.integer)):
        pix_ids = [int(pix_ids)]
      binary = np.zeros(seg.shape, dtype=bool)
      for pix_id in pix_ids:
        binary |= seg == pix_id
      if not binary.any():
        continue
      if gid not in self._slot_of_gid:
        if self._next_slot >= self.num_slots:
          continue  # no free slot, drop the object
        self._slot_of_gid[gid] = self._next_slot
        self._next_slot += 1
      slot_idx = self._slot_of_gid[gid]
      masked = rgb_image * binary[..., None]
      slots[slot_idx] = self._resize(masked.astype(np.uint8))

    return slots

  def _update_global_object_tracker(self):
    """Register any new world objects so they get a stable integer id."""
    hg = self._hg
    Wall = hg["Wall"]
    FloorWithObject = hg["FloorWithObject"]
    env = self.unwrapped
    grid = env.grid
    tracker = self._global_obj_tracker
    for j in range(grid.height):
      for i in range(grid.width):
        cell = grid.get(i, j)
        if cell is not None and not isinstance(cell, Wall):
          if id(cell) not in tracker:
            tracker[id(cell)] = len(tracker)
        floor = grid.get_floor(i, j)
        if floor is not None and isinstance(floor, FloorWithObject):
          if id(floor) not in tracker:
            tracker[id(floor)] = len(tracker)
    if env.carrying is not None and id(env.carrying) not in tracker:
      tracker[id(env.carrying)] = len(tracker)

  def _alpha_mask(self, texture):
    hg = self._hg
    ts = self.tile_size
    tex = hg["rendering"].resize(texture, (ts, ts))
    if tex.shape[-1] == 4:
      return tex[:, :, 3] == 255
    return np.ones((ts, ts), dtype=bool)

  def _gid_for(self, instance_key, tracker):
    if isinstance(instance_key, tuple):
      return instance_key
    if instance_key in tracker:
      return tracker[instance_key]
    return instance_key

  def _slot_group(self, name, instance_key, tracker):
    """Map an object to a stable semantic slot group key."""
    if name in self.SCENE_CLASSES:
      return ("scene",)
    if name == "agent":
      return ("agent",)
    if name in self.TRASHCAN_TYPES:
      return ("trashcan",)
    if name in self.TRASH_TYPES:
      return ("trash", name)
    return ("fixtures",)

  def _group_display_name(self, slot_group):
    kind = slot_group[0]
    if kind == "trash":
      return slot_group[1]
    return self.SLOT_GROUP_NAMES.get(kind, kind)

  def _obj_texture(self, obj):
    hg = self._hg
    if isinstance(obj, hg["Storage"]):
      return obj.textures[obj.state]
    if isinstance(obj, hg["Pickable"]):
      return None if obj.invisible else obj.texture
    if isinstance(obj, hg["Inanimate"]):
      return obj.texture
    return None

  def _build_composite_keys(self, grid, h, w):
    hg = self._hg
    FloorWithObject = hg["FloorWithObject"]
    groups = hg["_COMPOSITE_GROUPS"]
    cell_group = {}
    for j in range(h):
      for i in range(w):
        floor = grid.get_floor(i, j)
        if floor is not None and isinstance(floor, FloorWithObject) and "_" in floor.name:
          _, obj_name = floor.name.split("_", 1)
          if obj_name in groups:
            cell_group[(i, j)] = groups[obj_name]
    visited = set()
    composite_key = {}
    for pos in sorted(cell_group):
      if pos in visited:
        continue
      group = cell_group[pos]
      component = []
      queue = [pos]
      while queue:
        p = queue.pop()
        if p in visited or cell_group.get(p) != group:
          continue
        visited.add(p)
        component.append(p)
        x, y = p
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          nx, ny = x + dx, y + dy
          if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
            queue.append((nx, ny))
      key = min(component)
      for p in component:
        composite_key[p] = key
    return composite_key

  def _build_instance_mask(self):
    """Build an agent-POV instance segmentation mask.

    Returns:
      seg: (H*ts, W*ts) uint8 mask, 0 = background.
      gid_to_info: {slot_group -> (display_name, [pixel_ids])}.  Each slot
                   group may cover several seg pixel ids (OR'd in slot images).
    """
    hg = self._hg
    (FloorWithObject, Inanimate, Pickable, Storage, Wall) = (
        hg["FloorWithObject"], hg["Inanimate"], hg["Pickable"],
        hg["Storage"], hg["Wall"])
    CENTERED_VIEW = hg["CENTERED_VIEW"]
    USE_AGENT_TEXTURE = hg["USE_AGENT_TEXTURE"]
    AGENT_TEXTURE = hg["AGENT_TEXTURE"]
    PART_TO_WHOLE = hg["PART_TO_WHOLE"]
    point_in_triangle = hg["point_in_triangle"]
    rotate_fn = hg["rotate_fn"]

    self._update_global_object_tracker()
    tracker = self._global_obj_tracker

    env = self.unwrapped
    grid, vis_mask = env.gen_obs_grid()
    h, w = grid.height, grid.width
    if CENTERED_VIEW:
      agent_pos = (grid.width // 2, grid.height // 2)
    else:
      agent_pos = (grid.width // 2, grid.height - 1)

    world_grid = env.grid
    world_composite_key = self._build_composite_keys(
        world_grid, world_grid.height, world_grid.width)
    world_composite_to_key = {}
    for wj in range(world_grid.height):
      for wi in range(world_grid.width):
        world_floor = world_grid.get_floor(wi, wj)
        if world_floor is not None and isinstance(world_floor, FloorWithObject):
          if (wi, wj) in world_composite_key:
            world_composite_to_key[id(world_floor)] = world_composite_key[(wi, wj)]

    def obj_key(cell, floor):
      obj = cell if cell is not None else floor
      if obj is None:
        return None
      if floor is not None and isinstance(floor, FloorWithObject):
        if id(floor) in world_composite_to_key:
          return ('composite', world_composite_to_key[id(floor)])
      return id(obj)

    ts = self.tile_size
    seg = np.zeros((h * ts, w * ts), dtype=np.uint8)

    name_to_id = {}
    next_id = [1]
    group_to_pix = {}
    group_names = {}
    all_textures = env.textures

    def get_id(name, instance_key=None):
      name = PART_TO_WHOLE.get(name, name)
      if name in self.STUFF_CLASSES or instance_key is None:
        mask_key = name
      else:
        mask_key = (name, self._gid_for(instance_key, tracker))
      if mask_key not in name_to_id:
        name_to_id[mask_key] = next_id[0]
        next_id[0] += 1
      pix_id = name_to_id[mask_key]
      slot_group = self._slot_group(name, instance_key, tracker)
      if slot_group not in group_to_pix:
        group_to_pix[slot_group] = []
        group_names[slot_group] = self._group_display_name(slot_group)
      if pix_id not in group_to_pix[slot_group]:
        group_to_pix[slot_group].append(pix_id)
      return pix_id

    for j in range(h):
      for i in range(w):
        cell = grid.get(i, j)
        floor = grid.get_floor(i, j)
        tile = seg[j * ts:(j + 1) * ts, i * ts:(i + 1) * ts]

        if floor is not None:
          if isinstance(floor, FloorWithObject) and "_" in floor.name:
            base_name, obj_name = floor.name.split("_", 1)
            tile[:] = get_id(base_name)
            fkey = obj_key(None, floor)
            if obj_name in all_textures:
              alpha = self._alpha_mask(all_textures[obj_name])
              tile[alpha] = get_id(obj_name, instance_key=fkey)
            else:
              tile[:] = get_id(floor.name, instance_key=fkey)
          else:
            tile[:] = get_id(floor.name)

        if cell is not None and not isinstance(cell, Wall):
          tex = self._obj_texture(cell)
          okey = obj_key(cell, None)
          if tex is not None:
            alpha = self._alpha_mask(tex)
            tile[alpha] = get_id(cell.name, instance_key=okey)
          elif not (isinstance(cell, Pickable) and cell.invisible):
            tile[:] = get_id(cell.name, instance_key=okey)

        if cell is not None and isinstance(cell, Wall):
          tile[:] = get_id("wall")

        if (i, j) == agent_pos:
          carried = env.carrying
          if carried is not None:
            tex = self._obj_texture(carried)
            ckey = id(carried)
            if tex is not None:
              alpha = self._alpha_mask(tex)
              tile[alpha] = get_id(carried.name, instance_key=ckey)
            else:
              tile[:] = get_id(carried.name, instance_key=ckey)

        if (i, j) == agent_pos:
          agent_id = get_id("agent", instance_key="agent")
          if USE_AGENT_TEXTURE:
            alpha = self._alpha_mask(AGENT_TEXTURE)
            tile[alpha] = agent_id
          else:
            tile[:] = agent_id
          agent_dir = env.agent_dir
          if CENTERED_VIEW:
            tri_fn = point_in_triangle((0.65, 0.29), (0.87, 0.50), (0.65, 0.71))
          else:
            tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
          tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5,
                             theta=0.5 * 3.141592653589793 * agent_dir)
          for py in range(ts):
            for px in range(ts):
              if tri_fn((px + 0.5) / ts, (py + 0.5) / ts):
                tile[py, px] = agent_id

    gid_to_info = {
        slot_group: (group_names[slot_group], pix_ids)
        for slot_group, pix_ids in group_to_pix.items()
        if slot_group[0] != "background"
    }
    return seg, gid_to_info
