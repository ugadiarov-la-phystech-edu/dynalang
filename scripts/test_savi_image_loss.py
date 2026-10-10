"""Check real-checkpoint decoder parity, input gradients and WM integration."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'dynalang'), str(ROOT / 'scripts')]

import jax
import jax.numpy as jnp
import numpy as np
import torch

from embodied.torch.ocr.savi import SAViExtractor
from dynalang.savi_decoder import FrozenSAViDecoder

torch.set_num_threads(2)
CHECKPOINT = ROOT / 'checkpoints/savi/reach_red.ckpt'


class DecoderTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.extractor = SAViExtractor('', str(CHECKPOINT), 64, 'cpu')
    cls.decoder = FrozenSAViDecoder.from_torch(cls.extractor.decoder)
    images = np.load(ROOT / 'artifacts/savi_mof/frames.npz')['image'][:1]
    cls.slots, _ = cls.extractor.get_slots(images)
    cls.images = images.astype(np.float32) / 255

  def test_rgb_and_slot_gradient_match_torch(self):
    slots = torch.tensor(self.slots, requires_grad=True)
    rgb, masks = self.extractor.decoder(slots)
    expected = (rgb * masks).sum(1).permute(0, 2, 3, 1)
    target = torch.tensor(self.images)
    loss = (expected - target).square().mean()
    loss.backward()
    actual = jax.jit(self.decoder)(jnp.asarray(self.slots))
    actual_loss, gradient = jax.jit(jax.value_and_grad(
        lambda s: self.decoder.mse(s, jnp.asarray(self.images)).mean()))(
            jnp.asarray(self.slots))
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(actual_loss, loss.detach().numpy(), atol=2e-7, rtol=1e-4)
    np.testing.assert_allclose(gradient, slots.grad.numpy(), atol=2e-7, rtol=2e-3)
    self.assertGreater(float(jnp.linalg.norm(gradient)), 0)
    self.assertTrue(all(not p.requires_grad for p in self.extractor.parameters()))
    print('RGB max error:', float(np.max(np.abs(actual - expected.detach().numpy()))),
          'slot gradient norm:', float(jnp.linalg.norm(gradient)), flush=True)

  def test_world_model_image_loss_updates_projection(self):
    import embodied
    from dynalang.agent import Agent
    from dynalang import ninjax as nj
    from test_savi_encoder import make_config, FrameEnv
    config = make_config().update({'batch_size': 1, 'batch_length': 2,
        'run.pretrain_wm_only': True,
        'savi_image_loss.scale': 1.0, 'savi_image_loss.starts': 1})
    frames = np.repeat((self.images * 255).astype(np.uint8), 2, axis=0)
    slots = np.repeat(self.slots, 2, axis=0)
    env = FrameEnv(frames, np.ones(2, bool))
    spaces = {**env.obs_space, 'slot': embodied.Space(np.float32, (7, 128))}
    agent = Agent(spaces, env.act_space, embodied.Counter(), config)
    data = dict(image=frames[None], slot=slots[None],
        token=np.zeros((1, 2), np.int32), action=np.zeros((1, 2, 4), np.float32),
        reward=np.zeros((1, 2), np.float32), is_first=np.array([[True, False]]),
        is_last=np.zeros((1, 2), bool), is_terminal=np.zeros((1, 2), bool))
    converted = agent._convert_inps(data, agent.train_devices)
    def update(data):
      data = agent.agent.preprocess(data)
      return agent.agent.wm.train(data, agent.agent.wm.initial(1))
    before = agent.varibs
    (state, outputs, metrics), after = jax.jit(nj.pure(update))(
        before, jax.random.PRNGKey(4), converted)
    self.assertTrue(np.isfinite(float(metrics['savi_image_mse'])))
    keys = [k for k in before if '/dec/slot_proj/' in k and '/opt/' not in k]
    self.assertTrue(keys)
    self.assertTrue(any(not np.array_equal(before[k], after[k]) for k in keys))
    self.assertFalse(any('savi_decoder' in k for k in after))
    print('WM SAVi image MSE:', float(metrics['savi_image_mse']), flush=True)


if __name__ == '__main__':
  unittest.main()
