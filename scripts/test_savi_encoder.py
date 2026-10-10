"""Checkpoint-backed SAVi recurrence and OCTSSM integration tests.

Run: python scripts/test_savi_encoder.py
Uses artifacts/savi_mof/frames.npz when present, otherwise seeded RGB inputs.
No pytest dependency. Real-checkpoint tests deliberately fail if weights are absent.
"""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dynalang'))

import numpy as np
import torch
import embodied
from embodied.torch.ocr.savi import SAViExtractor

torch.set_num_threads(2)
CHECKPOINT = ROOT / 'checkpoints/savi/reach_red.ckpt'


def make_config():
    import ruamel.yaml as yaml
    presets = yaml.YAML(typ='safe').load((ROOT / 'dynalang/configs.yaml').read_text())
    config = embodied.Config(presets['defaults'])
    for name in ['mof_savi', 'min_debug']:
        config = config.update(presets[name])
    return config.update({
        'batch_size': 2, 'batch_length': 8, 'imag_horizon': 3, 'imag_starts': 4,
        'slot_extractor.device': 'cpu',
        'slot_extractor.checkpoint_path': str(CHECKPOINT),
        'jax': {'jit': True, 'platform': 'cpu', 'precision': 'float32',
                'prealloc': False, 'profiler': False, 'transfer_guard': False,
                'policy_devices': [0], 'train_devices': [0]},
        'octssm': {'deter': 16, 'units': 16, 'stoch': 4, 'classes': 4,
                   'tf_layers': 1, 'tf_heads': 2, 'feedforward_units': 32,
                   'tf_context_length': 4, 'pe': 'rope', 'kv_cache': True},
        'encoder.mlp_layers': 1, 'encoder.mlp_units': 32,
        'decoder.mlp_layers': 1, 'decoder.mlp_units': 32,
        'reward_head.mlp.layers': 1, 'reward_head.mlp.units': 32,
        'cont_head.mlp.layers': 1, 'cont_head.mlp.units': 32,
        'actor.mlp.layers': 1, 'actor.mlp.units': 32,
        'critic.mlp.layers': 1, 'critic.mlp.units': 32,
    })


class FrameEnv(embodied.Env):
    def __init__(self, images, resets):
        self.images, self.resets, self.index = images, resets, 0

    @property
    def obs_space(self):
        return {
            'image': embodied.Space(np.uint8, (64, 64, 3)),
            'token': embodied.Space(np.int32, (), 0, 16),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {'action': embodied.Space(np.float32, (4,), -1, 1),
                'reset': embodied.Space(bool)}

    def step(self, action):
        t = self.index
        self.index += 1
        return dict(image=self.images[t], token=np.int32(t % 16),
                    reward=np.float32(-0.1), is_first=np.bool_(self.resets[t]),
                    is_last=np.bool_(False), is_terminal=np.bool_(False))

    def close(self):
        pass


class SAViTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = SAViExtractor('', str(CHECKPOINT), 64, 'cpu')
        frames_file = ROOT / 'artifacts/savi_mof/frames.npz'
        if frames_file.is_file():
            frames = np.load(frames_file)['image'][:8]
            print(f'Using {len(frames)} real MOF frames', flush=True)
        else:
            frames = np.random.default_rng(0).integers(0, 256, (8, 64, 64, 3), np.uint8)
        cls.images = np.stack([frames, frames[::-1]])

    def reference(self, images, prior=None):
        frames = torch.tensor(images, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255
        actions = torch.zeros(frames.shape[0], frames.shape[1], 4)
        with torch.no_grad():
            return self.extractor.model.encode(frames, actions, prior).numpy()

    def test_checkpoint_and_factory(self):
        from dynalang.train import make_slot_extractor
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(digest, '9c937fee8edb9f5c9431cec3b4b911f28fd530242cc273d0a2ef90c3525031d8')
        extractor = make_slot_extractor(make_config())
        self.assertEqual((extractor.n_slots, extractor.dim), (7, 128))
        self.assertFalse(extractor.training)
        self.assertTrue(all(not p.requires_grad for p in extractor.parameters()))

    def test_sequence_matches_sold(self):
        expected = self.reference(self.images)
        state, outputs = None, []
        for t in range(self.images.shape[1]):
            slots, state = self.extractor.get_slots(self.images[:, t], state)
            outputs.append(slots)
        actual = np.stack(outputs, axis=1)
        error = np.abs(actual - expected).max()
        print(f'SOLD sequence max error: {error:.3g}', flush=True)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
        self.assertEqual(actual.shape, (2, 8, 7, 128))
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(np.abs(state - actual[:, -1]).max(), 0.01)
        # Restarting every frame must differ from temporal processing.
        cold, _ = self.extractor.get_slots(self.images[:, -1])
        self.assertGreater(np.abs(cold - actual[:, -1]).max(), 0.01)

    def test_chunk_continuation(self):
        prefix = self.reference(self.images[:, :4])
        expected = self.reference(self.images[:, 4:], torch.tensor(prefix[:, -1]))
        state = None
        for t in range(4):
            _, state = self.extractor.get_slots(self.images[:, t], state)
        tail = []
        for t in range(4, 8):
            slots, state = self.extractor.get_slots(self.images[:, t], state)
            tail.append(slots)
        np.testing.assert_allclose(np.stack(tail, 1), expected, atol=2e-5, rtol=1e-5)

    def test_mixed_batch_resets(self):
        resets = np.array([[1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1, 0, 0]], bool)
        env = embodied.BatchSlotExtractorEnv(
            [FrameEnv(self.images[i], resets[i]) for i in range(2)],
            False, self.extractor, use_previous_slots=True, initialize_twice=False)
        action = {'action': np.zeros((2, 4), np.float32), 'reset': np.zeros(2, bool)}
        actual = np.stack([env.step(action)['slot'] for _ in range(8)], 1)
        for b in range(2):
            boundaries = list(np.flatnonzero(resets[b])) + [8]
            for start, end in zip(boundaries, boundaries[1:]):
                expected = self.reference(self.images[b:b+1, start:end])[0]
                np.testing.assert_allclose(actual[b, start:end], expected, atol=3e-5, rtol=1e-5)
        self.assertEqual(env.obs_space['slot'].shape, (7, 128))
        env.close()

    def test_input_normalization_and_layout(self):
        image = self.images[0, 0]
        expected, _ = self.extractor.get_slots(image)
        actual, _ = self.extractor.get_slots(image.astype(np.float32) / 255)
        # NumPy/Torch float32 division can differ by one rounding step.
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-5)
        chw, _ = self.extractor(image.transpose(2, 0, 1)[None])
        np.testing.assert_allclose(chw[0].numpy(), expected, atol=1e-6)
        resized, _ = self.extractor.get_slots(np.repeat(np.repeat(image, 2, 0), 2, 1))
        self.assertEqual(resized.shape, (7, 128))
        with self.assertRaisesRegex(ValueError, r'\[0,1\]'):
            self.extractor.get_slots(image.astype(np.float32))
        with self.assertRaisesRegex(ValueError, 'carry shape'):
            self.extractor.get_slots(image, np.zeros((6, 128)))

    def test_masks_and_training_report(self):
        from embodied.run.slot_viz import add_slot_mask_report
        slots, _ = self.extractor.get_slots(self.images[:, 0])
        decoded = self.extractor.decode_slots(slots)
        np.testing.assert_allclose(decoded['masks'].sum(1), 1, atol=1e-6)
        self.assertEqual(decoded['reconstructions'].shape, (2, 3, 64, 64))
        self.assertEqual(self.extractor.decode_masks(slots).shape, (2, 7, 4096))
        self.assertEqual(self.extractor.decode_masks(slots[0]).shape, (7, 4096))
        repeated = np.repeat(slots[:1], 5, axis=0)
        # CPU convolution kernels accumulate float32 sums differently for a
        # four-frame batch and its one-frame tail; masks are probabilities.
        np.testing.assert_allclose(self.extractor.decode_masks(repeated),
                                   self.extractor.decode_masks(slots[:1]).repeat(5, 0),
                                   atol=1e-5, rtol=1e-5)
        env = embodied.BatchSlotExtractorEnv(
            [FrameEnv(self.images[0], np.ones(8, bool))], False, self.extractor)
        batch = {'image': self.images[:1, :2], 'slot': slots[:1, None].repeat(2, 1)}
        report = add_slot_mask_report(env, {'model_slot_raw': batch['slot'].copy()}, batch)
        self.assertIn('openl_slots_grid_hard', report)
        env.close()

    def test_lfs_pointer_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pointer.ckpt'
            path.write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                SAViExtractor('', str(path), 64, 'cpu')

    def test_octssm_train_policy_report(self):
        self._octssm_integration(pe='window', kv_cache=False)

    def test_octssm_cached_train_policy_report(self):
        self._octssm_integration(pe='rope', kv_cache=True)

    def _octssm_integration(self, pe, kv_cache):
        import jax
        from dynalang.agent import Agent
        config = make_config().update({'octssm.pe': pe, 'octssm.kv_cache': kv_cache})
        env = FrameEnv(self.images[0], np.ones(8, bool))
        spaces = {**env.obs_space, 'slot': embodied.Space(np.float32, (7, 128))}
        agent = Agent(spaces, env.act_space, embodied.Counter(), config)
        self.assertEqual(agent.agent.wm.rssm._num_slots, 8)
        slots = self.reference(self.images)
        B, T = slots.shape[:2]
        data = {
            'image': self.images, 'slot': slots,
            'token': np.tile(np.arange(T, dtype=np.int32), (B, 1)),
            'action': np.random.default_rng(1).uniform(-1, 1, (B, T, 4)).astype(np.float32),
            'reward': np.tile(np.array([-0.1, 0, 0.1, -0.2, 0, 0.2, -0.1, 0], np.float32), (B, 1)),
            'is_first': np.zeros((B, T), bool), 'is_last': np.zeros((B, T), bool),
            'is_terminal': np.zeros((B, T), bool),
        }
        data['is_first'][:, 0] = True
        data['is_first'][0, 4] = True
        data['is_terminal'][0, 3] = True
        data['is_terminal'][:, -1] = True
        before = {k: v.copy() for k, v in agent.save().items() if '/rssm/' in k and 'opt' not in k}
        state = None
        for _ in range(2):
            batch = agent._convert_inps(data, agent.train_devices, rng=True)
            _, state, _ = agent.train(batch, state)
        after = agent.save()
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(after)
                            if np.issubdtype(v.dtype, np.floating)),
                        'Non-finite model/optimizer state after training')
        self.assertTrue(before, 'No RSSM parameters found')
        self.assertTrue(any(not np.array_equal(v, after[k]) for k, v in before.items()),
                        'OCTSSM weights did not update')
        obs = {k: v[:, 0] for k, v in data.items() if k != 'action'}
        action, policy_state = agent.policy(obs, None, mode='eval')
        self.assertEqual(action['action'].shape, (B, 4))
        self.assertTrue(np.isfinite(action['action']).all())
        obs = {k: v[:, 1] for k, v in data.items() if k != 'action'}
        agent.policy(obs, policy_state, mode='eval')
        report = agent.report(agent._convert_inps(data, agent.train_devices, rng=True))
        report = {k: np.asarray(jax.device_get(v)) for k, v in report.items()}
        scalars = {k: float(v) for k, v in report.items() if v.size == 1}
        # Class-conditional diagnostics may be NaN for empty bins. Require
        # actual objective losses, predictions and model outputs to be finite.
        losses = {k: v for k, v in scalars.items()
                  if k.endswith(('_loss_mean', '_loss_std'))}
        self.assertTrue(losses)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()), losses)
        self.assertTrue(np.isfinite(report['model_slot_raw']).all())
        self.assertIn('slot_loss_mean', scalars)
        self.assertEqual(report['model_slot_raw'].shape, (B, T, 7, 128))
        print(f'OCTSSM ({pe}, cache={kv_cache}): weights updated, policy finite, '
              f'slot loss={scalars["slot_loss_mean"]:.4f}', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
