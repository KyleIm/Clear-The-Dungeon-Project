import importlib.util
import unittest

if all(importlib.util.find_spec(package) for package in ("numpy", "torch")):
    import numpy as np
    import torch

    from combat_rl.ppo import ActorCritic, Transition, advantages, update
else:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and NumPy are required for RL tests")
class CombatRewardPPOTests(unittest.TestCase):
    def test_masking_with_both_heads(self):
        model = ActorCritic([8])
        features = np.zeros((2, 583), dtype=np.float32)
        phase = np.array([0, 1])
        # Reward head must ignore the two padding entries even if set by the caller.
        mask = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1]], dtype=bool)
        for deterministic in [False, True]:
            action, log_prob, value = model.act((features, phase, mask), deterministic=deterministic)
            np.testing.assert_array_equal(action, [6, 2])
            np.testing.assert_allclose(log_prob, [0, 0])
            self.assertEqual(value.shape, (2,))
        self.assertEqual(model.values((features, phase, np.zeros_like(mask))).shape, (2,))

    def test_gae_separates_slots_and_stops_at_episode_boundary(self):
        def entry(slot, value, reward, next_value, done):
            return Transition(None, 0, None, 0, 0, value, reward, next_value, done, slot)

        first = entry(0, 1, 2, 3, False)
        other_slot = entry(1, 2, 1, 0, True)
        cutoff = entry(0, 3, 1, 8, True)
        next_episode = entry(0, 2, -1, 0, True)
        values = advantages([first, other_slot, cutoff, next_episode], gamma=0.5, gae_lambda=0.5)
        np.testing.assert_allclose(values, [3.0, -1.0, 2.0, -3.0])

    def test_profile_and_bundled_action_readback_preserve_sampling(self):
        class Recorder:
            detailed = True

            def __init__(self):
                self.values = {}

            def add(self, name, elapsed):
                self.values[name] = elapsed

        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            with self.subTest(device=device):
                model = ActorCritic([8]).to(device)
                inputs = (np.zeros((2, 583), dtype=np.float32), np.array([0, 1]),
                          np.ones((2, 7), dtype=bool))
                recorder = Recorder()
                torch.manual_seed(83)
                plain = model.act(inputs)
                torch.manual_seed(83)
                bundled = model.act(inputs, profile=recorder, bundle=True)
                for actual, expected in zip(bundled, plain):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(model.values(inputs, profile=recorder), model.values(inputs))
                self.assertEqual(set(recorder.values), {'host_to_device', 'device_to_host',
                                                         'actor_critic_inference', 'critic_inference'})
                self.assertTrue(all(elapsed >= 0 for elapsed in recorder.values.values()))

    def _check_update(self, device):
        torch.manual_seed(19)
        model = ActorCritic([8, 4]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        rng = np.random.default_rng(21)
        features = rng.normal(size=(4, 583)).astype(np.float32)
        phases = np.array([0, 1, 0, 1])
        masks = np.ones((4, 7), dtype=bool)
        masks[phases == 1, 5:] = False
        actions, log_probs, values = model.act((features, phases, masks))
        rollout = [Transition(features[i], phases[i], masks[i], actions[i], log_probs[i], values[i],
                              i / 2, 0.0, True, i) for i in range(4)]
        before = [parameter.detach().clone() for parameter in model.parameters()]
        stats = update(model, optimizer, rollout, dict(gamma=0.995, gae_lambda=0.95, clip=0.2,
                                                       entropy=0.01, minibatch=4, epochs=1))
        self.assertTrue(all(np.isfinite(stats[key]) for key in ("policy_loss", "value_loss", "entropy")))
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_cpu_updates_both_heads_and_critic(self):
        self._check_update("cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_updates_both_heads_and_critic(self):
        self._check_update("cuda")
