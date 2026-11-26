import importlib.util
import unittest

if all(importlib.util.find_spec(package) for package in ("numpy", "torch")):
    import numpy as np
    import torch

    from rl.ppo import ActorCritic, Rollout
else:
    torch = None


@unittest.skipIf(torch is None, "Install the optional RL requirements.")
class MaskedPPOTests(unittest.TestCase):
    def test_greedy_and_random_decisions_use_the_active_head_and_mask(self):
        model = ActorCritic(2, [0, 1, 2, 3, 0, 1, 2, 3], [4])
        obs = {
            "features": torch.zeros(4, 2),
            "phase": torch.arange(4),
            "action_mask": torch.eye(8, dtype=torch.bool)[4:],
        }
        for deterministic in (True, False):
            action, log_prob, value = model.act(obs, deterministic=deterministic)
            torch.testing.assert_close(action, torch.arange(4, 8))
            torch.testing.assert_close(log_prob, torch.zeros(4))
            self.assertEqual(value.shape, (4,))
        torch.testing.assert_close(model.greedy_actions(obs), torch.arange(4, 8))
        for draws in (None, [0, 0.1, 0.9, 0.999999999]):
            action, probabilities = model.evaluation_actions(obs, draws)
            torch.testing.assert_close(action, torch.arange(4, 8))
            torch.testing.assert_close(probabilities, obs['action_mask'].float())
        obs['action_mask'].fill_(True)
        legacy, _, _ = model.act(obs, deterministic=True)
        torch.testing.assert_close(model.greedy_actions(obs), legacy)
        single = torch.cat([model.greedy_actions({key: value[i:i + 1] for key, value in obs.items()})
                            for i in range(4)])
        torch.testing.assert_close(model.greedy_actions(obs), single)

    def test_gae_bootstraps_a_cutoff_without_leaking_into_next_episode(self):
        rollout = Rollout(3, 1, 2, 4)
        obs = {
            "features": np.zeros((1, 2), dtype=np.float32),
            "phase": np.zeros(1, dtype=np.int64),
            "action_mask": np.ones((1, 4), dtype=bool),
        }
        rollout.add(obs, [0], [0], [1], [2], [3], [False])
        rollout.add(obs, [0], [0], [3], [1], [8], [True])
        rollout.add(obs, [0], [0], [2], [-1], [0], [True])
        advantages, returns = rollout.compute_advantages(gamma=0.5, gae_lambda=0.5)
        np.testing.assert_allclose(advantages[:, 0], [3.0, 2.0, -3.0])
        np.testing.assert_allclose(returns[:, 0], [4.0, 5.0, -1.0])

    def test_seeded_draws_are_slot_local_and_masked(self):
        model = ActorCritic(2, [0, 1, 2, 3, 0], [4])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        obs = {'features': torch.zeros(2, 2), 'phase': torch.zeros(2, dtype=torch.long),
               'action_mask': torch.tensor([[True, False, False, False, True]] * 2)}
        action, mass = model.evaluation_actions(obs, [0.25, 0.75])
        torch.testing.assert_close(action, torch.tensor([0, 4]))
        torch.testing.assert_close(mass, obs['action_mask'].float() / 2)
        reversed_obs = {key: value.flip(0) for key, value in obs.items()}
        reversed_actions, _ = model.evaluation_actions(reversed_obs, [0.75, 0.25])
        torch.testing.assert_close(reversed_actions, action.flip(0))

    def _check_update(self, device):
        torch.manual_seed(11)
        model = ActorCritic(2, [0, 1, 2, 3, 0, 1, 2, 3], [8, 4]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        obs = {
            "features": torch.tensor([[1.0, -1.0]] * 4, device=device),
            "phase": torch.arange(4, device=device),
            "action_mask": torch.eye(4, dtype=torch.bool, device=device).repeat(1, 2),
        }
        action, log_prob, value = model.act(obs)
        rollout = Rollout(1, 4, 2, 8)
        rollout.add(
            {key: tensor.cpu().numpy() for key, tensor in obs.items()},
            action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy(),
            [1, 0, 1, 0], [0, 0, 0, 0], [True, True, True, True],
        )
        before = [param.detach().clone() for param in model.parameters()]
        metrics = rollout.update(model, optimizer, {
            "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.01,
            "batch_size": 4, "clip_range": 0.2, "update_epochs": 1,
        }, device)
        self.assertEqual(metrics["phase_steps"], dict.fromkeys(
            ("combat", "reward", "shop", "event"), 1,
        ))
        self.assertTrue(all(np.isfinite(metrics[key]) for key in ("policy_loss", "value_loss")))
        self.assertTrue(all(value > 0 for value in metrics["phase_entropy"].values()))
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_cpu_optimizer_updates_all_heads_and_critic(self):
        self._check_update("cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_optimizer_updates_all_heads_and_critic(self):
        self._check_update("cuda")


if __name__ == "__main__":
    unittest.main()
