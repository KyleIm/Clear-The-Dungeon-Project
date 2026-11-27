import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'Combat/Reward training dependencies required')
class WorkerTests(unittest.TestCase):
    def test_grouped_slots_follow_independent_trajectories_in_request_order(self):
        import numpy as np

        from combat_rl.environment import Run
        from combat_rl.workers import Workers

        timings = {}

        class Recorder:
            def add(self, name, seconds):
                timings.setdefault(name, []).append(seconds)

        requests = {slot: (100 + slot, 200 + slot, 4) for slot in (3, 0, 1, 2)}
        expected = {slot: Run(*args) for slot, args in requests.items()}
        with Workers(2, games_per_worker=2, profile=Recorder()) as pool:
            actual = pool.call('reset', requests)
            self.assertEqual(list(actual), list(requests))
            for slot, observed in actual.items():
                for field, value in zip(observed, expected[slot].observation()):
                    np.testing.assert_array_equal(field, value)
            self.assertEqual(pool.call('random', {}), {})
            for _ in range(2):
                subset = {slot: None for slot, run in expected.items() if not run.done}
                transitions = pool.call('random', subset)
                self.assertEqual(list(transitions), list(subset))
                for slot, (reward, delta, observation, result) in transitions.items():
                    run = expected[slot]
                    before = run.actions
                    self.assertEqual(reward, run.step(run.random_action()))
                    self.assertEqual(delta, run.actions - before)
                    if run.done:
                        self.assertIsNone(observation)
                        self.assertEqual(result, run.result())
                    else:
                        self.assertIsNone(result)
                        for field, value in zip(observation, run.observation()):
                            np.testing.assert_array_equal(field, value)
            snapshot = pool.call('snapshot', {3: None})
            pool.call('reset', {3: (9, 8, 1)})
            pool.call('restore', snapshot)
            restored = pool.call('snapshot', {3: None})[3]
            self.assertEqual(restored.actions, expected[3].actions)

        self.assertEqual(set(timings), {'worker_round', 'cpu_simulation', 'cpu_simulation_total', 'worker_ipc_sync'})
        self.assertEqual(len(timings['worker_round']), 2)
        for total, cpu, ipc in zip(timings['worker_round'], timings['cpu_simulation'], timings['worker_ipc_sync']):
            self.assertGreater(cpu, 0)
            self.assertGreaterEqual(ipc, 0)
            self.assertAlmostEqual(total, cpu + ipc)
