import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                         ('gymnasium', 'torch', 'matplotlib')), 'RL dependencies required')
class RunnerTests(unittest.TestCase):
    def test_training_validation_resume_and_report(self):
        from rl.runner import validate

        result = validate()
        self.assertEqual(result['steps'], 8)
        self.assertEqual(result['stage'], 2)
        self.assertTrue(result['report'])

    def test_split_seeds(self):
        from rl.runner import seed_for

        seeds = {seed_for(7, split, index) for split in ('game', 'validation', 'test')
                 for index in range(10)}
        self.assertEqual(len(seeds), 30)
